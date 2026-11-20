"""scripts/build_adaptive_dataset.py — build a target-specific ChEMBL training set.

Takes one inference target (its active-learning compounds) + a protein class, and carves a
domain-matched, leakage-free training set out of full ChEMBL.

Steps:
  1. Per-target assay selection: protein class (gpcr/protease) + confidence 9 +
     standard_type in {IC50, Ki, Kd} + target_transformed_type == pchembl_value.
  2. Structure-similarity adaptation (assay-level drop): drop an assay if its max
     ECFP4 Tanimoto to the AL compounds < 0.40, or its median MW falls outside the
     AL MW window [Q25 - 100, Q75 + 100] Da.
  3. Exact-structure leakage: remove any AL compound from every kept assay.
  4. Tidy: drop assays below MIN_ASSAY_SIZE; assay-level 90/10 train/test split.

Output: a parquet in the loader's schema (assay_x_type_x_doc, canonical_smiles,
target_transformed, split) — a drop-in CHEMBL_FULL_FILE for meta-training this
target. Run once per target. Settings are constants below. No flags / env vars — open and press Run.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402  (schema constants: ASSAY_COL, SMILES_COL, ...)

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule parse warnings

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
TARGET_NAME = "D2R"
AL_COMPOUNDS_CSV = "data/curated/D2R.csv"     # the inference / active-learning compounds
AL_SMILES_COL = "SMILES"                       # "SMILES" (D2R, USP7) | "Smiles" (S_aureus)

# --- assay selection (each filter is skipped when set to None) ---
# Protein-class targets (D2R=gpcr, USP7=protease) use the classification join +
# pchembl. Organism targets (S. aureus) have no protein class — select by organism
# + assay type + endpoint instead, and rely on leakage removal (below) to split the
# pre-training pool disjointly from the AL benchmark.
PROTEIN_CLASS = "gpcr"                          # "gpcr" | "protease" | None
ASSAY_ORGANISM_CONTAINS = None                  # e.g. "Staphylococcus aureus" | None
ASSAY_TYPE = None                               # e.g. "F" (functional/whole-cell) | None
CONFIDENCE_SCORE = 9                            # assay_confidence_score == this | None
STANDARD_TYPES = {"IC50", "Ki", "Kd"}          # {} / None to skip
TARGET_TRANSFORMED_TYPES = {"pchembl_value"}   # keeps the target scale consistent
# Row-level clip on the target value, e.g. (0, 12) for pMIC — the global
# |target|<=100 clean is far too loose for pMIC (which is normally ~2-9, but has
# unit-error outliers). None = no extra clip (pchembl_value is already sane).
TARGET_RANGE = None

# --- structure-similarity adaptation (Tanimoto + MW window) ---
# ON for foreign-pool targets (keeps only assays near the AL chemistry). OFF for a
# same-organism pool (S. aureus), where the pool is already on-target.
APPLY_STRUCTURE_ADAPTATION = True
TANIMOTO_MIN = 0.40                             # drop assay if max sim to AL < this
MW_MARGIN_DA = 100.0                            # median-MW window is [Q25 - m, Q75 + m]
FP_RADIUS, FP_BITS = 2, 2048                    # ECFP4
MIN_ASSAY_SIZE = 5                              # drop assays smaller than this after leakage

TEST_FRAC = 0.10                                # assay-level held-out fraction
SPLIT_SEED = 0

TARGET_CLASSIFICATION_CSV = "notes/target_classification.csv"  # target_chembl_id -> protein_class
OUTPUT_PARQUET = "data/curated/train_adaptive_D2R.parquet"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.time()
    sel = PROTEIN_CLASS or ASSAY_ORGANISM_CONTAINS or "all"
    print(f"=== adaptive dataset for {TARGET_NAME} (select={sel}, "
          f"adaptation={'on' if APPLY_STRUCTURE_ADAPTATION else 'off'}) ===")

    # --- AL compounds: canonical set (+ fingerprints/MW if adapting) -------- #
    al = pl.read_csv(_resolve(AL_COMPOUNDS_CSV))
    al_raw = al[AL_SMILES_COL].drop_nulls().unique().to_list()
    al_canon: set[str] = set()
    al_fps, al_mw = [], []
    for s in al_raw:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        al_canon.add(Chem.MolToSmiles(mol))
        if APPLY_STRUCTURE_ADAPTATION:
            al_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS))
            al_mw.append(Descriptors.MolWt(mol))
    if not al_canon:
        raise SystemExit("No valid AL compounds parsed — check AL_COMPOUNDS_CSV / AL_SMILES_COL.")
    mw_lo = mw_hi = None
    if APPLY_STRUCTURE_ADAPTATION:
        q25, q75 = (float(x) for x in np.quantile(al_mw, [0.25, 0.75]))
        mw_lo, mw_hi = q25 - MW_MARGIN_DA, q75 + MW_MARGIN_DA
        print(f"AL: {len(al_canon)} valid compounds | MW window [{mw_lo:.0f}, {mw_hi:.0f}] Da "
              f"(Q25={q25:.0f}, Q75={q75:.0f})")
    else:
        print(f"AL: {len(al_canon)} valid compounds (structure adaptation off)")

    # --- 1. Assay selection (each filter skipped when its setting is None) -- #
    lf = pl.read_parquet(
        _resolve(str(data.CHEMBL_FULL_FILE)),
        columns=[data.ASSAY_COL, data.SMILES_COL, data.TARGET_COL,
                 "target_transformed_type", "standard_type", "assay_confidence_score",
                 "assay_type", "assay_organism", "target_chembl_id"],
    )
    conds = []
    if PROTEIN_CLASS is not None:
        cls = pl.read_csv(_resolve(TARGET_CLASSIFICATION_CSV)).select(
            ["target_chembl_id", "protein_class"])
        lf = lf.join(cls, on="target_chembl_id", how="left")
        conds.append(pl.col("protein_class") == PROTEIN_CLASS)
    if ASSAY_ORGANISM_CONTAINS is not None:
        conds.append(pl.col("assay_organism").str.contains(f"(?i){ASSAY_ORGANISM_CONTAINS}"))
    if ASSAY_TYPE is not None:
        conds.append(pl.col("assay_type") == ASSAY_TYPE)
    if CONFIDENCE_SCORE is not None:
        conds.append(pl.col("assay_confidence_score") == CONFIDENCE_SCORE)
    if STANDARD_TYPES:
        conds.append(pl.col("standard_type").is_in(list(STANDARD_TYPES)))
    if TARGET_TRANSFORMED_TYPES:
        conds.append(pl.col("target_transformed_type").is_in(list(TARGET_TRANSFORMED_TYPES)))
    combined = conds[0]
    for c in conds[1:]:
        combined = combined & c
    df = lf.filter(combined).drop_nulls([data.SMILES_COL, data.TARGET_COL])
    if TARGET_RANGE is not None:
        lo, hi = TARGET_RANGE
        n_pre = len(df)
        df = df.filter(pl.col(data.TARGET_COL).is_between(lo, hi))
        print(f"   target clip to [{lo}, {hi}]: dropped {n_pre - len(df)} rows")
    print(f"1. selection: {df[data.ASSAY_COL].n_unique():>6} assays, {len(df):>8} rows")
    if len(df) == 0:
        raise SystemExit("Selection is empty — check the selection filters.")

    # --- 2. Per unique candidate compound: canonical SMILES (+ sim/MW) ------ #
    uniq = df[data.SMILES_COL].unique().to_list()
    rows = []
    for i, s in enumerate(uniq):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if APPLY_STRUCTURE_ADAPTATION:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
            max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, al_fps))
            rows.append((s, canon, max_sim, Descriptors.MolWt(mol)))
            if (i + 1) % 5000 == 0:
                print(f"   fingerprinted {i + 1}/{len(uniq)} unique compounds  ({time.time() - t0:.0f}s)")
        else:
            rows.append((s, canon))

    if APPLY_STRUCTURE_ADAPTATION:
        prop_df = pl.DataFrame(rows, schema=["_raw", "_canon", "_maxsim", "_mw"], orient="row")
        df = df.join(prop_df, left_on=data.SMILES_COL, right_on="_raw", how="inner")
        assay_stats = df.group_by(data.ASSAY_COL).agg([
            pl.col("_maxsim").max().alias("max_sim"),
            pl.col("_mw").median().alias("median_mw"),
        ])
        pass_tan = assay_stats.filter(pl.col("max_sim") >= TANIMOTO_MIN)
        pass_both = pass_tan.filter(
            (pl.col("median_mw") >= mw_lo) & (pl.col("median_mw") <= mw_hi))
        keep_assays = pass_both[data.ASSAY_COL].to_list()
        df = df.filter(pl.col(data.ASSAY_COL).is_in(keep_assays))
        print(f"2. Tanimoto>= {TANIMOTO_MIN}: {pass_tan.height:>6} assays kept "
              f"| + MW window: {pass_both.height:>6} assays, {len(df):>8} rows")
    else:
        prop_df = pl.DataFrame(rows, schema=["_raw", "_canon"], orient="row")
        df = df.join(prop_df, left_on=data.SMILES_COL, right_on="_raw", how="inner")
        print(f"2. structure adaptation skipped (same-organism pool)")

    # --- 3. Exact-structure leakage removal -------------------------------- #
    n_before = len(df)
    df = df.filter(~pl.col("_canon").is_in(list(al_canon)))
    print(f"3. leakage: removed {n_before - len(df)} compound-rows present in AL set")

    # --- 4. Tidy: min assay size + assay-level split ----------------------- #
    sizes = df.group_by(data.ASSAY_COL).agg(pl.len().alias("n"))
    big = sizes.filter(pl.col("n") >= MIN_ASSAY_SIZE)[data.ASSAY_COL].to_list()
    df = df.filter(pl.col(data.ASSAY_COL).is_in(big))

    assays = sorted(df[data.ASSAY_COL].unique().to_list())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(assays)
    n_test = max(1, int(round(len(assays) * TEST_FRAC)))
    test_set = set(assays[:n_test])
    df = df.with_columns(
        pl.when(pl.col(data.ASSAY_COL).is_in(list(test_set)))
        .then(pl.lit("test")).otherwise(pl.lit("train")).alias(data.SPLIT_COL)
    )
    n_train = len(assays) - n_test
    print(f"4. tidy: {len(assays)} assays >= {MIN_ASSAY_SIZE} mols "
          f"({n_train} train / {n_test} test), {len(df)} rows")

    # --- Write in the loader's schema (RDKit-canonical SMILES) ------------- #
    out = df.select([
        pl.col(data.ASSAY_COL),
        pl.col("_canon").alias(data.SMILES_COL),
        pl.col(data.TARGET_COL),
        pl.col(data.SPLIT_COL),
    ])
    out_path = _resolve(OUTPUT_PARQUET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}  "
          f"({len(out)} rows, {out[data.ASSAY_COL].n_unique()} assays)  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
