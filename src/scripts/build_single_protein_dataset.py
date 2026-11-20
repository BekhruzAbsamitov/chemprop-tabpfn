"""scripts/build_single_protein_dataset.py — the 2026-08-20 curated set. Press Run.

Converts the supervisor's `chembl36_curated_single_protein.csv` into the 4-column
parquet `data_utils/data.py` loads. That file is a much stricter cut of ChEMBL 36 than
the one every previous run used — binding assays only, single-protein targets only,
confidence 9 only, IC50/Ki/Kd only, >= 5 measurements per assay, all SMILES RDKit-parseable.

WHY IT MATTERS: 39.5k assays against ~180k before, so this is *less* data but far cleaner
— and it carries almost no label leakage against the AL benchmark pools. Overlap of the
training split with each pool, full ChEMBL -> this corpus, as a fraction of the pool
(src/scripts/leakage_by_target.py, RDKit canonical SMILES on both sides):

                 any training assay       assay on the pool's own target
    S. aureus    77.2 %  ->   4.4 %       58.8 %  ->  0.0 %
    TYK2          0.15 % ->   0.15 %       0.15 % ->  0.15 %
    USP7          4.4 %  ->   2.9 %        3.7 %  ->  2.6 %
    D2R          91.1 %  ->  91.4 %        3.8 %  ->  4.4 %

Only the right-hand column can leak benchmark labels, so this corpus removes the one real
leakage of the full-ChEMBL encoder (S. aureus). The overlap once quoted here (D2R 1.2 %,
USP7 and TYK2 0 %) was a raw-string match, which misses molecules the Gorantla pools
write in non-canonical form.

TWO DECISIONS THIS SCRIPT MAKES, both deliberate:

1. **It ignores the file's own `test` column and builds an ASSAY-LEVEL split instead.**
   The supplied `test` flag is a *molecule-level* split: all 39,837 assays appear in both
   halves. That suits ordinary supervised training, but our task is meta-learning —
   `make_episode` draws a context and a query from ONE assay, and validation has to ask
   whether the encoder helps on an assay it has never seen. Splitting by molecule would
   let the encoder train on every assay it is later validated against, making validation
   far easier and risking a false positive on RQ4. So assays are partitioned whole, and
   the script asserts afterwards that none spans both splits.

2. **It drops rows whose label is a raw `standard_value`.** 17,434 rows (2 %) carry
   untransformed values ranging to 3.4e13, against a pchembl median of 7.0. TabPFN
   z-normalises per episode, so a single such row would dominate its episode and the
   +-5 clip would swallow the rest. `pchembl_value` and `log10_transformed` are both on
   sane log scales and are kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
SOURCE_CSV = "data/curated/chembl36_curated_single_protein.csv"
DEST_PARQUET = "data/curated/chembl36_single_protein_clean.parquet"

ASSAY_COL = "assay_x_type_x_doc"    # what data.py keys episodes on
SMILES_COL = "canonical_smiles"
TARGET_COL = "target_transformed"
SPLIT_COL = "split"

# Label scales we keep — see decision 2 in the module docstring.
KEEP_TARGET_TYPES = ["pchembl_value", "log10_transformed"]

TEST_FRAC = 0.10      # assay-level holdout, matching the adaptive datasets' 90/10
SPLIT_SEED = 0        # fixed so the split is reproducible run to run
MIN_ASSAY_SIZE = 5    # the source already guarantees this; asserted, not applied


def main() -> None:
    src = REPO_ROOT / SOURCE_CSV
    if not src.exists():
        raise SystemExit(f"source not found: {src}")

    print(f"reading {src.name} ...")
    df = pl.read_csv(
        src,
        columns=[ASSAY_COL, SMILES_COL, TARGET_COL, "target_transformed_type",
                 "assay_chembl_id", "target_chembl_id"],
        infer_schema_length=10000,
    )
    print(f"  {df.height:,} rows, {df[ASSAY_COL].n_unique():,} assays")

    # 1. Keep only sanely-scaled labels, and drop anything null in the columns we need.
    before = df.height
    df = (df.filter(pl.col("target_transformed_type").is_in(KEEP_TARGET_TYPES))
            .drop_nulls([SMILES_COL, TARGET_COL]))
    print(f"  label-scale filter: {before:,} -> {df.height:,} rows "
          f"(dropped {before - df.height:,} raw standard_value / null)")

    # 2. Assays can lose rows to the filter above, so re-apply the >= 5 rule.
    sizes = df.group_by(ASSAY_COL).len()
    keep = sizes.filter(pl.col("len") >= MIN_ASSAY_SIZE)[ASSAY_COL].implode()
    df = df.filter(pl.col(ASSAY_COL).is_in(keep))
    print(f"  >= {MIN_ASSAY_SIZE} measurements: {df[ASSAY_COL].n_unique():,} assays kept")

    # 3. ASSAY-LEVEL split — the whole point of this script (decision 1 above).
    assays = df[ASSAY_COL].unique().sort()
    shuffled = assays.to_frame().sample(fraction=1.0, shuffle=True, seed=SPLIT_SEED)
    n_test = int(round(shuffled.height * TEST_FRAC))
    test_assays = set(shuffled[ASSAY_COL][:n_test].to_list())
    df = df.with_columns(
        pl.when(pl.col(ASSAY_COL).is_in(test_assays))
          .then(pl.lit("test")).otherwise(pl.lit("train")).alias(SPLIT_COL)
    )

    # 4. The assertion that makes decision 1 real rather than intended.
    spanning = (df.group_by(ASSAY_COL)
                  .agg(pl.col(SPLIT_COL).n_unique().alias("n"))
                  .filter(pl.col("n") > 1).height)
    assert spanning == 0, f"{spanning} assays span both splits — split is not assay-level"

    out = df.select([ASSAY_COL, SMILES_COL, TARGET_COL, SPLIT_COL])
    dest = REPO_ROOT / DEST_PARQUET
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)

    print("\n--- written ---")
    for split in ("train", "test"):
        s = out.filter(pl.col(SPLIT_COL) == split)
        print(f"  {split:5s} {s.height:8,} rows  {s[ASSAY_COL].n_unique():6,} assays")
    print(f"  no assay spans both splits: verified")
    print(f"  label range: {out[TARGET_COL].min():.2f} .. {out[TARGET_COL].max():.2f}"
          f"  (median {out[TARGET_COL].median():.2f})")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
