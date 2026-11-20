"""scripts/build_saureus_corrected.py — the S. aureus pool with the pMIC its curator defined.

The supervisor's file (data/raw/s_aureus_curated_with_scores_filtered.csv; data/curated/
S_aureus.csv is a byte-identical copy) describes `pMIC` as -log10(MIC in mol/L), "higher is
better". The column actually holds log10(MIC + 1) with MIC in ug/mL, which INCREASES with MIC.
The benchmark maximised it with hits = pMIC >= 2.41 (MIC >= ~256 ug/mL), i.e. the least active
5 % of the pool: none of those molecules is marked susceptible in the file. The pool was also
meant to contain exact measurements only, but no step ever filtered on `Standard Relation`.
See notes/saureus-label-correction.md.

This script rebuilds the pool as intended, leaving both source files untouched:
  1. keep exact measurements only (Standard Relation "="); censored values ("> 256", "< 0.5")
     are bounds, not measurements
  2. pMIC = -log10(MIC [mol/L]),  MIC [mol/L] = max(MIC, MIC_FLOOR) [ug/mL] * 1e-3 / MW [g/mol],
     i.e. pMIC = -log10(MIC [ug/mL] / MW) + 3 (the supervisor's form, 2026-10-07; pMIC 5 = 10 uM).
     MW is RDKit's average molecular weight (Descriptors.MolWt) of the SMILES. The file's own
     `mol_weight` column is the exact (monoisotopic) mass, which differs by up to 6 g/mol
     (at most 0.004 in pMIC).
  3. data.load_target() then de-duplicates by median per SMILES, as for every pool
Hits: the top 5 % by pMIC (hit_top_frac=0.05), the rule used for TYK2, USP7 and D2R.

Output: data/curated/S_aureus_corrected.csv (refuses to overwrite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
SOURCE_CSV = "data/raw/s_aureus_curated_with_scores_filtered.csv"   # the supervisor's file
OUTPUT_CSV = "data/curated/S_aureus_corrected.csv"
HIT_TOP_FRAC = 0.05
# Exact MICs below 0.001 ug/mL lie below standard broth-dilution series; 13 measurements report
# MIC = 0 (presumably tiny values rounded down), for which -log10 would be infinite. They are
# capped here. All of them are hits either way.
MIC_FLOOR = 1e-3                      # ug/mL
KEEP = ["Smiles", "Standard Value", "Standard Units", "Standard Relation", "mol_weight",
        "Assay ChEMBL ID", "Document ChEMBL ID", "is_susceptible", "pMIC"]


def main() -> None:
    out = REPO_ROOT / OUTPUT_CSV
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    raw = pl.read_csv(REPO_ROOT / SOURCE_CSV, infer_schema_length=10000)

    # Check the source is what this script assumes, so a changed file cannot slip through.
    units = raw["Standard Units"].unique().to_list()
    if units != ["ug.mL-1"]:
        raise SystemExit(f"expected every MIC in ug/mL, found {units}")
    v = raw["Standard Value"].cast(pl.Float64).to_numpy()
    p = raw["pMIC"].cast(pl.Float64).to_numpy()
    ok = ~np.isnan(v) & ~np.isnan(p)
    if not np.allclose(p[ok], np.log10(v[ok] + 1), atol=1e-6):
        raise SystemExit("the source pMIC is no longer log10(MIC + 1); re-check before using")

    exact = raw.filter((pl.col("Standard Relation") == "'='") & pl.col("Standard Value").is_not_null())
    mic = exact["Standard Value"].cast(pl.Float64)
    # Average molecular weight in g/mol, from the SMILES (one RDKit parse per distinct molecule).
    RDLogger.DisableLog("rdApp.*")
    mw_of = {}
    for smi in exact["Smiles"].unique().to_list():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise SystemExit(f"RDKit cannot parse {smi!r}")
        mw_of[smi] = Descriptors.MolWt(mol)
    mw = exact["Smiles"].replace_strict(mw_of, return_dtype=pl.Float64)
    df = (exact.select(KEEP)
          .rename({"pMIC": "pMIC_original_log10_mic_plus_1"})
          .with_columns(mw.alias("mw_rdkit_average"))
          .with_columns((-(mic.clip(lower_bound=MIC_FLOOR) * 1e-3 / mw).log10()).alias("pMIC")))
    df.write_csv(out)

    smiles, y = data.load_target(out, smiles_col="Smiles", target_col="pMIC")
    y = y.numpy()
    k = max(1, round(HIT_TOP_FRAC * len(y)))
    hits = np.argsort(-y, kind="stable")[:k]
    sus = (df.group_by("Smiles", maintain_order=True).agg(pl.col("is_susceptible").mean())
           ["is_susceptible"].to_numpy())
    print(f"wrote {out}\n"
          f"  {len(raw):,} measurements -> {len(exact):,} exact ('=') "
          f"({len(raw) - len(exact):,} censored or missing dropped)\n"
          f"  {len(smiles):,} molecules after de-duplication by median\n"
          f"  MIC capped at {MIC_FLOOR} ug/mL on {int((mic < MIC_FLOOR).sum())} measurements\n"
          f"  pMIC range {y.min():.2f} to {y.max():.2f}, median {np.median(y):.2f}\n"
          f"  hits = top {HIT_TOP_FRAC:.0%}: {k:,} molecules, pMIC >= {y[hits].min():.3f}; "
          f"marked susceptible: {sus[hits].mean():.1%} (rest of the pool: "
          f"{np.delete(sus, hits).mean():.1%})")


if __name__ == "__main__":
    main()
