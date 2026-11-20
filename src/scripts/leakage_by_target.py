"""scripts/leakage_by_target.py — which pool molecules the encoder saw measured on the SAME target.

The leakage audit (Methodology Table 3.5) counts a pool molecule as leaked if it occurs
anywhere in the encoder's training split. That is
not the problematic case: a molecule measured against an unrelated protein tells the encoder
nothing about its activity on the benchmark target. Leakage that matters is a (molecule,
target) pair, the same molecule measured against the pool's own target.

For every training corpus (training split only, as the encoder saw it) and every pool this
reports the fraction of pool molecules that occur
  any_assay      in any training assay (the current Table 3.5 number)
  same_target    in a training assay on the pool's own ChEMBL target
  same_organism  S. aureus only: in any assay on the organism or one of its proteins
Both sides are canonicalised with RDKit before matching.

The domain-matched corpora are not included: benchmark molecules were removed from them at
construction, so all three numbers are zero by design.

Output: experiments/leakage_by_target.csv (refuses to overwrite).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import polars as pl
from rdkit import Chem, RDLogger

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
# pool name -> (csv, SMILES column, own ChEMBL target id). The ids were checked against
# the data: they are the targets holding the most TYK2 / USP7 pool molecules, and the
# S. aureus file's only target. CHEMBL217 is the human dopamine D2 receptor.
POOLS = {
    "tyk2": ("data/curated/TYK2.csv", "SMILES", "CHEMBL3553"),
    "usp7": ("data/curated/USP7.csv", "SMILES", "CHEMBL2157850"),
    "d2r": ("data/curated/D2R.csv", "SMILES", "CHEMBL217"),
    "saureus_original": ("data/curated/S_aureus.csv", "Smiles", "CHEMBL352"),
    "saureus_corrected": ("data/curated/S_aureus_corrected.csv", "Smiles", "CHEMBL352"),
}
SAUREUS_ORGANISM = "Staphylococcus aureus"
ALL_CHEMBL = "data/curated/chembl36_curated_clean.parquet"
SINGLE_PROTEIN = "data/curated/chembl36_single_protein_clean.parquet"
SINGLE_PROTEIN_SOURCE = "data/curated/chembl36_curated_single_protein.csv"   # has target ids
OUTPUT_CSV = "experiments/leakage_by_target.csv"


def canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def canonicalise(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a `can` column (RDKit canonical SMILES), computed once per distinct input."""
    distinct = frame["canonical_smiles"].unique().to_list()
    mapping = pl.DataFrame({"canonical_smiles": distinct, "can": [canon(s) for s in distinct]})
    return frame.join(mapping, on="canonical_smiles", how="left")


def training_pairs() -> dict[str, pl.DataFrame]:
    """Per corpus: the distinct (molecule, target, organism) triples of the training split."""
    cols = ["canonical_smiles", "target_chembl_id", "assay_organism"]
    all_chembl = (pl.scan_parquet(REPO_ROOT / ALL_CHEMBL).filter(pl.col("split") == "train")
                  .select(cols).unique().collect())
    sp = (pl.scan_parquet(REPO_ROOT / SINGLE_PROTEIN).filter(pl.col("split") == "train")
          .select("assay_x_type_x_doc", "canonical_smiles").unique().collect())
    meta = (pl.scan_csv(REPO_ROOT / SINGLE_PROTEIN_SOURCE, infer_schema_length=10000)
            .select("assay_x_type_x_doc", "target_chembl_id", "assay_organism").unique()
            .collect())
    if meta["assay_x_type_x_doc"].n_unique() != len(meta):
        raise SystemExit("an assay key maps to several targets in the single-protein source")
    single = sp.join(meta, on="assay_x_type_x_doc", how="left")
    missing = single["target_chembl_id"].null_count()
    if missing:
        raise SystemExit(f"{missing} single-protein rows have no target id after the join")
    return {"all_chembl": all_chembl, "single_protein": single.select(cols).unique()}


def main() -> None:
    out = REPO_ROOT / OUTPUT_CSV
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    corpora = {name: canonicalise(df) for name, df in training_pairs().items()}
    rows = []
    for pool, (path, col, target) in POOLS.items():
        smiles = pl.read_csv(REPO_ROOT / path, infer_schema_length=10000)[col].unique().to_list()
        pool_can = {c for c in (canon(s) for s in smiles) if c is not None}
        for corpus, df in corpora.items():
            seen = df.filter(pl.col("can").is_in(list(pool_can)))
            any_assay = seen["can"].n_unique()
            same_target = seen.filter(pl.col("target_chembl_id") == target)["can"].n_unique()
            row = dict(corpus=corpus, pool=pool, target=target, pool_molecules=len(pool_can),
                       any_assay=any_assay, same_target=same_target,
                       any_assay_frac=f"{any_assay / len(pool_can):.4f}",
                       same_target_frac=f"{same_target / len(pool_can):.4f}")
            if pool.startswith("saureus"):
                org = seen.filter((pl.col("target_chembl_id") == target)
                                  | pl.col("assay_organism").str.starts_with(SAUREUS_ORGANISM))
                row["same_organism"] = org["can"].n_unique()
                row["same_organism_frac"] = f"{row['same_organism'] / len(pool_can):.4f}"
            # Which targets hold the leaked molecules, for the text (top three).
            top = (seen.group_by("target_chembl_id").agg(pl.col("can").n_unique().alias("n"))
                   .sort("n", descending=True).head(3).rows())
            row["top_targets"] = "; ".join(f"{t} ({n})" for t, n in top)
            rows.append(row)
            print(f"{corpus:15s} {pool:18s} any {row['any_assay_frac']:>7}  same target "
                  f"{row['same_target_frac']:>7}"
                  + (f"  same organism {row['same_organism_frac']:>7}" if "same_organism" in row else "")
                  + f"  | top: {row['top_targets']}")
    fields = ["corpus", "pool", "target", "pool_molecules", "any_assay", "any_assay_frac",
              "same_target", "same_target_frac", "same_organism", "same_organism_frac",
              "top_targets"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
