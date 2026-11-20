"""scripts/clean_chembl_targets.py — remove raw/untransformed-scale assays. Press Run.

WHY
    The ChEMBL `target_transformed` column is NOT consistently transformed: ~90%
    of assays are on a clean log scale (values ~0-15) or a percent scale (0-100),
    but ~10% are on RAW scales with values up to 10^22. Those raw assays make
    poor training episodes: TabPFN z-normalizes each assay by its own std, and a
    few astronomical values collapse the whole episode to the ±clip, giving the
    encoder a near-zero-information gradient.

WHAT THIS DOES
    Assay-level filter: DROP an entire assay if ANY of its transformed values
    falls outside [-MAX_ABS_TARGET, +MAX_ABS_TARGET]. We drop the whole assay (not
    just the offending molecules) so we never keep a truncated fragment of a
    raw-scale assay. Clean log assays (0-15) and percent assays (0-100) are
    untouched. Assays never cross the train/val/test split, so filtering by assay
    is automatically split-consistent.

OUTPUT
    Writes a cleaned copy of each input parquet (same 42-column schema, rows for
    dropped assays removed) next to the original, and prints a before/after
    summary so you can verify what was kept.

All settings are constants below. No flags / env vars — open and press Run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
MAX_ABS_TARGET = 100.0  # keep an assay only if all its values lie in [-this, +this]

# Fold the validation split into train, leaving a 2-way train/test split (the
# external targets — S. aureus, D2R, USP7 — are the real held-out evaluation, and
# ChEMBL 'test' serves as the in-domain held-out set, so a separate 'val' is
# redundant here).
MERGE_VAL_INTO_TRAIN = True

# (input parquet, output parquet, slim?) — slim drops all but KEEP_COLUMNS.
FILES = [
    ("data/curated/chembl36_curated.parquet", "data/curated/chembl36_curated_clean.parquet", False),
    ("data/curated/chembl36_dev.parquet",     "data/curated/chembl36_dev_clean.parquet",     True),
]

# Column names (must match src/data_utils/data.py).
ASSAY_COL = "assay_x_type_x_doc"
SMILES_COL = "canonical_smiles"
TARGET_COL = "target_transformed"
SPLIT_COL = "split"

# When slimming, keep only the columns the pipeline actually reads.
KEEP_COLUMNS = [ASSAY_COL, SMILES_COL, TARGET_COL, SPLIT_COL]


def _split_summary(lf: pl.LazyFrame, label: str) -> None:
    """Print rows + unique assays per split for a (lazy) frame."""
    summ = (
        lf.group_by(SPLIT_COL)
        .agg(pl.len().alias("rows"), pl.col(ASSAY_COL).n_unique().alias("assays"))
        .sort(SPLIT_COL)
        .collect()
    )
    print(f"  {label}:")
    for row in summ.iter_rows(named=True):
        print(f"    {str(row[SPLIT_COL]):5s}: {row['rows']:>10,} rows | {row['assays']:>8,} assays")


def clean_one(inp: Path, out: Path, slim: bool) -> None:
    print(f"\n=== {inp.name}  ->  {out.name} ===")
    if not inp.exists():
        print(f"  [skip] input not found: {inp}")
        return

    lf = pl.scan_parquet(inp)

    # 1. Per-assay min/max of the transformed target; keep only in-range assays.
    good = (
        lf.group_by(ASSAY_COL)
        .agg(pl.col(TARGET_COL).min().alias("mn"), pl.col(TARGET_COL).max().alias("mx"))
        .filter((pl.col("mx") <= MAX_ABS_TARGET) & (pl.col("mn") >= -MAX_ABS_TARGET))
        .select(ASSAY_COL)
        .collect()
    )
    good_ids = good[ASSAY_COL].to_list()

    # 2. Keep every row belonging to a good assay.
    cleaned = lf.filter(pl.col(ASSAY_COL).is_in(good_ids))

    # 3. Fold 'val' into 'train' -> a 2-way train/test split.
    if MERGE_VAL_INTO_TRAIN:
        cleaned = cleaned.with_columns(
            pl.when(pl.col(SPLIT_COL) == "val")
            .then(pl.lit("train"))
            .otherwise(pl.col(SPLIT_COL))
            .alias(SPLIT_COL)
        )

    # 4. Optionally drop all but the columns the pipeline reads.
    if slim:
        cleaned = cleaned.select(KEEP_COLUMNS)

    # 5. Summaries before/after.
    _split_summary(lf, "before")
    _split_summary(cleaned, "after ")

    # 4. Sanity of the cleaned target distribution (train split).
    tstats = (
        cleaned.filter(pl.col(SPLIT_COL) == "train")
        .select(
            pl.col(TARGET_COL).min().alias("min"),
            pl.col(TARGET_COL).median().alias("median"),
            pl.col(TARGET_COL).max().alias("max"),
            pl.col(TARGET_COL).std().alias("std"),
        )
        .collect()
    )
    r = tstats.row(0, named=True)
    print(f"  cleaned train target: min {r['min']:.2f} | median {r['median']:.2f} | "
          f"max {r['max']:.2f} | std {r['std']:.2f}")

    # 5. Write. Try streaming sink; fall back to a full collect if unsupported.
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        cleaned.sink_parquet(out)
    except Exception as e:  # noqa: BLE001 — polars streaming may reject is_in sinks
        print(f"  (streaming sink unavailable [{type(e).__name__}]; collecting in memory)")
        cleaned.collect().write_parquet(out)
    size_mb = out.stat().st_size / 1e6
    print(f"  wrote {out}  ({size_mb:.1f} MB)")


def main() -> None:
    t0 = time.time()
    print(f"cleaning ChEMBL: keep assays with all |{TARGET_COL}| <= {MAX_ABS_TARGET:g}"
          f" | merge val->train: {MERGE_VAL_INTO_TRAIN}")
    for rel_in, rel_out, slim in FILES:
        clean_one(REPO_ROOT / rel_in, REPO_ROOT / rel_out, slim)
    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
