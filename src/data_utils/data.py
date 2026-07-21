"""src/data.py — load the datasets the pipeline uses.

Two data sources:
  1. ChEMBL (parquet) -> "episodes" for training/evaluation. One assay = one task.
  2. S. aureus (csv)  -> the real evaluation target (SMILES + pMIC).

Of the 42 columns in the ChEMBL file we load only the FOUR the pipeline needs
(the molecule, its activity value, the assay id, and the split). Chemprop builds
all its features from the SMILES, so the other 38 columns are ignored here.

You can press Run on this file directly: its __main__ block prints a quick
summary so you can confirm the data loads correctly. No terminal flags needed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

# --------------------------------------------------------------------------- #
# SETTINGS — file locations and the column names we read. Edit if files move.  #
# Paths are anchored to the repo, so running from any folder still works.      #
# --------------------------------------------------------------------------- #
# This file lives at src/data_utils/data.py, so go up three levels to the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Cleaned files (raw-scale assays removed, val folded into train) — see
# scripts/clean_chembl_targets.py. Dev is also slimmed to the 4 columns used here.
CHEMBL_DEV_FILE = REPO_ROOT / "data" / "curated" / "chembl36_dev_clean.parquet"  # small, fast
CHEMBL_FULL_FILE = REPO_ROOT / "data" / "curated" / "chembl36_curated_clean.parquet"  # full, cleaned
S_AUREUS_FILE = REPO_ROOT / "data" / "curated" / "s_aureus_curated_with_scores_filtered.csv"

# The only 4 ChEMBL columns the pipeline uses:
ASSAY_COL = "assay_x_type_x_doc"  # task id: one assay = one prediction task
SMILES_COL = "canonical_smiles"  # the molecule
TARGET_COL = "target_transformed"  # the value to predict
SPLIT_COL = "split"  # "train" / "val" / "test"

# Column names inside the S. aureus csv:
S_AUREUS_SMILES_COL = "Smiles"
S_AUREUS_TARGET_COL = "pMIC"

# An assay whose context labels are all (nearly) identical is unusable: TabPFN
# normalizes the context by its standard deviation, which would be ~0.
_MIN_LABEL_STD = 1e-6


# --------------------------------------------------------------------------- #
# 1. ChEMBL: molecules grouped into assays (tasks)                            #
# --------------------------------------------------------------------------- #
@dataclass
class Assay:
    """All molecules measured in one assay = one prediction task."""
    assay_id: str
    smiles: list[str]
    y: torch.Tensor  # one activity value per molecule


def load_chembl_assays(split: str, use_full: bool = False) -> list[Assay]:
    path = CHEMBL_FULL_FILE if use_full else CHEMBL_DEV_FILE
    df = (
        pl.read_parquet(path, columns=[ASSAY_COL, SMILES_COL, TARGET_COL, SPLIT_COL])
        .filter(pl.col(SPLIT_COL) == split)
        .drop_nulls([SMILES_COL, TARGET_COL])
    )
    assays: list[Assay] = []
    for key, group in df.group_by(ASSAY_COL, maintain_order=True):
        assay_id = key[0] if isinstance(key, (tuple, list)) else key
        assays.append(
            Assay(
                assay_id=str(assay_id),
                smiles=group[SMILES_COL].to_list(),
                y=torch.tensor(group[TARGET_COL].to_list(), dtype=torch.float32),
            )
        )
    return assays


# --------------------------------------------------------------------------- #
# 2. Episodes: split one assay into a context (support) set + a query set      #
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    """One in-context task: labeled context molecules + query molecules to predict."""
    context_smiles: list[str]
    context_y: torch.Tensor
    query_smiles: list[str]
    query_y: torch.Tensor


def make_episode(
        assay: Assay,
        context_frac: float = 0.5,
        *,
        min_context: int = 5,
        min_query: int = 1,
        max_context: int | None = None,
        max_query: int | None = None,
        rng: random.Random | None = None,
) -> Episode | None:
    """Randomly split one assay into context + query.

    Returns None if the assay is too small, or if the context labels are almost
    all identical (unusable — see _MIN_LABEL_STD).

    ``max_context`` / ``max_query`` bound the episode size. Full ChEMBL has
    assays with thousands of molecules; without a cap, one giant assay produces
    a TabPFN forward+backward large enough to exhaust GPU memory. Because the
    indices are already shuffled, truncating keeps a uniform random subset.
    """
    n = len(assay.smiles)
    if n < min_context + min_query:
        return None
    rng = rng or random.Random(0)
    order = list(range(n))
    rng.shuffle(order)
    n_context = min(max(min_context, round(context_frac * n)), n - min_query)
    c_idx, q_idx = order[:n_context], order[n_context:]
    if max_context is not None:
        c_idx = c_idx[:max_context]
    if max_query is not None:
        q_idx = q_idx[:max_query]
    context_y = assay.y[c_idx]
    if float(context_y.std(unbiased=False)) < _MIN_LABEL_STD:
        return None
    return Episode(
        context_smiles=[assay.smiles[i] for i in c_idx],
        context_y=context_y,
        query_smiles=[assay.smiles[i] for i in q_idx],
        query_y=assay.y[q_idx],
    )


def iter_episodes(
        assays: list[Assay],
        *,
        context_frac: float = 0.5,
        max_context: int | None = None,
        max_query: int | None = None,
        rng: random.Random | None = None,
):
    """Yield episodes by cycling through assays WITHOUT replacement.

    Each "epoch" shuffles the assay list and walks it once, so no assay repeats
    until every other assay has been used — this removes the redundancy of
    with-replacement sampling (where some assays are drawn many times before
    others appear at all). Assays too small to form an episode are skipped. When
    the list is exhausted it reshuffles and continues, so a caller can pull as
    many episodes as it wants (e.g. more steps than there are assays) while
    keeping coverage balanced. Infinite generator — the caller decides how many
    to take.
    """
    rng = rng or random.Random(0)
    order = list(assays)
    while True:
        rng.shuffle(order)
        for assay in order:
            episode = make_episode(
                assay,
                context_frac=context_frac,
                max_context=max_context,
                max_query=max_query,
                rng=rng,
            )
            if episode is not None:
                yield episode


# --------------------------------------------------------------------------- #
# 3. S. aureus: the real evaluation target                                    #
# --------------------------------------------------------------------------- #
def load_target(
        csv_path: Path = S_AUREUS_FILE,
        *,
        smiles_col: str = S_AUREUS_SMILES_COL,
        target_col: str = S_AUREUS_TARGET_COL,
) -> tuple[list[str], torch.Tensor]:
    """Load the S. aureus target as (unique SMILES, values).

    De-duplicates by SMILES (median value) so a molecule can't land in both the
    context and the query of the same evaluation split.
    """
    df = (
        pl.read_csv(csv_path, infer_schema_length=10000)
        .select(
            [
                pl.col(smiles_col).alias("smiles"),
                pl.col(target_col).cast(pl.Float64).alias("y"),
            ]
        )
        .drop_nulls()
        .group_by("smiles")
        .agg(pl.col("y").median())
    )
    return df["smiles"].to_list(), torch.tensor(df["y"].to_list(), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Press Run to sanity-check that everything loads.                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("ChEMBL dev — molecules per split:")
    for split_name in ("train", "val", "test"):
        assays = load_chembl_assays(split_name)
        n_mol = sum(len(a.smiles) for a in assays)
        print(f"  {split_name:5s}: {len(assays):5d} assays, {n_mol:7d} molecules")

    example = load_chembl_assays("train")[0]
    print(
        f"\nexample assay '{example.assay_id}': {len(example.smiles)} molecules | "
        f"first SMILES {example.smiles[0]!r} | target {float(example.y[0]):.3f}"
    )

    smiles, y = load_target()
    print(
        f"\nS. aureus target: {len(smiles)} unique molecules | "
        f"pMIC mean {float(y.mean()):.2f} (min {float(y.min()):.2f}, max {float(y.max()):.2f})"
    )
