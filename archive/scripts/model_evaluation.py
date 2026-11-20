"""scripts/model_evaluation.py — Experiment A: static ranking of representations.

Compares the molecular representations — a random-init Chemprop encoder, the
ChEMBL-trained encoder, Chemeleon (and optionally Morgan) — by how well a frozen
TabPFN ranks each target dataset in one shot. Reports mean Spearman ± std over
N_SAMPLES splits, all arms scored on the same splits (a fair comparison).

The heavy lifting lives in reusable modules:
  * representations.build_arms  — the arms to compare
  * ranking_eval.run_static_ranking — the splits, scoring, and results tables

Settings are constants below. No flags / env vars — open and press Run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.tabpfn_regressor import _load_hf_token  # noqa: E402
from ranking_eval import run_static_ranking  # noqa: E402
from representations import build_arms  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Each dataset: (display name, csv path rel. to repo root, smiles col, target col).
DATASETS = [
    ("S. aureus", "data/curated/s_aureus_curated_with_scores_filtered.csv", "Smiles", "pMIC"),
    ("D2R", "data/curated/D2R_final.csv", "SMILES", "affinity"),
    ("USP7", "data/curated/USP7_final.csv", "SMILES", "affinity"),
]

N_CONTEXT = 200          # labeled molecules TabPFN learns from, per split
N_QUERY = 200            # molecules it ranks, per split
N_SAMPLES = 3            # random context/query draws per dataset

RANDOM_HIDDEN_SIZE = 128                                    # random-init arm width
TRAINED_ENCODER_PATH = "experiments/trained_encoder_full.pt"  # None to skip trained arm
INCLUDE_CHEMELEON = True
INCLUDE_MORGAN = False

N_ESTIMATORS = 8         # TabPFN passes to average (8 = fair across arms; 1 = fast)
LOSS_FN = "nll"
SEED = 0


def main() -> None:
    _load_hf_token()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    arms = build_arms(
        device,
        random_hidden_size=RANDOM_HIDDEN_SIZE,
        trained_path=TRAINED_ENCODER_PATH,
        include_chemeleon=INCLUDE_CHEMELEON,
        include_morgan=INCLUDE_MORGAN,
        repo_root=REPO_ROOT,
    )
    run_static_ranking(
        arms, DATASETS,
        n_context=N_CONTEXT, n_query=N_QUERY, n_samples=N_SAMPLES,
        n_estimators=N_ESTIMATORS, loss_fn=LOSS_FN, seed=SEED, repo_root=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
