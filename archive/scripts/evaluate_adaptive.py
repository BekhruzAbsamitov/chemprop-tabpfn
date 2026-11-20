"""scripts/evaluate_adaptive.py — does the ADAPTIVE-trained encoder beat the baselines?

The payoff evaluation for the 2026-07-23 adaptive datasets. For each target we pair
it with ITS OWN adaptively-trained encoder (trained_encoder_<target>.pt) and compare,
under a frozen TabPFN, against the baseline representations:

  * Chemprop-random   — untrained GNN, SAME width as the trained encoder (the control:
                        the only difference from Chemprop-trained is the training)
  * Chemprop-trained  — the per-target adaptive encoder
  * Chemeleon         — PubChem-pretrained foundation encoder
  * Morgan            — ECFP4 fingerprints (the classic baseline)

Static one-shot ranking: draw N_SAMPLES context/query splits (the SAME splits for
every arm), score each with TabPFN, report mean Spearman +/- std. Higher = better.
The adaptive encoders never saw these compounds (leakage was removed at build time),
so this is a fair held-out comparison. No flags / env vars — open and press Run.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402
from metrics import mean_std  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402
from ranking_eval import _evaluate_arm, make_splits  # noqa: E402
from representations import build_arms  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Each target: (display name, csv rel. path, smiles col, target col, adaptive encoder).
TARGETS = [
    ("D2R", "data/curated/D2R.csv", "SMILES", "affinity",
     "experiments/trained_encoder_d2r.pt"),
    ("USP7", "data/curated/USP7.csv", "SMILES", "affinity",
     "experiments/trained_encoder_usp7.pt"),
    ("S. aureus", "data/curated/S_aureus.csv", "Smiles", "pMIC",
     "experiments/trained_encoder_saureus.pt"),
]

N_CONTEXT = 200          # labeled molecules TabPFN learns from, per split
N_QUERY = 200            # molecules it ranks, per split
N_SAMPLES = 3            # random context/query draws per target
TRAINED_HIDDEN_SIZE = 300  # width the adaptive encoders were trained at; the random
                           # baseline uses the SAME width so training is the only diff
N_ESTIMATORS = 8         # TabPFN passes to average (8 = fair across arms)
INCLUDE_CHEMELEON = True
INCLUDE_MORGAN = True
LOSS_FN = "nll"
SEED = 0

ARM_ORDER = ["Chemprop-random", "Chemprop-trained", "Chemeleon", "Morgan"]


def _print_combined(title: str, table: dict, target_names: list[str]) -> None:
    width = 22 + 16 * len(target_names) + 12
    print("=" * width)
    print(title)
    print("=" * width)
    print(f"  {'arm':18s}" + "".join(f"{t:>15s} " for t in target_names) + f"{'overall':>10s}")
    for arm in ARM_ORDER:
        if arm not in table:
            continue
        row, means = f"  {arm:18s}", []
        for t in target_names:
            if t in table[arm]:
                m, s = table[arm][t]
                row += f"{m:7.3f} +/-{s:5.3f} "
                means.append(m)
            else:
                row += f"{'-':>15s} "
        overall, _ = mean_std(means)
        row += f"{overall:>10.3f}"
        print(row)
    print()


def main() -> None:
    _load_hf_token()
    torch.manual_seed(SEED)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | {N_SAMPLES} samples x {N_CONTEXT} ctx / {N_QUERY} qry | "
          f"n_estimators={N_ESTIMATORS}\n")

    heads: dict = {}          # one frozen TabPFN head per arm, reused across targets
    rho_table: dict = {}      # {arm: {target: (mean, std)}}
    r2_table: dict = {}
    done_targets: list[str] = []

    for name, rel_path, smiles_col, target_col, enc_rel in TARGETS:
        enc_path = REPO_ROOT / enc_rel
        if not enc_path.exists():
            print(f"=== {name}: SKIPPED — adaptive encoder not found ({enc_rel}) ===\n")
            continue
        print(f"=== {name} (encoder: {enc_rel}) ===")
        smiles, y = data.load_target(
            REPO_ROOT / rel_path, smiles_col=smiles_col, target_col=target_col)
        print(f"  {len(smiles)} unique molecules | target '{target_col}' mean {float(y.mean()):.2f}")
        if len(smiles) < N_CONTEXT + N_QUERY:
            print(f"  WARNING: only {len(smiles)} molecules; skipping.\n")
            continue

        arms = build_arms(
            device, random_hidden_size=TRAINED_HIDDEN_SIZE, trained_path=enc_path,
            include_chemeleon=INCLUDE_CHEMELEON, include_morgan=INCLUDE_MORGAN,
            repo_root=REPO_ROOT)
        splits = make_splits(len(smiles), random.Random(SEED),
                             n_context=N_CONTEXT, n_query=N_QUERY, n_samples=N_SAMPLES)
        for arm_name, featurize in arms:
            if arm_name not in heads:
                heads[arm_name] = TabPFNHead(n_estimators=N_ESTIMATORS, loss_fn=LOSS_FN)
            rhos, r2s = _evaluate_arm(featurize, heads[arm_name], smiles, y, splits)
            rho_table.setdefault(arm_name, {})[name] = mean_std(rhos)
            r2_table.setdefault(arm_name, {})[name] = mean_std(r2s)
            m, s = rho_table[arm_name][name]
            print(f"  {arm_name:18s} Spearman {m:6.3f} +/- {s:.3f}  ({time.time() - t0:.0f}s)")
        print()
        done_targets.append(name)

    print()
    _print_combined("Spearman (mean +/- std over samples) — higher is better",
                    rho_table, done_targets)
    _print_combined("R2 (mean +/- std over samples) — higher is better",
                    r2_table, done_targets)
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
