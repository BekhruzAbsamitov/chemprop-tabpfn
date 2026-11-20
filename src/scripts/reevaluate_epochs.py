"""scripts/reevaluate_epochs.py — re-run the full validation on saved epoch checkpoints.

WHY THIS EXISTS. The 50-epoch single-protein run completed epochs
0-26, was interrupted by cluster maintenance, and was then requeued by HTCondor, which
restarted it from scratch and overwrote its curve CSV and stdout log. The
values of epochs 25 and 26 were therefore never recorded, but their checkpoints survive in
experiments/trained_encoder_full_sp_50ep_epochs/. This script recomputes them.

It also re-evaluates checkpoints whose ORIGINAL values are known (epochs 0, 23 and 24).
GPU inference is not bit-reproducible, so a recomputed value differs slightly from the
recorded one; the known epochs measure by how much, which says how far the recomputed
epochs 25-26 can be compared with the recorded curve.

The validation set and the TabPFN head are built exactly as run_experiment.py builds them
in epoch mode for the full_sp_50ep preset: every held-out assay of the single-protein
corpus, uncapped, seed 0, TabPFN with 8 ensemble members.

Writes one row per checkpoint, as it goes, to OUTPUT_CSV (refuses to overwrite it).
Cluster: condor_submit cluster/run_gpu_reevaluate_epochs.sub   (~70 min per checkpoint)
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402
from models.checkpoints import load_encoder  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402
from training import build_val_episodes, evaluate_val_full  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
TRAIN_DATA_FILE = "data/curated/chembl36_single_protein_clean.parquet"
EPOCHS_DIR = "experiments/trained_encoder_full_sp_50ep_epochs"
# Known epochs first (they calibrate the GPU noise), then the two missing ones.
EPOCHS = [0, 23, 24, 25, 26]
# Values recorded by the original run, for the comparison printed at the end.
RECORDED = {0: (1.5442, 0.3009), 23: (1.4801, 0.3363), 24: (1.4837, 0.3390)}
N_ESTIMATORS = 8          # as in the full_sp_50ep preset
SEED = 0                  # as in the full_sp_50ep preset
OUTPUT_CSV = "experiments/val_reevaluated_full_sp_50ep.csv"


def main() -> None:
    _load_hf_token()
    out = REPO_ROOT / OUTPUT_CSV
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    torch.manual_seed(SEED)

    test_assays = data.load_chembl_assays("test", path=REPO_ROOT / TRAIN_DATA_FILE)
    val_episodes = build_val_episodes(test_assays, n_episodes=len(test_assays),
                                      max_context=None, max_query=None,
                                      rng=random.Random(SEED))
    head = TabPFNHead(n_estimators=N_ESTIMATORS, loss_fn="nll")
    print(f"{len(val_episodes)} validation episodes from {len(test_assays)} held-out assays "
          f"| device {head.device}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_loss", "val_spearman", "val_points",
                                "recorded_val_loss", "recorded_val_spearman", "seconds"])

    for epoch in EPOCHS:
        ckpt = REPO_ROOT / EPOCHS_DIR / f"epoch_{epoch:03d}.pt"
        if not ckpt.exists():
            print(f"epoch {epoch:>3}: no checkpoint at {ckpt} — skipped")
            continue
        t0 = time.time()
        encoder = load_encoder(ckpt, device=head.device)
        loss, rho, n_pts = evaluate_val_full(encoder, head, val_episodes)
        secs = time.time() - t0
        rec_loss, rec_rho = RECORDED.get(epoch, ("", ""))
        with out.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{loss:.4f}", f"{rho:.4f}", n_pts,
                                    rec_loss, rec_rho, f"{secs:.0f}"])
        note = (f" | recorded {rec_loss:.4f} / {rec_rho:+.4f} -> diff "
                f"{loss - rec_loss:+.4f} / {rho - rec_rho:+.4f}") if epoch in RECORDED else ""
        print(f"epoch {epoch:>3}: val_loss {loss:.4f} | val_spearman {rho:+.4f} | "
              f"{n_pts:,} points | {secs / 60:.1f} min{note}")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
