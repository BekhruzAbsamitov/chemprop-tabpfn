"""scripts/calibration_study.py — direct evaluation of predictive calibration (E1).

The 2026-09-29 supervision asked for calibration to be assessed DIRECTLY, not only
indirectly through EI vs greedy: "for each dataset, train a model on a random split and
evaluate ECE on the test set; 3 models x 5 datasets x (1 representation)".

For a regression model, calibration means that its predictive intervals contain the true
value as often as they claim to. For each nominal level q (10 %, 20 %, ..., 90 %, 95 %) the
central q-interval is formed and its EMPIRICAL COVERAGE on the test set is measured.
  coverage_q      fraction of test molecules whose true value lies in the q-interval
  ECE             mean over the levels of |coverage_q - q|   (0 = perfectly calibrated)
  signed error    mean of (coverage_q - q): negative = overconfident, positive = underconfident
  NLL             negative log-likelihood of the test values (lower is better), as the mean,
                  the median, and the mean with each molecule's NLL capped at NLL_CAP. The plain
                  mean is dominated by a few molecules: a test value outside TabPFN's bar range
                  has zero density (infinite NLL), and a label spike (USP7: 166 molecules at
                  exactly 11.301) makes the Gaussian sigma tiny, so one miss costs hundreds of nats.
  spearman, rmse  accuracy of the predicted mean, for context

Each surrogate is evaluated with the intervals that acquisition actually uses:
  rf, gp, gp_tanimoto  Gaussian intervals mu +- z sigma (EI/PI assume a Gaussian)
  tabpfn  "gauss"      Gaussian intervals mu +- z sigma, sigma = half the 68 % width (as in EI)
  tabpfn  "full"       the quantiles of TabPFN's full bar distribution
Comparing TabPFN's two rows measures what the Gaussian approximation costs (also E5).

Datasets: the four benchmark pools plus Mpro (the fifth dataset of Gorantla et al. 2024).
Representations: CheMeleon for all models; Morgan in addition, so the Tanimoto-kernel GP
(defined on binary fingerprints only) can be evaluated as well; and the random-init
Chemprop encoder, the weak-representation control of the primary benchmark.

Output: experiments/calibration_study.csv (refuses to overwrite).
Cluster: condor_submit cluster/run_gpu_calibration.sub
"""

from __future__ import annotations

import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SRC_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_al import TARGETS, featurize_pool, load_pool  # noqa: E402
from data_utils import data  # noqa: E402
from models.baselines import (  # noqa: E402
    GaussianProcessHead, GaussianProcessTanimotoHead, InferenceTabPFN, RandomForestHead)
from models.tabpfn_regressor import _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
DATASETS = ["saureus", "tyk2", "usp7", "d2r", "mpro"]
MPRO = dict(csv="data/curated/Mpro.csv", smiles_col="SMILES", value_col="affinity")
REPRESENTATIONS = {"chemeleon": ("chemeleon", None), "morgan": ("morgan", None),
                   "chemprop_untrained": ("untrained", None)}   # random-init D-MPNN, 300-d
MODELS = ["tabpfn", "rf", "gp", "gp_tanimoto"]
TRAIN_SIZES = [100, 1000]       # the active-learning regime: 100 = early, 1,000 = late
N_SPLITS = 5                    # random train/test splits per setting
MAX_TEST = 2000                 # cap on test molecules (the largest pool has 40k)
MIN_TEST = 200                  # skip a train size that would leave fewer test molecules
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N_ESTIMATORS = 4
NLL_CAP = 10.0                  # nats per molecule, for nll_capped only
OUTPUT_CSV = "experiments/calibration_study.csv"

COLUMNS = (["dataset", "representation", "model", "intervals", "train_size", "test_size",
            "split", "spearman", "rmse", "nll", "nll_median", "nll_capped", "ece",
            "signed_calibration_error",
            "width_68"] + [f"coverage_{int(q * 100)}" for q in LEVELS])


def _normal_ppf(p: float) -> float:
    return float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(p)))


Z = {q: _normal_ppf(0.5 + q / 2) for q in LEVELS}      # half-width multipliers


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def summarise(y, mu, lo_hi: dict, nll: np.ndarray) -> dict:
    """Coverage-based calibration metrics from per-level interval bounds and per-molecule NLL."""
    cov = {q: float(np.mean((y >= lo) & (y <= hi))) for q, (lo, hi) in lo_hi.items()}
    lo68, hi68 = lo_hi.get(0.7, (None, None))
    nll = np.nan_to_num(np.asarray(nll, dtype=float), nan=np.inf, posinf=np.inf)
    return dict(
        spearman=spearman(mu, y), rmse=float(np.sqrt(np.mean((mu - y) ** 2))),
        nll=float(np.mean(nll)), nll_median=float(np.median(nll)),
        nll_capped=float(np.mean(np.minimum(nll, NLL_CAP))),
        ece=float(np.mean([abs(cov[q] - q) for q in LEVELS])),
        signed_calibration_error=float(np.mean([cov[q] - q for q in LEVELS])),
        width_68=float(np.mean(hi68 - lo68)) if lo68 is not None else float("nan"),
        **{f"coverage_{int(q * 100)}": cov[q] for q in LEVELS})


def gaussian_rows(y, mu, sigma) -> dict:
    sigma = np.clip(sigma, 1e-6, None)
    lo_hi = {q: (mu - Z[q] * sigma, mu + Z[q] * sigma) for q in LEVELS}
    nll = 0.5 * np.log(2 * math.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2
    return summarise(y, mu, lo_hi, nll)


def tabpfn_full_rows(model: InferenceTabPFN, y, x_tr, y_tr, x_te) -> tuple[dict, dict]:
    """TabPFN evaluated with its full bar distribution and with the Gaussian approximation."""
    mu, sigma, criterion, logits = model.predict_full(x_tr, y_tr, x_te)
    mu, sigma = mu.numpy(), sigma.numpy()
    qs = sorted({0.5 - q / 2 for q in LEVELS} | {0.5 + q / 2 for q in LEVELS})
    quant = model.reg.predict(x_te.numpy(), output_type="quantiles", quantiles=qs)
    qmap = {round(q, 6): np.asarray(v).ravel() for q, v in zip(qs, quant)}
    lo_hi = {q: (qmap[round(0.5 - q / 2, 6)], qmap[round(0.5 + q / 2, 6)]) for q in LEVELS}
    y_t = torch.tensor(y, dtype=torch.float32, device=logits.device)
    nll = criterion(logits, y_t).detach().float().cpu().numpy().ravel()
    return summarise(y, mu, lo_hi, nll), gaussian_rows(y, mu, sigma)


def load_dataset(name: str):
    if name == "mpro":
        smiles, yv = data.load_target(REPO_ROOT / MPRO["csv"], smiles_col=MPRO["smiles_col"],
                                      target_col=MPRO["value_col"])
        return smiles, yv.flatten().float()
    smiles, yv, _ = load_pool(TARGETS[name])
    return smiles, yv.flatten().float()


def make_model(name: str, device: torch.device):
    return {"tabpfn": lambda: InferenceTabPFN(n_estimators=N_ESTIMATORS, device=str(device)),
            "rf": RandomForestHead, "gp": GaussianProcessHead,
            "gp_tanimoto": GaussianProcessTanimotoHead}[name]()


def main() -> None:
    out = REPO_ROOT / OUTPUT_CSV
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    out.parent.mkdir(parents=True, exist_ok=True)
    _load_hf_token()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with out.open("w", newline="") as f:
        csv.writer(f).writerow(COLUMNS)

    t0 = time.time()
    for ds in DATASETS:
        smiles, y = load_dataset(ds)
        for rep, (kind, ckpt) in REPRESENTATIONS.items():
            X = featurize_pool(kind, ckpt, smiles, device).float().cpu()
            print(f"{ds} ({len(smiles):,}) / {rep} {tuple(X.shape)}")
            for n_train in TRAIN_SIZES:
                if len(smiles) - n_train < MIN_TEST:
                    print(f"  train {n_train}: pool too small for a test set — skipped")
                    continue
                for split in range(N_SPLITS):
                    rng = random.Random(1000 * split + n_train)
                    perm = list(range(len(smiles)))
                    rng.shuffle(perm)
                    tr = perm[:n_train]
                    te = perm[n_train:n_train + MAX_TEST]
                    x_tr, y_tr, x_te = X[tr], y[tr], X[te]
                    y_te = y[te].numpy()
                    for m in MODELS:
                        if m == "gp_tanimoto" and rep != "morgan":
                            continue
                        model = make_model(m, device)
                        if m == "tabpfn":
                            full, gauss = tabpfn_full_rows(model, y_te, x_tr, y_tr, x_te)
                            results = [("full", full), ("gauss", gauss)]
                        else:
                            mu, sigma = model.predict_dist(x_tr, y_tr, x_te)
                            results = [("gauss", gaussian_rows(y_te, mu.numpy(), sigma.numpy()))]
                        with out.open("a", newline="") as f:
                            w = csv.writer(f)
                            for intervals, res in results:
                                w.writerow([ds, rep, m, intervals, n_train, len(te), split]
                                           + [f"{res[c]:.4f}" for c in COLUMNS[7:]])
                        msg = " | ".join(f"{iv}: ECE {r['ece']:.3f} rho {r['spearman']:+.3f}"
                                         for iv, r in results)
                        print(f"  train {n_train:>4} split {split} {m:11s} {msg} "
                              f"| {(time.time() - t0) / 60:.0f} min")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
