"""scripts/cka_debiased.py — does the CKA result survive the debiased estimator? Press Run.

Linear CKA as computed by cka_study.py (Kornblith et al. 2019) uses the biased HSIC estimator,
which overstates the similarity of two representations when the number of molecules is small
relative to their width: two unrelated random matrices score well above zero. Every CKA value in
Chapter 5 is computed per held-out assay of 25 or more molecules, against representations of 300
to 2,048 columns, so the bias could matter. This script recomputes the same comparisons with
both estimators:

  linear     the estimator of cka_study.py (its values should be reproduced to ~3 decimals)
  debiased   CKA built on the unbiased HSIC estimator of Song et al. (2012), as used for
             representation similarity by Nguyen et al. (2021)

for the same encoders, held-out assays and anchors as cka_study.py:
  STATIC      the four short-run encoders against their initialisation (rebuilt from the seed,
              exactly as cka_study.py does) and the three anchors;
  TRAJECTORY  CKA(E_initial, E_step_k) for every checkpoint of the three 100,000-step runs, with
              E_initial their saved step_000000.pt, and the three anchors.

It writes experiments/cka_debiased.csv and never touches cka_study.csv or cka_trajectory.csv.
Runs on the CPU in a few minutes; no GPU is needed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from active_learning import batched_encode  # noqa: E402
from data_utils import data  # noqa: E402
from metrics import mean_std  # noqa: E402
from models.checkpoints import load_encoder  # noqa: E402
from models.chemeleon import CheMeleonFingerprint  # noqa: E402
from models.encoder import MoleculeEncoder  # noqa: E402
from models.morgan import morgan_features  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — the same encoders, assays and seeds as cka_study.py.             #
# --------------------------------------------------------------------------- #
STATIC = {
    "d2r": ("experiments/trained_encoder_d2r.pt", "data/curated/train_adaptive_D2R.parquet"),
    "usp7": ("experiments/trained_encoder_usp7.pt", "data/curated/train_adaptive_USP7.parquet"),
    "saureus": ("experiments/trained_encoder_saureus.pt", "data/curated/train_adaptive_S_aureus.parquet"),
    "full": ("experiments/trained_encoder_full.pt", ""),       # all of ChEMBL (large parquet, slow to load)
}
TRAJECTORIES = {
    "saureus_long": ("experiments/trained_encoder_saureus_long_steps", "data/curated/train_adaptive_S_aureus.parquet"),
    "d2r_long": ("experiments/trained_encoder_d2r_long_steps", "data/curated/train_adaptive_D2R.parquet"),
    "usp7_long": ("experiments/trained_encoder_usp7_long_steps", "data/curated/train_adaptive_USP7.parquet"),
}
N_ASSAYS = 20
MIN_ASSAY_MOLS = 25
INIT_SEED = 0
OTHER_INIT_SEED = 1
DEVICE = "cpu"
SAVE_TO = "experiments/cka_debiased.csv"


def _centre(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x - x.mean(dim=0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """The estimator of cka_study.py: ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on centred features."""
    x, y = _centre(x), _centre(y)
    scale = (x.T @ x).norm() * (y.T @ y).norm()
    return float(((y.T @ x).norm() ** 2) / scale) if scale > 0 else float("nan")


def _hsic_unbiased(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012) for Gram matrices k, l of n >= 4 samples."""
    n = k.shape[0]
    k, l = k.clone(), l.clone()
    k.fill_diagonal_(0.0)
    l.fill_diagonal_(0.0)
    kl = k @ l
    return (torch.trace(kl) + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2.0 * kl.sum() / (n - 2)) / (n * (n - 3))


def debiased_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """CKA with linear kernels built on the unbiased HSIC estimator."""
    x, y = _centre(x), _centre(y)
    k, l = x @ x.T, y @ y.T
    xy, xx, yy = _hsic_unbiased(k, l), _hsic_unbiased(k, k), _hsic_unbiased(l, l)
    if xx <= 0 or yy <= 0:
        return float("nan")
    return float(xy / torch.sqrt(xx * yy))


def held_out_assays(parquet: str) -> list:
    path = (REPO_ROOT / parquet) if parquet else None
    assays = data.load_chembl_assays("test", use_full=not parquet, path=path)
    return [a for a in assays if len(a.smiles) >= MIN_ASSAY_MOLS][:N_ASSAYS]


def random_encoder(seed: int, hidden: int, device) -> MoleculeEncoder:
    torch.manual_seed(seed)
    return MoleculeEncoder(hidden_size=hidden).eval().to(device)


def anchors(smiles, other, chemeleon, device) -> dict:
    return {"other-init": batched_encode(other, smiles, device, warn=False),
            "chemeleon": batched_encode(chemeleon, smiles, device, warn=False),
            "morgan": batched_encode(lambda s: morgan_features(s), smiles, device, warn=False)}


def summarise(part, run, compared_to, step, lin, deb) -> list:
    ml, sl = mean_std(lin)
    md, sd = mean_std(deb)
    return [part, run, compared_to, step, f"{ml:.4f}", f"{sl:.4f}", f"{md:.4f}", f"{sd:.4f}", len(lin)]


def main() -> None:
    device = torch.device(DEVICE)
    chemeleon = CheMeleonFingerprint(device=str(device))
    rows = []

    print("STATIC — trained encoder against its initialisation (rebuilt from the seed, as in cka_study.py)")
    for name, (ckpt, parquet) in STATIC.items():
        path = REPO_ROOT / ckpt
        if not path.exists():
            print(f"  {name}: skipped, no checkpoint")
            continue
        trained = load_encoder(path, str(device))
        hidden = torch.load(path, map_location="cpu", weights_only=False)["hidden_size"]
        initial = random_encoder(INIT_SEED, hidden, device)
        other = random_encoder(OTHER_INIT_SEED, hidden, device)
        scores = {k: ([], []) for k in ("trained", "other-init", "chemeleon", "morgan")}
        for assay in held_out_assays(parquet):
            e0 = batched_encode(initial, assay.smiles, device, warn=False)
            embs = {"trained": batched_encode(trained, assay.smiles, device, warn=False)}
            embs.update(anchors(assay.smiles, other, chemeleon, device))
            for k, e in embs.items():
                scores[k][0].append(linear_cka(e0, e))
                scores[k][1].append(debiased_cka(e0, e))
        for k, (lin, deb) in scores.items():
            rows.append(summarise("static", name, k, "", lin, deb))
            print(f"  {name:8s} {k:11s} linear {rows[-1][4]}  debiased {rows[-1][6]}  ({len(lin)} assays)")

    print("\nTRAJECTORY — every checkpoint of the 100,000-step runs against step_000000.pt")
    for name, (steps_dir, parquet) in TRAJECTORIES.items():
        sdir = REPO_ROOT / steps_dir
        ckpts = sorted(sdir.glob("step_*.pt"))
        if not ckpts or not (sdir / "step_000000.pt").exists():
            print(f"  {name}: skipped, no checkpoints")
            continue
        initial = load_encoder(sdir / "step_000000.pt", str(device))
        hidden = torch.load(sdir / "step_000000.pt", map_location="cpu", weights_only=False)["hidden_size"]
        other = random_encoder(OTHER_INIT_SEED, hidden, device)
        encoders = [(int(p.stem.split("_")[1]), load_encoder(p, str(device))) for p in ckpts]
        per_step = {s: ([], []) for s, _ in encoders}
        per_ref = {k: ([], []) for k in ("other-init", "chemeleon", "morgan")}
        assays = held_out_assays(parquet)
        for assay in assays:
            e0 = batched_encode(initial, assay.smiles, device, warn=False)
            for k, e in anchors(assay.smiles, other, chemeleon, device).items():
                per_ref[k][0].append(linear_cka(e0, e))
                per_ref[k][1].append(debiased_cka(e0, e))
            for s, enc in encoders:
                e = batched_encode(enc, assay.smiles, device, warn=False)
                per_step[s][0].append(linear_cka(e0, e))
                per_step[s][1].append(debiased_cka(e0, e))
        for s, (lin, deb) in sorted(per_step.items()):
            rows.append(summarise("trajectory", name, "trained", s, lin, deb))
        for k, (lin, deb) in per_ref.items():
            rows.append(summarise("trajectory", name, k, "", lin, deb))
        last = [r for r in rows if r[1] == name and r[2] == "trained"][-1]
        che = [r for r in rows if r[1] == name and r[2] == "chemeleon"][-1]
        for i, label in ((4, "linear"), (6, "debiased")):
            moved, ref = 1 - float(last[i]), 1 - float(che[i])
            print(f"  {name:13s} {label:8s} step {last[3]}: CKA {last[i]} (1 - CKA {moved:.3f}); "
                  f"CheMeleon {che[i]} (1 - CKA {ref:.3f}); share of CheMeleon's distance {moved / ref:.0%}")

    out = REPO_ROOT / SAVE_TO
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "run", "compared_to", "step", "cka_linear_mean", "cka_linear_std",
                    "cka_debiased_mean", "cka_debiased_std", "n_assays"])
        w.writerows(rows)
    print(f"\nsaved -> {SAVE_TO}")


if __name__ == "__main__":
    main()
