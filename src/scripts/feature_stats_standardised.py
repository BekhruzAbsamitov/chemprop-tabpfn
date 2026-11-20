"""scripts/feature_stats_standardised.py — the effective rank of the embeddings AS TABPFN SEES THEM. Press Run.

feature_stats.py reports the effective rank of raw, column-centred embeddings. TabPFN standardises
every column before it sees them (Methodology 3.5.1), so the variance of a single column is
invisible to it, while the raw effective rank still moves with that variance. This script
recomputes the effective rank AFTER standardising every column (centred, scaled to unit
standard deviation; columns with a standard deviation below DEAD_SD are dropped), on the same
held-out molecules and the same checkpoints as feature_stats.py, plus the two anchors.

The raw effective rank and the mean absolute correlation are written next to it. The raw value
reproduces feature_stats.csv / feature_stats_50ep.csv, which the script checks and prints, so
the standardised column can be trusted to sit on the same molecules. The mean absolute
correlation is scale-free and is the same under both treatments.

Output: one row per checkpoint and anchor, in SAVE_TO. Read by plot_training.py (Table 5.5 and
Figure 5.2 of the thesis). CPU is enough: a few forward passes over at most 2,000 molecules per
checkpoint, no TabPFN inference.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from active_learning import batched_encode  # noqa: E402
from feature_stats import mean_abs_correlation, molecule_set  # noqa: E402  (same molecules, same |r|)
from models.checkpoints import load_encoder  # noqa: E402
from models.chemeleon import CheMeleonFingerprint  # noqa: E402
from models.morgan import morgan_features  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
SINGLE_PROTEIN = "data/curated/chembl36_single_protein_clean.parquet"
# run -> (directory of step- or epoch-tagged checkpoints, parquet whose `test` split gives the molecules)
TRAJECTORIES = {
    "saureus_long": ("experiments/trained_encoder_saureus_long_steps", "data/curated/train_adaptive_S_aureus.parquet"),
    "d2r_long": ("experiments/trained_encoder_d2r_long_steps", "data/curated/train_adaptive_D2R.parquet"),
    "usp7_long": ("experiments/trained_encoder_usp7_long_steps", "data/curated/train_adaptive_USP7.parquet"),
    "full_sp": ("experiments/trained_encoder_full_sp_steps", SINGLE_PROTEIN),
    "full_sp_10ep": ("experiments/trained_encoder_full_sp_10ep_steps", SINGLE_PROTEIN),
    "full_sp_50ep": ("experiments/trained_encoder_full_sp_50ep_epochs", SINGLE_PROTEIN),
    "full_sp_50ep_seed1": ("experiments/trained_encoder_full_sp_50ep_seed1_epochs", SINGLE_PROTEIN),
}
ONLY_RUNS: list[str] | None = None        # None = all of TRAJECTORIES
DEAD_SD = 1e-4                            # columns with a smaller standard deviation are dropped
CHECK_AGAINST = ["experiments/feature_stats.csv", "experiments/feature_stats_50ep.csv"]
SAVE_TO = "experiments/feature_stats_standardised.csv"

FIELDS = ["run", "series", "unit", "step", "n_molecules", "n_features", "n_features_kept",
          "effective_rank", "effective_rank_std", "eff_rank_std_frac", "mean_abs_corr"]


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of the column-centred matrix."""
    x = x.double()
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    p = sv / sv.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


def standardised(x: torch.Tensor) -> torch.Tensor:
    """Every column centred and scaled to unit standard deviation; dead columns dropped."""
    x = x.double()
    sd = x.std(dim=0)
    keep = sd > DEAD_SD
    return (x[:, keep] - x[:, keep].mean(dim=0)) / sd[keep]


def row(run: str, series: str, unit: str, step: int, emb: torch.Tensor) -> dict:
    z = standardised(emb)
    return {"run": run, "series": series, "unit": unit, "step": step,
            "n_molecules": emb.shape[0], "n_features": emb.shape[1], "n_features_kept": z.shape[1],
            "effective_rank": effective_rank(emb), "effective_rank_std": effective_rank(z),
            "eff_rank_std_frac": effective_rank(z) / emb.shape[1],
            "mean_abs_corr": mean_abs_correlation(emb)}


def checkpoints(directory: Path) -> tuple[list[Path], str]:
    steps = sorted(directory.glob("step_*.pt"))
    if steps:
        return steps, "step"
    # epoch_NNN.pt only: the restart's renamed copies (epoch_001_restart_oct01.pt) are skipped
    return sorted(p for p in directory.glob("epoch_*.pt") if re.fullmatch(r"epoch_\d{3}\.pt", p.name)), "epoch"


def check_against_raw(rows: list[dict]) -> None:
    """The raw effective rank must reproduce the earlier CSVs, or the molecules differ."""
    ref = {}
    for rel in CHECK_AGAINST:
        path = REPO_ROOT / rel
        if path.exists():
            with path.open() as fh:
                for r in csv.DictReader(fh):
                    ref[(r["run"], r["series"] if r["series"] in ("chemeleon", "morgan") else "ckpt",
                         int(float(r["step"])))] = float(r["effective_rank"])
    worst, n = 0.0, 0
    for r in rows:
        key = (r["run"], r["series"] if r["series"] in ("chemeleon", "morgan") else "ckpt", r["step"])
        if key in ref:
            worst, n = max(worst, abs(ref[key] - r["effective_rank"])), n + 1
    print(f"raw effective rank vs earlier CSVs: {n} matching rows, largest difference {worst:.3f}")


def main() -> None:
    out = REPO_ROOT / SAVE_TO
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; change SAVE_TO or move it away first")
    device = torch.device("cpu")
    chemeleon = CheMeleonFingerprint()
    rows: list[dict] = []
    for name, (directory, parquet) in TRAJECTORIES.items():
        if ONLY_RUNS is not None and name not in ONLY_RUNS:
            continue
        ckpts, unit = checkpoints(REPO_ROOT / directory)
        if not ckpts:
            print(f"=== {name}: SKIPPED, no checkpoints in {directory}")
            continue
        smiles = molecule_set(parquet)
        print(f"=== {name}: {len(ckpts)} checkpoints, {len(smiles)} held-out molecules")
        for label, encode in (("chemeleon", chemeleon), ("morgan", lambda s: morgan_features(s))):
            rows.append(row(name, label, unit, -1, batched_encode(encode, smiles, device, warn=False)))
        for ckpt in ckpts:
            step = int(ckpt.stem.split("_")[1])
            emb = batched_encode(load_encoder(ckpt, str(device)), smiles, device, warn=False)
            r = row(name, "init" if step == 0 else "trained", unit, step, emb)
            rows.append(r)
            print(f"    {unit} {step:>7d}: raw rank {r['effective_rank']:6.1f}   standardised {r['effective_rank_std']:6.1f}"
                  f"   |r| {r['mean_abs_corr']:.3f}")

    check_against_raw(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
