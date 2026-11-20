"""scripts/feature_stats.py — WHY is the trained encoder worse than random init? Press Run.

The 2026-08-13 supervision asked the question the AL benchmark forced on us: training
the encoder does not merely fail to help, it makes the representation *worse* than the
random initialisation it started from (report §5e Result 3 — untrained Chemprop beats
every trained variant on 23 of 24 S. aureus cells). "It didn't learn" explains a tie.
It does not explain a loss.

THE HYPOTHESIS. TabPFN is a prior-fitted network: it was meta-trained on synthetic
tabular datasets and does in-context inference against that prior. It never updates its
weights, so it cannot adapt to a feature distribution the prior does not cover — it can
only be fed features that suit it or features that don't. A randomly initialised D-MPNN
is, in effect, a random projection: its 300 outputs are near-Gaussian, weakly
correlated, and spread across a full-rank subspace — which is close to what TabPFN's
synthetic prior expects. Gradient descent through a frozen TabPFN optimises the
*prediction*, with no term anywhere rewarding the encoder for keeping that shape. So
the prediction is free to improve while the feature distribution degrades, and once
the degradation costs more than the fit gains, the trained encoder falls behind its own
starting point.

That predicts specific, measurable damage. This script measures it, for the SAME
molecules, across every step-tagged checkpoint of a run:

  effective rank      exp(entropy of the singular-value spectrum). How many dimensions
                      the representation actually uses. Collapse = information thrown
                      away that TabPFN cannot recover.
  mean |correlation|  redundancy between feature dimensions. TabPFN's prior assumes
                      informative, not duplicated, columns.
  excess kurtosis     tail weight per feature, averaged. Heavy tails are the classic
                      way to fall off a synthetic prior: TabPFN z-normalises each
                      column, so an outlier-dominated column normalises to a spike at
                      zero and stops carrying signal.
  |skewness|          asymmetry, same reasoning.
  dead fraction       features whose variance has collapsed to ~0 — literally unused
                      columns of the prompt.

A statistic is meaningless on its own, exactly as the CKA study found for CKA, so
every curve is printed next to the same three ANCHORS measured on the same molecules:
the step-0 random init (the thing we are trying to beat), Chemeleon (a pretrained
encoder that genuinely works), and Morgan (a different representation family). Read a
trained-encoder value by where it falls relative to those.

WHAT WOULD FALSIFY THE HYPOTHESIS: flat curves. If effective rank, correlation and
kurtosis sit still while AL performance drops, the damage is not distributional and
the explanation is elsewhere — report that, do not bury it.

CPU-only and a few minutes per run: this is 51 forward passes over a fixed molecule
set, no TabPFN inference anywhere.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from active_learning import batched_encode  # noqa: E402
from data_utils import data  # noqa: E402
from models.checkpoints import load_encoder  # noqa: E402
from models.chemeleon import CheMeleonFingerprint  # noqa: E402
from models.morgan import morgan_features  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Step-tagged checkpoint directories written by run_experiment.py, and the parquet
# whose `test` split supplies the molecules. Same sources as the CKA trajectory, so
# the two studies are directly comparable. A missing directory is skipped with a note.
TRAJECTORIES = {
    # The 2026-08-24 clean single-protein run — the FIRST run whose val_spearman rises
    # significantly above its own baseline (§5h). §9 predicts feature collapse causes the
    # damage in the failing runs, so this one is the direct test of the mechanism: if the
    # encoder genuinely improved, its effective rank should NOT have collapsed.
    # 10 epochs on the same set (§5h). §9 says training collapses effective rank; the
    # 3.19-epoch run showed it collapsing then partially RECOVERING, so this asks whether
    # longer training recovers further — which would qualify §9 — or settles low.
    "full_sp_10ep": ("experiments/trained_encoder_full_sp_10ep_steps",
                     "data/curated/chembl36_single_protein_clean.parquet"),
    "full_sp": ("experiments/trained_encoder_full_sp_steps",
                "data/curated/chembl36_single_protein_clean.parquet"),
    "saureus_long": ("experiments/trained_encoder_saureus_long_steps",
                     "data/curated/train_adaptive_S_aureus.parquet"),
    "d2r_long": ("experiments/trained_encoder_d2r_long_steps",
                 "data/curated/train_adaptive_D2R.parquet"),
    "usp7_long": ("experiments/trained_encoder_usp7_long_steps",
                  "data/curated/train_adaptive_USP7.parquet"),
    # The epoch-level run at lr 3e-5 (§5h): the first run with a STEADY in-domain gain.
    # §9's collapse was measured on runs at 1e-4; this asks whether it also happens at the
    # lower rate. Same molecules as full_sp_10ep (same parquet), so the two are directly
    # comparable. epoch_001 of the original run was lost to the 2026-10-01 restart (only the
    # restart's copy survives, renamed, and the 3-digit pattern below skips it); epoch_000
    # was rewritten by the restart but is bit-identical to the seeded initialisation.
    "full_sp_50ep": ("experiments/trained_encoder_full_sp_50ep_epochs",
                     "data/curated/chembl36_single_protein_clean.parquet"),
    "full_sp_50ep_seed1": ("experiments/trained_encoder_full_sp_50ep_seed1_epochs",
                           "data/curated/chembl36_single_protein_clean.parquet"),
}
# Runs to compute this time (None = all of TRAJECTORIES), and where to write them. A run
# with epoch checkpoints (epoch_NNN.pt) records the epoch in `step` and "epoch" in `unit`.
ONLY_RUNS: list[str] | None = ["full_sp_50ep", "full_sp_50ep_seed1"]

N_ASSAYS = 20         # held-out assays to pool molecules from
MIN_ASSAY_MOLS = 25   # skip tiny assays
MAX_MOLS = 2000       # cap the molecule set; stats converge well below this
DEAD_VAR_EPS = 1e-8   # variance below this counts as a dead feature
SAVE_TO = "experiments/feature_stats_50ep.csv"   # the earlier runs are in feature_stats.csv


def effective_rank(x: torch.Tensor) -> float:
    """exp(Shannon entropy of the normalized singular-value spectrum) (Roy & Vetterli).

    A soft count of how many directions the representation actually spans: equals d for
    a perfectly isotropic cloud, 1 if every molecule maps onto a single direction. Read
    it against the anchors — a random init sits near its ceiling, so any drop is
    dimensions being discarded.
    """
    x = x.double()
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    total = sv.sum()
    if total <= 0:
        return float("nan")
    p = sv / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def mean_abs_correlation(x: torch.Tensor) -> float:
    """Mean |Pearson r| over distinct feature PAIRS — redundancy between columns.

    Dead (zero-variance) columns are dropped first: their correlation is undefined and
    would otherwise enter as NaN and poison the mean.
    """
    x = x.double()
    sd = x.std(dim=0)
    keep = sd > DEAD_VAR_EPS ** 0.5
    if int(keep.sum()) < 2:
        return float("nan")
    x = x[:, keep]
    c = torch.corrcoef(x.T)
    d = c.shape[0]
    off = ~torch.eye(d, dtype=torch.bool, device=c.device)
    return float(c[off].abs().nanmean())


def tail_stats(x: torch.Tensor) -> tuple[float, float]:
    """(mean excess kurtosis, mean |skewness|) over live features.

    Both are computed per column and then averaged, because TabPFN normalises each
    column independently — a single heavy-tailed column is a broken column, and
    averaging the moments keeps that visible instead of cancelling it out.
    """
    x = x.double()
    sd = x.std(dim=0)
    keep = sd > DEAD_VAR_EPS ** 0.5
    if int(keep.sum()) == 0:
        return float("nan"), float("nan")
    z = (x[:, keep] - x[:, keep].mean(dim=0)) / sd[keep]
    kurt = (z ** 4).mean(dim=0) - 3.0   # excess: 0 for a Gaussian
    skew = (z ** 3).mean(dim=0)
    return float(kurt.mean()), float(skew.abs().mean())


def dead_fraction(x: torch.Tensor) -> float:
    """Fraction of feature dimensions whose variance has collapsed to ~0."""
    return float((x.double().var(dim=0) < DEAD_VAR_EPS).double().mean())


def describe(x: torch.Tensor) -> dict[str, float]:
    """All five statistics for one [N, d] embedding of the molecule set."""
    kurt, skew = tail_stats(x)
    return {
        "n_features": float(x.shape[1]),
        "effective_rank": effective_rank(x),
        "eff_rank_frac": effective_rank(x) / x.shape[1],
        "mean_abs_corr": mean_abs_correlation(x),
        "excess_kurtosis": kurt,
        "abs_skewness": skew,
        "dead_fraction": dead_fraction(x),
    }


def molecule_set(parquet: str) -> list[str]:
    """Molecules from the `test` split of an encoder's own training file — held out
    from that encoder, and the same pool the CKA study scores."""
    path = (REPO_ROOT / parquet) if parquet else None
    assays = data.load_chembl_assays("test", use_full=not parquet, path=path)
    big = [a for a in assays if len(a.smiles) >= MIN_ASSAY_MOLS][:N_ASSAYS]
    smiles: list[str] = []
    for assay in big:
        smiles.extend(assay.smiles)
    # De-duplicate, preserving order: repeated molecules would inflate correlations.
    seen, unique = set(), []
    for smi in smiles:
        if smi not in seen:
            seen.add(smi)
            unique.append(smi)
    return unique[:MAX_MOLS]


FIELDS = ["run", "series", "unit", "step", "n_features", "effective_rank", "eff_rank_frac",
          "mean_abs_corr", "excess_kurtosis", "abs_skewness", "dead_fraction"]


def run_trajectory(name: str, steps_dir: Path, parquet: str, chemeleon,
                   device: torch.device) -> list[dict]:
    """Feature statistics for every checkpoint of one run, plus the three anchors."""
    ckpts = sorted(steps_dir.glob("step_*.pt"))
    unit = "step"
    if not ckpts:
        ckpts = sorted(steps_dir.glob("epoch_[0-9][0-9][0-9].pt"))
        unit = "epoch"
    if not ckpts:
        print(f"=== {name}: SKIPPED — no checkpoints in {steps_dir.name} yet\n")
        return []

    smiles = molecule_set(parquet)
    print(f"=== {name}: {len(ckpts)} checkpoints, {len(smiles)} held-out molecules")

    rows: list[dict] = []

    # Anchors first, so the trajectory below can be read against them immediately.
    anchors = {
        "chemeleon": batched_encode(chemeleon, smiles, device, warn=False),
        "morgan": batched_encode(lambda s: morgan_features(s), smiles, device, warn=False),
    }
    for label, emb in anchors.items():
        stats = describe(emb)
        rows.append({"run": name, "series": label, "unit": unit, "step": -1, **stats})
        print(f"    {label:<12} rank {stats['effective_rank']:7.1f}"
              f" ({stats['eff_rank_frac']:.2f} of {int(stats['n_features'])})"
              f"  |r| {stats['mean_abs_corr']:.3f}"
              f"  kurt {stats['excess_kurtosis']:+8.2f}"
              f"  dead {stats['dead_fraction']:.2f}")

    print(f"    {unit:<12} {'eff.rank':>9} {'frac':>6} {'|r|':>7} "
          f"{'kurtosis':>9} {'|skew|':>7} {'dead':>6}")
    for ckpt in ckpts:
        step = int(ckpt.stem.split("_")[1])
        encoder = load_encoder(ckpt, str(device))
        stats = describe(batched_encode(encoder, smiles, device, warn=False))
        # step 0 is the random init — the baseline the trained encoder must beat.
        series = "init" if step == 0 else "trained"
        rows.append({"run": name, "series": series, "unit": unit, "step": step, **stats})
        print(f"    {step:<12d} {stats['effective_rank']:9.1f} "
              f"{stats['eff_rank_frac']:6.2f} {stats['mean_abs_corr']:7.3f} "
              f"{stats['excess_kurtosis']:9.2f} {stats['abs_skewness']:7.3f} "
              f"{stats['dead_fraction']:6.2f}")

    first = next(r for r in rows if r["series"] == "init")
    last = rows[-1]
    print(f"    --> init -> final: rank {first['effective_rank']:.1f} -> "
          f"{last['effective_rank']:.1f}, |r| {first['mean_abs_corr']:.3f} -> "
          f"{last['mean_abs_corr']:.3f}, kurtosis {first['excess_kurtosis']:.2f} -> "
          f"{last['excess_kurtosis']:.2f}\n")
    return rows


def main() -> None:
    out = REPO_ROOT / SAVE_TO
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; change SAVE_TO or move it away first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    chemeleon = CheMeleonFingerprint()

    rows: list[dict] = []
    for name, (steps_dir, parquet) in TRAJECTORIES.items():
        if ONLY_RUNS is not None and name not in ONLY_RUNS:
            continue
        rows.extend(run_trajectory(name, REPO_ROOT / steps_dir, parquet, chemeleon, device))

    if not rows:
        print("nothing to write — no step-tagged checkpoints found")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
