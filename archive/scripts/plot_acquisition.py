"""scripts/plot_acquisition.py — the acquisition-map figure of the Background chapter. Press Run.

One real campaign state (TabPFN on Morgan fingerprints, N_LABELLED random labelled molecules,
N_CANDIDATES random candidates) drawn in the plane of predicted mean (x) and predicted
uncertainty (y, log scale). Every candidate is a dot; the true hits (top 5 % of the pool) are
coloured. Each panel shows one criterion against one reference value tau:

    rows:    Probability of Improvement, Expected Improvement
    columns: tau = y+ (best activity among the labelled molecules), tau = y_thresh (hit threshold)

The scores use the Gaussian approximation of the benchmark (mean, and half the width of the
central 68 % interval). The solid curve is the batch boundary: the curve of equal score through
the BATCH-th best candidate, so the batch is everything on its shaded side. The dashed line is
the same boundary for greedy selection. The box counts the true hits in each batch.

D2R is drawn because it shows the four cases most clearly; the script can print the counts for
every target (PRINT_ALL_TARGETS) so that choice is visible. One state and one batch illustrate
the mechanism only; the benchmark's averages are in Chapter 4.

matplotlib is deliberately NOT a project dependency (it is needed by the plot scripts alone).
Run it with:

    uv run --with matplotlib python src/scripts/plot_acquisition.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SRC_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
SAVE = "experiments/fig_acquisition_map.pdf"
SAVE_PNG_TOO = True

TARGET = "d2r"                    # the target drawn in the figure
PRINT_ALL_TARGETS = False         # print batch hit counts for every target first
N_LABELLED = 100
N_CANDIDATES = 1000
BATCH = 10
STATE_SEED = 0
N_ESTIMATORS = 4                  # as in the benchmark
X_RANGE = (2.5, 10.5)             # predicted mean shown (a few candidates far left are cut)
Y_MIN = 0.03                      # lower end of the log-scaled uncertainty axis

COLOUR_HIT = "#c45a00"
COLOUR_OTHER = "0.72"
COLOUR_PICK = "#1f4e79"


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.constrained_layout.use": True})


def save(fig, rel: str) -> None:
    out = REPO_ROOT / rel
    fig.savefig(out)
    print(f"wrote {out}")
    if SAVE_PNG_TOO:
        fig.savefig(out.with_suffix(".png"), dpi=200)


def Phi(z):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2)))


def phi(z):
    z = np.asarray(z)
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def pi_score(mu, s, tau):
    return Phi((mu - tau) / s)


def ei_score(mu, s, tau):
    z = (mu - tau) / s
    return (mu - tau) * Phi(z) + s * phi(z)


def campaign_state(target: str):
    from benchmark_al import TARGETS, featurize_pool, load_pool
    from models.baselines import InferenceTabPFN
    from models.tabpfn_regressor import _load_hf_token

    _load_hf_token()
    smiles, y, hit_set = load_pool(TARGETS[target])
    y_thresh = float(min(float(y[i]) for i in hit_set))
    X = featurize_pool("morgan", None, smiles, torch.device("cpu"))
    g = torch.Generator().manual_seed(STATE_SEED)
    perm = torch.randperm(len(smiles), generator=g)
    ctx, cand = perm[:N_LABELLED], perm[N_LABELLED:N_LABELLED + N_CANDIDATES]
    mu, sigma = InferenceTabPFN(n_estimators=N_ESTIMATORS, device="cpu").predict_dist(
        X[ctx], y[ctx], X[cand])
    is_hit = np.array([int(i) in hit_set for i in cand])
    y_plus = float(y[ctx].max())
    return mu.numpy().ravel(), sigma.numpy().ravel(), is_hit, y_plus, y_thresh


def top(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores)[:BATCH]


def batches(mu, s, y_plus, y_thresh) -> dict:
    out = {"greedy": top(mu)}
    for tname, tau in (("y+", y_plus), ("thresh", y_thresh)):
        out[f"PI@{tname}"] = top(pi_score(mu, s, tau))
        out[f"EI@{tname}"] = top(ei_score(mu, s, tau))
    return out


def print_counts(target, mu, s, is_hit, y_plus, y_thresh) -> None:
    b = batches(mu, s, y_plus, y_thresh)
    counts = "  ".join(f"{k} {int(is_hit[v].sum())}" for k, v in b.items())
    print(f"{target:8s} y+ {y_plus:6.2f}  y_thresh {y_thresh:6.2f}  "
          f"hits among candidates {int(is_hit.sum()):3d}  hits per batch of {BATCH}: {counts}")


def figure(mu, s, is_hit, y_plus, y_thresh) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0), sharex=True, sharey=True)
    b = batches(mu, s, y_plus, y_thresh)
    greedy_cut = np.sort(mu)[-BATCH]
    x0, x1 = X_RANGE
    y1 = s.max() * 1.3
    gx, gy = np.meshgrid(np.linspace(x0, x1, 500), np.geomspace(Y_MIN, y1, 500))

    score_fn = {"PI": pi_score, "EI": ei_score}
    for r, crit in enumerate(("PI", "EI")):
        for c, (tname, tau, label) in enumerate((("y+", y_plus, r"$\tau = y^{+}$"),
                                                 ("thresh", y_thresh, r"$\tau = y_{\mathrm{thresh}}$"))):
            ax = axes[r, c]
            pick = b[f"{crit}@{tname}"]
            # The batch boundary: the curve of equal score through the BATCH-th candidate.
            # Everything on the shaded side of it is acquired.
            cut = np.sort(score_fn[crit](mu, s, tau))[-BATCH]
            grid = score_fn[crit](gx, gy, tau)
            ax.contourf(gx, gy, grid, levels=[cut, np.inf], colors=[COLOUR_PICK], alpha=0.10)
            ax.contour(gx, gy, grid, levels=[cut], colors=[COLOUR_PICK], linewidths=1.1)
            ax.axvline(greedy_cut, color="0.2", lw=0.9, ls="--", zorder=1)
            ax.scatter(mu[~is_hit], s[~is_hit], s=5, c=COLOUR_OTHER, lw=0, zorder=2)
            ax.scatter(mu[is_hit], s[is_hit], s=9, c=COLOUR_HIT, lw=0, zorder=3)
            ax.scatter(mu[pick], s[pick], s=42, facecolors="none", edgecolors=COLOUR_PICK,
                       lw=1.0, zorder=4)
            ax.plot([tau], [Y_MIN], marker="^", ms=6, color=COLOUR_PICK, clip_on=False, zorder=6)
            ax.set_title(f"{crit}, {label}", fontsize=9, loc="left")
            ax.text(0.03, 0.05,
                    f"hits in batch: {int(is_hit[pick].sum())} of {BATCH}"
                    f"\ngreedy: {int(is_hit[b['greedy']].sum())} of {BATCH}",
                    transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8"), zorder=5)
            ax.set_xlim(x0, x1)
            ax.set_yscale("log")
            ax.set_ylim(Y_MIN, y1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(r"predicted mean $\mu$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"predicted uncertainty $\sigma$")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls="", marker="o", ms=3, color=COLOUR_OTHER, label="candidate"),
        Line2D([], [], ls="", marker="o", ms=3.5, color=COLOUR_HIT, label="true hit"),
        Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=COLOUR_PICK, label=f"acquired batch ({BATCH})"),
        Line2D([], [], color=COLOUR_PICK, lw=1.1, label="batch boundary"),
        Line2D([], [], color="0.2", ls="--", lw=0.9, label="greedy batch boundary"),
        Line2D([], [], ls="", marker="^", ms=6, color=COLOUR_PICK, label=r"reference value $\tau$"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    save(fig, SAVE)


def main() -> None:
    if PRINT_ALL_TARGETS:
        from benchmark_al import TARGETS
        for t in TARGETS:
            print_counts(t, *campaign_state(t))
    state = campaign_state(TARGET)
    print_counts(TARGET, *state)
    figure(*state)


if __name__ == "__main__":
    main()
