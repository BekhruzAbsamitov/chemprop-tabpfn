"""scripts/plot_curves.py — the 10-epoch validation curves. Press Run.

The 2026-08-27 supervision asked for validation loss and Spearman plotted over the first
ten epochs, with an explicit answer to whether there is a trend to convergence. This draws
that figure from the curve CSV written by run_experiment.py.

WHAT THE FIGURE HAS TO SHOW HONESTLY. Single evaluations are noisy — the eval-to-eval
standard deviation is about +-0.045 on Spearman, which is the same size as the entire
effect. Plotting only a smoothed line would hide that; plotting only the raw points would
hide the trend. So both are drawn: the raw measurements faintly, and a centred rolling
mean on top.

THE BASELINE IS THE POINT. Every curve is drawn against a dashed horizontal line at the
step-0 value — the untrained encoder the run started from. Without it a reader sees a
falling loss and infers success; with it they see where the curve ends up relative to
doing nothing at all, which is the actual result.

matplotlib is deliberately NOT a project dependency (it is needed by this script alone).
Run it with:

    uv run --with matplotlib python src/scripts/plot_curves.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display on the cluster or in CI
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
CURVE_CSV = "experiments/train_val_curve_full_sp_10ep.csv"
SAVE_TO = "experiments/fig_val_curves_10ep.pdf"
SAVE_PNG_TOO = True          # a PNG alongside the PDF, for slides
STEPS_PER_EPOCH = 31384      # usable episodes per pass over the curated set
SMOOTH_WINDOW = 8            # rolling-mean width, in evaluations (8 = one epoch)
CONVERGED_FROM = 5.0         # epoch at which the trend test finds no further movement


def read_curve(path: Path) -> tuple[list[float], list[float], list[float]]:
    """(epochs, val_loss, val_spearman) — step 0 included as the baseline."""
    epochs, loss, rho = [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["step"]) / STEPS_PER_EPOCH)
            loss.append(float(row["val_loss"]))
            rho.append(float(row["val_spearman"]))
    return epochs, loss, rho


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Centred rolling mean, shrinking at the ends so the line spans the whole axis."""
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def panel(ax, epochs, values, baseline, *, colour, ylabel, better):
    """One metric: raw points, rolling mean, and the step-0 baseline."""
    ax.axvspan(CONVERGED_FROM, max(epochs), color="0.93", zorder=0)
    ax.plot(epochs[1:], values[1:], color=colour, lw=0.8, alpha=0.35, zorder=2,
            marker="o", ms=2.2, mew=0, label="per evaluation")
    ax.plot(epochs[1:], rolling_mean(values[1:], SMOOTH_WINDOW), color=colour, lw=2.0,
            zorder=3, label="rolling mean (1 epoch)")
    ax.axhline(baseline, color="0.25", ls="--", lw=1.1, zorder=1,
               label=f"untrained baseline ({baseline:.3f})")
    # The baseline value differs per panel, so each panel carries its own legend.
    # loc="best" keeps it clear of the data — an annotation pinned to the line masked
    # the Spearman curve, which runs through the right-hand margin.
    ax.legend(loc="best", fontsize=7.5, framealpha=0.92, borderpad=0.4,
              handlelength=1.6, labelspacing=0.3).set_zorder(5)
    ax.set_ylabel(f"{ylabel}\n({better})", fontsize=9)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    path = REPO_ROOT / CURVE_CSV
    if not path.exists():
        raise SystemExit(f"curve not found: {path}")
    epochs, loss, rho = read_curve(path)
    print(f"{len(epochs)} evaluations spanning {max(epochs):.2f} epochs")

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.constrained_layout.use": True})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True)

    panel(ax1, epochs, loss, loss[0], colour="#1f4e79",
          ylabel="validation loss (NLL)", better="lower is better")
    panel(ax2, epochs, rho, rho[0], colour="#8c3b00",
          ylabel="validation Spearman", better="higher is better")

    ax2.set_xlabel("training epochs", fontsize=9)
    ax2.set_xticks(range(0, int(max(epochs)) + 1))
    ax1.set_title("Validation curves over 10 epochs, clean single-protein ChEMBL",
                  fontsize=10, loc="left")
    # Label the shaded band once, in the lower panel: the upper one's legend sits in the
    # only free corner there, and this panel has clear space along its floor.
    lo, hi = ax2.get_ylim()
    ax2.annotate(f"no significant trend after epoch {CONVERGED_FROM:.0f}",
                 xy=(CONVERGED_FROM + 0.2, lo + 0.04 * (hi - lo)),
                 fontsize=8, color="0.35", va="bottom", ha="left")

    out = REPO_ROOT / SAVE_TO
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")
    if SAVE_PNG_TOO:
        png = out.with_suffix(".png")
        fig.savefig(png, dpi=200)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
