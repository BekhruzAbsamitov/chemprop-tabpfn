"""scripts/plot_lr_comparison.py — do lower learning rates quiet the curve? Press Run.

The 2026-09-03 supervision observed that the per-evaluation line in the 10-epoch
validation figure fluctuates heavily and suggested a lower learning rate. This draws the
three runs that test it, all identical apart from the rate:

    baseline   lr = 1e-4, constant   (the run that prompted the comment)
    lowlr      lr = 3e-5, constant   (the literal suggestion)
    cosine     lr = 1e-4 -> 0        (annealed; sweeps the whole range inside ONE run)

WHY A SCHEDULE AND NOT ONLY A LOWER RATE. Gradients exceed grad_clip_norm=1.0 on 100 %
of steps, so every update is rescaled to exactly lr x 1.0 and its magnitude never decays
on its own. Comparing two constant rates confounds "smaller steps" with "a different
run" — and the step-0 spread below shows run-to-run variation is large. The cosine run
avoids that entirely: the rate falls from 1e-4 to ~0 within a single run, so the
fluctuation can be measured against the rate with everything else held fixed.

WHAT THE FIGURE SHOWS. Left column: the curves themselves, raw evaluations plus a
one-epoch rolling mean, each against its own step-0 baseline. Right column: the
fluctuation *around* the local trend, binned by epoch — this is the quantity the
supervision was actually commenting on, and plotting it directly is what turns an
impression into a measurement.

matplotlib is deliberately not a project dependency. Run with:

    uv run --with matplotlib python src/scripts/plot_lr_comparison.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
RUNS = [
    ("baseline: lr 1e-4 constant", "train_val_curve_full_sp_10ep.csv",       "#1f4e79"),
    ("lower:    lr 3e-5 constant", "train_val_curve_full_sp_10ep_lowlr.csv", "#8c3b00"),
    # The annealed run is measured but not plotted — its numbers are reported as a table
    # in §5h instead. Re-add the line below to restore the third panel:
    # ("cosine:   lr 1e-4 -> 0",   "train_val_curve_full_sp_10ep_cosine.csv", "#2d6a3e"),
]
CURVE_DIR = "experiments"
SAVE_TO = "experiments/fig_lr_comparison.pdf"
SAVE_PNG_TOO = True
STEPS_PER_EPOCH = 31384
SMOOTH_WINDOW = 8      # evaluations; 8 = one epoch
BIN_EPOCHS = 2         # width of the fluctuation bins on the right


def read(path: Path):
    steps, loss, rho, lr = [], [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]) / STEPS_PER_EPOCH)
            loss.append(float(row["val_loss"]))
            rho.append(float(row["val_spearman"]))
            lr.append(float(row["lr"]) if row.get("lr") else float("nan"))
    return steps, loss, rho, lr


def rolling_mean(v, w):
    h = w // 2
    return [sum(v[max(0, i - h):min(len(v), i + h + 1)])
            / len(v[max(0, i - h):min(len(v), i + h + 1)]) for i in range(len(v))]


def residual_std(v, w=SMOOTH_WINDOW):
    """Spread of the raw points AROUND their local trend — the 'fluctuation'."""
    m = rolling_mean(v, w)
    return st.stdev([a - b for a, b in zip(v, m)]) if len(v) > 2 else float("nan")


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.constrained_layout.use": True})
    n = len(RUNS)
    fig, axes = plt.subplots(n, 3, figsize=(11.4, 2.3 * n),
                             gridspec_kw={"width_ratios": [1.5, 1.5, 1.0]})
    if n == 1:
        axes = [axes]

    def curve(ax, ep, v, baseline, colour, ylabel, ylim):
        """Raw evaluations, a one-epoch rolling mean, and this run's own step-0 value."""
        ax.plot(ep[1:], v[1:], color=colour, lw=0.7, alpha=0.33, marker="o", ms=2, mew=0)
        ax.plot(ep[1:], rolling_mean(v[1:], SMOOTH_WINDOW), color=colour, lw=2.0)
        ax.axhline(baseline, color="0.25", ls="--", lw=1.0)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for row, (label, fname, colour) in enumerate(RUNS):
        path = REPO_ROOT / CURVE_DIR / fname
        if not path.exists():
            print(f"skipping {fname} — not found")
            continue
        ep, loss, rho, lr = read(path)

        curve(axes[row][0], ep, loss, loss[0], colour,
              "val loss (NLL)\n(lower is better)", (1.90, 2.30))
        axes[row][0].set_title(f"{label}   —   NLL, baseline {loss[0]:.3f}",
                               fontsize=9, loc="left")

        curve(axes[row][1], ep, rho, rho[0], colour,
              "val Spearman\n(higher is better)", (0.18, 0.44))
        axes[row][1].set_title(f"Spearman, baseline {rho[0]:.3f}", fontsize=9, loc="left")

        # Fluctuation for BOTH metrics side by side, so the difference between them is
        # readable: NLL is a mean over episodes, Spearman a rank correlation over ~7
        # query molecules each, and the second is intrinsically the noisier statistic.
        axf = axes[row][2]
        centres, f_loss, f_rho = [], [], []
        for lo in range(0, 10, BIN_EPOCHS):
            idx = [i for i, e in enumerate(ep) if lo < e <= lo + BIN_EPOCHS]
            if len(idx) >= 4:
                centres.append(lo + BIN_EPOCHS / 2)
                f_loss.append(residual_std([loss[i] for i in idx]))
                f_rho.append(residual_std([rho[i] for i in idx]))
        w = BIN_EPOCHS * 0.36
        axf.bar([c - w / 2 for c in centres], f_loss, width=w, color=colour, alpha=0.45,
                label="NLL")
        axf.bar([c + w / 2 for c in centres], f_rho, width=w, color=colour, alpha=0.95,
                label="Spearman")
        axf.set_ylim(0, 0.06)
        axf.set_ylabel("fluctuation", fontsize=8.5)
        axf.grid(axis="y", color="0.92", lw=0.6)
        axf.set_axisbelow(True)
        for s in ("top", "right"):
            axf.spines[s].set_visible(False)
        axf.set_title(f"whole run: NLL {residual_std(loss[1:]):.4f}, "
                      f"rho {residual_std(rho[1:]):.4f}", fontsize=8, loc="left",
                      color="0.3")
        if row == 0:
            axf.legend(fontsize=7, loc="upper left", framealpha=0.9, handlelength=1.2)

    for c in range(3):
        axes[-1][c].set_xlabel("training epochs", fontsize=9)
        for r in range(n):
            axes[r][c].set_xticks(range(0, 11, 2))

    out = REPO_ROOT / SAVE_TO
    fig.savefig(out)
    print(f"wrote {out}")
    if SAVE_PNG_TOO:
        fig.savefig(out.with_suffix(".png"), dpi=200)
        print(f"wrote {out.with_suffix('.png')}")

    print("\nstep-0 = the SAME untrained encoder on the SAME val episodes:")
    br, bl = [], []
    for label, fname, _ in RUNS:
        q = REPO_ROOT / CURVE_DIR / fname
        if q.exists():
            _, loss, rho, _ = read(q)
            br.append(rho[0]); bl.append(loss[0])
            print(f"  {label:28s} rho {rho[0]:.4f}   NLL {loss[0]:.4f}")
    if len(br) > 1:
        print(f"  spread: rho {max(br)-min(br):.4f}, NLL {max(bl)-min(bl):.4f}")


if __name__ == "__main__":
    main()
