"""scripts/plot_epoch_curves.py — the epoch-level validation curves. Press Run.

Draws the figure the 2026-09-11 supervision specified for an epoch-mode run
(run_experiment.py with n_epochs > 0):

    thin line with dots : validation value at each epoch (on the WHOLE validation set)
    solid line          : windowed average over k = 3 epochs

for both validation loss and validation Spearman, each against the untrained
encoder's epoch-0 value (dashed). The best epoch by validation loss is marked, and the
reason the run stopped (early stopping, deadline, or all epochs done) is printed in the
title. Works on a run that is still going — it plots whatever epochs exist.

matplotlib is deliberately not a project dependency. Run with:

    uv run --with matplotlib python src/scripts/plot_epoch_curves.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
CURVE_CSV = "experiments/val_curve_full_sp_50ep.csv"
SAVE_TO = "experiments/fig_val_epochs_full_sp_50ep.pdf"
SAVE_PNG_TOO = True
WINDOW = 3                      # k for the windowed average, in epochs
TITLE = "Validation over epochs — clean single-protein ChEMBL, lr 3e-5"


def windowed_mean(v: list[float], k: int) -> list[float]:
    """Centred windowed average; the window shrinks at the ends so no epoch is dropped."""
    h = k // 2
    return [sum(v[max(0, i - h):i + h + 1]) / len(v[max(0, i - h):i + h + 1])
            for i in range(len(v))]


def main() -> None:
    path = REPO_ROOT / CURVE_CSV
    if not path.exists():
        raise SystemExit(f"curve not found: {path}")
    rows = list(csv.DictReader(path.open()))
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["val_loss"]) for r in rows]
    rho = [float(r["val_spearman"]) for r in rows]
    events = {int(r["epoch"]): r["event"] for r in rows}
    best_ep = min(range(len(loss)), key=lambda i: loss[i])
    last_event = rows[-1]["event"]
    status = {"early_stop": f"early stopping at epoch {ep[-1]}",
              "deadline": f"stopped at epoch {ep[-1]} by the deadline"}.get(
        last_event, f"{ep[-1]} epochs so far")
    print(f"{len(rows)} rows (epochs 0..{ep[-1]}), best val loss {loss[best_ep]:.4f} at "
          f"epoch {ep[best_ep]}; {status}")

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.constrained_layout.use": True})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True)

    for ax, v, colour, label, better in (
            (ax1, loss, "#1f4e79", "validation loss (NLL)", "lower is better"),
            (ax2, rho, "#8c3b00", "validation Spearman", "higher is better")):
        ax.plot(ep, v, color=colour, lw=0.8, alpha=0.45, marker="o", ms=3.2, mew=0,
                label="per epoch", zorder=2)
        ax.plot(ep, windowed_mean(v, WINDOW), color=colour, lw=2.0, zorder=3,
                label=f"windowed average (k = {WINDOW})")
        ax.axhline(v[0], color="0.25", ls="--", lw=1.0, zorder=1,
                   label=f"untrained, epoch 0 ({v[0]:.3f})")
        ax.axvline(ep[best_ep], color="0.55", ls=":", lw=1.0, zorder=1)
        ax.set_ylabel(f"{label}\n({better})", fontsize=8.5)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.92, handlelength=1.6)

    # Label the best-epoch marker along the panel FLOOR: the top edge is where the
    # untrained baseline usually sits, and a label there collides with its dashed line.
    ax1.annotate(f"best epoch {ep[best_ep]}", xy=(ep[best_ep], ax1.get_ylim()[0]),
                 xytext=(3, 4), textcoords="offset points", fontsize=7.5, color="0.4")
    ax1.set_title(f"{TITLE}\n({status})", fontsize=9.5, loc="left")
    ax2.set_xlabel("epoch", fontsize=9)
    step = 1 if ep[-1] <= 20 else 5
    ax2.set_xticks(range(0, ep[-1] + 1, step))

    out = REPO_ROOT / SAVE_TO
    fig.savefig(out)
    print(f"wrote {out}")
    if SAVE_PNG_TOO:
        fig.savefig(out.with_suffix(".png"), dpi=200)
        print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
