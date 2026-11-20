"""scripts/plot_epoch_lr_comparison.py — epoch-level curves, lr 3e-5 vs lr 1e-4. Press Run.

The 2026-09-11 supervision's figure (thin line with a dot per epoch, solid windowed
average k = 3), with both learning rates on the same axes.

THE TWO RUNS DO NOT SHARE A VALIDATION PROTOCOL, so raw values cannot be overlaid:
  lr 3e-5  full_sp_50ep  epoch mode: the WHOLE validation set (40,966 points, point-
                         weighted NLL) once per epoch.
  lr 1e-4  full_sp_10ep  step mode: 64 validation episodes every 3,923 steps (1/8
                         epoch). Only the evaluation at each epoch boundary is used,
                         so it has one dot per epoch like the other run.
Each run is therefore drawn as the CHANGE FROM ITS OWN UNTRAINED ENCODER (epoch 0).
The zero line is "no better than untrained" for both. The absolute epoch-0 values are
in the legend.

matplotlib is deliberately not a project dependency. Run with:

    uv run --with matplotlib python src/scripts/plot_epoch_lr_comparison.py
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
STEPS_PER_EPOCH = 31384         # one pass over the 34,192 training assays (usable ones)
WINDOW = 3                      # k for the windowed average, in epochs
SAVE_TO = "experiments/fig_val_epochs_lr_comparison.pdf"
SAVE_PNG_TOO = True
# (label, csv, mode, colour). mode "epoch" = epoch-mode CSV; "step" = step-mode CSV.
RUNS = [
    ("lr 3e-5 — full validation (40,966 pts)", "experiments/val_curve_full_sp_50ep.csv",
     "epoch", "#1f4e79"),
    ("lr 1e-4 — 64 validation episodes", "experiments/train_val_curve_full_sp_10ep.csv",
     "step", "#b5451b"),
]


def windowed_mean(v: list[float], k: int) -> list[float]:
    """Centred windowed average; the window shrinks at the ends so no epoch is dropped."""
    h = k // 2
    return [sum(v[max(0, i - h):i + h + 1]) / len(v[max(0, i - h):i + h + 1])
            for i in range(len(v))]


def load(path: str, mode: str) -> tuple[list[int], list[float], list[float]]:
    rows = list(csv.DictReader((REPO_ROOT / path).open()))
    if mode == "epoch":
        return ([int(r["epoch"]) for r in rows], [float(r["val_loss"]) for r in rows],
                [float(r["val_spearman"]) for r in rows])
    # Step mode: keep the evaluation nearest each epoch boundary. The final step
    # (313,840) is 0.0006 epochs past boundary 10 and counts as epoch 10.
    by_epoch: dict[int, dict] = {}
    for r in rows:
        s = int(r["step"])
        e = round(s / STEPS_PER_EPOCH)
        if abs(s - e * STEPS_PER_EPOCH) <= 100:
            by_epoch[e] = r
    ep = sorted(by_epoch)
    return (ep, [float(by_epoch[e]["val_loss"]) for e in ep],
            [float(by_epoch[e]["val_spearman"]) for e in ep])


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "figure.constrained_layout.use": True})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)
    last_epoch = 0

    for label, path, mode, colour in RUNS:
        ep, loss, rho = load(path, mode)
        last_epoch = max(last_epoch, ep[-1])
        best = min(range(len(loss)), key=lambda i: loss[i])
        d_loss = [v - loss[0] for v in loss]
        d_rho = [v - rho[0] for v in rho]
        print(f"{label}: epochs 0..{ep[-1]}, epoch-0 loss {loss[0]:.4f} rho {rho[0]:.4f}; "
              f"best loss at epoch {ep[best]} ({d_loss[best]:+.4f}, rho {d_rho[best]:+.4f}); "
              f"last epoch {d_loss[-1]:+.4f} / rho {d_rho[-1]:+.4f}")
        for ax, d, base in ((ax1, d_loss, loss[0]), (ax2, d_rho, rho[0])):
            ax.plot(ep, d, color=colour, lw=0.8, alpha=0.45, marker="o", ms=3.2, mew=0,
                    zorder=2)
            ax.plot(ep, windowed_mean(d, WINDOW), color=colour, lw=2.0, zorder=3,
                    label=f"{label}  (epoch 0 = {base:.3f})")
            ax.plot(ep[best], d[best], marker="*", ms=10, color=colour, mec="white",
                    mew=0.6, zorder=4)

    for ax, name, better in ((ax1, "Δ validation loss (NLL)", "lower is better"),
                             (ax2, "Δ validation Spearman", "higher is better")):
        ax.axhline(0, color="0.25", ls="--", lw=1.0, zorder=1)
        ax.set_ylabel(f"{name}\nvs untrained ({better})", fontsize=8.5)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # One legend explains the encoding; the per-run legends name the runs.
    ax1.legend(loc="upper right", fontsize=7.3, framealpha=0.92, handlelength=1.6)
    style = [plt.Line2D([], [], color="0.4", lw=0.8, marker="o", ms=3.2, alpha=0.6,
                        label="per epoch"),
             plt.Line2D([], [], color="0.4", lw=2.0, label=f"windowed average (k = {WINDOW})"),
             plt.Line2D([], [], color="0.4", ls="none", marker="*", ms=9,
                        label="best epoch (val loss)"),
             plt.Line2D([], [], color="0.25", ls="--", lw=1.0, label="untrained encoder")]
    # Below the axes: inside, every corner covers some run's dot.
    ax2.legend(handles=style, loc="upper center", bbox_to_anchor=(0.5, -0.2),
               fontsize=7.3, frameon=False, handlelength=1.8, ncol=4)

    ax1.set_title("Validation over epochs — clean single-protein ChEMBL\n"
                  "each run relative to its own untrained encoder (validation protocols differ)",
                  fontsize=9.5, loc="left")
    ax2.set_xlabel("epoch", fontsize=9)
    ax2.set_xticks(range(0, last_epoch + 1, 1 if last_epoch <= 20 else 5))

    out = REPO_ROOT / SAVE_TO
    fig.savefig(out)
    print(f"wrote {out}")
    if SAVE_PNG_TOO:
        fig.savefig(out.with_suffix(".png"), dpi=200)
        print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
