"""scripts/plot_training.py — the figures of Chapter 5 and Appendix D (training through a frozen TabPFN). Press Run.

Chapter 5 follows ONE encoder, the epoch-level run at its last epoch (26), averaged over its two seeds, and shows it
in three figures: fig:transfer_gain, fig:val_epoch_level and fig:epoch_change. The figures of the other training
settings in Appendix D are fig:appd_transfer_gain_all and fig:rank_transfer. Three further figures on those
settings are still written but no longer used in the thesis: fig:val_single_protein, fig:cka_trajectory and
fig:effective_rank.

FIGURE fig:transfer_gain (transfer, Chapter 5). The epoch-level encoder at epoch 26 minus its own random
initialisation: the mean of the two seeds (per matched pair, the mean of the two differences, each seed against
its own control), each seed, and CheMeleon. Same layout and units as fig:appd_transfer_gain_all below.

FIGURE fig:val_epoch_level (in-domain validation, Chapter 5). Validation NLL and Spearman correlation of the
epoch-level run on every held-out point, once per epoch, as the change from epoch 0, for both seeds; a star
marks each seed's lowest validation loss.

FIGURE fig:epoch_change (change and geometry of the epoch-level run, Chapter 5). Left: linear CKA between each
epoch's checkpoint and the run's own random initialisation, both seeds (cka_epoch_level.csv, written by
cka_epoch_level.py), with the CKA between the initialisation and CheMeleon (dashed, per seed) and between the
two initialisations (dotted). Right: effective rank of the standardised embeddings relative to the random
initialisation, both seeds (feature_stats_standardised.csv).

FIGURE fig:val_single_protein (in-domain validation; not in the thesis). Validation NLL and Spearman correlation on the held-out
assays of the single-protein corpus, each run drawn as the change from its OWN random initialisation
encoder (step or epoch 0), because the validation protocols differ:
  lr 1e-4   full_sp_10ep        64 validation episodes every 1/8 epoch (step-based)
  lr 3e-5   full_sp_10ep_lowlr  the same 64 episodes, same schedule (step-based)
  lr 3e-5   full_sp_50ep        every held-out assay, 40,966 points, once per epoch
The step-based runs are drawn as faint per-evaluation dots and a one-epoch rolling mean.
Epochs 25 and 26 of the epoch-level run were not logged (cluster restart) and are taken
from their re-evaluation (val_reevaluated_full_sp_50ep.csv).

FIGURE fig:effective_rank (effective rank; not in the thesis). Effective rank of the embeddings of a fixed set of held-out
molecules AFTER every column has been standardised, which is what TabPFN receives
(feature_stats_standardised.py), relative to the encoder's own random initialisation.
Left: the four 100,000-step runs at lr 1e-4. Right: the two single-protein runs over
epochs, lr 1e-4 (10 epochs) and lr 3e-5 (the epoch-level run).

FIGURE fig:cka_trajectory (CKA trajectory; not in the thesis). CKA between each checkpoint of the three 100,000-step runs and the
run's own initialisation, with the CKA between the initialisation and CheMeleon, and between the
initialisation and a second random initialisation, as reference lines. CKA_ESTIMATOR selects the
linear estimator (cka_trajectory.csv) or the debiased one (cka_debiased.csv, cka_debiased.py).

FIGURE fig:appd_transfer_gain_all (transfer of every encoder, Appendix D). Every trained encoder minus ITS OWN random initialisation (the
seed-1 encoders against the seed-1 control, all others against the seed-0 control), in final CEF of
the final benchmark run (experiments/benchmark_final: final_*.csv and seed1_control_*.csv). A
MATCHED PAIR is one target, surrogate (TabPFN full distribution, RF, RBF GP), strategy (greedy, EI and
PI at y+ and y_thresh) and configuration (initial-set size, batch size); the unit of inference is
the SCENARIO, one target in one configuration (4 x 4 = 16), whose value is the mean of its 15 pairs.
All-targets column: mean of the scenarios +- 2 SE over scenarios, scenarios won out of 16. Target
columns: the mean of the target's four scenarios, which are drawn as small dots (no SE from four
values); a marker is filled when all four have the sign of the mean.

FIGURE fig:rank_transfer (rank against transfer, Appendix D). For every benchmarked trained encoder whose geometry
was measured: the change in standardised effective rank from its random initialisation (x) against its
transfer over its own random initialisation (y, mean over scenarios +- 2 SE). Filled: the single-protein
checkpoints (circles: first seed, which share the corpus, the molecules of the rank and the initial
weights; diamonds: second seed, same corpus and molecules, other initial weights); hollow: the long
domain-matched encoders, whose ranks come from other molecule sets and whose transfer is measured on
their own target only (bar = range of its four scenarios).

The script also prints the numbers Chapter 5 quotes from these data: the validation
table, the noise of the curves and the feature-statistics table.

matplotlib is deliberately NOT a project dependency (it is needed by the plot scripts alone).
Run it with:

    uv run --with matplotlib python src/scripts/plot_training.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
EXP = "experiments"
EPOCH_CSV = "experiments/val_curve_full_sp_50ep.csv"
EPOCH_REEVAL_CSV = "experiments/val_reevaluated_full_sp_50ep.csv"
EPOCH_SEED2_CSV = "experiments/val_curve_full_sp_50ep_seed1.csv"
FEATURE_STD_CSV = "experiments/feature_stats_standardised.csv"   # run feature_stats_standardised.py first
SAVE_VALIDATION = "experiments/fig_val_single_protein.pdf"      # Appendix D: the single-protein runs
SAVE_VALIDATION_EPOCH = "experiments/fig_val_epoch_level.pdf"   # Chapter 5: the epoch-level run, both seeds
SAVE_RANK = "experiments/fig_effective_rank.pdf"
SAVE_EPOCH_CHANGE = "experiments/fig_epoch_change.pdf"          # Chapter 5: CKA and effective rank per epoch
CKA_EPOCH_CSV = "experiments/cka_epoch_level.csv"               # run cka_epoch_level.py first
SAVE_CKA = "experiments/fig_cka_trajectory.pdf"
SAVE_GEOMETRY_TRANSFER = "experiments/fig_rank_transfer.pdf"
SAVE_TRANSFER = "experiments/fig_transfer_gain.pdf"             # Chapter 5: the epoch-level encoder and CheMeleon
SAVE_TRANSFER_ALL = "experiments/fig_transfer_gain_all.pdf"     # Appendix D: every trained encoder
SAVE_PNG_TOO = True

CKA_TRAJECTORY_CSV = "experiments/cka_trajectory.csv"      # linear CKA (cka_study.py)
CKA_DEBIASED_CSV = "experiments/cka_debiased.csv"          # both estimators (cka_debiased.py)
CKA_ESTIMATOR = "linear"         # "linear" (as in the chapter's tables) or "debiased"

# The final benchmark run (benchmark_extended.py "final") and the second-seed control (experiment "seed1_control").
FINAL_DIR = "experiments/benchmark_final"
RUN_GLOBS = ["final_*.csv", "seed1_control_*.csv"]
N_SEEDS = 5
N_ROUNDS = 20
CANDIDATES = "1000"              # the main protocol
NATIVE_DIST = {"tabpfn": "full", "rf": "gauss", "gp": "gauss"}   # the three main surrogates, native predictive form
PAIR_KEY = ["target", "model", "criterion", "init_size", "batch_size"]
SCENARIO = ["target", "init_size", "batch_size"]
PAIRS_PER_SCENARIO = 15          # 3 surrogates x 5 strategies
CONTROL = "chemprop_untrained"           # random initialisation of the first seed
CONTROL_SEED1 = "chemprop_untrained_seed1"   # random initialisation of the second seed (seed1_control)

# Figures "transfer": a trained encoder minus ITS OWN random initialisation, per target and over all targets.
# (row label, representation, control; a string = a group heading). An averaged row has a tuple of
# (representation, control) arms in place of the representation and control None: the mean of the differences.
E26 = ("chemprop_trained_full_sp_50ep_final", CONTROL)
S26 = ("chemprop_trained_full_sp_50ep_seed1_final", CONTROL_SEED1)
TRANSFER_MAIN_ROWS = [           # Chapter 5
    r"epoch-level encoder, epoch 26",
    ("Mean of the two seeds", (E26, S26), None),
    ("First seed", *E26),
    ("Second seed", *S26),
    "reference",
    ("CheMeleon", "chemeleon", CONTROL),
]
TRANSFER_TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
TRANSFER_TARGET_NAMES = {"saureus": r"$S.\ aureus$", "tyk2": "TYK2", "usp7": "USP7", "d2r": "D2R", "all": "all targets"}
TRANSFER_ROWS = [                # Appendix D: every trained encoder
    r"rate $10^{-4}$",
    ("All of ChEMBL", "chemprop_trained_full", CONTROL),
    ("Domain-matched", "chemprop_trained_own", CONTROL),
    ("Domain-matched, long", "chemprop_trained_own_long", CONTROL),
    ("100k steps, final", "chemprop_trained_full_sp", CONTROL),
    ("100k steps, best val.", "chemprop_trained_full_sp_best", CONTROL),
    ("10 epochs, final", "chemprop_trained_full_sp_10ep", CONTROL),
    ("10 epochs, best val.", "chemprop_trained_full_sp_10ep_best", CONTROL),
    r"rate $3 \times 10^{-5}$ (epoch-level)",
    ("First seed, best (epoch 23)", "chemprop_trained_full_sp_50ep_best", CONTROL),
    ("First seed, last (epoch 26)", "chemprop_trained_full_sp_50ep_final", CONTROL),
    ("Second seed, best (epoch 21)", "chemprop_trained_full_sp_50ep_seed1_best", CONTROL_SEED1),
    ("Second seed, last (epoch 26)", "chemprop_trained_full_sp_50ep_seed1_final", CONTROL_SEED1),
    "reference",
    ("CheMeleon", "chemeleon", CONTROL),
]
NOT_INTERPRETED = {("chemprop_trained_full", "saureus")}   # 72 % same-target overlap (Section 3.2)
GAIN, LOSS, REFERENCE_GREY = "#1f4e79", "#b2182b", "0.55"

# Figure "rank against transfer": every benchmarked trained encoder whose geometry was measured.
# (label, feature_stats run, step or epoch, representation, control, target or None = all, rate, seed)
RANK_TRANSFER_POINTS = [
    ("D2R, long", "d2r_long", 100_000, "chemprop_trained_own_long", CONTROL, "d2r", "1e-4", 0),
    ("S. aureus, long", "saureus_long", 100_000, "chemprop_trained_own_long", CONTROL, "saureus", "1e-4", 0),
    ("USP7, long", "usp7_long", 100_000, "chemprop_trained_own_long", CONTROL, "usp7", "1e-4", 0),
    ("100k steps, final", "full_sp", 100_000, "chemprop_trained_full_sp", CONTROL, None, "1e-4", 0),
    ("100k steps, best", "full_sp", 64_000, "chemprop_trained_full_sp_best", CONTROL, None, "1e-4", 0),
    ("10 epochs, final", "full_sp_10ep", 313_840, "chemprop_trained_full_sp_10ep", CONTROL, None, "1e-4", 0),
    ("10 epochs, best", "full_sp_10ep", 156_920, "chemprop_trained_full_sp_10ep_best", CONTROL, None, "1e-4", 0),
    ("epoch 23, seed 0", "full_sp_50ep", 23, "chemprop_trained_full_sp_50ep_best", CONTROL, None, "3e-5", 0),
    ("epoch 26, seed 0", "full_sp_50ep", 26, "chemprop_trained_full_sp_50ep_final", CONTROL, None, "3e-5", 0),
    ("epoch 21, seed 1", "full_sp_50ep_seed1", 21, "chemprop_trained_full_sp_50ep_seed1_best", CONTROL_SEED1, None, "3e-5", 1),
    ("epoch 26, seed 1", "full_sp_50ep_seed1", 26, "chemprop_trained_full_sp_50ep_seed1_final", CONTROL_SEED1, None, "3e-5", 1),
]

FIG_WIDTH = 5.5                  # inches; the thesis text block is 398 pt = 5.51 in
STEPS_PER_EPOCH = 31_384         # single-protein corpus: assays that can form an episode
EVALS_PER_EPOCH = 8              # step-based single-protein runs evaluate every 1/8 epoch
BEST_EPOCH = 23                  # lowest validation loss of the epoch-level run
BEST_EPOCH_SEED2 = 21            # and of its second seed
SEED2_COLOUR = "#6a9fca"         # the second seed, a lighter tone of LR_LOW

LR_HIGH, LR_LOW = "#c45a00", "#1f4e79"
LONG_RUNS = [  # (feature_stats run, curve file, label, colour, line style); orange and blue mean lr
    ("d2r_long", "d2r_long", "D2R, domain-matched", "#2e7d32", "-"),
    ("saureus_long", "saureus_long", r"$S.\ aureus$, domain-matched", "#7b3294", "--"),
    ("usp7_long", "usp7_long", "USP7, domain-matched", "#555555", "-."),
    ("full_sp", "full_sp", "single-protein, 100,000 steps", "#1b9e9e", ":"),
]


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "mathtext.fontset": "dejavuserif",
                         "figure.constrained_layout.use": True})


def clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, rel: str) -> None:
    out = REPO_ROOT / rel
    fig.savefig(out)
    print(f"wrote {out}")
    if SAVE_PNG_TOO:
        fig.savefig(out.with_suffix(".png"), dpi=200)


def curve(name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / EXP / f"train_val_curve_{name}.csv")


def epoch_run() -> pd.DataFrame:
    """Epoch-level run, epochs 0-26: logged values, plus the re-evaluated epochs 25-26."""
    e = pd.read_csv(REPO_ROOT / EPOCH_CSV)[["epoch", "val_loss", "val_spearman"]]
    r = pd.read_csv(REPO_ROOT / EPOCH_REEVAL_CSV)[["epoch", "val_loss", "val_spearman"]]
    return pd.concat([e, r[~r.epoch.isin(e.epoch)]]).sort_values("epoch").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figure 5.1: in-domain validation on the single-protein corpus               #
# --------------------------------------------------------------------------- #
def figure_validation() -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.45), sharex=True)
    e = epoch_run()
    step_runs = [("full_sp_10ep", r"$10^{-4}$, 64 episodes", LR_HIGH, "-"),
                 ("full_sp_10ep_lowlr", r"$3 \times 10^{-5}$, 64 episodes", LR_LOW, "--")]
    for ax, col, ylab in [(axes[0], "val_loss", "$\\Delta$ validation NLL"),
                          (axes[1], "val_spearman", "$\\Delta$ validation Spearman")]:
        for name, label, c, ls in step_runs:
            q = curve(name)
            x = q.step / STEPS_PER_EPOCH
            d = q[col] - q[col].iloc[0]
            ax.plot(x[1:], d[1:], ls="none", marker="o", ms=1.4, color=c, alpha=0.18)
            ax.plot(x, d.rolling(EVALS_PER_EPOCH, center=True, min_periods=EVALS_PER_EPOCH // 2).mean(),
                    color=c, lw=1.3, ls=ls, label=f"{label}, one-epoch mean")
        d = e[col] - e[col].iloc[0]
        ax.plot(e.epoch, d, color=LR_LOW, lw=1.3, marker="o", ms=2.6,
                label=r"$3 \times 10^{-5}$, every held-out point, per epoch")
        b = e[e.epoch == BEST_EPOCH]
        ax.plot(b.epoch, b[col] - e[col].iloc[0], marker="*", ms=10, color=LR_LOW, ls="none", mec="white", mew=0.6, zorder=6,
                label=f"lowest validation loss (epoch {BEST_EPOCH})")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(ylab, fontsize=8.5, loc="left")
        ax.set_xlabel("training epochs")
        ax.set_xlim(0, 26.5)
        clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    save(fig, SAVE_VALIDATION)


# --------------------------------------------------------------------------- #
# Figure (Chapter 5): in-domain validation of the epoch-level run, both seeds #
# --------------------------------------------------------------------------- #
def figure_validation_epoch() -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.25), sharex=True)
    second = pd.read_csv(REPO_ROOT / EPOCH_SEED2_CSV)[["epoch", "val_loss", "val_spearman"]]
    runs = [(epoch_run(), "first seed", LR_LOW, "o", BEST_EPOCH),
            (second, "second seed", SEED2_COLOUR, "D", BEST_EPOCH_SEED2)]
    print("\nFIGURE VALIDATION, EPOCH-LEVEL RUN — validation NLL and Spearman: epoch 0 -> best epoch -> epoch 26")
    for ax, col, ylab in [(axes[0], "val_loss", "$\\Delta$ validation NLL"),
                          (axes[1], "val_spearman", "$\\Delta$ validation Spearman")]:
        for e, label, c, marker, best in runs:
            d = e[col] - e[col].iloc[0]
            ax.plot(e.epoch, d, color=c, lw=1.2, marker=marker, ms=2.4, label=label)
            b = e[e.epoch == best]
            ax.plot(b.epoch, b[col] - e[col].iloc[0], marker="*", ms=10, color=c, ls="none", mec="white", mew=0.6,
                    zorder=6, label=f"lowest validation loss, {label} (epoch {best})")
            last = e[e.epoch == 26][col].iloc[0]
            print(f"  {col:13s} {label:11s} {e[col].iloc[0]:.4f} -> {b[col].iloc[0]:.4f} (epoch {best}, "
                  f"{100 * (b[col].iloc[0] / e[col].iloc[0] - 1):+.1f} %) -> {last:.4f} (epoch 26, "
                  f"{100 * (last / e[col].iloc[0] - 1):+.1f} %);  median |change| between epochs "
                  f"{e[col].diff().abs().median():.4f}")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(ylab, fontsize=8.5, loc="left")
        ax.set_xlabel("training epochs")
        ax.set_xlim(0, 26.5)
        clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    save(fig, SAVE_VALIDATION_EPOCH)


# --------------------------------------------------------------------------- #
# Figure (Chapter 5): CKA with the initialisation and effective rank, per epoch #
# --------------------------------------------------------------------------- #
def figure_epoch_change() -> None:
    style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.35), sharex=True)
    cka = pd.read_csv(REPO_ROOT / CKA_EPOCH_CSV)
    fs = pd.read_csv(REPO_ROOT / FEATURE_STD_CSV)
    runs = [("full_sp_50ep", "first seed", LR_LOW, "o"), ("full_sp_50ep_seed1", "second seed", SEED2_COLOUR, "D")]
    print("\nFIGURE EPOCH CHANGE — epoch 26: linear CKA with the random init. (share of CheMeleon's distance, times a "
          "change of seed); standardised effective rank relative to the random init. (lowest)")
    for run, label, c, marker in runs:
        g = cka[cka.run == run]
        t = g[g.compared_to == "trained"].assign(step=lambda x: x.step.astype(int)).sort_values("step")
        ref = g[g.compared_to != "trained"].set_index("compared_to").cka_linear_mean
        axes[0].plot(np.r_[0, t.step], np.r_[1.0, t.cka_linear_mean], color=c, lw=1.2, marker=marker, ms=2.4, label=label)
        axes[0].axhline(ref["chemeleon"], color=c, lw=0.9, ls="--", label=f"CheMeleon, {label}")
        last = t.cka_linear_mean.iloc[-1]
        r = fs[fs.run == run]
        r0 = r[r.series == "init"].effective_rank_std.iloc[0]
        q = r[r.series == "trained"].sort_values("step")
        axes[1].plot(np.r_[0, q.step], np.r_[1.0, q.effective_rank_std / r0], color=c, lw=1.2, marker=marker, ms=2.4, label=label)
        print(f"  {label:11s} CKA {last:.3f} ({(1 - last) / (1 - ref['chemeleon']):.0%}, {(1 - last) / (1 - ref['other-init']):.1f} x); "
              f"rank {q.effective_rank_std.iloc[-1] / r0:.2f} (lowest {q.effective_rank_std.min() / r0:.2f} at epoch "
              f"{int(q.step[q.effective_rank_std.idxmin()])})")
    axes[0].axhline(ref["other-init"], color="0.35", lw=0.9, ls=":", label="the other seed's random init.")
    axes[0].set_title("CKA with the random initialisation", fontsize=8.5, loc="left")
    axes[0].set_ylim(0.6, 1.02)
    axes[1].axhline(1.0, color="0.5", lw=0.8)
    axes[1].set_title("effective rank, relative to random init.", fontsize=8.5, loc="left")
    axes[1].set_ylim(0, 1.2)
    for ax in axes:
        ax.set_xlabel("training epochs")
        ax.set_xlim(0, 26.5)
        clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 1, 3, 4]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside lower center", ncol=3, fontsize=7,
               frameon=False, handlelength=2.4)
    save(fig, SAVE_EPOCH_CHANGE)


# --------------------------------------------------------------------------- #
# Figure 5.2: effective rank relative to the random initialisation            #
# --------------------------------------------------------------------------- #
def figure_rank() -> None:
    fs = pd.read_csv(REPO_ROOT / FEATURE_STD_CSV)
    f50 = fs[fs.run == "full_sp_50ep"]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.75), sharey=True)

    ax = axes[0]
    for run, _, label, c, ls in LONG_RUNS:
        g = fs[fs.run == run]
        r0 = g[g.series == "init"].effective_rank_std.iloc[0]
        t = g[g.series == "trained"].sort_values("step")
        x = np.r_[0, t.step.to_numpy()] / 1000
        y = np.r_[1.0, t.effective_rank_std.to_numpy() / r0]
        ax.plot(x, y, color=c, ls=ls, lw=1.3, label=label)
    ax.set_title(r"100,000 steps at $10^{-4}$", fontsize=8.5, loc="left")
    ax.set_xlabel(r"optimisation steps ($\times 1{,}000$)")
    ax.set_ylabel("effective rank,\nrelative to random init.")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])

    ax = axes[1]
    g = fs[fs.run == "full_sp_10ep"]
    r0 = g[g.series == "init"].effective_rank_std.iloc[0]
    t = g[g.series == "trained"].sort_values("step")
    ax.plot(np.r_[0, t.step / STEPS_PER_EPOCH], np.r_[1.0, t.effective_rank_std / r0],
            color=LR_HIGH, lw=1.2, label=r"single-protein, $10^{-4}$, 10 epochs")
    r0 = f50[f50.series == "init"].effective_rank_std.iloc[0]
    t = f50[f50.series == "trained"].sort_values("step")
    ax.plot(np.r_[0, t.step], np.r_[1.0, t.effective_rank_std / r0], color=LR_LOW, lw=1.2, marker="o", ms=2.4,
            label=r"single-protein, $3 \times 10^{-5}$, epoch-level run")
    b = t[t.step == BEST_EPOCH]
    ax.plot(b.step, b.effective_rank_std / r0, marker="*", ms=11, color=LR_LOW, ls="none", mec="white", mew=0.6, zorder=6,
            label=f"best epoch of the epoch-level run ({BEST_EPOCH})")
    ax.set_title("single-protein corpus", fontsize=8.5, loc="left")
    ax.set_xlabel("training epochs")
    ax.set_xlim(0, 26.5)

    handles, labels = [], []
    for ax in axes:
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_ylim(0, 1.45)
        clean(ax)
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False, handlelength=2.6)
    save(fig, SAVE_RANK)


# --------------------------------------------------------------------------- #
# Numbers quoted in Chapter 5                                                 #
# --------------------------------------------------------------------------- #
def numbers() -> None:
    print("\nTABLE tab:in_domain — in-domain validation: initial -> end (best), Spearman and NLL")
    for name in ["d2r_long", "saureus_long", "usp7_long", "full_sp", "full_sp_10ep", "full_sp_10ep_lowlr"]:
        q = curve(name)
        i = q.val_spearman.idxmax()
        print(f"  {name:20s} Spearman {q.val_spearman.iloc[0]:.3f} -> {q.val_spearman.iloc[-1]:.3f} "
              f"(max {q.val_spearman.max():.3f} at step {q.step[i]:,});  NLL {q.val_loss.iloc[0]:.3f} -> "
              f"{q.val_loss.iloc[-1]:.3f}")
    e = epoch_run()
    b = e.loc[e.val_loss.idxmin()]
    print(f"  epoch-level run      NLL {e.val_loss.iloc[0]:.4f} -> {b.val_loss:.4f} at epoch {int(b.epoch)} "
          f"({100 * (b.val_loss / e.val_loss.iloc[0] - 1):+.1f} %), Spearman {e.val_spearman.iloc[0]:.4f} -> "
          f"{b.val_spearman:.4f}; epochs 24-26: "
          + ", ".join(f"{r.val_loss:.4f}/{r.val_spearman:.4f}" for r in e[e.epoch >= 24].itertuples()))

    print("\nDOMAIN-MATCHED TRENDS — least-squares slope per 100,000 steps over the 50 post-baseline evaluations")
    from scipy import stats
    for name in ["d2r_long", "saureus_long", "usp7_long"]:
        q = curve(name)
        base, e50 = q.iloc[0], q.iloc[1:]
        h = len(e50) // 2
        parts = []
        for col in ["val_spearman", "val_loss", "train_loss"]:
            s = stats.linregress(e50.step / 1e5, e50[col])
            parts.append(f"{col} {s.slope:+.3f} (p = {s.pvalue:.3f})")
        print(f"  {name:13s} " + ";  ".join(parts)
              + f";  Spearman baseline {base.val_spearman:.3f}, second-half mean {e50.val_spearman.iloc[h:].mean():.3f}"
              + f";  NLL {base.val_loss:.3f} -> {q.val_loss.iloc[-1]:.3f}")

    print("\nA/B TESTS (Section 5.1) — validation Spearman at steps 0 / 100 / 200 / 300, 300 steps, D2R, one ensemble member")
    for arm in ["ab_clip_1p0", "ab_clip_10p0", "ab_clip_1000p0", "ab_loss_mse", "ab_loss_huber"]:
        a = pd.read_csv(REPO_ROOT / EXP / f"{arm}.csv")
        print(f"  {arm:15s} " + " / ".join(f"{v:.3f}" for v in a.val_spearman)
              + f"   clip fraction {a.grad_clip_frac.dropna().mean():.2f}")

    print("\nGRADIENTS (Section 5.1) — mean pre-clip norm per evaluation interval, and share of clipped steps")
    for name in ["d2r_long", "saureus_long", "usp7_long", "full_sp"]:
        q = curve(name)
        print(f"  {name:13s} mean norm {q.grad_mean.min():.0f}-{q.grad_mean.max():.0f}, single steps up to "
              f"{q.grad_max.max():,.0f}, clipped steps at least {100 * q.grad_clip_frac.min():.1f} %")

    print("\nFIRST RUN (100k steps) — second half minus first half of the 50 evaluations")
    q = curve("full_sp").iloc[1:]
    a, c = q.iloc[:len(q) // 2], q.iloc[len(q) // 2:]
    for col in ["val_spearman", "val_loss"]:
        se = np.sqrt(a[col].var(ddof=1) / len(a) + c[col].var(ddof=1) / len(c))
        print(f"  {col}: {c[col].mean() - a[col].mean():+.3f} ± {se:.3f}")

    print("\nNOISE — median |change| in validation NLL between consecutive epochs")
    for name, label in [("full_sp_10ep", "lr 1e-4, 64 episodes"), ("full_sp_10ep_lowlr", "lr 3e-5, 64 episodes")]:
        q = curve(name).iloc[::EVALS_PER_EPOCH]
        print(f"  {label:28s} {q.val_loss.diff().abs().median():.4f}")
    print(f"  {'lr 3e-5, every point':28s} {e.val_loss.diff().abs().median():.4f}")
    q = curve("full_sp_10ep_lowlr")
    w = q[(q.step / STEPS_PER_EPOCH >= 5)]
    x = w.step / STEPS_PER_EPOCH
    X = np.c_[np.ones(len(w)), x]
    beta, *_ = np.linalg.lstsq(X, w.val_loss, rcond=None)
    resid = w.val_loss - X @ beta
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    print(f"  lr 3e-5, 64 episodes: NLL slope over epochs 5-10 {beta[1]:+.4f} ± {se:.4f} per epoch")

    print("\nTABLE tab:geometry — standardised effective rank: init -> first checkpoint -> lowest (step) -> final "
          "[raw in brackets]; mean |corr| init -> final (max); anchors as rank/width and |corr|")
    fs = pd.read_csv(REPO_ROOT / FEATURE_STD_CSV)
    for run in ["saureus_long", "d2r_long", "usp7_long", "full_sp", "full_sp_10ep", "full_sp_50ep", "full_sp_50ep_seed1"]:
        g = fs[fs.run == run]
        i0 = g[g.series == "init"].iloc[0]
        t = g[g.series == "trained"].sort_values("step")
        lo = t.loc[t.effective_rank_std.idxmin()]
        anchors = "  ".join(f"{s} {g[g.series == s].effective_rank_std.iloc[0]:.1f} "
                            f"({g[g.series == s].eff_rank_std_frac.iloc[0]:.3f} of width)/"
                            f"{g[g.series == s].mean_abs_corr.iloc[0]:.3f}" for s in ["chemeleon", "morgan"])
        fin = t.iloc[-1]
        if run == "full_sp_50ep":
            fin = t[t.step == BEST_EPOCH].iloc[0]
        print(f"  {run:19s} {int(i0.n_molecules):5d} molecules; rank {i0.effective_rank_std:.1f} [{i0.effective_rank:.1f}] -> "
              f"{t.effective_rank_std.iloc[0]:.1f} (step {int(t.step.iloc[0]):,}) -> "
              f"{lo.effective_rank_std:.1f} (step {int(lo.step):,}) -> {fin.effective_rank_std:.1f} [{fin.effective_rank:.1f}] "
              f"(step {int(fin.step):,}; {100 * (fin.effective_rank_std / i0.effective_rank_std - 1):+.0f} % "
              f"[raw {100 * (fin.effective_rank / i0.effective_rank - 1):+.0f} %]);  |corr| {i0.mean_abs_corr:.3f} -> "
              f"{fin.mean_abs_corr:.3f} (max {t.mean_abs_corr.max():.3f});  initial {i0.eff_rank_std_frac:.3f} of width;  "
              f"anchors {anchors}")
    t = fs[(fs.run == "full_sp_50ep") & (fs.series == "trained")].set_index("step")
    print("  full_sp_50ep standardised rank by epoch: "
          + " ".join(f"{int(s)}:{r:.1f}" for s, r in t.effective_rank_std.items()))
    t = fs[(fs.run == "full_sp_50ep_seed1")].sort_values("step")
    print("  full_sp_50ep_seed1 standardised rank by epoch: "
          + " ".join(f"{int(r.step)}:{r.effective_rank_std:.1f} (|r| {r.mean_abs_corr:.3f})"
                     for r in t[t.series.isin(["init", "trained"])].itertuples()))

# --------------------------------------------------------------------------- #
# Figure: CKA with the initialisation across training (the three long runs)   #
# --------------------------------------------------------------------------- #
def cka_trajectory() -> pd.DataFrame:
    """Columns run, compared_to, step, cka: from cka_trajectory.csv (linear) or cka_debiased.csv."""
    if CKA_ESTIMATOR == "linear":
        t = pd.read_csv(REPO_ROOT / CKA_TRAJECTORY_CSV).rename(columns={"cka_mean": "cka"})
    else:
        t = pd.read_csv(REPO_ROOT / CKA_DEBIASED_CSV)
        t = t[t.part == "trajectory"].rename(columns={"cka_debiased_mean": "cka"})
    return t[["run", "compared_to", "step", "cka", "n_assays"]]


def figure_cka() -> None:
    t = cka_trajectory()
    style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.35), sharey=True)
    for ax, (run, _, label, colour, _) in zip(axes, LONG_RUNS[:3]):
        g = t[t.run == run]
        tr = g[g.compared_to == "trained"].copy()
        tr["step"] = tr.step.astype(float)
        tr = tr.sort_values("step")
        ax.plot(tr.step / 1000, tr.cka, color=colour, lw=1.4, label="trained encoder")
        for ref, ls, name in [("chemeleon", "--", "CheMeleon"), ("other-init", ":", "second random init.")]:
            v = g[g.compared_to == ref].cka.iloc[0]
            ax.axhline(v, color="0.35", lw=0.9, ls=ls, label=name)
        n = int(t[(t.run == run) & (t.compared_to == "chemeleon")].n_assays.iloc[0])
        ax.set_title(f"{label.split(',')[0]} ({n} assays)", fontsize=8.5, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        clean(ax)
    fig.supxlabel(r"optimisation steps ($\times 1{,}000$)", fontsize=8.5)
    axes[0].set_ylabel(f"CKA with the random\ninitialisation ({CKA_ESTIMATOR} estimator)")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False, handlelength=2.2)
    save(fig, SAVE_CKA)


# --------------------------------------------------------------------------- #
# Figure: change in effective rank against change in transfer, per encoder    #
# --------------------------------------------------------------------------- #
def rank_change(run: str, step: int) -> float:
    fs = pd.read_csv(REPO_ROOT / FEATURE_STD_CSV)
    g = fs[fs.run == run]
    r0 = g[g.series == "init"].effective_rank_std.iloc[0]
    return 100 * (g[g.series == "trained"].set_index("step").effective_rank_std.loc[step] / r0 - 1)


_WIDE: pd.DataFrame | None = None


def final_wide() -> pd.DataFrame:
    """Final-round CEF of the final run (mean over the five seeds), one column per representation, indexed by
    matched pair (target, surrogate, criterion, configuration). TabPFN, RF and RBF GP in their native predictive
    form. The per-target domain-matched arms are pooled under one name: each exists on its own target only."""
    global _WIDE
    if _WIDE is None:
        files = [f for pattern in RUN_GLOBS for f in sorted((REPO_ROOT / FINAL_DIR).glob(pattern))]
        df = pd.concat([pd.read_csv(f, dtype={"candidates": str, "tau": str, "dist": str}) for f in files],
                       ignore_index=True)
        df["criterion"] = np.where(df.acquisition == "greedy", "greedy", df.acquisition + "_" + df.tau)
        df = df[(df["round"] == N_ROUNDS) & (df.candidates == CANDIDATES) & df.model.isin(NATIVE_DIST)]
        df = df[(df.criterion == "greedy") | (df.dist == df.model.map(NATIVE_DIST))]
        g = df.groupby(PAIR_KEY + ["representation", "dist"]).agg(cef=("cef", "mean"), n=("seed", "nunique")).reset_index()
        g = g[g.n == N_SEEDS]
        for kind in ("", "_long"):
            g["representation"] = g.representation.replace(
                {f"chemprop_trained_own_{t}{kind}": f"chemprop_trained_own{kind}" for t in TRANSFER_TARGETS})
        _WIDE = g.pivot_table(index=PAIR_KEY, columns="representation", values="cef")
    return _WIDE


def encoder_pairs(representation: str, control: str) -> pd.Series:
    """Encoder minus its control, final CEF: one value per matched pair."""
    w = final_wide()
    return (w[representation] - w[control]).dropna()


def scenario_means(d: pd.Series) -> pd.Series:
    """The unit of inference: the mean of the matched pairs inside each scenario (a target in one configuration)."""
    n = d.groupby(SCENARIO).size()
    assert (n == PAIRS_PER_SCENARIO).all(), n[n != PAIRS_PER_SCENARIO]
    return d.groupby(SCENARIO).mean()


def scenario_stat(s: pd.Series) -> tuple[float, float, int, int]:
    """Mean over scenarios, its standard error over scenarios, scenarios won, scenarios."""
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s)), int((s > 0).sum()), len(s)


def rank_transfer_value(representation: str, control: str, target: str | None) -> tuple[pd.Series, float, float]:
    """Scenario means of one encoder (restricted to its own target when given), their mean and SE over scenarios."""
    s = scenario_means(encoder_pairs(representation, control))
    if target:
        s = s.xs(target, level="target", drop_level=False)
    m, se, _, _ = scenario_stat(s)
    return s, m, se


LABEL_AT = {  # label -> (x, y, horizontal alignment) in data coordinates, chosen so that no two labels overlap;
    "10 epochs, best": (-45, 0.07, "left"), "10 epochs, final": (-56, -0.27, "left"),
    "100k steps, final": (-70, -0.04, "right"), "100k steps, best": (-78, -0.47, "right"),
    "D2R, long": (-67, 0.13, "right"), "S. aureus, long": (-70, -0.67, "left"), "USP7, long": (-28, -0.35, "left"),
}


def row_scenarios(rep, ctrl) -> pd.Series:
    """Scenario means of one figure row; an averaged row (a tuple of arms) is the mean of its arms' differences."""
    if isinstance(rep, tuple):
        return scenario_means(sum(encoder_pairs(r, c) for r, c in rep) / len(rep))
    return scenario_means(encoder_pairs(rep, ctrl))


def figure_transfer(rows: list, out: str, height: float, title: str) -> None:
    """Target panels: the mean of the target's four scenarios (marker) and their range (line), on ONE shared scale.
    All-targets panel, on its own scale: mean +- 2 SE over scenarios, filled where established, scenarios won at
    the right edge. Fill carries a meaning in the all-targets panel only; hollow grey in a target panel = not
    interpreted (NOT_INTERPRETED)."""
    style()
    cols = TRANSFER_TARGETS + ["all"]
    ypos = [-float(i) for i in range(len(rows))]               # one slot per row from the top down, headings included
    fig, axes = plt.subplots(1, 5, figsize=(FIG_WIDTH, height), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.45]})
    for ax in axes[1:4]:
        ax.sharex(axes[0])                                     # the four targets on one scale; all targets on its own
    print(f"\nFIGURE {title} — trained encoder minus its own random init., final CEF; scenario level")
    for ax, t in zip(axes, cols):
        for yy, r in zip(ypos, rows):
            if isinstance(r, str):
                continue
            label, rep, ctrl = r
            s = row_scenarios(rep, ctrl)
            if t != "all":
                if t not in s.index.get_level_values("target"):
                    ax.annotate("n/a", xy=(0, yy), xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                                fontsize=5.6, color="0.45")           # no encoder for this target
                    continue
                s = s.xs(t, level="target")
            m, se, won, n = scenario_stat(s)
            hollow = (rep, t) in NOT_INTERPRETED
            c = REFERENCE_GREY if (rep == "chemeleon" or hollow) else (GAIN if m > 0 else LOSS)
            if t == "all":
                sig = abs(m) > 2 * se
                ax.errorbar(m, yy, xerr=2 * se, fmt="none", ecolor=c, elinewidth=1.0, capsize=0, zorder=3)
                ax.plot(m, yy, "o", ms=3.0, color=c, mfc=c if sig else "white", mew=0.9, zorder=4)
                ax.text(1.03, yy, f"{won}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=6, color="0.3")
            else:                      # four scenarios: their range, a standard error from four values means little
                sig = bool((s * np.sign(m) > 0).all())          # printed only: all four scenarios have the sign of the mean
                ax.plot([s.min(), s.max()], [yy, yy], color=c, lw=1.0, solid_capstyle="butt", zorder=3)
                ax.plot(m, yy, "o", ms=3.2, color=c, mfc="white" if hollow else c, mew=0.9, zorder=4)
                if hollow:
                    ax.annotate(r"$\dagger$", xy=(m, yy), xytext=(0, 3.0), textcoords="offset points", ha="center",
                                va="bottom", fontsize=8, color="0.2")
            print(f"  {t:8s} {label:26s} {m:+.2f}" + (f" ± {se:.2f}{'*' if sig else ' '}" if t == "all" else f"{'*' if sig else ' '}      ")
                  + f" ({won}/{n})")
        ax.axvline(0, color="0.3", lw=0.8, zorder=1)
        for yy, r in zip(ypos, rows):
            if isinstance(r, str):
                ax.axhline(yy + 0.0, color="0.9", lw=0.5, zorder=0)
        ax.set_title(TRANSFER_TARGET_NAMES[t], fontsize=8.5)
        ax.margins(x=0.12)
        ax.set_ylim(ypos[-1] - 0.7, 0.6)
        ax.tick_params(axis="x", labelsize=7)
        clean(ax)
    axes[0].set_xticks([-1, 0, 1, 2])
    axes[4].text(1.03, ypos[0], "won", transform=axes[4].get_yaxis_transform(), ha="left", va="center",
                 fontsize=6, color="0.4", fontstyle="italic")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([r if isinstance(r, str) else r[0] for r in rows], fontsize=7.2)
    for lab, r in zip(axes[0].get_yticklabels(), rows):
        if isinstance(r, str):
            lab.set_color("0.4")
            lab.set_fontstyle("italic")
        elif r[1] == "chemeleon":
            lab.set_color("0.4")
    for ax in axes:                    # no tick marks on the heading rows
        for tick, r in zip(ax.yaxis.get_major_ticks(), rows):
            if isinstance(r, str):
                tick.tick1line.set_visible(False)
    fig.supxlabel("encoder minus its own random initialisation, final CEF", fontsize=8)
    save(fig, out)


def figure_rank_transfer() -> None:
    style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.7))
    print("\nRANK AGAINST TRANSFER — change in standardised effective rank (%) and trained − own random init. (CEF)")
    sp_x, sp_y, sp_seed = [], [], []
    for label, run, step, rep, ctrl, target, lr, seed in RANK_TRANSFER_POINTS:
        rc = rank_change(run, step)
        s, m, se = rank_transfer_value(rep, ctrl, target)
        single = target is None
        colour = (LR_LOW if lr == "3e-5" else LR_HIGH) if single else "0.45"
        if single:
            ax.errorbar(rc, m, yerr=2 * se, fmt="D" if seed else "o", ms=4.3 if seed else 4.5, color=colour,
                        elinewidth=0.9, capsize=0)
        else:                          # four scenarios only: show their range, not a standard error
            ax.plot([rc, rc], [s.min(), s.max()], color=colour, lw=0.9)
            ax.plot(rc, m, "o", ms=4.5, color=colour, mfc="white")
        if label in LABEL_AT:
            lx, ly, ha = LABEL_AT[label]
            ax.annotate(label.replace("S. aureus", r"$S.\ aureus$"), (rc, m), xytext=(lx, ly), textcoords="data", fontsize=6.8, color="0.25", ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55", shrinkA=1, shrinkB=3))
        if single:
            sp_x.append(rc)
            sp_y.append(m)
            sp_seed.append(seed)
        print(f"  {label:20s} lr {lr:5s} rank {rc:+6.1f} %   transfer {m:+.2f} ± {se:.2f} ({len(s)} scenarios)")
    x, yv, sd = pd.Series(sp_x), pd.Series(sp_y), pd.Series(sp_seed)
    rho = lambda a, b: a.rank().corr(b.rank())
    print(f"  Spearman over the six first-seed single-protein checkpoints: {rho(x[sd == 0], yv[sd == 0]):+.2f};  "
          f"over all eight: {rho(x, yv):+.2f};  four at 1e-4: {rho(x[:4], yv[:4]):+.2f};  four at 3e-5: {rho(x[4:], yv[4:]):+.2f};  "
          f"second seed, two checkpoints: {x[sd == 1].tolist()[0]:+.1f} % / {x[sd == 1].tolist()[1]:+.1f} %")
    ax.annotate("epochs 21 to 26,\nrate $3 \\times 10^{-5}$", (-4, 0.22), xytext=(-25, 0.36), textcoords="data", fontsize=6.8,
                color="0.25", ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55", shrinkA=1, shrinkB=2))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(-98, 16)
    ax.set_xlabel("standardised effective rank, change from the random initialisation (%)")
    ax.set_ylabel("trained minus own random init.,\nfinal CEF (mean $\\pm$ 2 SE)")
    keys = [plt.Line2D([], [], marker="o", ls="", color=LR_HIGH, label="rate $10^{-4}$"),
            plt.Line2D([], [], marker="o", ls="", color=LR_LOW, label=r"rate $3 \times 10^{-5}$, first seed"),
            plt.Line2D([], [], marker="D", ls="", color=LR_LOW, label=r"rate $3 \times 10^{-5}$, second seed"),
            plt.Line2D([], [], marker="o", ls="", color="0.45", mfc="white", label="domain-matched")]
    ax.legend(handles=keys, loc="lower right", fontsize=6.8, frameon=False)
    clean(ax)
    save(fig, SAVE_GEOMETRY_TRANSFER)


def main() -> None:
    figure_transfer(TRANSFER_MAIN_ROWS, SAVE_TRANSFER, 2.15, "TRANSFER, CHAPTER 5")   # Chapter 5
    figure_validation_epoch()
    figure_epoch_change()
    figure_validation()                                                               # Appendix D
    figure_transfer(TRANSFER_ROWS, SAVE_TRANSFER_ALL, 3.9, "TRANSFER, ALL ENCODERS")
    figure_cka()
    figure_rank()
    figure_rank_transfer()
    numbers()


if __name__ == "__main__":
    main()
