"""scripts/plot_benchmark.py — the figures of Chapter 4 (active learning benchmark). Press Run.

All active learning figures come from ONE run, the final benchmark
(experiments/benchmark_final/final_<target>_<representation>.csv, per seed and per round).
Conventions:
  - each surrogate is used in its NATIVE predictive form: TabPFN's EI and PI are computed from
    its full (bar) distribution, those of the random forest and the RBF Gaussian process from
    their mean and standard deviation (NATIVE_DIST). TabPFN's Gaussian form is an ablation and
    has a figure of its own;
  - the main surrogates are TabPFN, the random forest and the RBF GP (MAIN_SURROGATES); the
    Tanimoto GP is reported in the appendix only, and its numbers are printed at the end;
  - Chapter 4 uses the five representations NOT trained through TabPFN (CH4_REPRESENTATIONS);
    the trained encoders are the subject of Chapter 5 (plot_training.py);
  - 1,000 candidates per round (the full-pool control is printed, not drawn);
  - the unit is an AL experiment: one arm in one budget configuration on one target and one
    representation, final-round value averaged over the five seeds. Comparisons are matched
    pairs that differ in one choice: mean +- 2 SE, with the pairs won.

FIGURE 4.1 (hit discovery). Cumulative hits against measured molecules for TabPFN, the random
forest and the RBF GP, one panel per target, for one representation, budget configuration and
criterion (CURVE_*). Band: mean +- SE over the five seeds; dotted: random selection (5 %).

FIGURE (acquisition). Each criterion minus greedy selection, final CEF: one column per target
and one for all; rows 1-4 pooled over the three main surrogates, rows 5-7 PI(y_thresh) for each.

FIGURE (surrogates, RQ1). TabPFN minus the random forest and minus the RBF GP under the same
criterion: all five criteria pooled, then greedy, PI(y_thresh) and EI(y_thresh) alone.

FIGURE (ablation). TabPFN's EI and PI from its full distribution minus the same criterion from
its Gaussian form (mean and the half-width of the central 68 % interval).

FIGURE (representation, RQ3). Mean final CEF of the five representations, one heatmap per main
surrogate, one row per target plus the mean; averaged over the five criteria and the four
budget configurations. Shade = distance below the best representation of the row, capped at
HEAT_CAP. Panel title: the spread, max minus min over representations, averaged over targets,
criteria and configurations.

FIGURE (calibration). Observed minus nominal coverage of the central intervals, per surrogate and
training-set size (experiments/calibration_study.csv; not part of the AL run). TabPFN is drawn
with intervals from its full distribution and, dashed, from its Gaussian form.

The script also prints every number the chapter quotes from these data.

matplotlib is deliberately NOT a project dependency (it is needed by the plot scripts alone).
Run it with:

    uv run --with matplotlib python src/scripts/plot_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
FINAL_DIR = "experiments/benchmark_final"          # the final run (benchmark_extended.py "final")
REPLICATE_DIR = "experiments/benchmark_extended"   # the earlier extended run, an exact replicate
CALIBRATION_CSV = "experiments/calibration_study.csv"
RUNTIME_CSV = "experiments/runtime_study.csv"
SAVE_CURVES = "experiments/fig_hit_discovery.pdf"
SAVE_ACQUISITION = "experiments/fig_acquisition_gain.pdf"
SAVE_SURROGATES = "experiments/fig_surrogate_gain.pdf"
SAVE_ABLATION = "experiments/fig_tabpfn_full_vs_gauss.pdf"
SAVE_REPRESENTATION = "experiments/fig_representation.pdf"
SAVE_CALIBRATION = "experiments/fig_calibration.pdf"
SAVE_PNG_TOO = True
FIG_WIDTH = 5.5                  # inches; the thesis text block is 398 pt = 5.51 in, so figures are not rescaled

TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
TARGET_NAMES = {"saureus": r"$S.\ aureus$", "tyk2": "TYK2", "usp7": "USP7", "d2r": "D2R"}
N_SEEDS = 5
N_ROUNDS = 20
BASE_RATE = 0.05
CANDIDATES = "1000"              # the main protocol; "all" = the full-pool control (USP7, D2R)

MAIN_SURROGATES = ["tabpfn", "rf", "gp"]
NATIVE_DIST = {"tabpfn": "full", "rf": "gauss", "gp": "gauss", "gp_tanimoto": "gauss"}
CH4_REPRESENTATIONS = ["chemeleon", "mordred3d", "chemprop_untrained", "morgan", "physchem"]
CRITERIA = ["greedy", "ei_yplus", "pi_yplus", "ei_ythresh", "pi_ythresh"]
CRITERION_NAMES = {"greedy": "greedy", "ei_yplus": r"EI($y^{+}$)", "pi_yplus": r"PI($y^{+}$)",
                   "ei_ythresh": r"EI($y_{\mathrm{thresh}}$)", "pi_ythresh": r"PI($y_{\mathrm{thresh}}$)"}

CURVE_REPRESENTATION = "chemeleon"
CURVE_CRITERION = "greedy"
CURVE_INIT = 100
CURVE_BATCH = "large"            # "small" or "large": the target's smaller or larger batch size

SURROGATES = {"tabpfn": ("TabPFN", "#1f4e79"), "rf": ("Random forest", "#c45a00"),
              "gp": ("Gaussian process (RBF)", "#2e7d32")}
SURROGATE_SHORT = {"tabpfn": "TabPFN", "rf": "RF", "gp": "RBF GP", "gp_tanimoto": "Tanimoto GP"}

# Acquisition figure rows: (label, criterion, surrogate or None = the three main surrogates pooled)
ACQ_ROWS = [
    (r"EI($y^{+}$)", "ei_yplus", None),
    (r"PI($y^{+}$)", "pi_yplus", None),
    (r"EI($y_{\mathrm{thresh}}$)", "ei_ythresh", None),
    (r"PI($y_{\mathrm{thresh}}$)", "pi_ythresh", None),
    (r"PI($y_{\mathrm{thresh}}$), TabPFN", "pi_ythresh", "tabpfn"),
    (r"PI($y_{\mathrm{thresh}}$), RF", "pi_ythresh", "rf"),
    (r"PI($y_{\mathrm{thresh}}$), RBF GP", "pi_ythresh", "gp"),
]
ACQ_SEPARATORS = [3]             # draw a separator below these row indices

# Surrogate figure rows: (label, alternative surrogate, criterion or None = all five pooled)
SURROGATE_ROWS = [
    ("vs RF, all criteria", "rf", None),
    ("vs RF, greedy", "rf", "greedy"),
    (r"vs RF, PI($y_{\mathrm{thresh}}$)", "rf", "pi_ythresh"),
    (r"vs RF, EI($y_{\mathrm{thresh}}$)", "rf", "ei_ythresh"),
    ("vs RBF GP, all criteria", "gp", None),
    ("vs RBF GP, greedy", "gp", "greedy"),
    (r"vs RBF GP, PI($y_{\mathrm{thresh}}$)", "gp", "pi_ythresh"),
    (r"vs RBF GP, EI($y_{\mathrm{thresh}}$)", "gp", "ei_ythresh"),
]
SURROGATE_SEPARATORS = [3]

ABLATION_ROWS = ["ei_yplus", "pi_yplus", "ei_ythresh", "pi_ythresh"]

# Representation heatmap
REPRESENTATION_NAMES = {"chemeleon": "CheMeleon", "mordred3d": "Mordred", "chemprop_untrained": "Random init.",
                        "morgan": "Morgan", "physchem": "Physchem."}
HEAT_CAP = 2.0                   # CEF below the best representation at which the shade saturates
HEAT_COLOURS = ("#ffffff", "#595959")   # one neutral hue, light -> dark (blue and red mean gain/loss elsewhere)
GAIN, LOSS = "#1f4e79", "#b2182b"

EXPERIMENT_KEY = ["target", "representation", "model", "init_size", "batch_size"]


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


def paired(x: pd.Series) -> tuple[float, float, int, int]:
    x = x.dropna()
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), int((x > 0).sum()), len(x)


def fmt(x: pd.Series) -> str:
    m, se, won, n = paired(x)
    return f"{m:+.2f} ± {se:.2f}{'*' if abs(m) > 2 * se else ' '}({won}/{n})"


def by_target(x: pd.Series) -> str:
    return "  ".join(f"{t} {x.xs(t, level='target').mean():+.2f}" for t in TARGETS
                     if t in x.index.get_level_values("target"))


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def read_runs(directory: str, pattern: str) -> pd.DataFrame:
    files = sorted((REPO_ROOT / directory).glob(pattern))
    df = pd.concat([pd.read_csv(f, dtype={"candidates": str, "tau": str, "dist": str}) for f in files],
                   ignore_index=True)
    df["criterion"] = np.where(df.acquisition == "greedy", "greedy", df.acquisition + "_" + df.tau)
    return df


def experiments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per AL experiment: final-round means over the five seeds (complete ones only)."""
    f = df[df["round"] == N_ROUNDS]
    g = f.groupby(EXPERIMENT_KEY + ["criterion", "dist", "candidates"]).agg(
        cef=("cef", "mean"), best_y=("best_y", "mean"), mean_y=("mean_y_acquired", "mean"),
        sd=("cef", "std"), n=("seed", "nunique")).reset_index()
    return g[g.n == N_SEEDS]


def native(e: pd.DataFrame) -> pd.DataFrame:
    """Greedy plus each surrogate's EI and PI in its native predictive form."""
    return e[(e.criterion == "greedy") | (e.dist == e.model.map(NATIVE_DIST))]


def wide(e: pd.DataFrame, value: str = "cef") -> pd.DataFrame:
    """Criterion columns, indexed by AL-experiment key (main protocol)."""
    q = e[e.candidates == CANDIDATES]
    return q.pivot_table(index=EXPERIMENT_KEY, columns="criterion", values=value)


def by_model(e: pd.DataFrame, value: str = "cef") -> pd.DataFrame:
    """Surrogate columns, indexed by target, representation, criterion and configuration."""
    q = e[e.candidates == CANDIDATES]
    return q.pivot_table(index=["target", "representation", "criterion", "init_size", "batch_size"],
                         columns="model", values=value)


def subset(x: pd.DataFrame | pd.Series, reps=None, models=None):
    keep = np.ones(len(x), dtype=bool)
    if reps is not None:
        keep &= x.index.get_level_values("representation").isin(reps)
    if models is not None and "model" in x.index.names:
        keep &= x.index.get_level_values("model").isin(models)
    return x[keep]


# --------------------------------------------------------------------------- #
# Generic dot plot: one column per target and one for all, one row per series #
# --------------------------------------------------------------------------- #
def dot_figure(rows: list[tuple[str, pd.Series]], separators: list[int], xlabel: str, path: str,
               height: float, title: str) -> None:
    """One panel per target, on ONE shared x scale, and a wider all-targets panel on its own scale. Every point is
    a mean paired difference +- 2 SE, filled where established. The pairs won are written for all targets only, in
    a column at the right edge; the counts per target are in the Appendix C tables."""
    style()
    cols = TARGETS + ["all"]
    fig, axes = plt.subplots(1, 5, figsize=(FIG_WIDTH, height), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3]})
    for ax in axes[1:4]:
        ax.sharex(axes[0])                      # the four targets on one scale; all targets on its own
    print(f"\n{title} (mean ± SE, * = more than 2 SE, pairs won)")
    ypos = np.arange(len(rows))[::-1]
    lim = {"targets": [0.0, 0.0], "all": [0.0, 0.0]}     # x range: the zero line and every bar
    for ax, t in zip(axes, cols):
        r = lim["all" if t == "all" else "targets"]
        for yy, (label, series) in zip(ypos, rows):
            d = series if t == "all" else series.xs(t, level="target")
            m, se, won, n = paired(d)
            r[0], r[1] = min(r[0], m - 2 * se), max(r[1], m + 2 * se)
            sig = abs(m) > 2 * se
            c = GAIN if m > 0 else LOSS
            ax.errorbar(m, yy, xerr=2 * se, fmt="o", ms=3.2, color=c, mfc=c if sig else "white",
                        mew=0.9, elinewidth=1.0, capsize=0)
            if t == "all":
                ax.text(1.04, yy, f"{won}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=6, color="0.3")
            print(f"  {t:8s} {label:42s} {fmt(d)}")
        ax.axvline(0, color="0.3", lw=0.8, zorder=0)
        for s in separators:
            ax.axhline(ypos[s] - 0.5, color="0.85", lw=0.6)
        ax.set_title(TARGET_NAMES.get(t, "all targets"), fontsize=8.5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if t == "all" else 4, integer=True))
        ax.tick_params(axis="x", labelsize=7)
        clean(ax)
    for ax, key in ((axes[0], "targets"), (axes[4], "all")):
        lo, hi = lim[key]
        pad = 0.08 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
    axes[4].text(1.04, 1.0, "won", transform=axes[4].transAxes, ha="left", va="bottom", fontsize=6,
                 color="0.4", fontstyle="italic")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7.2)
    fig.supxlabel(xlabel, fontsize=8)
    save(fig, path)


# --------------------------------------------------------------------------- #
# Figure 4.1: hit-discovery curves                                            #
# --------------------------------------------------------------------------- #
def figure_curves(df: pd.DataFrame) -> None:
    style()
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, 2.1))
    print(f"\nFIGURE HIT DISCOVERY — {CURVE_REPRESENTATION}, {CURVE_CRITERION}, N_init {CURVE_INIT}, "
          f"{CURVE_BATCH} batch: final cumulative hits, mean ± SE over seeds")
    for ax, t in zip(axes, TARGETS):
        d = df[(df.target == t) & (df.representation == CURVE_REPRESENTATION) & (df.criterion == CURVE_CRITERION)
               & (df.init_size == CURVE_INIT) & (df.candidates == CANDIDATES)]
        sizes = sorted(d.batch_size.unique())
        d = d[d.batch_size == (sizes[0] if CURVE_BATCH == "small" else sizes[-1])]
        if CURVE_CRITERION != "greedy":
            d = d[d.dist == d.model.map(NATIVE_DIST)]
        for m in MAIN_SURROGATES:
            q = d[d.model == m]
            hits = q.pivot_table(index="round", columns="seed", values="cum_hits")
            budget = q.groupby("round").budget.first()
            first = q[q["round"] == 1]
            y0 = (first.cum_hits - first.new_hits).mean()
            x = np.r_[CURVE_INIT, budget.to_numpy()]
            y = np.r_[y0, hits.mean(axis=1).to_numpy()]
            se = np.r_[np.nan, (hits.std(axis=1, ddof=1) / np.sqrt(hits.shape[1])).to_numpy()]
            name, c = SURROGATES[m]
            ax.plot(x, y, color=c, lw=1.3, label=name)
            ax.fill_between(x, y - se, y + se, color=c, alpha=0.18, lw=0)
            print(f"  {t:8s} {m:7s} budget {int(x[-1]):>5}: {y[-1]:6.1f} ± {se[-1]:4.1f}")
        xs = np.linspace(CURVE_INIT, x[-1], 50)
        ax.plot(xs, BASE_RATE * xs, color="0.4", lw=0.9, ls=":", label="random selection")
        ax.set_title(f"{TARGET_NAMES[t]} ($K = {int(d.batch_size.iloc[0])}$)", fontsize=8.5, loc="left")
        ax.set_xlabel("measured molecules", fontsize=7.5)
        ax.set_xlim(CURVE_INIT, x[-1])
        ax.set_ylim(0, None)
        clean(ax)
    axes[0].set_ylabel("cumulative hits")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    save(fig, SAVE_CURVES)


# --------------------------------------------------------------------------- #
# Figures: acquisition, surrogates, ablation                                  #
# --------------------------------------------------------------------------- #
def figure_acquisition(nat: pd.DataFrame) -> None:
    p = subset(wide(nat), CH4_REPRESENTATIONS, MAIN_SURROGATES)
    rows = []
    for label, crit, model in ACQ_ROWS:
        q = p if model is None else p.xs(model, level="model", drop_level=False)
        rows.append((label, (q[crit] - q.greedy).dropna()))
    dot_figure(rows, ACQ_SEPARATORS, "final CEF minus greedy selection (mean ± 2 SE)", SAVE_ACQUISITION, 2.75,
               "FIGURE ACQUISITION — criterion minus greedy, native forms, five representations")


def figure_surrogates(nat: pd.DataFrame) -> None:
    w = subset(by_model(nat), CH4_REPRESENTATIONS)
    rows = []
    for label, alt, crit in SURROGATE_ROWS:
        q = w if crit is None else w.xs(crit, level="criterion", drop_level=False)
        rows.append((label, (q.tabpfn - q[alt]).dropna()))
    dot_figure(rows, SURROGATE_SEPARATORS, "TabPFN minus the alternative surrogate, final CEF (mean ± 2 SE)",
               SAVE_SURROGATES, 2.95, "FIGURE SURROGATES — TabPFN minus alternative, same criterion, native forms")


def figure_ablation(e: pd.DataFrame) -> None:
    q = e[(e.model == "tabpfn") & (e.candidates == CANDIDATES) & (e.criterion != "greedy")]
    w = subset(q.pivot_table(index=EXPERIMENT_KEY + ["criterion"], columns="dist", values="cef"),
               CH4_REPRESENTATIONS)
    rows = [(CRITERION_NAMES[c], (w.full - w.gauss).xs(c, level="criterion", drop_level=False).dropna())
            for c in ABLATION_ROWS]
    dot_figure(rows, [], "TabPFN, full distribution minus Gaussian form, final CEF (mean ± 2 SE)",
               SAVE_ABLATION, 2.0, "FIGURE ABLATION — TabPFN full distribution minus Gaussian form, five representations")


# --------------------------------------------------------------------------- #
# Representation heatmap (RQ3)                                                #
# --------------------------------------------------------------------------- #
def figure_representation(nat: pd.DataFrame) -> None:
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap

    q = nat[(nat.candidates == CANDIDATES) & nat.representation.isin(CH4_REPRESENTATIONS)
            & nat.model.isin(MAIN_SURROGATES)]
    means = q.pivot_table(index=["model", "target"], columns="representation", values="cef")[CH4_REPRESENTATIONS]
    k = ["target", "model", "criterion", "init_size", "batch_size"]
    spread = q.groupby(k).cef.agg(lambda x: x.max() - x.min()).groupby(level="model").mean()
    cmap = LinearSegmentedColormap.from_list("below_best", HEAT_COLOURS)

    style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.45), sharey=True)
    rows = TARGETS + ["mean"]
    print("\nFIGURE REPRESENTATION — mean final CEF over five criteria and four configurations; spread = mean max-min")
    for ax, mod in zip(axes, MAIN_SURROGATES):
        name = SURROGATES[mod][0].replace("Gaussian process (RBF)", "RBF GP")
        v = np.vstack([means.loc[(mod, t)].to_numpy() for t in TARGETS])
        v = np.vstack([v, v.mean(axis=0)])
        below = v.max(axis=1, keepdims=True) - v
        im = ax.imshow(np.clip(below, 0, HEAT_CAP), cmap=cmap, vmin=0, vmax=HEAT_CAP, aspect="auto")
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                best = below[i, j] == 0
                ink = "white" if below[i, j] > 0.6 * HEAT_CAP else "black"
                # White labels are drawn as outlines, not as text: white text in the PDF is reported as
                # "hidden text" by Turnitin. A path effect makes matplotlib convert the label to a path.
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color=ink,
                        fontweight="bold" if best else "normal",
                        path_effects=[patheffects.Normal()] if ink == "white" else None)
            print(f"  {name:13s} {rows[i]:8s} " + "  ".join(
                f"{REPRESENTATION_NAMES[r]} {v[i, j]:.2f}" for j, r in enumerate(CH4_REPRESENTATIONS))
                  + f"   [best: {REPRESENTATION_NAMES[CH4_REPRESENTATIONS[int(np.argmax(v[i]))]]}]")
        ax.axhline(len(TARGETS) - 0.5, color="white", lw=3)
        ax.set_xticks(range(len(CH4_REPRESENTATIONS)))
        ax.set_xticklabels([REPRESENTATION_NAMES[r] for r in CH4_REPRESENTATIONS], fontsize=6.3, rotation=40,
                           ha="right", rotation_mode="anchor")
        ax.set_title(f"{name}\nspread {spread[mod]:.2f}", fontsize=8, linespacing=1.3)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        print(f"  {name:13s} spread {spread[mod]:.2f}")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([TARGET_NAMES.get(t, "mean") for t in rows], fontsize=7.5)
    bar = fig.colorbar(im, ax=axes, shrink=0.75, aspect=18, pad=0.02, ticks=[0, 1, HEAT_CAP])
    bar.ax.set_yticklabels(["0 (best)", "1", f"≥ {HEAT_CAP:g}"], fontsize=6.5)
    bar.set_label("CEF below the best\nrepresentation", fontsize=7)
    bar.outline.set_visible(False)
    save(fig, SAVE_REPRESENTATION)


# --------------------------------------------------------------------------- #
# Calibration (separate study)                                                #
# --------------------------------------------------------------------------- #
def figure_calibration() -> None:
    c = pd.read_csv(REPO_ROOT / CALIBRATION_CSV)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cols = [f"coverage_{int(round(q * 100))}" for q in levels]
    curves = [("tabpfn", "full", "TabPFN, full distribution", "#1f4e79", "-"),
              ("tabpfn", "gauss", "TabPFN, Gaussian form (ablation)", "#1f4e79", "--"),
              ("rf", "gauss", "Random forest", "#c45a00", "-"),
              ("gp", "gauss", "Gaussian process (RBF)", "#2e7d32", "-")]
    style()
    sizes = sorted(c.train_size.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(FIG_WIDTH, 2.3), sharey=True)
    for ax, n in zip(axes, sizes):
        for m, iv, name, col, ls in curves:
            q = c[(c.model == m) & (c.intervals == iv) & (c.train_size == n)]
            err = q[cols].mean().to_numpy() - np.array(levels)
            ax.plot(levels, err, color=col, ls=ls, lw=1.3, marker="o", ms=2.5, label=name)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(f"{n:,} training molecules", fontsize=8.5, loc="left")
        ax.set_xlabel("nominal coverage of the central interval", fontsize=7.5)
        clean(ax)
    axes[0].set_ylabel("observed − nominal\ncoverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    save(fig, SAVE_CALIBRATION)

    print("\nCALIBRATION — ECE: TabPFN full distribution lower than ... (per dataset, representation, size, split)")
    w = c.pivot_table(index=["dataset", "representation", "train_size", "split"],
                      columns=["model", "intervals"], values="ece")
    tf = w[("tabpfn", "full")]
    for m, iv in [("tabpfn", "gauss"), ("rf", "gauss"), ("gp", "gauss"), ("gp_tanimoto", "gauss")]:
        d = (w[(m, iv)] - tf).dropna()
        print(f"  than {m}/{iv}: {(d > 0).sum()}/{len(d)}")
    s = c.groupby(["model", "intervals"]).agg(ece_median=("ece", "median"), cov90=("coverage_90", "mean"),
                                               signed=("signed_calibration_error", "mean"),
                                               nll_median=("nll_median", "median"),
                                               nll_capped=("nll_capped", "median"))
    print(s.round(3).to_string())


# --------------------------------------------------------------------------- #
# Numbers quoted in the text that come from no figure                         #
# --------------------------------------------------------------------------- #
def numbers(df: pd.DataFrame, e: pd.DataFrame, nat: pd.DataFrame) -> None:
    p = subset(wide(nat), CH4_REPRESENTATIONS)
    w = subset(by_model(nat), CH4_REPRESENTATIONS)

    print("\nSIZE — campaigns, AL experiments (all files)")
    print(f"  campaigns {df.groupby(['target', 'representation'] + ['model', 'criterion', 'dist', 'candidates', 'init_size', 'batch_size', 'seed']).ngroups:,}"
          f"  AL experiments {len(e):,}  ({(e.candidates == CANDIDATES).sum():,} with {CANDIDATES} candidates)")

    print("\nMEAN FINAL CEF per criterion and surrogate (native forms, five representations)")
    print(p.groupby(level="model")[CRITERIA].mean().round(2).to_string())

    print("\nCRITERION MINUS GREEDY, per surrogate (native forms)")
    for m in MAIN_SURROGATES:
        q = p.xs(m, level="model", drop_level=False)
        for c in CRITERIA[1:]:
            d = (q[c] - q.greedy).dropna()
            print(f"  {SURROGATE_SHORT[m]:8s} {c:11s} {fmt(d):26s} | {by_target(d)}")

    print("\nY_THRESH MINUS Y+, per surrogate and criterion: CEF | best compound found (activity units)")
    pb = subset(wide(nat, "best_y"), CH4_REPRESENTATIONS)
    for m in MAIN_SURROGATES:
        for a in ["ei", "pi"]:
            d = (p.xs(m, level="model", drop_level=False)[f"{a}_ythresh"] - p.xs(m, level="model", drop_level=False)[f"{a}_yplus"]).dropna()
            b = (pb.xs(m, level="model", drop_level=False)[f"{a}_ythresh"] - pb.xs(m, level="model", drop_level=False)[f"{a}_yplus"]).dropna()
            print(f"  {SURROGATE_SHORT[m]:8s} {a.upper()}  CEF {fmt(d):26s} best_y {b.mean():+.3f} | {by_target(d)}")

    print("\nBEST COMPOUND FOUND, criterion minus greedy (activity units), pooled over the three surrogates")
    for c in CRITERIA[1:]:
        b = subset(pb, models=MAIN_SURROGATES)
        print(f"  {c:11s} {fmt((b[c] - b.greedy).dropna())}")

    print("\nMEAN ACTIVITY OF THE ACQUIRED MOLECULES, criterion minus greedy, pooled; sign agreement with CEF")
    py = subset(wide(nat, "mean_y"), CH4_REPRESENTATIONS, MAIN_SURROGATES)
    pc = subset(p, models=MAIN_SURROGATES)
    for c in CRITERIA[1:]:
        dy, dc = (py[c] - py.greedy).dropna(), (pc[c] - pc.greedy).dropna()
        agree = (np.sign(dy) == np.sign(dc.reindex(dy.index))).mean()
        print(f"  {c:11s} {fmt(dy):26s} sign agrees with CEF in {agree:.0%}")

    print("\nTABPFN MINUS ALTERNATIVE, per criterion and representation (native forms)")
    for alt in ["rf", "gp"]:
        for c in CRITERIA:
            q = w.xs(c, level="criterion")
            d = (q.tabpfn - q[alt]).dropna()
            print(f"  vs {SURROGATE_SHORT[alt]:7s} {c:11s} {fmt(d):26s} | {by_target(d)}")
        for r in CH4_REPRESENTATIONS:
            q = w.xs(r, level="representation")
            print(f"  vs {SURROGATE_SHORT[alt]:7s} all criteria, {r:18s} {fmt((q.tabpfn - q[alt]).dropna())}")

    print("\nTABPFN MINUS RF BY ROUND (all criteria pooled, native forms, five representations)")
    f = df[(df.candidates == CANDIDATES) & df.representation.isin(CH4_REPRESENTATIONS) & df.model.isin(["tabpfn", "rf"])]
    f = f[(f.criterion == "greedy") | (f.dist == f.model.map(NATIVE_DIST))]
    a = f.groupby(["target", "representation", "criterion", "init_size", "batch_size", "round", "model"]).cef.mean().unstack("model")
    d = (a.tabpfn - a.rf).reset_index(name="d")
    for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 20)]:
        x = d[(d["round"] >= lo) & (d["round"] <= hi)].d
        print(f"  rounds {lo:>2}–{hi:<2} {x.mean():+.2f}  won {(x > 0).mean():.0%}  (n = {len(x)})")

    print("\nTABPFN ABLATION over ALL sixteen representations (full minus Gaussian)")
    q = e[(e.model == "tabpfn") & (e.candidates == CANDIDATES) & (e.criterion != "greedy")]
    wa = q.pivot_table(index=EXPERIMENT_KEY + ["criterion"], columns="dist", values="cef")
    for c in ABLATION_ROWS:
        d = (wa.full - wa.gauss).xs(c, level="criterion", drop_level=False).dropna()
        print(f"  {c:11s} {fmt(d):26s} | {by_target(d)}")

    print("\nSEED-TO-SEED SD of final CEF within one AL experiment (native, five representations)")
    s = nat[(nat.candidates == CANDIDATES) & nat.representation.isin(CH4_REPRESENTATIONS)]
    s = s.assign(budget=s.init_size + N_ROUNDS * s.batch_size)
    for (ini, bs), g in s.groupby(["init_size", s.groupby("target").batch_size.rank(method="dense").astype(int)]):
        print(f"  N_init {ini:>3}, {'small' if bs == 1 else 'large'} batch: median SD {g.sd.median():.2f}, max {g.sd.max():.2f}")

    print("\nCANDIDATE SUBSAMPLING — full pool minus 1,000 candidates (USP7, D2R; CheMeleon; native forms)")
    k = ["target", "representation", "model", "criterion", "init_size", "batch_size"]
    full = nat[nat.candidates == "all"].set_index(k)
    sub = nat[nat.candidates == CANDIDATES].set_index(k)
    j = full[["cef", "best_y"]].join(sub[["cef", "best_y"]], rsuffix="_1k", how="inner")
    for m in MAIN_SURROGATES:
        jj = j.xs(m, level="model")
        print(f"  {SURROGATE_SHORT[m]:8s} CEF {fmt(jj.cef - jj.cef_1k):26s} best_y {fmt(jj.best_y - jj.best_y_1k)}")
    for t in ["usp7", "d2r"]:
        jj = j.xs(t, level="target")
        print(f"  {t}: rank correlation of the AL experiments, full pool vs 1,000: "
              f"{jj.cef.rank().corr(jj.cef_1k.rank()):.2f} (n = {len(jj)})")

    print("\nREPRODUCIBILITY — the same configurations in the earlier extended run (exact replicate)")
    old = experiments(read_runs(REPLICATE_DIR, "acquisition_*.csv"))
    k = EXPERIMENT_KEY + ["criterion", "dist", "candidates"]
    j = e.set_index(k)[["cef"]].join(old.set_index(k)[["cef"]], rsuffix="_old", how="inner")
    j["d"] = (j.cef - j.cef_old).abs()
    j["features"] = np.where(j.index.get_level_values("representation") == "morgan", "Morgan (CPU)", "GPU-featurised")
    print(f"  {len(j)} matched AL experiments, correlation {j.cef.corr(j.cef_old):.3f}")
    print(j.groupby(["features", j.index.get_level_values("model")]).d.agg(
        identical=lambda x: (x == 0).mean(), median="median", max="max").round(3).to_string())

    print("\nAPPENDIX: TANIMOTO GP (all sixteen representations, native)")
    wall = by_model(nat)
    for c in ["greedy", "pi_ythresh"]:
        q = wall.xs(c, level="criterion")
        print(f"  {c:11s} Tanimoto − RBF GP {fmt((q.gp_tanimoto - q.gp).dropna()):26s}"
              f" TabPFN − Tanimoto {fmt((q.tabpfn - q.gp_tanimoto).dropna())}")
    q = wall.xs("greedy", level="criterion")
    print("  greedy, Tanimoto − RBF GP by representation:")
    for r, g in (q.gp_tanimoto - q.gp).dropna().groupby(level="representation"):
        print(f"    {r:42s} {fmt(g)}")
    print("  mean final CEF of the Tanimoto GP per criterion (five representations): "
          + "  ".join(f"{c} {v:.2f}" for c, v in subset(wide(nat), CH4_REPRESENTATIONS).xs("gp_tanimoto", level="model")[CRITERIA].mean().items()))

    print("\nRUNTIME — median seconds per round (min–max over representations)")
    r = pd.read_csv(REPO_ROOT / RUNTIME_CSV)
    r = r[r.status == "ok"]
    r["cands"] = np.where(r.candidates == 1000, "1000", "pool")
    lo = r.pivot_table(index=["model", "cands"], columns="context_size", values="median_seconds", aggfunc="min")
    hi = r.pivot_table(index=["model", "cands"], columns="context_size", values="median_seconds", aggfunc="max")
    print((lo.round(2).astype(str) + "–" + hi.round(2).astype(str)).to_string())


def main() -> None:
    df = read_runs(FINAL_DIR, "final_*.csv")
    e = experiments(df)
    nat = native(e)
    figure_curves(df)
    figure_acquisition(nat)
    figure_surrogates(nat)
    figure_ablation(e)
    figure_representation(nat)
    figure_calibration()
    numbers(df, e, nat)


if __name__ == "__main__":
    main()
