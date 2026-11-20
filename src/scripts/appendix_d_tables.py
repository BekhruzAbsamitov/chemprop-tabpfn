"""scripts/appendix_d_tables.py — every value of Appendix D (detailed results of Chapter 5). Press Run.

Prints Tables D.1-D.10 and writes them as LaTeX to OUT_TEX, ready to copy into the thesis as
Content/Appendix_D_tables.tex (input by Content/Appendix_D.tex). Chapter 5 cites each table by label.

  D.1  transfer per target       trained encoder minus its OWN random initialisation, final CEF
  D.2  transfer per surrogate    the same differences under TabPFN, the random forest and the RBF GP
  D.3  transfer per strategy     the same differences under greedy, EI and PI at y+ and at y_thresh
  D.4  between encoders          checkpoints, learning rates and seeds against each other; the two controls; CheMeleon
  D.5  in-domain validation      NLL and Spearman per run: initial, end, best
  D.6  in-domain trends, noise   slopes, split halves, counts against the initial value, epoch-to-epoch noise
  D.7  controls                  A/B tests, gradient norms, fit of the training data
  D.8  geometry                  standardised and raw effective rank, redundancy, anchors
  D.9  CKA, short runs           linear and debiased CKA with the initialisation, and the three anchors
  D.10 CKA, long runs            the same along the trajectories of the 100,000-step runs
  D.11 CKA, epoch-level run      the same per epoch of the epoch-level run, both seeds (cka_epoch_level.py)
  D.12 encoder vs representations the encoder of Chapter 5 minus each representation of Chapter 4, per surrogate

Transfer values come from the final benchmark run (experiments/benchmark_final: final_*.csv and the second-seed
control seed1_control_*.csv). A MATCHED PAIR is one target, surrogate (TabPFN with its full predictive distribution,
RF and RBF GP, each in its native form), strategy (greedy; EI and PI at y+ and at y_thresh) and configuration
(initial-set size, batch size), final round, mean of the five seeds. The unit of inference is the SCENARIO, one
target in one configuration (4 x 4 = 16; 12 for the domain-matched encoders, which have no TYK2 arm), whose value is
the mean of its pairs (15: three surrogates x five strategies; five or three in the per-surrogate and per-strategy
splits). The seed-1 encoders are compared with the seed-1 control, all others with the seed-0 control.
MEAN26, the encoder Chapter 5 follows, is the epoch-level encoder at its last epoch (26) averaged over the two seeds:
per matched pair, the mean of the two differences, each seed against its own control.
Per target: the mean of the four scenarios and the scenarios won (four values give no useful standard error).
All targets, per surrogate, per strategy: mean +- SE over scenarios, scenarios won, * = |mean| > 2 SE.
In-domain values are those of plot_training.py; geometry reads feature_stats_standardised.csv; CKA reads
cka_study.csv, cka_trajectory.csv and cka_debiased.csv.

Run with:  uv run python src/scripts/appendix_d_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
EXP = "experiments"
OUT_TEX = "experiments/appendix_d_tables.tex"
TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
TARGET_TEX = {"saureus": r"\emph{S.~aureus}", "tyk2": "TYK2", "usp7": "USP7", "d2r": "D2R"}
MODELS = [("tabpfn", "TabPFN"), ("rf", "RF"), ("gp", "RBF GP")]
STRATEGIES = [("greedy", "greedy"), ("ei_yplus", r"EI($y^{+}$)"), ("pi_yplus", r"PI($y^{+}$)"),
              ("ei_ythresh", r"EI($y_{\mathrm{thresh}}$)"), ("pi_ythresh", r"PI($y_{\mathrm{thresh}}$)")]

# The final benchmark run (benchmark_extended.py "final") and the second-seed control (experiment "seed1_control").
FINAL_DIR = "experiments/benchmark_final"
RUN_GLOBS = ["final_*.csv", "seed1_control_*.csv"]
N_SEEDS = 5
N_ROUNDS = 20
CANDIDATES = "1000"              # the main protocol
NATIVE_DIST = {"tabpfn": "full", "rf": "gauss", "gp": "gauss"}   # the three main surrogates, native predictive form
PAIR_KEY = ["target", "model", "criterion", "init_size", "batch_size"]   # a matched pair of two arms shares these
SCENARIO = ["target", "init_size", "batch_size"]    # the unit of inference: one target in one configuration
PAIRS_PER_SCENARIO = 15          # 3 surrogates x 5 strategies
CONTROL = "chemprop_untrained"               # random initialisation of the first seed
CONTROL_SEED1 = "chemprop_untrained_seed1"   # random initialisation of the second seed (seed1_control)

# The encoders of the transfer tables: (group heading, [(row label, representation, control)]).
BEST100, FINAL100 = "chemprop_trained_full_sp_best", "chemprop_trained_full_sp"
BEST10, FINAL10 = "chemprop_trained_full_sp_10ep_best", "chemprop_trained_full_sp_10ep"
E23, E26 = "chemprop_trained_full_sp_50ep_best", "chemprop_trained_full_sp_50ep_final"
S21, S26 = "chemprop_trained_full_sp_50ep_seed1_best", "chemprop_trained_full_sp_50ep_seed1_final"
# An averaged arm: a tuple of (representation, control) arms in place of the representation; its gain is the mean of
# the gains of its arms, each over its own control. Used with control None.
MEAN26 = ((E26, CONTROL), (S26, CONTROL_SEED1))     # the encoder Chapter 5 follows
TRANSFER_GROUPS = [
    (r"Rate $10^{-4}$", [
        ("All of ChEMBL", "chemprop_trained_full", CONTROL),
        ("Domain-matched", "chemprop_trained_own", CONTROL),
        ("Domain-matched, long", "chemprop_trained_own_long", CONTROL),
        ("Single-protein, $100$k steps, final", FINAL100, CONTROL),
        ("Single-protein, $100$k steps, best", BEST100, CONTROL),
        ("Single-protein, $10$ epochs, final", FINAL10, CONTROL),
        ("Single-protein, $10$ epochs, best", BEST10, CONTROL)]),
    (r"Rate $3 \times 10^{-5}$, epoch-level run", [
        ("Mean of the two seeds, last epoch ($26$)", MEAN26, None),
        ("First seed, best epoch ($23$)", E23, CONTROL),
        ("First seed, last epoch ($26$)", E26, CONTROL),
        ("Second seed, best epoch ($21$)", S21, CONTROL_SEED1),
        ("Second seed, last epoch ($26$)", S26, CONTROL_SEED1)]),
    ("Reference", [("CheMeleon", "chemeleon", CONTROL)]),
]
# The representations of Chapter 4, for the table that places the encoder of Chapter 5 among them.
CHAPTER4_REPRESENTATIONS = [("CheMeleon", "chemeleon"), ("Mordred", "mordred3d"), ("Morgan", "morgan"),
                            ("Physicochemical", "physchem"), ("Random initialisation", CONTROL)]
NOT_INTERPRETED = {("chemprop_trained_full", "saureus")}   # 72 % same-target overlap (Section 3.2)

# Comparisons between encoders (Table D.4): (group heading, [(row label, (arm, control), (arm, control) or None)]).
# A comparison is the gain of the first arm over its control minus the gain of the second over its control; where the
# second is None the first is a raw difference between two representations (control None).
WITHIN_RUN = [
    ("Best validation minus final checkpoint", [
        (r"$100$k steps", (BEST100, CONTROL), (FINAL100, CONTROL)),
        (r"$10$ epochs", (BEST10, CONTROL), (FINAL10, CONTROL)),
        (r"Epoch-level, first seed (epoch $23$ minus $26$)", (E23, CONTROL), (E26, CONTROL)),
        (r"Epoch-level, second seed (epoch $21$ minus $26$)", (S21, CONTROL_SEED1), (S26, CONTROL_SEED1))]),
    (r"Rate $3 \times 10^{-5}$ minus rate $10^{-4}$ (the $10$-epoch run, final)", [
        (r"First seed, best epoch ($23$)", (E23, CONTROL), (FINAL10, CONTROL)),
        (r"First seed, last epoch ($26$)", (E26, CONTROL), (FINAL10, CONTROL)),
        (r"Second seed, best epoch ($21$)", (S21, CONTROL_SEED1), (FINAL10, CONTROL)),
        (r"Second seed, last epoch ($26$)", (S26, CONTROL_SEED1), (FINAL10, CONTROL))]),
    ("Second seed minus first seed, each over its own random initialisation", [
        (r"Best epochs ($21$ and $23$)", (S21, CONTROL_SEED1), (E23, CONTROL)),
        (r"Last epochs ($26$ and $26$)", (S26, CONTROL_SEED1), (E26, CONTROL))]),
    (r"Mean of the two seeds at the last epoch ($26$), each seed over its own random initialisation", [
        (r"Minus the $10$-epoch run at $10^{-4}$ (final)", (MEAN26, None), (FINAL10, CONTROL)),
        (r"Minus CheMeleon", (MEAN26, None), ("chemeleon", CONTROL))]),
    ("Second random initialisation minus first, raw", [
        ("Control, second seed minus first", (CONTROL_SEED1, None), (CONTROL, None))]),
    ("Epoch-level encoder minus CheMeleon, raw", [
        (r"First seed, best epoch ($23$)", (E23, None), ("chemeleon", None)),
        (r"First seed, last epoch ($26$)", (E26, None), ("chemeleon", None)),
        (r"Second seed, best epoch ($21$)", (S21, None), ("chemeleon", None)),
        (r"Second seed, last epoch ($26$)", (S26, None), ("chemeleon", None))]),
]

# In-domain curves: (row label, curve CSV, rate, validation, x unit)
STEP_RUNS = [
    ("D2R, long", "train_val_curve_d2r_long.csv", r"$10^{-4}$", "64 ep.", "step"),
    (r"\emph{S.~aureus}, long", "train_val_curve_saureus_long.csv", r"$10^{-4}$", "64 ep.", "step"),
    ("USP7, long", "train_val_curve_usp7_long.csv", r"$10^{-4}$", "16 ep.", "step"),
    ("100k steps", "train_val_curve_full_sp.csv", r"$10^{-4}$", "64 ep.", "step"),
    ("10 epochs", "train_val_curve_full_sp_10ep.csv", r"$10^{-4}$", "64 ep.", "epoch"),
    ("10 epochs, lower rate", "train_val_curve_full_sp_10ep_lowlr.csv", r"$3 \times 10^{-5}$", "64 ep.", "epoch"),
]
EPOCH_CSV = "val_curve_full_sp_50ep.csv"
EPOCH_REEVAL_CSV = "val_reevaluated_full_sp_50ep.csv"   # epochs 25-26 were not logged (cluster restart)
EPOCH_SEED2_CSV = "val_curve_full_sp_50ep_seed1.csv"
STEPS_PER_EPOCH = 31_384
EVALS_PER_EPOCH = 8
DOMAIN_RUNS = [("D2R", "d2r_long", "D2R"), (r"\emph{S.~aureus}", "saureus_long", "S_aureus"),
               ("USP7", "usp7_long", "USP7")]
AB_ARMS = [("Clip at $1$ (default)", "ab_clip_1p0"), ("Clip at $10$", "ab_clip_10p0"),
           ("Clip at $1{,}000$", "ab_clip_1000p0"), ("Mean squared error", "ab_loss_mse"),
           ("Huber loss", "ab_loss_huber")]

# Geometry: (row label, feature_stats run, end step: None = last checkpoint)
GEOMETRY_RUNS = [
    (r"\emph{S.~aureus}, long", "saureus_long", None), ("D2R, long", "d2r_long", None),
    ("USP7, long", "usp7_long", None), ("100k steps", "full_sp", None), ("10 epochs", "full_sp_10ep", None),
    ("Epoch-level, best (23)", "full_sp_50ep", 23), ("Epoch-level, last (26)", "full_sp_50ep", 26),
    ("Second seed, best (21)", "full_sp_50ep_seed1", 21), ("Second seed, last (26)", "full_sp_50ep_seed1", 26),
]
FEATURE_STD_CSV = "feature_stats_standardised.csv"

CKA_STATIC = [("All of ChEMBL", "full"), ("D2R", "d2r"), (r"\emph{S.~aureus}", "saureus"), ("USP7", "usp7")]
CKA_LONG = [("D2R", "d2r_long"), (r"\emph{S.~aureus}", "saureus_long"), ("USP7", "usp7_long")]
CKA_STEPS = [10_000, 40_000, 100_000]
CKA_DEBIASED_CSV = "cka_debiased.csv"
CKA_EPOCH_CSV = "cka_epoch_level.csv"        # written by cka_epoch_level.py
CKA_EPOCH_RUNS = [("First seed", "full_sp_50ep"), ("Second seed", "full_sp_50ep_seed1")]
CKA_EPOCHS = [1, 5, 10, 20, 26]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def txt(c) -> str:
    """One cell for the printed report: mean [± SE over scenarios] (scenarios won / scenarios)."""
    if c is None:
        return "—"
    m, se, won, n = c
    if se is None:
        return f"{m:+.2f}              ({won}/{n})"
    return f"{m:+.2f} ± {se:.2f}{'*' if abs(m) > 2 * se else ' '}({won}/{n})"


def tex(c, dagger: bool = False) -> str:
    """One LaTeX cell. A per-target cell (no SE: four scenarios) is the mean and the scenarios won; an all-targets,
    per-surrogate or per-strategy cell is the mean and SE over scenarios, * above two standard errors."""
    if c is None:
        return "---"
    m, se, won, n = c
    m = 0.0 if abs(m) < 0.005 else m   # print a mean that rounds to zero as 0.00, not -0.00
    value = "0.00" if m == 0 else f"{m:+.2f}"
    if se is None:
        mark = r"^{\dagger}" if dagger else ""
    else:
        mark = "^{*}" if abs(m) > 2 * se else ""
        value = f"{value} \\pm {se:.2f}"
    return rf"\appcell{{{value}{mark}}}{{{won}/{n}}}"

def num(x: float, d: int = 3, sign: bool = False) -> str:
    return f"${x:+.{d}f}$" if sign else f"${x:.{d}f}$"


def pct(x: float) -> str:
    return "$0\\%$" if round(x) == 0 else f"${x:+.0f}\\%$"


class Doc:
    """Collects the printed report and the LaTeX tables side by side."""

    def __init__(self) -> None:
        self.tex: list[str] = []

    def heading(self, title: str) -> None:
        print(f"\n=== {title}")

    def table(self, label: str, caption: str, colspec: str, header: list[str], rows: list[str],
              size: str = r"\footnotesize") -> None:
        self.tex += [r"\begin{table}[!htbp]", r"\centering", size, r"\setlength{\tabcolsep}{2pt}",
                     rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                     rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} {colspec} @{{}}}}",
                     r"\toprule", *header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}",
                     r"\end{table}", ""]


# Indented row label under a group heading. The kern keeps the label on the first line (a \quad lets TeX break
# right after it, dropping a long label to the second line) and the hanging indent keeps wrapped lines indented.
SUB = r"\hangindent=0.5em\hangafter=1\leavevmode\kern0.5em "   # 0.5em: "Physicochemical" must fit a 2.7cm label column
def group(title: str, ncols: int, first: bool = False) -> list[str]:
    return ([] if first else [r"\addlinespace[2pt]"]) + [
        rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\", r"\addlinespace[2pt]"]


def plain(title: str) -> str:
    """A group title without LaTeX, for the printed report."""
    return title.replace("$10^{-4}$", "1e-4").replace(r"$3 \times 10^{-5}$", "3e-5")


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
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
                {f"chemprop_trained_own_{t}{kind}": f"chemprop_trained_own{kind}" for t in TARGETS})
        _WIDE = g.pivot_table(index=PAIR_KEY, columns="representation", values="cef")
    return _WIDE


def gain(arm: tuple[str, str | None]) -> pd.Series:
    """An arm minus its control (no control: the arm itself), final CEF, one value per matched pair. An averaged
    arm (a tuple of arms in place of the representation) is the mean of the gains of its arms."""
    rep, ctrl = arm
    if isinstance(rep, tuple):
        return sum(gain(a) for a in rep) / len(rep)
    w = final_wide()
    return w[rep] - (w[ctrl] if ctrl else 0.0)


def comparison(a: tuple[str, str | None], b: tuple[str, str | None] | None = None) -> pd.Series:
    """Gain of a minus gain of b (b None: the gain of a), per matched pair; pairs missing in either arm are dropped."""
    return (gain(a) if b is None else gain(a) - gain(b)).dropna()


def scenario_means(d: pd.Series, model: str | None = None, criterion: str | None = None) -> pd.Series:
    """The unit of inference: the mean of the matched pairs inside each scenario (a target in one configuration),
    optionally restricted to one surrogate (5 pairs per scenario) or one strategy (3 pairs per scenario)."""
    expected = PAIRS_PER_SCENARIO
    if model:
        d, expected = d.xs(model, level="model", drop_level=False), expected // 3
    if criterion:
        d, expected = d.xs(criterion, level="criterion", drop_level=False), expected // 5
    n = d.groupby(SCENARIO).size()
    assert (n == expected).all(), n[n != expected]
    return d.groupby(SCENARIO).mean()


def scen_stat(s: pd.Series) -> tuple[float, float, int, int]:
    """Mean over scenarios, its standard error over scenarios, scenarios won, scenarios."""
    return s.mean(), s.std(ddof=1) / np.sqrt(len(s)), int((s > 0).sum()), len(s)


def target_cells(d: pd.Series) -> list:
    """Per target: the mean of its four scenarios and the scenarios won (no SE from four values; None where the
    encoder has no arm); then all targets: mean, SE over scenarios, scenarios won."""
    s = scenario_means(d)
    present = set(s.index.get_level_values("target"))
    cells = []
    for t in TARGETS:
        if t in present:
            x = s.xs(t, level="target")
            cells.append((x.mean(), None, int((x > 0).sum()), len(x)))
        else:
            cells.append(None)
    return cells + [scen_stat(s)]


def curve(csv: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / EXP / csv)


def epoch_run() -> pd.DataFrame:
    e = curve(EPOCH_CSV)
    r = curve(EPOCH_REEVAL_CSV)[["epoch", "val_loss", "val_spearman"]]
    return pd.concat([e, r[~r.epoch.isin(e.epoch)]]).sort_values("epoch").reset_index(drop=True)


def where(x: float, unit: str) -> str:
    if unit == "epoch":
        return f"epoch {x / STEPS_PER_EPOCH:.2f}".rstrip("0").rstrip(".")
    return f"step {x / 1000:.0f}k"


TARGET_HEADER = [r"\textbf{Encoder} & " + " & ".join(rf"\textbf{{{TARGET_TEX[t]}}}" for t in TARGETS)
                 + r" & \textbf{All} \\"]


# --------------------------------------------------------------------------- #
# D.1 Transfer per target                                                     #
# --------------------------------------------------------------------------- #
def label_name(label: str, rep: str) -> str:
    return SUB + rf"\textit{{{label}}}" if rep == "chemeleon" else SUB + rf"{label}"


def table_d1(doc: Doc) -> None:
    doc.heading("D.1 Transfer per target: encoder minus its own random init., final CEF (scenarios: per target the "
                "mean of four and scenarios won; all targets mean ± SE over 16, 12 without TYK2)")
    rows = []
    for gi, (title, arms) in enumerate(TRANSFER_GROUPS):
        print(f"  [{plain(title)}]")
        rows += group(title, 6, first=gi == 0)
        for label, rep, ctrl in arms:
            cells = target_cells(comparison((rep, ctrl)))
            print(f"    {label.replace('$', ''):44s} " + "  ".join(f"{txt(c):>26s}" for c in cells))
            marks = [(rep, t) in NOT_INTERPRETED for t in TARGETS] + [False]
            rows.append(label_name(label, rep) + " & " + " & ".join(tex(c, m) for c, m in zip(cells, marks))
                        + r" \\[2pt]")
    doc.table("tab:appd_transfer_targets",
              r"Each encoder minus its own randomly initialised encoder (the second-seed encoders minus the "
              r"second-seed control), in final CEF. The unit is a scenario, one target in one configuration; its "
              r"value is the mean of the matched pairs it contains, over three surrogates and five strategies. "
              r"Target columns: the mean of the target's four scenarios and, below it, the scenarios in which the "
              r"encoder ended higher. All: the mean $\pm$ standard error over the $16$ scenarios ($12$ for the "
              r"domain-matched encoders, which have no TYK2 arm) and the scenarios won; $^{*}$more than two standard "
              r"errors. CheMeleon (italic) is the reference. $^{\dagger}$Not interpreted: $72\%$ of the "
              r"\emph{S.~aureus} pool occurs in this encoder's training split on the same target "
              r"(Section~\ref{subsec:leakage}); the All value includes it.",
              r"L{3.35cm} l l l l l", TARGET_HEADER, rows)


# --------------------------------------------------------------------------- #
# D.2 Transfer per surrogate                                                  #
# --------------------------------------------------------------------------- #
def split_table(doc: Doc, heading: str, label: str, caption: str, splits: list[tuple[str, dict]],
                with_all: bool = True, label_width: str = "3.35cm") -> None:
    """Every transfer encoder against its control, one column per split of the matched pairs (and, if with_all,
    one for all pairs)."""
    doc.heading(heading)
    cols = len(splits) + (1 if with_all else 0)
    rows = []
    for gi, (title, arms) in enumerate(TRANSFER_GROUPS):
        print(f"  [{plain(title)}]")
        rows += group(title, cols + 1, first=gi == 0)
        for name, rep, ctrl in arms:
            d = comparison((rep, ctrl))
            cells = [scen_stat(scenario_means(d, **kw)) for _, kw in splits]
            if with_all:
                cells.append(scen_stat(scenario_means(d)))
            print(f"    {name.replace('$', ''):44s} " + "  ".join(f"{txt(c):>20s}" for c in cells))
            rows.append(label_name(name, rep) + " & " + " & ".join(tex(c) for c in cells) + r" \\[3pt]")
    header = [r"\textbf{Encoder} & " + " & ".join(rf"\textbf{{{n}}}" for n, _ in splits)
              + (r" & \textbf{All} \\" if with_all else r" \\")]
    doc.table(label, caption, rf"L{{{label_width}}} " + " ".join(["l"] * cols), header, rows)


def table_d2(doc: Doc) -> None:
    split_table(doc, "D.2 Transfer per surrogate (mean ± SE over scenarios, scenarios won)",
                "tab:appd_transfer_surrogates",
                r"The differences of Table~\ref{tab:appd_transfer_targets} split by surrogate. The value of a scenario "
                r"is the mean of the five matched pairs of that surrogate, one per strategy; the mean $\pm$ standard "
                r"error is taken over the $16$ scenarios ($12$ for the domain-matched encoders) and the scenarios won "
                r"are given below; $^{*}$more than two standard errors.",
                [(n, {"model": m}) for m, n in MODELS])


def table_d3(doc: Doc) -> None:
    split_table(doc, "D.3 Transfer per strategy (mean ± SE over scenarios, scenarios won)",
                "tab:appd_transfer_strategies",
                r"The differences of Table~\ref{tab:appd_transfer_targets} split by strategy. The value of a scenario "
                r"is the mean of the three matched pairs of that strategy, one per surrogate; the mean $\pm$ standard "
                r"error is taken over the $16$ scenarios ($12$ for the domain-matched encoders) and the scenarios won "
                r"are given below; $^{*}$more than two standard errors. Over all five strategies the values are those "
                r"of the last column of Table~\ref{tab:appd_transfer_targets}.",
                [(n, {"criterion": c}) for c, n in STRATEGIES], with_all=False, label_width="2.72cm")   # no All column (it is that of D.1) and a narrower label: six columns do not fit


# --------------------------------------------------------------------------- #
# D.4 Comparisons between encoders                                            #
# --------------------------------------------------------------------------- #
def table_d4(doc: Doc) -> None:
    doc.heading("D.4 Comparisons between encoders of the final run (scenario level)")
    rows = []
    for gi, (title, comparisons) in enumerate(WITHIN_RUN):
        print(f"  [{title.replace('$', '')}]")
        rows += group(title, 6, first=gi == 0)
        for label, a, b in comparisons:
            cells = target_cells(comparison(a, b))
            print(f"    {label.replace('$', ''):44s} " + "  ".join(f"{txt(c):>26s}" for c in cells))
            rows.append(SUB + rf"{label} & " + " & ".join(tex(c) for c in cells) + r" \\[2pt]")
    header = [r"\textbf{Comparison} & " + " & ".join(rf"\textbf{{{TARGET_TEX[t]}}}" for t in TARGETS)
              + r" & \textbf{All} \\"]
    doc.table("tab:appd_within_run",
              r"Comparisons between encoders, in final CEF, as in Table~\ref{tab:appd_transfer_targets}. Each encoder "
              r"is first taken minus the control of its own seed; the last two groups are raw differences between "
              r"two arms.",
              r"L{4.6cm} l l l l l", header, rows)


# --------------------------------------------------------------------------- #
# D.5 In-domain validation per run                                            #
# --------------------------------------------------------------------------- #
def table_d5(doc: Doc) -> None:
    doc.heading("D.5 In-domain validation: NLL and Spearman, initial -> end, best")
    rows = []
    for label, csv, rate, val, unit in STEP_RUNS:
        q = curve(csv)
        lo, hi = q.val_loss.idxmin(), q.val_spearman.idxmax()
        vals = (q.val_loss.iloc[0], q.val_loss.iloc[-1], q.val_loss[lo], where(q.step[lo], unit),
                q.val_spearman.iloc[0], q.val_spearman.iloc[-1], q.val_spearman[hi], where(q.step[hi], unit))
        rows.append(f"{label} & {rate} & {val} & {num(vals[0])} & {num(vals[1])} & {num(vals[2])} & {vals[3]} & "
                    f"{num(vals[4])} & {num(vals[5])} & {num(vals[6])} & {vals[7]} \\\\")
        print(f"  {csv:40s} NLL {vals[0]:.3f} -> {vals[1]:.3f} (lowest {vals[2]:.3f}, {vals[3]});  "
              f"Spearman {vals[4]:.3f} -> {vals[5]:.3f} (highest {vals[6]:.3f}, {vals[7]})")
    for label, e in [("Epoch-level", epoch_run()), ("Second seed", curve(EPOCH_SEED2_CSV))]:
        lo, hi = e.val_loss.idxmin(), e.val_spearman.idxmax()
        vals = (e.val_loss.iloc[0], e.val_loss.iloc[-1], e.val_loss[lo], f"epoch {int(e.epoch[lo])}",
                e.val_spearman.iloc[0], e.val_spearman.iloc[-1], e.val_spearman[hi], f"epoch {int(e.epoch[hi])}")
        rows.append(f"{label} & $3 \\times 10^{{-5}}$ & all & {num(vals[0])} & {num(vals[1])} & {num(vals[2])} & "
                    f"{vals[3]} & {num(vals[4])} & {num(vals[5])} & {num(vals[6])} & {vals[7]} \\\\")
        print(f"  {label:40s} NLL {vals[0]:.4f} -> {vals[1]:.4f} (lowest {vals[2]:.4f}, {vals[3]}, "
              f"{100 * (vals[2] / vals[0] - 1):+.1f} %);  Spearman {vals[4]:.4f} -> {vals[5]:.4f} "
              f"(highest {vals[6]:.4f}, {vals[7]}); last epoch {int(e.epoch.iloc[-1])}")
    header = [r" & & & \multicolumn{4}{c}{\textbf{Validation NLL}} & \multicolumn{4}{c}{\textbf{Validation Spearman}} \\",
              r"\cmidrule(lr){4-7} \cmidrule(l){8-11}",
              r"\textbf{Run} & \textbf{Rate} & \textbf{Validation} & \textbf{init.} & \textbf{end} & \textbf{lowest} & "
              r"\textbf{at} & \textbf{init.} & \textbf{end} & \textbf{highest} & \textbf{at} \\"]
    doc.table("tab:appd_in_domain",
              r"In-domain validation on the held-out assays of each run's own corpus: for the random initialisation "
              r"(init.), at the end of training, and at the best point. The step-based runs are validated on a "
              r"fixed set of episodes, so single values are noisy (Table~\ref{tab:appd_trends}); the epoch-level "
              r"runs score every held-out point once per epoch. The epoch-level run was ended by a cluster maintenance after epoch~$26$ (epochs "
              r"$25$ and $26$ re-evaluated from their checkpoints, Appendix~\ref{app:epoch_run}); its second seed "
              r"stopped early after epoch~$26$.",
              r"l l l r r r l r r r l", header, rows, size=r"\scriptsize")


# --------------------------------------------------------------------------- #
# D.6 In-domain trends and noise                                              #
# --------------------------------------------------------------------------- #
def slope_per_epoch(q: pd.DataFrame, start: float = 5) -> tuple[float, float]:
    w = q[q.step / STEPS_PER_EPOCH >= start]
    x = w.step / STEPS_PER_EPOCH
    X = np.c_[np.ones(len(w)), x]
    beta, *_ = np.linalg.lstsq(X, w.val_loss, rcond=None)
    resid = w.val_loss - X @ beta
    return beta[1], np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def table_d6(doc: Doc) -> None:
    doc.heading("D.6 In-domain trends and noise")
    rows = []

    def add(run: str, quantity: str, value: str) -> None:
        rows.append(f"{run} & {quantity} & {value} \\\\")
        print(f"  {run.replace(chr(92), ''):34s} {quantity:62s} {value.replace('$', '')}")

    for (label, csv, *_), (_, _, _) in zip(STEP_RUNS[:3], DOMAIN_RUNS):
        q = curve(csv)
        base, e50 = q.iloc[0], q.iloc[1:]
        ss = stats.linregress(e50.step / 1e5, e50.val_spearman)
        sl = stats.linregress(e50.step / 1e5, e50.val_loss)
        add(label, r"Slope per $100{,}000$ steps: Spearman; NLL",
            f"${ss.slope:+.3f}$ (${p_text(ss.pvalue)}$); ${sl.slope:+.3f}$ (${p_text(sl.pvalue)}$)")
        add("", "Evaluations with NLL below / Spearman above the initial value",
            f"${(e50.val_loss < base.val_loss).sum()}/{len(e50)}$; ${(e50.val_spearman > base.val_spearman).sum()}/{len(e50)}$")
        add("", "Spearman: initial; mean of the second half; mean $\\pm$ SD of all $50$",
            f"${base.val_spearman:.3f}$; ${e50.val_spearman.iloc[len(e50) // 2:].mean():.3f}$; "
            f"${e50.val_spearman.mean():.3f} \\pm {e50.val_spearman.std(ddof=1):.3f}$")
    q = curve("train_val_curve_full_sp.csv")
    base, e50 = q.iloc[0], q.iloc[1:]
    a, c = e50.iloc[:len(e50) // 2], e50.iloc[len(e50) // 2:]
    halves = []
    for col in ["val_spearman", "val_loss"]:
        se = np.sqrt(a[col].var(ddof=1) / len(a) + c[col].var(ddof=1) / len(c))
        halves.append(f"${c[col].mean() - a[col].mean():+.3f} \\pm {se:.3f}$")
    add("100k steps", "Second minus first half: Spearman; NLL", "; ".join(halves))
    add("", "NLL: initial; mean of the second half; end",
        f"${base.val_loss:.3f}$; ${c.val_loss.mean():.3f}$; ${q.val_loss.iloc[-1]:.3f}$")
    for label, csv in [("10 epochs", "train_val_curve_full_sp_10ep.csv"),
                       ("10 epochs, lower rate", "train_val_curve_full_sp_10ep_lowlr.csv")]:
        q = curve(csv)
        b, se = slope_per_epoch(q)
        noise = q.iloc[::EVALS_PER_EPOCH].val_loss.diff().abs().median()
        add(label, "NLL slope per epoch over epochs $5$--$10$; median change between epochs",
            f"${b:+.4f} \\pm {se:.4f}$; ${noise:.3f}$")
    e = epoch_run()
    noise = e.val_loss.diff().abs().median()
    gain = e.val_loss.iloc[0] - e.val_loss.min()
    add("Epoch-level", "Median change between epochs; gain to the best epoch; ratio",
        f"${noise:.4f}$; ${gain:.3f}$; ${gain / noise:.0f}\\times$")
    add("", "NLL at epochs $23$--$26$",
        "; ".join(f"${v:.4f}$" for v in e[e.epoch >= 23].val_loss))
    header = [r"\textbf{Run} & \textbf{Quantity} & \textbf{Value} \\"]
    doc.table("tab:appd_trends",
              r"Trends and noise of the in-domain validation curves. Slopes are least-squares fits over the $50$ "
              r"evaluations after the random initialisation, with two-sided $p$-values from the slope's $t$-test; "
              r"the halves are evaluations $1$--$25$ and $26$--$50$. Noise is the median absolute change in "
              r"validation NLL between consecutive epochs (step-based runs: one evaluation per epoch).",
              r"L{2.3cm} L{6.0cm} L{4.9cm}", header, rows)


# --------------------------------------------------------------------------- #
# D.7 Controls                                                                #
# --------------------------------------------------------------------------- #
def table_d7(doc: Doc) -> None:
    doc.heading("D.7 Controls: A/B tests, gradient norms, fit of the training data")
    rows = group("A/B tests: validation Spearman at steps 0 / 100 / 200 / 300", 5, first=True)
    for label, arm in AB_ARMS:
        a = curve(f"{arm}.csv")
        v = " / ".join(f"{x:.3f}" for x in a.val_spearman)
        rows.append(SUB + rf"{label} & \multicolumn{{3}}{{l}}{{${v}$}} & ${a.grad_clip_frac.dropna().mean():.2f}$ \\")
        print(f"  {arm:15s} {v}   clipped share {a.grad_clip_frac.dropna().mean():.2f}")
    rows += group("Gradient norm before clipping: interval means, largest step, smallest clipped share", 5)
    for label, csv, *_ in STEP_RUNS:
        q = curve(csv)
        rows.append(SUB + rf"{label} & ${q.grad_mean.min():.0f}$--${q.grad_mean.max():.0f}$ & "
                    rf"${q.grad_max.max():,.0f}$ & & ${100 * q.grad_clip_frac.min():.1f}\%$ \\".replace(",", "{,}"))
        print(f"  {csv:40s} mean norm {q.grad_mean.min():.0f}-{q.grad_mean.max():.0f}, largest {q.grad_max.max():,.0f}, "
              f"clipped at least {100 * q.grad_clip_frac.min():.1f} %")
    for label, e in [("Epoch-level", curve(EPOCH_CSV)), ("Epoch-level, second seed", curve(EPOCH_SEED2_CSV))]:
        g = e.dropna(subset=["grad_mean"])
        late = g[g.epoch >= 19].grad_mean
        rows.append(SUB + rf"{label} & ${g.grad_mean.iloc[0]:.0f}$ (epoch $1$), ${late.min():.0f}$--${late.max():.0f}$ "
                    rf"(epochs $\ge 19$) & --- & & ${100 * g.grad_clip_frac.min():.1f}\%$ \\")
        print(f"  {label:40s} mean norm {g.grad_mean.iloc[0]:.1f} at epoch 1, {late.min():.1f}-{late.max():.1f} from epoch 19; "
              f"clipped at least {100 * g.grad_clip_frac.min():.1f} %")
    rows += group(r"Fit of the training data: training and validation NLL, first to last evaluation", 5)
    for label, run, corpus in DOMAIN_RUNS:
        q = curve(f"train_val_curve_{run}.csv")
        tr = q.dropna(subset=["train_loss"])
        n = pl.read_parquet(REPO_ROOT / "data" / "curated" / f"train_adaptive_{corpus}.parquet",
                            columns=["assay_x_type_x_doc", "split"]).filter(pl.col("split") == "train")[
            "assay_x_type_x_doc"].n_unique()
        r2 = stats.linregress(tr.step, tr.train_loss).rvalue ** 2
        gap = tr.train_loss.iloc[-1] - q.val_loss.iloc[-1]
        rows.append(SUB + rf"{label}, {100_000 / n:.0f} passes & "
                    rf"${tr.train_loss.iloc[0]:.3f} \to {tr.train_loss.iloc[-1]:.3f}$ ($r^2 = {r2:.2f}$) & "
                    rf"${q.val_loss.iloc[0]:.3f} \to {q.val_loss.iloc[-1]:.3f}$ & & ${gap:+.2f}$ \\".replace(",", "{,}"))
        print(f"  {run:13s} {n} training assays, {100_000 / n:.0f} passes; train {tr.train_loss.iloc[0]:.3f} -> "
              f"{tr.train_loss.iloc[-1]:.3f} (r2 {r2:.2f}); val {q.val_loss.iloc[0]:.3f} -> {q.val_loss.iloc[-1]:.3f}; gap {gap:+.2f}")
    header = [r"\textbf{Run or arm} & \multicolumn{3}{l}{\textbf{Values (see the group heading)}} & \textbf{Share / gap} \\"]
    doc.table("tab:appd_controls",
              r"Detail of the controls of Section~\ref{sec:pipeline_trains}. A/B tests: $300$ steps on the "
              r"domain-matched D2R corpus with one TabPFN ensemble member, the same seed and episode stream; last "
              r"column, the share of clipped steps. Gradient norms: the mean pre-clip norm per evaluation interval "
              r"(per epoch for the epoch-level runs), the largest single step and the smallest share of clipped "
              r"steps in any interval, at a clip threshold of $1$. Fit: last column, training minus validation NLL "
              r"at the end; passes are the $100{,}000$ steps divided by the training assays ($3{,}064$ for D2R, "
              r"$802$ for \emph{S.~aureus}, $139$ for USP7).",
              r"L{3.9cm} l l l r", header, rows)


# --------------------------------------------------------------------------- #
# D.8 Geometry                                                                #
# --------------------------------------------------------------------------- #
def table_d8(doc: Doc) -> None:
    doc.heading("D.8 Geometry: standardised effective rank, raw effective rank, mean |r|")
    fs = pd.read_csv(REPO_ROOT / EXP / FEATURE_STD_CSV)
    rows = []
    for label, run, end in GEOMETRY_RUNS:
        g = fs[fs.run == run]
        i0 = g[g.series == "init"].iloc[0]
        t = g[g.series == "trained"].sort_values("step")
        first, lo = t.iloc[0], t.loc[t.effective_rank_std.idxmin()]
        fin = t.iloc[-1] if end is None else t[t.step == end].iloc[0]
        unit = "epoch" if run.startswith("full_sp_50ep") else "step"

        def at(r) -> str:
            return f"{int(r.step)}" if unit == "epoch" else f"{r.step / 1000:,.1f}k".replace(".0k", "k")
        ch = 100 * (fin.effective_rank_std / i0.effective_rank_std - 1)
        ch_raw = 100 * (fin.effective_rank / i0.effective_rank - 1)
        rows.append(rf"{label} & ${i0.effective_rank_std:.1f}$ & ${first.effective_rank_std:.1f}$ "
                    rf"({at(first)}) & ${lo.effective_rank_std:.1f}$ ({at(lo)}) & ${fin.effective_rank_std:.1f}$ ({pct(ch)}) & "
                    rf"${i0.effective_rank:.1f} \to {fin.effective_rank:.1f}$ ({pct(ch_raw)}) & "
                    rf"${i0.mean_abs_corr:.3f}$ & ${t.mean_abs_corr.max():.3f}$ & ${fin.mean_abs_corr:.3f}$ \\".replace(",", "{,}"))
        print(f"  {run:19s} {int(i0.n_molecules):5d} mol; std rank {i0.effective_rank_std:.1f} -> first {first.effective_rank_std:.1f} "
              f"({at(first)}, {100 * (first.effective_rank_std / i0.effective_rank_std - 1):+.0f} %) -> lowest "
              f"{lo.effective_rank_std:.1f} ({at(lo)}, {100 * lo.effective_rank_std / i0.effective_rank_std:.0f} % of initial) -> "
              f"end {fin.effective_rank_std:.1f} ({ch:+.0f} %); raw {i0.effective_rank:.1f} -> {fin.effective_rank:.1f} "
              f"({ch_raw:+.0f} %); |r| {i0.mean_abs_corr:.3f}, max {t.mean_abs_corr.max():.3f}, end {fin.mean_abs_corr:.3f}; "
              f"initial {i0.eff_rank_std_frac:.3f} of width, end {fin.eff_rank_std_frac:.3f}")
    rows += group(r"Anchors on the same molecules: rank (fraction of width), mean $\lvert r \rvert$", 9)
    for label, run in [(r"\emph{S.~aureus}", "saureus_long"), ("D2R", "d2r_long"), ("USP7", "usp7_long"),
                       ("Single-protein", "full_sp")]:
        g = fs[fs.run == run]
        cells = []
        for s in ["chemeleon", "morgan"]:
            r = g[g.series == s].iloc[0]
            cells.append(rf"${r.effective_rank_std:.1f}$ (${r.eff_rank_std_frac:.3f}$), ${r.mean_abs_corr:.3f}$")
            print(f"  anchor {run:13s} {s:9s} rank {r.effective_rank_std:.1f} ({r.eff_rank_std_frac:.3f} of width), |r| {r.mean_abs_corr:.3f}")
        rows.append(SUB + rf"{label} & \multicolumn{{4}}{{l}}{{CheMeleon {cells[0]}}} & \multicolumn{{4}}{{l}}{{Morgan {cells[1]}}} \\")
    header = [r" & \multicolumn{4}{c}{\textbf{Effective rank, standardised}} & \textbf{Raw rank} & "
              r"\multicolumn{3}{c}{\textbf{Mean $\lvert r \rvert$}} \\",
              r"\cmidrule(lr){2-5} \cmidrule(lr){6-6} \cmidrule(l){7-9}",
              r"\textbf{Run} & \textbf{init.} & \textbf{first (at)} & \textbf{lowest (at)} & "
              r"\textbf{end (change)} & \textbf{init. $\to$ end} & \textbf{init.} & \textbf{max} & \textbf{end} \\"]
    doc.table("tab:appd_geometry",
              r"Geometry of the embeddings of a fixed set of held-out molecules per run ($495$ for \emph{S.~aureus}, "
              r"$1{,}254$ for D2R, $104$ for USP7, $731$ for the single-protein runs): effective rank after "
              r"standardising every column, as TabPFN receives them, for the random initialisation (init.), at the first "
              r"checkpoint, at its lowest point and at the end; the effective rank of the raw columns; and the mean "
              r"absolute correlation between features. Steps are optimisation steps; the epoch-level rows are "
              r"indexed by epoch. Ranks are comparable only within a molecule set.",
              r"l r l l l l r r r", header, rows, size=r"\scriptsize")


# --------------------------------------------------------------------------- #
# D.8 CKA                                                                     #
# --------------------------------------------------------------------------- #
def table_d9(doc: Doc) -> None:
    doc.heading("D.9 CKA of the short runs with their initialisation: linear / debiased")
    c = pd.read_csv(REPO_ROOT / EXP / CKA_DEBIASED_CSV)

    def pair(r) -> str:
        return rf"${r.cka_linear_mean:.3f}$ / ${r.cka_debiased_mean:.3f}$"
    rows = []
    for label, run in CKA_STATIC:
        g = c[(c.part == "static") & (c.run == run)].set_index("compared_to")
        moved = [(1 - g.loc["trained", f"cka_{e}_mean"]) / (1 - g.loc["other-init", f"cka_{e}_mean"])
                 for e in ("linear", "debiased")]
        rows.append(rf"{label} & {pair(g.loc['trained'])} & {pair(g.loc['other-init'])} & {pair(g.loc['chemeleon'])} & "
                    rf"{pair(g.loc['morgan'])} & ${moved[0]:.1f}$ / ${moved[1]:.1f}$ & ${int(g.loc['trained'].n_assays)}$ \\")
        print(f"  {run:13s} trained {g.loc['trained'].cka_linear_mean:.3f}/{g.loc['trained'].cka_debiased_mean:.3f}  "
              f"second init {g.loc['other-init'].cka_linear_mean:.3f}/{g.loc['other-init'].cka_debiased_mean:.3f}  "
              f"CheMeleon {g.loc['chemeleon'].cka_linear_mean:.3f}/{g.loc['chemeleon'].cka_debiased_mean:.3f}  "
              f"Morgan {g.loc['morgan'].cka_linear_mean:.3f}/{g.loc['morgan'].cka_debiased_mean:.3f}  "
              f"(1-CKA)/(1-CKA second init) {moved[0]:.1f}/{moved[1]:.1f}")
    shift = (c.cka_linear_mean - c.cka_debiased_mean)
    header = [r"\textbf{Encoder} & \textbf{Trained} & \textbf{Second init.} & \textbf{CheMeleon} & \textbf{Morgan} & "
              r"\textbf{Ratio} & \textbf{Assays} \\"]
    doc.table("tab:appd_cka",
              r"CKA between each short-run encoder, or an anchor, and the encoder's random initialisation, rebuilt "
              r"from the training seed; each value is given as linear / debiased estimator~\cite{song2012feature,"
              r"nguyen2021wide}, the mean over held-out assays with at least $25$ molecules. Ratio: the distance "
              r"$1 - \mathrm{CKA}$ of the trained encoder divided by that of a second random initialisation. "
              rf"Over this table and Table~\ref{{tab:appd_cka_long}}, the two estimators differ by at most ${shift.max():.2f}$.",
              r"l l l l l l r", header, rows, size=r"\scriptsize")

    doc.heading("D.10 CKA of the 100,000-step runs with their saved initialisation: linear / debiased")
    rows, corr = [], []
    for label, run in CKA_LONG:
        g = c[(c.part == "trajectory") & (c.run == run)]
        t = g[g.compared_to == "trained"].assign(step=lambda x: x.step.astype(float)).set_index("step")
        corr.append(t.cka_linear_mean.corr(t.cka_debiased_mean))
        lo_l, lo_d = t.cka_linear_mean.idxmin(), t.cka_debiased_mean.idxmin()
        che = g[g.compared_to == "chemeleon"].iloc[0]
        share = [(1 - t.loc[100_000, f"cka_{e}_mean"]) / (1 - che[f"cka_{e}_mean"]) for e in ("linear", "debiased")]
        rows.append(rf"{label} & " + " & ".join(pair(t.loc[s]) for s in CKA_STEPS)
                    + rf" & ${t.cka_linear_mean.min():.3f}$ ({lo_l / 1000:.0f}k) / ${t.cka_debiased_mean.min():.3f}$ "
                    rf"({lo_d / 1000:.0f}k) & {pair(che)} & ${100 * share[0]:.0f}\%$ / ${100 * share[1]:.0f}\%$ \\")
        print(f"  {run:13s} " + "  ".join(f"{int(s)}: {t.loc[s].cka_linear_mean:.3f}/{t.loc[s].cka_debiased_mean:.3f}" for s in CKA_STEPS)
              + f"  lowest {t.cka_linear_mean.min():.3f} ({lo_l:.0f}) / {t.cka_debiased_mean.min():.3f} ({lo_d:.0f})"
              f"  CheMeleon {che.cka_linear_mean:.3f}/{che.cka_debiased_mean:.3f}  share of CheMeleon's distance "
              f"{100 * share[0]:.0f} % / {100 * share[1]:.0f} %")
    print(f"  linear minus debiased: largest {shift.max():.3f}; trajectories correlate at {min(corr):.3f} or above")
    header = [r" & \multicolumn{3}{c}{\textbf{CKA at step}} & & & \\", r"\cmidrule(lr){2-4}",
              r"\textbf{Run} & \textbf{10k} & \textbf{40k} & \textbf{100k} & \textbf{Lowest (step)} & "
              r"\textbf{CheMeleon} & \textbf{Share} \\"]
    doc.table("tab:appd_cka_long",
              r"CKA between the checkpoints of the three $100{,}000$-step domain-matched runs and the saved "
              r"initialisation of each run, as linear / debiased estimator: at three steps, at its lowest point, "
              r"and for CheMeleon on the same assays. Share: the distance $1 - \mathrm{CKA}$ of the final encoder as "
              r"a fraction of CheMeleon's. Held-out assays: $20$ (D2R), $17$ (\emph{S.~aureus}), $2$ (USP7). The "
              rf"two estimators give trajectories that correlate at ${min(corr):.3f}$ or above.",
              r"l l l l l l l", header, rows, size=r"\scriptsize")


# --------------------------------------------------------------------------- #
# D.11 CKA of the epoch-level run                                             #
# --------------------------------------------------------------------------- #
def table_d10(doc: Doc) -> None:
    doc.heading("D.11 CKA of the epoch-level run with its own random initialisation: linear / debiased")
    c = pd.read_csv(REPO_ROOT / EXP / CKA_EPOCH_CSV)

    def pair(r) -> str:
        return rf"${r.cka_linear_mean:.3f}$ / ${r.cka_debiased_mean:.3f}$"
    cols, corr = [], []
    for label, run in CKA_EPOCH_RUNS:
        g = c[c.run == run]
        t = g[g.compared_to == "trained"].assign(step=lambda x: x.step.astype(int)).set_index("step")
        ref = g[g.compared_to != "trained"].set_index("compared_to")
        corr.append(t.cka_linear_mean.corr(t.cka_debiased_mean))
        last = t.loc[CKA_EPOCHS[-1]]
        share = [(1 - last[f"cka_{e}_mean"]) / (1 - ref.loc["chemeleon", f"cka_{e}_mean"]) for e in ("linear", "debiased")]
        times = [(1 - last[f"cka_{e}_mean"]) / (1 - ref.loc["other-init", f"cka_{e}_mean"]) for e in ("linear", "debiased")]
        lo_l, lo_d = t.cka_linear_mean.idxmin(), t.cka_debiased_mean.idxmin()
        cols.append([pair(t.loc[e]) for e in CKA_EPOCHS]
                    + [rf"${t.cka_linear_mean.min():.3f}$ ({lo_l}) / ${t.cka_debiased_mean.min():.3f}$ ({lo_d})",
                       pair(ref.loc["other-init"]), pair(ref.loc["chemeleon"]), pair(ref.loc["morgan"]),
                       rf"${100 * share[0]:.0f}\%$ / ${100 * share[1]:.0f}\%$", rf"${times[0]:.1f}$ / ${times[1]:.1f}$"])
        print(f"  {run:19s} " + "  ".join(f"{e}: {t.loc[e].cka_linear_mean:.3f}/{t.loc[e].cka_debiased_mean:.3f}" for e in CKA_EPOCHS)
              + f"  lowest {t.cka_linear_mean.min():.3f} (epoch {lo_l}) / {t.cka_debiased_mean.min():.3f} (epoch {lo_d})  change of seed "
              f"{ref.loc['other-init'].cka_linear_mean:.3f}/{ref.loc['other-init'].cka_debiased_mean:.3f}  CheMeleon "
              f"{ref.loc['chemeleon'].cka_linear_mean:.3f}/{ref.loc['chemeleon'].cka_debiased_mean:.3f}  Morgan "
              f"{ref.loc['morgan'].cka_linear_mean:.3f}/{ref.loc['morgan'].cka_debiased_mean:.3f}  share of CheMeleon's "
              f"distance {100 * share[0]:.0f} % / {100 * share[1]:.0f} %  times a change of seed {times[0]:.1f} / {times[1]:.1f}"
              f"  ({int(last.n_assays)} assays)")
    shift = c.cka_linear_mean - c.cka_debiased_mean
    print(f"  linear minus debiased: largest {shift.max():.3f}; trajectories correlate at {min(corr):.3f} or above")
    names = ([rf"Encoder at epoch ${e}$" for e in CKA_EPOCHS]
             + ["Encoder, lowest value (epoch)", "Random init. of the other seed", "CheMeleon",
                "Morgan fingerprints", r"Share of CheMeleon's distance, epoch $26$", r"Multiple of a change of seed, epoch $26$"])
    rows = []
    for i, name in enumerate(names):
        if i in (6, 9):
            rows.append(r"\addlinespace[3pt]")
        rows.append(rf"{name} & " + " & ".join(col[i] for col in cols) + r" \\")
    header = [r"\textbf{CKA with the random initialisation} & " + " & ".join(rf"\textbf{{{label}}}" for label, _ in CKA_EPOCH_RUNS) + r" \\"]
    doc.table("tab:appd_cka_epoch",
              r"CKA between the epoch-level encoder, or an anchor, and the run's own random initialisation (its saved "
              r"epoch~$0$), as linear / debiased estimator, the mean over $20$ held-out assays of the single-protein "
              r"corpus with at least $25$ molecules. Distance is $1 - \mathrm{CKA}$. The two estimators differ by at "
              rf"most ${shift.max():.2f}$ and give trajectories that correlate at ${min(corr):.3f}$ or above.",
              r"l l l", header, rows)


# --------------------------------------------------------------------------- #
# D.12 The encoder of Chapter 5 among the representations of Chapter 4        #
# --------------------------------------------------------------------------- #
def table_d11(doc: Doc) -> None:
    doc.heading("D.12 Encoder of Chapter 5 (epoch 26, mean of the two seeds) minus each representation of Chapter 4, "
                "per surrogate (mean ± SE over scenarios, scenarios won)")
    rows = []
    for label, rep in CHAPTER4_REPRESENTATIONS:
        # (encoder - its own control, mean of the seeds) - (representation - first-seed control)
        d = comparison((MEAN26, None), (rep, CONTROL))
        cells = [scen_stat(scenario_means(d, model=m)) for m, _ in MODELS]
        print(f"    {label:22s} " + "  ".join(f"{txt(c):>20s}" for c in cells))
        rows.append(rf"{label} & " + " & ".join(tex(c) for c in cells) + r" \\[3pt]")
    header = [r"\textbf{Encoder minus} & " + " & ".join(rf"\textbf{{{n}}}" for _, n in MODELS) + r" \\"]
    doc.table("tab:appd_encoder_vs_reps",
              r"The epoch-level encoder at its last epoch, the mean of the two seeds, minus each representation of "
              r"Chapter~\ref{ch:results_al}, in final CEF, per surrogate. Each seed's encoder is first taken minus the "
              r"random initialisation of its own seed and each representation minus that of the first seed, so the last "
              r"row repeats Table~\ref{tab:appd_transfer_surrogates}. The value of a scenario is the mean of the five "
              r"matched pairs of that surrogate; mean $\pm$ standard error over the $16$ scenarios, scenarios won "
              r"below; $^{*}$more than two standard errors. The RBF Gaussian process fails on Mordred descriptors "
              r"(Section~\ref{subsec:mordred}), so no value over the three surrogates is given.",
              r"l l l l", header, rows)


def main() -> None:
    doc = Doc()
    table_d1(doc)
    table_d2(doc)
    table_d3(doc)
    table_d4(doc)
    table_d5(doc)
    table_d6(doc)
    table_d7(doc)
    table_d8(doc)
    table_d9(doc)
    table_d10(doc)
    table_d11(doc)
    out = REPO_ROOT / OUT_TEX
    out.write_text("% Written by src/scripts/appendix_d_tables.py; do not edit by hand.\n"
                   r"\providecommand{\appcell}[2]{\begin{tabular}[t]{@{}l@{}}$#1$\\[-1pt]{\scriptsize (#2)}\end{tabular}}"
                   "\n\n" + "\n".join(doc.tex))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
