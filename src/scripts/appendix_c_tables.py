"""scripts/appendix_c_tables.py — every value of Appendix C (detailed results of Chapter 4). Press Run.

Prints the tables of Appendix C and writes them as LaTeX to OUT_TEX, ready to copy into the thesis as
Content/Appendix_C_tables.tex. Source: the final run only, experiments/benchmark_final/final_*.csv
(benchmark_extended.py, experiments "final" and "final_subsampling"), one row per seed and round, plus
experiments/calibration_study.csv for the calibration table.

Units, those of plot_benchmark.py (the figures of Chapter 4):
  - an AL experiment is one arm in one budget configuration on one target: its final-round CEF averaged
    over the five seeds; only complete experiments (five seeds) are used;
  - the main protocol (1,000 candidates per round) and the five representations not trained through
    TabPFN, with each surrogate in its native predictive form (TabPFN: the full distribution; RF, RBF GP and
    Tanimoto GP: Gaussian). "All sixteen" in a caption means every representation of the run;
  - a pair is two AL experiments that differ in one thing only (the surrogate, the criterion or the
    representation) and agree in target, configuration and everything else; a difference is "established"
    when its mean exceeds two standard errors, and the count beneath it is the pairs in which the first
    option ended higher;
  - Table C.3(c), the Mordred diagnosis, needs molecular embeddings and Gaussian-process fits; its values
    are printed by gp_mordred_geometry.py and copied into MORDRED_DIAGNOSIS below.

Run with:  uv run python src/scripts/appendix_c_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
FINAL_DIR = "experiments/benchmark_final"
CALIBRATION_CSV = "experiments/calibration_study.csv"
OUT_TEX = "experiments/appendix_c_tables.tex"

TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
TARGET_TEX = {"saureus": r"\emph{S.~aureus}", "tyk2": "TYK2", "usp7": "USP7", "d2r": "D2R"}
N_SEEDS = 5
N_ROUNDS = 20
CANDIDATES = "1000"                       # the main protocol; "all" is the full-pool control
MAIN_SURROGATES = ["tabpfn", "rf", "gp"]
NATIVE_DIST = {"tabpfn": "full", "rf": "gauss", "gp": "gauss", "gp_tanimoto": "gauss"}
SURROGATE_TEX = {"tabpfn": "TabPFN", "rf": "RF", "gp": "RBF GP", "gp_tanimoto": "Tanimoto GP"}
FIVE_REPRESENTATIONS = ["chemeleon", "mordred3d", "chemprop_untrained", "morgan", "physchem"]
REPRESENTATION_TEX = {"chemeleon": "CheMeleon", "mordred3d": "Mordred", "chemprop_untrained": "Random init.",
                      "morgan": "Morgan", "physchem": "Physicochemical"}
CRITERIA = ["greedy", "ei_yplus", "pi_yplus", "ei_ythresh", "pi_ythresh"]
CRITERION_TEX = {"greedy": "greedy", "ei_yplus": r"EI($y^{+}$)", "pi_yplus": r"PI($y^{+}$)",
                 "ei_ythresh": r"EI($y_{\text{thresh}}$)", "pi_ythresh": r"PI($y_{\text{thresh}}$)"}
# The sixteen representations of the run, grouped as in Table 3.4 (the domain-matched encoders exist per
# target, so one row pools the three targets that have one).
ALL_REPRESENTATIONS_TEX = [
    ("Morgan", ["morgan"]), ("Physicochemical", ["physchem"]), ("Mordred", ["mordred3d"]),
    ("CheMeleon", ["chemeleon"]), ("Random init.", ["chemprop_untrained"]),
    ("All of ChEMBL", ["chemprop_trained_full"]),
    ("Domain-matched, standard length", [f"chemprop_trained_own_{t}" for t in ("saureus", "usp7", "d2r")]),
    ("Domain-matched, $100$k steps", [f"chemprop_trained_own_{t}_long" for t in ("saureus", "usp7", "d2r")]),
    ("Single-protein, $100$k steps, final", ["chemprop_trained_full_sp"]),
    ("Single-protein, $100$k steps, best", ["chemprop_trained_full_sp_best"]),
    ("Single-protein, $10$ epochs, final", ["chemprop_trained_full_sp_10ep"]),
    ("Single-protein, $10$ epochs, best", ["chemprop_trained_full_sp_10ep_best"]),
    ("Epoch-level, epoch $23$", ["chemprop_trained_full_sp_50ep_best"]),
    ("Epoch-level, epoch $26$", ["chemprop_trained_full_sp_50ep_final"]),
    ("Epoch-level, second seed, epoch $21$", ["chemprop_trained_full_sp_50ep_seed1_best"]),
    ("Epoch-level, second seed, epoch $26$", ["chemprop_trained_full_sp_50ep_seed1_final"]),
]
COVERAGE_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
ROUND_BINS = [(1, 5), (6, 10), (11, 15), (16, 20)]

# Printed by gp_mordred_geometry.py on 2026-10-19 (D2R pool, 100 labelled molecules, RBF GP):
# width, effective rank, median distance, CV of distances, p99/p1 of distances, fitted length scale,
# fitted signal variance, mean off-diagonal kernel value, SD of predicted mean / SD of activity.
MORDRED_DIAGNOSIS = [
    ("morgan", 2048, 559.4, "9.17", 0.100, 1.75, "17.1", 29.6, 0.87, 0.799),
    ("physchem", 200, 96.5, "5.53", 0.154, 2.43, "4.58", 8.93, 0.492, 0.770),
    ("mordred3d", 1826, 5.4, r"4.06 \times 10^{4}", 3.480, 57.80, "10", 50.3, 0.0, 0.349),
    ("chemeleon", 2048, 368.5, "2.56", 0.289, 4.98, "3.25", 16.6, 0.693, 0.836),
    ("chemprop_untrained", 300, 57.2, "4.12", 0.489, 8.59, "3.27", 16.7, 0.42, 0.943),
]
MORDRED_STANDARDISED = {"median_distance": 47.9, "length_scale": 55.2, "spearman_raw": -0.081,
                        "spearman_standardised": 0.283}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def stat(x) -> tuple[float, float, int, int] | None:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return None
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), int((x > 0).sum()), len(x)


def txt(s) -> str:
    if s is None:
        return "—"
    m, se, won, n = s
    return f"{m:+.2f} ± {se:.2f}{'*' if abs(m) > 2 * se else ' '}({won}/{n})"


def tex(s) -> str:
    if s is None:
        return "---"
    m, se, won, n = s
    star = "^{*}" if abs(m) > 2 * se else ""
    return rf"\appcell{{{m:+.2f} \pm {se:.2f}{star}}}{{{won}/{n}}}"


def tex3(s) -> str:
    """Three decimals, for differences in activity units (they are an order of magnitude smaller)."""
    if s is None:
        return "---"
    m, se, won, n = s
    star = "^{*}" if abs(m) > 2 * se else ""
    return rf"\appcell{{{m:+.3f} \pm {se:.3f}{star}}}{{{won}/{n}}}"


def per_target(d: pd.Series) -> list:
    """Stats for each target and for all targets; d is indexed with a 'target' level."""
    present = set(d.index.get_level_values("target"))
    return [stat(d.xs(t, level="target")) if t in present else None for t in TARGETS] + [stat(d)]


def without_usp7(d: pd.Series) -> pd.Series:
    return d[d.index.get_level_values("target") != "usp7"]


def subset(x, reps=None, models=None):
    keep = np.ones(len(x), dtype=bool)
    if reps is not None:
        keep &= x.index.get_level_values("representation").isin(reps)
    if models is not None and "model" in x.index.names:
        keep &= x.index.get_level_values("model").isin(models)
    return x[keep]


class Doc:
    """Collects the printed report and the LaTeX tables side by side."""

    def __init__(self) -> None:
        self.tex: list[str] = []

    def heading(self, title: str) -> None:
        print(f"\n=== {title}")

    def row(self, label: str, cells: list) -> None:
        print(f"  {label:50s} " + "  ".join(f"{txt(c):>24s}" for c in cells))

    def table(self, label: str, caption: str, colspec: str, header: str, rows: list[str],
              size: str = r"\footnotesize") -> None:
        self.tex += [r"\begin{table}[!htbp]", r"\centering", size, r"\setlength{\tabcolsep}{2pt}",
                     rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                     rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} {colspec} @{{}}}}",
                     r"\toprule", header + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}",
                     r"\end{table}", ""]


TARGET_HEADER = " & ".join([r"\textbf{Comparison}"] + [rf"\textbf{{{TARGET_TEX[t]}}}" for t in TARGETS]
                           + [r"\textbf{All}"])


# Indented row label under a group heading. The kern keeps the label on the first line (a \quad lets TeX break
# right after it, dropping a long label to the second line) and the hanging indent keeps wrapped lines indented.
SUB = r"\hangindent=0.5em\hangafter=1\leavevmode\kern0.5em "   # 0.5em: "Physicochemical" must fit a 2.7cm label column
def tex_row(label: str, cells: list, fmt=tex) -> str:
    return label + " & " + " & ".join(fmt(c) for c in cells) + r" \\[3pt]"


def group(title: str, ncols: int) -> list[str]:
    """A group heading spanning the table; the rows below it are indented."""
    return [rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\", r"\addlinespace[2pt]"]


LABEL_COL = "L{2.7cm}"           # wrapping label column for tables with five value columns


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def read_runs() -> pd.DataFrame:
    files = sorted((REPO_ROOT / FINAL_DIR).glob("final_*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"candidates": str, "tau": str, "dist": str}) for f in files],
                   ignore_index=True)
    df["criterion"] = np.where(df.acquisition == "greedy", "greedy", df.acquisition + "_" + df.tau)
    print(f"read {len(files)} files, {len(df):,} rows")
    return df


EXPERIMENT_KEY = ["target", "representation", "model", "init_size", "batch_size"]


def experiments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per AL experiment: final-round means over the five seeds (complete ones only)."""
    f = df[df["round"] == N_ROUNDS]
    g = f.groupby(EXPERIMENT_KEY + ["criterion", "dist", "candidates"]).agg(
        cef=("cef", "mean"), best_y=("best_y", "mean"), mean_y=("mean_y_acquired", "mean"),
        n=("seed", "nunique")).reset_index()
    return g[g.n == N_SEEDS]


def native(e: pd.DataFrame) -> pd.DataFrame:
    """Greedy plus each surrogate's EI and PI in its native predictive form."""
    return e[(e.criterion == "greedy") | (e.dist == e.model.map(NATIVE_DIST))]


def wide(e: pd.DataFrame, value: str = "cef") -> pd.DataFrame:
    """Criterion columns, indexed by AL-experiment key (main protocol)."""
    return e[e.candidates == CANDIDATES].pivot_table(index=EXPERIMENT_KEY, columns="criterion", values=value)


def by_model(e: pd.DataFrame, value: str = "cef") -> pd.DataFrame:
    """Surrogate columns, indexed by target, representation, criterion and configuration."""
    return e[e.candidates == CANDIDATES].pivot_table(
        index=["target", "representation", "criterion", "init_size", "batch_size"], columns="model", values=value)


def round_level(df: pd.DataFrame) -> pd.DataFrame:
    """CEF per round (mean over seeds) for TabPFN, RF and RBF GP: the main protocol, five representations."""
    f = df[(df.candidates == CANDIDATES) & df.representation.isin(FIVE_REPRESENTATIONS)
           & df.model.isin(MAIN_SURROGATES)]
    f = f[(f.criterion == "greedy") | (f.dist == f.model.map(NATIVE_DIST))]
    g = f.groupby(["target", "representation", "criterion", "init_size", "batch_size", "round", "model"]).agg(
        cef=("cef", "mean"), n=("seed", "nunique")).reset_index()
    assert (g.n == N_SEEDS).all(), "incomplete experiments in the round-level data"
    return g.pivot_table(index=["target", "representation", "criterion", "init_size", "batch_size", "round"],
                         columns="model", values="cef")


# --------------------------------------------------------------------------- #
# Tables                                                                      #
# --------------------------------------------------------------------------- #
def table_surrogates(doc: Doc, B5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_surrogates — TabPFN minus the RF and the RBF GP, final CEF")
    out = []
    for alt in ("rf", "gp"):
        out += group(f"TabPFN minus {SURROGATE_TEX[alt]}", 6)
        d = B5.tabpfn - B5[alt]
        rows = [("all criteria", d)] + [(CRITERION_TEX[c], d.xs(c, level="criterion", drop_level=False))
                                        for c in CRITERIA]
        for label, x in rows:
            cells = per_target(x)
            doc.row(f"vs {SURROGATE_TEX[alt]}: {label}", cells)
            out.append(tex_row(SUB + label, cells))
    weakest = []
    for t in TARGETS:
        q = B5.xs(t, level="target")
        weakest.append(f"{TARGET_TEX[t]} {int(((q.tabpfn < q.rf) & (q.tabpfn < q.gp)).sum())}/{len(q)}")
    note = ", ".join(weakest)
    print("  TabPFN strictly the weakest of the three surrogates:", note)
    doc.table("tab:app_surrogates",
              "TabPFN minus the random forest (RF) and the RBF Gaussian process in final CEF: mean paired "
              r"difference $\pm$ standard error and, beneath it, the pairs in which TabPFN ended higher; "
              r"$^{*}$ marks an established difference. Five representations not trained through TabPFN, four "
              "configurations, each surrogate in its native predictive form; ``all criteria'' pools greedy "
              r"selection and the four improvement criteria. TabPFN is the weakest of the three surrogates in "
              rf"{note} of the AL experiments of a target and criterion.",
              f"{LABEL_COL} l l l l l", TARGET_HEADER, out)


def table_representation_pairs(doc: Doc, B5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_representation_pairs — TabPFN minus RF / RBF GP per representation")
    cols = [("all criteria", None), ("greedy", "greedy"), (CRITERION_TEX["pi_ythresh"], "pi_ythresh")]
    rows = []
    for r in FIVE_REPRESENTATIONS:
        q = B5.xs(r, level="representation", drop_level=False)
        cells = []
        for alt in ("rf", "gp"):
            d = q.tabpfn - q[alt]
            for _, c in cols:
                cells.append(stat(d if c is None else d.xs(c, level="criterion")))
        doc.row(REPRESENTATION_TEX[r], cells)
        rows.append(tex_row(REPRESENTATION_TEX[r], cells))
    nm = B5[B5.index.get_level_values("representation") != "mordred3d"]
    s_all, s_gr = stat(nm.tabpfn - nm.gp), stat(nm.xs("greedy", level="criterion").pipe(lambda z: z.tabpfn - z.gp))
    print("  TabPFN − RBF GP without Mordred: all criteria", txt(s_all), "| greedy", txt(s_gr))
    header = (r"& \multicolumn{3}{c}{\textbf{TabPFN minus RF}} & \multicolumn{3}{c}{\textbf{TabPFN minus RBF GP}} \\"
              "\n" r"\cmidrule(lr){2-4} \cmidrule(l){5-7}" "\n"
              r"\textbf{Representation} & " + " & ".join(rf"\textbf{{{n.capitalize() if n[0].islower() else n}}}"
                                                          for _ in ("rf", "gp") for n, _ in cols))
    doc.table("tab:app_representation_pairs",
              "TabPFN minus the random forest and the RBF Gaussian process per representation, pooled over the "
              r"four targets and four configurations ($16$ pairs per criterion, $80$ for all criteria). Without "
              r"Mordred, TabPFN minus the RBF Gaussian process is "
              rf"${s_all[0]:+.2f} \pm {s_all[1]:.2f}$ (${s_all[2]}/{s_all[3]}$) over all criteria and "
              rf"${s_gr[0]:+.2f} \pm {s_gr[1]:.2f}$ (${s_gr[2]}/{s_gr[3]}$) under greedy selection.",
              "l l l l l l l", header, rows, size=r"\scriptsize")


def table_init(doc: Doc, B5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_init — TabPFN − RF margin by initial-set size")
    rf = (B5.tabpfn - B5.rf).unstack("init_size")
    rows = []
    for label, t in [(TARGET_TEX[t], t) for t in TARGETS] + [("All", None)]:
        q = rf if t is None else rf.xs(t, level="target")
        cells = [stat(q[10]), stat(q[100]), stat(q[100] - q[10])]
        doc.row(label, cells)
        rows.append(tex_row(label, cells))
    doc.table("tab:app_init",
              r"TabPFN's margin over the random forest in final CEF with $N_{\text{init}} = 10$ and $100$, and "
              r"the paired difference of the two margins (pairs over representation, criterion and batch size).",
              "l l l l", r"\textbf{Target} & $\boldsymbol{N_{\text{init}} = 10}$ & "
              r"$\boldsymbol{N_{\text{init}} = 100}$ & \textbf{Difference}", rows)


def table_budget_and_rounds(doc: Doc, L: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_rounds_batches — margin by round, batch size fixed")
    d = (L.tabpfn - L.rf).reset_index(name="d")
    rows_b = []
    for lo, hi in ROUND_BINS:
        x = d[(d["round"] >= lo) & (d["round"] <= hi)].d
        print(f"  rounds {lo:>2}-{hi:<2} {x.mean():+.2f}  ahead {(x > 0).mean():.0%}  (n = {len(x)})")
        rows_b.append(rf"Rounds {lo}--{hi} & ${x.mean():+.2f}$ & {(x > 0).mean() * 100:.0f}\% & "
                      rf"{len(x):,}".replace(",", "{,}"))

    doc.heading("TABLE tab:app_matched_budget — matched budget")
    rows_c, diffs, per_model = [], [], {m: [] for m in MAIN_SURROGATES}
    for t in TARGETS:
        small, large = sorted(L.xs(t, level="target").index.get_level_values("batch_size").unique())
        r_large = N_ROUNDS * small // large
        assert N_ROUNDS * small % large == 0, (t, small, large)
        for init in sorted(L.index.get_level_values("init_size").unique()):
            lv = ["target", "init_size", "batch_size", "round"]
            sa = L.xs((t, init, small, N_ROUNDS), level=lv)
            la = L.xs((t, init, large, r_large), level=lv)
            diff = (sa.tabpfn - sa.rf) - (la.tabpfn - la.rf)
            diffs.append(diff)
            for m in per_model:
                per_model[m].append(sa[m] - la[m])
            budget = init + N_ROUNDS * small
            gs, gl = (sa.tabpfn - sa.rf).mean(), (la.tabpfn - la.rf).mean()
            s = stat(diff)
            print(f"  {t:8s} N_init {init:>3} budget {budget:>4}: K {small} (r{N_ROUNDS}) {gs:+.2f} vs K {large} "
                  f"(r{r_large}) {gl:+.2f}; difference {txt(s)}")
            rows_c.append(rf"{TARGET_TEX[t]} & {init} & {budget:,} & {small} ({N_ROUNDS}) vs {large} ({r_large}) & "
                          .replace(",", "{,}") + rf"${gs:+.2f}$ & ${gl:+.2f}$ & {tex(s)} \\[3pt]")
    s_all = stat(pd.concat(diffs))
    print("  all", txt(s_all))
    rows_c += [r"\midrule", rf"All & & & & & & {tex(s_all)} \\[3pt]"]
    doc.table("tab:app_matched_budget",
              "TabPFN minus the random forest at a matched budget. The smaller batch is read after round $20$, "
              "the larger batch after the round (in brackets) at which it has measured the same number of "
              "molecules. Margins are means over representations and criteria; the last column is their paired "
              "difference, smaller minus larger batch, with the pairs in which the smaller batch gave TabPFN "
              "the larger margin.",
              "l r r l r r l", r"\textbf{Target} & $\boldsymbol{N_{\text{init}}}$ & \textbf{Budget} & "
              r"\textbf{Batch (round)} & \textbf{Smaller} & \textbf{Larger} & \textbf{Difference}", rows_c)

    doc.heading("per surrogate, smaller minus larger batch at a matched budget")
    rows_d = []
    for m in MAIN_SURROGATES:
        s = stat(pd.concat(per_model[m]))
        doc.row(SURROGATE_TEX[m], [s])
        rows_d.append(rf"{SURROGATE_TEX[m]} & {tex(s)}")
    pad = rows_d + [" & "] * (len(rows_b) - len(rows_d))
    doc.table("tab:app_rounds_batches",
              "Left: TabPFN's margin over the random forest by round, with the batch size fixed, pooled over all "
              "AL experiments, and the share of pairs in which TabPFN is ahead. Right: final CEF with the "
              "smaller minus the larger batch at a matched budget, per surrogate (pairs over target, "
              r"$N_{\text{init}}$, representation and criterion).",
              r"l r r r @{\hspace{2.5em}} l l",
              r"\textbf{Rounds} & \textbf{Margin} & \textbf{Ahead} & \textbf{Pairs} & \textbf{Surrogate} & "
              r"\textbf{Smaller $-$ larger}",
              [b + " & " + p + r" \\[3pt]" for b, p in zip(rows_b, pad)])


def table_criteria(doc: Doc, W5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_criteria_surrogate — criterion minus greedy, per surrogate")
    out = []
    for c in CRITERIA[1:]:
        out += group(f"{CRITERION_TEX[c]} minus greedy selection", 6)
        d = (W5[c] - W5.greedy).dropna()
        for m in MAIN_SURROGATES:
            cells = per_target(d.xs(m, level="model", drop_level=False))
            doc.row(f"{c}: {SURROGATE_TEX[m]}", cells)
            out.append(tex_row(SUB + SURROGATE_TEX[m], cells))
        cells = per_target(d)
        doc.row(f"{c}: all three", cells)
        out.append(tex_row(SUB + r"All three", cells))
    doc.table("tab:app_criteria_surrogate",
              "Each improvement criterion minus greedy selection in final CEF, per surrogate and target (pairs "
              "over representation and configuration; each surrogate in its native predictive form), and "
              "pooled over the three surrogates. The Tanimoto-kernel Gaussian process is in "
              r"Table~\ref{tab:app_tanimoto}.",
              f"{LABEL_COL} l l l l l", TARGET_HEADER.replace("Comparison", "Surrogate"), out)


def table_threshold(doc: Doc, W5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_threshold — y_thresh minus y+, and by representation")
    out = group(r"Criterion at $y_{\text{thresh}}$ minus the same criterion at $y^{+}$", 6)
    for m in MAIN_SURROGATES:
        for a in ("ei", "pi"):
            d = (W5[f"{a}_ythresh"] - W5[f"{a}_yplus"]).xs(m, level="model", drop_level=False).dropna()
            cells = per_target(d)
            doc.row(f"{a.upper()}: {SURROGATE_TEX[m]}", cells)
            out.append(tex_row(SUB + rf"{a.upper()}, {SURROGATE_TEX[m]}", cells))
    for c in ("pi_ythresh", "ei_ythresh"):
        out += group(rf"{CRITERION_TEX[c]} minus greedy selection, by representation (three surrogates)", 6)
        d = (W5[c] - W5.greedy).dropna()
        for r in FIVE_REPRESENTATIONS:
            cells = per_target(d.xs(r, level="representation", drop_level=False))
            doc.row(f"{c}: {REPRESENTATION_TEX[r]}", cells)
            out.append(tex_row(SUB + REPRESENTATION_TEX[r], cells))
    doc.table("tab:app_threshold",
              r"Upper part: each criterion anchored at the hit threshold minus the same criterion anchored at the "
              r"best value so far, per surrogate. Lower parts: PI($y_{\text{thresh}}$) and "
              r"EI($y_{\text{thresh}}$) minus greedy selection, per representation, pooled over the three "
              "surrogates. Final CEF; pairs over representation and configuration (over configuration and "
              "surrogate in the lower parts).",
              f"{LABEL_COL} l l l l l", TARGET_HEADER, out)


def table_full_distribution(doc: Doc, E1k: pd.DataFrame, W5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_full_distribution — TabPFN, full distribution minus Gaussian form")
    q = E1k[(E1k.model == "tabpfn") & (E1k.criterion != "greedy")]
    wa = q.pivot_table(index=EXPERIMENT_KEY + ["criterion"], columns="dist", values="cef")
    d_all = wa.full - wa.gauss
    out = []
    for title, d in [("Five representations not trained through TabPFN", subset(d_all, FIVE_REPRESENTATIONS)),
                     ("All sixteen representations", d_all)]:
        out += group(f"Full distribution minus Gaussian form: {title[0].lower() + title[1:]}", 6)
        for c in CRITERIA[1:]:
            cells = per_target(d.xs(c, level="criterion", drop_level=False).dropna())
            doc.row(f"{title[:12]}: {c}", cells)
            out.append(tex_row(SUB + CRITERION_TEX[c], cells))
    out += group(r"TabPFN minus RF, both under PI($y_{\text{thresh}}$)", 6)
    t = E1k[(E1k.model == "tabpfn") & (E1k.criterion == "pi_ythresh")].pivot_table(
        index=["target", "representation", "init_size", "batch_size"], columns="dist", values="cef")
    rf = E1k[(E1k.model == "rf") & (E1k.criterion == "pi_ythresh") & (E1k.dist == "gauss")].set_index(
        ["target", "representation", "init_size", "batch_size"]).cef
    for label, x in [("full form", t.full), ("Gaussian form", t.gauss)]:
        d = subset(x - rf, FIVE_REPRESENTATIONS).dropna()
        cells = per_target(d)
        doc.row(f"TabPFN − RF, {label}", cells)
        out.append(tex_row(SUB + label, cells))
    doc.table("tab:app_full_distribution",
              "Upper parts: TabPFN's criteria computed from its full predictive distribution minus the same "
              "criteria from its Gaussian approximation (mean and half-width of the central $68\\%$ interval), in "
              "final CEF; pairs over representation and configuration. Lower part: TabPFN minus the random "
              r"forest, both under PI($y_{\text{thresh}}$) (the random forest in its Gaussian form), on the five "
              "representations.",
              f"{LABEL_COL} l l l l l", TARGET_HEADER, out)


def table_means(doc: Doc, E1k: pd.DataFrame, W5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_tabpfn_means — mean final CEF")
    t = E1k[(E1k.model == "tabpfn") & E1k.representation.isin(FIVE_REPRESENTATIONS)]
    full = t[(t.criterion == "greedy") | (t.dist == "full")].pivot_table(
        index="representation", columns="criterion", values="cef")[CRITERIA]
    gauss = t[t.dist == "gauss"].pivot_table(index="representation", columns="criterion", values="cef")
    cols = [(c, CRITERION_TEX[c]) for c in CRITERIA] + [("pi_ythresh_gauss", r"PI($y_{\text{thresh}}$), Gaussian")]
    rows = group("TabPFN, by representation (full distribution)", 7)
    for r in FIVE_REPRESENTATIONS:
        vals = [full.loc[r, c] for c in CRITERIA] + [gauss.loc[r, "pi_ythresh"]]
        print(f"  {r:20s} " + "  ".join(f"{v:.2f}" for v in vals))
        rows.append(SUB + REPRESENTATION_TEX[r] + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    rows += group("Mean over the five representations, by surrogate (native forms)", 7)
    mean = W5.groupby(level="model")[CRITERIA].mean()
    for m in ("tabpfn", "rf", "gp", "gp_tanimoto"):
        vals = [mean.loc[m, c] for c in CRITERIA]
        extra = f"{gauss['pi_ythresh'].mean():.2f}" if m == "tabpfn" else "---"
        print(f"  {m:20s} " + "  ".join(f"{v:.2f}" for v in vals))
        rows.append(SUB + SURROGATE_TEX[m] + " & " + " & ".join(f"{v:.2f}" for v in vals) + f" & {extra} \\\\")
    doc.table("tab:app_tabpfn_means",
              "Mean final CEF per criterion, extended over surrogates: TabPFN by representation, and every "
              "surrogate averaged over the five representations (all targets and configurations). The Gaussian "
              r"column is TabPFN's PI($y_{\text{thresh}}$) computed from its Gaussian approximation. The "
              "Tanimoto-kernel Gaussian process is the one of Table~\\ref{tab:app_tanimoto}.",
              "l r r r r r r", r"\textbf{Representation} & " + " & ".join(rf"\textbf{{{n}}}" for _, n in cols), rows)


def table_without_usp7(doc: Doc, W5: pd.DataFrame, B5: pd.DataFrame, E1k: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_without_usp7 — main comparisons with and without USP7")
    prf, pgp = B5.tabpfn - B5.rf, B5.tabpfn - B5.gp
    cr = {c: (W5[c] - W5.greedy).dropna() for c in CRITERIA[1:]}
    tq = E1k[(E1k.model == "tabpfn") & (E1k.criterion == "pi_ythresh")].pivot_table(
        index=["target", "representation", "init_size", "batch_size"], columns="dist", values="cef")
    gr = E1k[(E1k.model == "tabpfn") & (E1k.criterion == "greedy")].set_index(
        ["target", "representation", "init_size", "batch_size"]).cef
    tq, gr = subset(tq, FIVE_REPRESENTATIONS), subset(gr, FIVE_REPRESENTATIONS)
    xs = lambda s, c: s.xs(c, level="criterion", drop_level=False)  # noqa: E731
    groups = [
        ("Surrogates", [
            (r"TabPFN $-$ RF, all criteria", prf),
            (r"TabPFN $-$ RF, greedy", xs(prf, "greedy")),
            (r"TabPFN $-$ RF, PI($y_{\text{thresh}}$)", xs(prf, "pi_ythresh")),
            (r"TabPFN $-$ RF, EI($y^{+}$)", xs(prf, "ei_yplus")),
            (r"TabPFN $-$ RBF GP, all criteria", pgp),
            (r"TabPFN $-$ RBF GP, greedy", xs(pgp, "greedy")),
        ]),
        ("Criteria, three surrogates", [
            (r"EI($y^{+}$) $-$ greedy", cr["ei_yplus"]),
            (r"EI($y^{+}$) $-$ greedy, RF", cr["ei_yplus"].xs("rf", level="model", drop_level=False)),
            (r"EI($y_{\text{thresh}}$) $-$ greedy", cr["ei_ythresh"]),
            (r"PI($y_{\text{thresh}}$) $-$ greedy", cr["pi_ythresh"]),
        ]),
        ("TabPFN", [
            (r"full $-$ Gaussian PI($y_{\text{thresh}}$)", tq.full - tq.gauss),
            (r"full PI($y_{\text{thresh}}$) $-$ greedy", tq.full - gr),
            (r"Gaussian PI($y_{\text{thresh}}$) $-$ greedy", tq.gauss - gr),
        ]),
    ]
    rows = []
    for title, items in groups:
        rows += group(title, 3)
        for label, d in items:
            cells = [stat(d), stat(without_usp7(d))]
            doc.row(label.replace("$-$", "−"), cells)
            rows.append(tex_row(SUB + label, cells))
    doc.table("tab:app_without_usp7",
              r"The main comparisons of Chapter~\ref{ch:results_al} over all four targets and without USP7, "
              "in final CEF (same pairing as in the chapter).",
              "l l l", r"\textbf{Comparison} & \textbf{All targets} & \textbf{Without USP7}", rows)


def table_activity(doc: Doc, df_e: pd.DataFrame, W5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_activity — mean activity of the acquired molecules and best activity found")
    nat = native(df_e)
    out = []
    for value, title in [("mean_y", "Mean activity of the acquired molecules"), ("best_y", "Best activity found")]:
        Wv = subset(wide(nat, value), FIVE_REPRESENTATIONS, MAIN_SURROGATES)
        Wc = subset(W5, models=MAIN_SURROGATES)
        out += group(f"{title}, criterion minus greedy selection (activity units)", 6)
        for c in CRITERIA[1:]:
            dy, dc = (Wv[c] - Wv.greedy).dropna(), (Wc[c] - Wc.greedy).dropna()
            agree = (np.sign(dy) == np.sign(dc.reindex(dy.index))).mean()
            cells = [stat(dy.xs(t, level="target")) for t in TARGETS]
            doc.row(f"{value}: {c}  (sign agrees with CEF in {agree:.0%})", cells + [stat(dy)])
            out.append(SUB + CRITERION_TEX[c] + " & " + " & ".join(tex3(x) for x in cells)
                       + rf" & {agree * 100:.0f}\% \\[3pt]")
    header = (r"\textbf{Criterion} & " + " & ".join(rf"\textbf{{{TARGET_TEX[t]}}}" for t in TARGETS)
              + r" & \textbf{Agree}")
    doc.table("tab:app_activity",
              "Mean measured activity of the acquired molecules and best activity found (the highest activity "
              "among the measured molecules), each criterion minus greedy selection, in the units of each "
              "pool's activity scale and therefore not averaged across targets. Mean $\\pm$ standard error over "
              "$60$ matched pairs per target (three surrogates, five representations, four configurations) "
              "and, beneath it, the pairs in which the criterion acquired the more potent molecules; "
              "$^{*}$ marks an established difference. The last column gives the share of all $240$ pairs in "
              "which the difference in this metric and the difference in final CEF have the same sign.",
              f"{LABEL_COL} l l l l r", header, out, size=r"\scriptsize")


def table_representation(doc: Doc, B5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_representation — representation spread and Mordred")
    spread = {}
    for m in MAIN_SURROGATES:
        x = B5[m].unstack("representation")[FIVE_REPRESENTATIONS]
        spread[m] = (x.max(axis=1) - x.min(axis=1)).groupby(level="target").mean()
    spread = pd.DataFrame(spread)
    mor = B5.xs("mordred3d", level="representation")
    che = B5.xs("chemeleon", level="representation")
    rows = []
    for t in TARGETS + [None]:
        s = spread.mean() if t is None else spread.loc[t]
        mt = mor if t is None else mor.xs(t, level="target")
        ct = che if t is None else che.xs(t, level="target")
        label = "All" if t is None else TARGET_TEX[t]
        a, b = stat(mt.tabpfn - ct.tabpfn), stat(mt.gp - mt.tabpfn)
        print(f"  {label:20s} spread TabPFN {s.tabpfn:.2f} RF {s.rf:.2f} GP {s.gp:.2f}; Mordred − CheMeleon "
              f"(TabPFN) {txt(a)}; RBF GP − TabPFN on Mordred {txt(b)}; RBF GP CEF on Mordred {mt.gp.mean():.2f}")
        rows.append(rf"{label} & {s.tabpfn:.2f} & {s.rf:.2f} & {s.gp:.2f} & {tex(a)} & {tex(b)} & {mt.gp.mean():.2f} \\[3pt]")
        if t == TARGETS[-1]:
            rows.append(r"\midrule")
    doc.table("tab:app_representation",
              "Left: spread of final CEF across the five representations not trained through TabPFN (best minus "
              "worst), averaged over criteria and configurations, per surrogate. Right: Mordred minus CheMeleon "
              "under TabPFN, the RBF Gaussian process minus TabPFN on Mordred, and the RBF Gaussian process's "
              "mean final CEF on Mordred (random selection scores $1$).",
              "l r r r l l r",
              r"\textbf{Target} & \textbf{TabPFN} & \textbf{RF} & \textbf{RBF GP} & \textbf{Mordred $-$ CheMeleon} & "
              r"\textbf{RBF GP $-$ TabPFN} & \textbf{RBF GP CEF}", rows, size=r"\scriptsize")


def table_chemeleon(doc: Doc, B5: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_chemeleon — CheMeleon minus each other representation")
    out = []
    for m in MAIN_SURROGATES:
        out += group(f"{SURROGATE_TEX[m]}: CheMeleon minus", 6)
        x = B5[m].unstack("representation")
        for r in FIVE_REPRESENTATIONS:
            if r == "chemeleon":
                continue
            cells = per_target((x["chemeleon"] - x[r]).dropna())
            doc.row(f"{SURROGATE_TEX[m]}: CheMeleon − {REPRESENTATION_TEX[r]}", cells)
            out.append(tex_row(SUB + REPRESENTATION_TEX[r], cells))
    doc.table("tab:app_chemeleon",
              "CheMeleon minus each other representation not trained through TabPFN in final CEF, per surrogate "
              "(pairs over criterion and configuration: $20$ per target, $80$ in all).",
              f"{LABEL_COL} l l l l l", TARGET_HEADER.replace("Comparison", "Representation"), out)


def table_tanimoto(doc: Doc, B16: pd.DataFrame) -> None:
    doc.heading("TABLE tab:app_tanimoto — the Tanimoto-kernel Gaussian process")
    cols = [("greedy", "greedy"), ("pi_ythresh", CRITERION_TEX["pi_ythresh"])]

    def cells_for(d_tan_rbf: pd.Series, d_tab_tan: pd.Series) -> list:
        return [stat(d_tan_rbf.xs(c, level="criterion")) for c, _ in cols] + \
               [stat(d_tab_tan.xs(c, level="criterion")) for c, _ in cols]

    tan_rbf, tab_tan = B16.gp_tanimoto - B16.gp, B16.tabpfn - B16.gp_tanimoto
    out = group("By representation (pairs over target and configuration)", 5)
    for label, reps in ALL_REPRESENTATIONS_TEX:
        sel = lambda s: s[s.index.get_level_values("representation").isin(reps)]  # noqa: E731
        cells = cells_for(sel(tan_rbf), sel(tab_tan))
        doc.row(label.replace("$", ""), cells)
        out.append(tex_row(SUB + label, cells))
    out += group("By target (all sixteen representations)", 5)
    for t in TARGETS:
        cells = cells_for(tan_rbf.xs(t, level="target", drop_level=False), tab_tan.xs(t, level="target", drop_level=False))
        doc.row(TARGET_TEX[t].replace("\\emph{S.~aureus}", "S. aureus"), cells)
        out.append(tex_row(SUB + TARGET_TEX[t], cells))
    for label, reps in [("All sixteen representations", None), ("Five not trained through TabPFN", FIVE_REPRESENTATIONS)]:
        a, b = (tan_rbf, tab_tan) if reps is None else (subset(tan_rbf, reps), subset(tab_tan, reps))
        cells = cells_for(a, b)
        doc.row(label, cells)
        out.append(tex_row(SUB + label, cells))
    header = (r"& \multicolumn{2}{c}{\textbf{Tanimoto GP minus RBF GP}} & \multicolumn{2}{c}{\textbf{TabPFN minus Tanimoto GP}} \\"
              "\n" r"\cmidrule(lr){2-3} \cmidrule(l){4-5}" "\n"
              r"\textbf{Representation} & " + " & ".join(
                  rf"\textbf{{{g.capitalize() if g[0].islower() else g}}}" for _ in range(2) for _, g in cols))
    doc.table("tab:app_tanimoto",
              "The Gaussian process with a Tanimoto kernel, run on every representation, against the RBF Gaussian "
              "process and TabPFN in final CEF (each surrogate in its native predictive form; pairs over target and "
              "configuration, $16$ per representation and $12$ for a domain-matched encoder, "
              "which exists on three targets).",
              "L{4.8cm} l l l l", header, out, size=r"\scriptsize")


def table_mordred(doc: Doc) -> None:
    doc.heading("TABLE tab:app_mordred — Mordred diagnosis (values from gp_mordred_geometry.py)")
    rows = []
    for r, w, er, md, cv, p991, ell, sf2, k, sdr in MORDRED_DIAGNOSIS:
        print(f"  {r:20s} width {w}, eff. rank {er}, median dist {md}, CV {cv}, p99/p1 {p991}, ell {ell}, "
              f"sig_f^2 {sf2}, mean k {k}, pred SD / y SD {sdr}")
        rows.append(rf"{REPRESENTATION_TEX[r]} & {w:,} & {er} & ${md}$ & {cv:.3f} & {p991:.2f} & {ell} & {sf2} & "
                    .replace(",", "{,}", 1) + rf"{k:.3f} & {sdr:.3f} \\")
    s = MORDRED_STANDARDISED            # quoted in Section 4.4.2 of the thesis, not in the caption
    print(f"  Mordred standardised: median distance {s['median_distance']}, length scale {s['length_scale']}, "
          f"Spearman {s['spearman_raw']:+.2f} -> {s['spearman_standardised']:+.2f}")
    doc.table("tab:app_mordred",
              "Why the RBF Gaussian process fails on Mordred descriptors (Section~\\ref{subsec:mordred}): geometry "
              "of each representation on the D2R pool with $100$ labelled molecules, and the RBF Gaussian "
              "process fitted to them. Effective rank as in Equation~\\eqref{eq:effective_rank}; distances are "
              r"Euclidean between molecules; $\ell$ and $\sigma_f^2$ are the fitted length scale (bounded at "
              r"$100$) and signal variance; $\bar{k}$ is the mean off-diagonal kernel entry; the last column is the "
              r"spread of the predicted mean relative to that of the activities.",
              "l r r r r r r r r r",
              r"\textbf{Representation} & \textbf{Width} & \textbf{Eff.\ rank} & \textbf{Median dist.} & "
              r"\textbf{CV} & \textbf{p99/p1} & $\boldsymbol{\ell}$ & $\boldsymbol{\sigma_f^2}$ & "
              r"$\boldsymbol{\bar{k}}$ & \textbf{SD ratio}", rows)


def table_calibration(doc: Doc) -> None:
    doc.heading("TABLE tab:app_calibration — calibration and ranking")
    c = pd.read_csv(REPO_ROOT / CALIBRATION_CSV)
    c["mi"] = c.model + "/" + c.intervals
    names = {"tabpfn/full": "TabPFN, full", "tabpfn/gauss": "TabPFN, Gaussian", "rf/gauss": "Random forest",
             "gp/gauss": "RBF GP", "gp_tanimoto/gauss": "Tanimoto GP (Morgan)"}
    rows = []
    for mi, name in names.items():
        cells = []
        for n in sorted(c.train_size.unique()):
            q = c[(c.mi == mi) & (c.train_size == n)]
            dev = [q[f"coverage_{lv}"].mean() - lv / 100 for lv in COVERAGE_LEVELS]
            cells.append((q.coverage_90.mean(), min(dev), max(dev), q.ece.median(), q.spearman.mean()))
        print(f"  {name:22s} " + "   ".join(
            f"n={n}: cov90 {a:.3f}, dev {lo:+.3f}..{hi:+.3f}, ECE {e_:.3f}, rho {r_:.3f}"
            for n, (a, lo, hi, e_, r_) in zip(sorted(c.train_size.unique()), cells)))
        rows.append(name + " & " + " & ".join(
            rf"{a:.3f} & ${lo:+.3f}$ to ${hi:+.3f}$ & {e_:.3f} & {r_:.2f}" for a, lo, hi, e_, r_ in cells) + r" \\")

    w = c.pivot_table(index=["dataset", "representation", "train_size", "split"], columns="mi", values="ece")
    counts = {other: f"{int((w[other] > w['tabpfn/full']).sum())}/{int((w[other] - w['tabpfn/full']).notna().sum())}"
              for other in ["tabpfn/gauss", "rf/gauss", "gp/gauss", "gp_tanimoto/gauss"]}
    per_ds = (w["tabpfn/full"] < w["tabpfn/gauss"]).groupby(level="dataset").agg(["sum", "count"])
    sp = c[c.intervals == "gauss"].pivot_table(index=["dataset", "representation", "train_size", "split"],
                                              columns="model", values="spearman")
    morgan = sp[sp.index.get_level_values("representation") == "morgan"]
    rank = {"rf": f"{int((sp.tabpfn > sp.rf).sum())}/{int(sp.rf.notna().sum())}",
            "gp": f"{int((sp.tabpfn > sp.gp).sum())}/{int(sp.gp.notna().sum())}",
            "tanimoto": f"{int((morgan.gp_tanimoto > morgan.tabpfn).sum())}/{int(morgan.gp_tanimoto.notna().sum())}"}
    print("  TabPFN full ECE lower than:", counts)
    print("  full < Gaussian per dataset:", {d: f"{r['sum']}/{r['count']}" for d, r in per_ds.iterrows()})
    print("  Spearman: TabPFN > RF", rank["rf"], " TabPFN > RBF GP", rank["gp"], " Tanimoto > TabPFN (Morgan)",
          rank["tanimoto"])
    ds_names = {"saureus": r"\emph{S.~aureus}", "tyk2": "TYK2", "usp7": "USP7", "d2r": "D2R", "mpro": r"M$^{\text{pro}}$"}
    per_ds_tex = ", ".join(f"{ds_names.get(d, d)} {r['sum']}/{r['count']}" for d, r in per_ds.iterrows())
    sizes = sorted(c.train_size.unique())
    header = (r"& \multicolumn{4}{c}{\textbf{" + f"{sizes[0]:,}".replace(",", "{,}") + r" training molecules}} & "
              r"\multicolumn{4}{c}{\textbf{" + f"{sizes[1]:,}".replace(",", "{,}") + r" training molecules}} \\" + "\n"
              r"\cmidrule(lr){2-5} \cmidrule(l){6-9}" + "\n"
              r"\textbf{Surrogate} & \textbf{Cov.\ 90} & \textbf{Obs.\ $-$ nom.} & \textbf{ECE} & $\boldsymbol{\rho}$ & "
              r"\textbf{Cov.\ 90} & \textbf{Obs.\ $-$ nom.} & \textbf{ECE} & $\boldsymbol{\rho}$")
    doc.table("tab:app_calibration",
              "Calibration and ranking by training-set size: mean coverage of the central $90\\%$ interval, the "
              "range of observed minus nominal coverage over the ten nominal levels, median ECE and mean Spearman "
              "correlation $\\rho$ on held-out molecules. Settings in which the full distribution's ECE is "
              f"lower than that of TabPFN's Gaussian approximation, per dataset: {per_ds_tex}.",
              "l r r r r r r r r", header, rows, size=r"\scriptsize")


# --------------------------------------------------------------------------- #
def main() -> None:
    df = read_runs()
    e = experiments(df)
    print(f"{len(e):,} complete AL experiments ({(e.candidates == CANDIDATES).sum():,} with {CANDIDATES} candidates)")
    nat = native(e)
    E1k = e[e.candidates == CANDIDATES]                           # every form of every surrogate, main protocol
    W5all = subset(wide(nat), FIVE_REPRESENTATIONS)               # criterion columns, native forms, with the Tanimoto GP
    W5 = subset(W5all, models=MAIN_SURROGATES)                    # the same for the three main surrogates
    B5 = subset(by_model(nat), FIVE_REPRESENTATIONS)              # surrogate columns, native forms
    B16 = by_model(nat)                                           # the same for all sixteen representations
    assert len(B5.dropna(subset=MAIN_SURROGATES)) == 400, "expected 400 pairs: 4 targets x 5 representations x 5 criteria x 4 configurations"
    doc = Doc()
    table_surrogates(doc, B5)
    table_representation_pairs(doc, B5)
    table_init(doc, B5)
    table_budget_and_rounds(doc, round_level(df))
    table_criteria(doc, W5)
    table_threshold(doc, W5)
    table_full_distribution(doc, E1k, W5)
    table_means(doc, E1k, W5all)
    table_without_usp7(doc, W5, B5, E1k)
    table_activity(doc, e, W5)
    table_representation(doc, B5)
    table_chemeleon(doc, B5)
    table_tanimoto(doc, B16)
    table_mordred(doc)
    table_calibration(doc)
    out = REPO_ROOT / OUT_TEX
    out.write_text("% Written by src/scripts/appendix_c_tables.py; do not edit by hand.\n"
                   r"\providecommand{\appcell}[2]{\begin{tabular}[t]{@{}l@{}}$#1$\\[-1pt]{\scriptsize (#2)}\end{tabular}}"
                   "\n\n" + "\n".join(doc.tex))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
