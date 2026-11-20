"""scripts/appendix_b_tables.py — every value of Appendix B (runtime and candidate subsampling). Press Run.

Prints the numbers of Appendix B and writes its tables as LaTeX to OUT_TEX, ready to copy into the thesis as
Content/Appendix_B_tables.tex.

Capacity. CAPACITY_CSV (runtime_capacity.py) holds ONE TabPFN predict call for 1,000 to 400,000 candidates on the
same GPU model, with its peak GPU memory; Table tab:app_capacity shows that 1,000 candidates are the protocol's
subsample and not a limit of the GPU.

Runtime. RUNTIME_CSV (runtime_study.py) holds the wall-clock time of one acquisition round,
fitting (or conditioning on) the labelled set and predicting for the candidates, on the S. aureus pool: three
representations, labelled sets of 10 to 1,100 molecules, scoring 1,000 candidates and scoring the whole
remaining pool. Two points per setting identify the two parts of the cost model of the 2026-10-29 supervision,

    time of a round = a(n) + m * b(n),

a fixed part a(n) that depends on the labelled set (n molecules; fitting or conditioning) and a part b(n) per
candidate, for m candidates scored. a(n) and b(n) are linearly interpolated between the measured labelled-set
sizes. The cost of a campaign is the sum over its rounds, each at that round's labelled-set size, and its
extrapolation to larger pools assumes that b(n) stays constant, which is not measured. TabPFN ran on the GPU
(NVIDIA A100-PCIe-40GB, the model on which 50 of the 64 benchmark jobs ran), the other surrogates on the CPU of the
same host (four cores requested; the job sees all cores of the host, so the CPU times depend on its load).
The first sweep (experiments/runtime_study.csv, 2026-10-07) ran on an H200 and was about twice as fast; it is no
longer used in the thesis.

Check. The cost model is compared with the rounds of the benchmark's own S. aureus campaigns that ran on the same
GPU model (CHECK_REPRESENTATIONS; greedy selection, 1,000 candidates, all four configurations), each with the
curve a(n), b(n) of its own representation.

Candidate subsampling. final_subsampling of the final run (USP7 and D2R, CheMeleon) scores every untested
molecule; its 1,000-candidate counterparts are the same arms of `final`. A pair is one AL experiment scored both
ways (same target, surrogate, criterion, N_init and batch size); the surrogates are in their native predictive
forms and "established" means a mean beyond two standard errors, as in Appendix C.

Run with:  uv run python src/scripts/appendix_b_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
RUNTIME_CSV = "experiments/runtime_study_a100.csv"   # the sweep on an A100-PCIe-40GB (2026-11-09); the first, on an H200, is experiments/runtime_study.csv
FINAL_DIR = "experiments/benchmark_final"
OUT_TEX = "experiments/appendix_b_tables.tex"
CAPACITY_CSV = "experiments/runtime_capacity_a100.csv"   # one predict call per candidate-set size (runtime_capacity.py)
CAPACITY_REPRESENTATIONS = [("chemeleon", "CheMeleon ($2{,}048$)"),
                            ("chemprop_untrained", "Random-init Chemprop ($300$)")]
CAPACITY_CONTEXTS = [100, 1100]
CAPACITY_SIZES = [1_000, 10_000, 30_000, 100_000, 200_000, 400_000]
# Settings whose attempt ended the job: the CSV has their `started` row only, the error is in the job's output file
# (runlogs/run_gpu_capacity_a100.237508.out: TabPFNCUDAOutOfMemoryError, 33.75 GiB allocated, 7.45 GiB more requested).
CAPACITY_OUT_OF_MEMORY = {("chemprop_untrained", 100, 400_000)}

SUBSAMPLE = 1000                                  # candidates per round in the benchmark
POOL_SIZES = [10**4, 10**5, 10**6, 10**7, 10**9]  # extrapolation: candidates scored every round
CAMPAIGN_INIT, CAMPAIGN_BATCH, CAMPAIGN_ROUNDS = 100, 50, 20    # the labelled sets of the extrapolated campaign
# S. aureus campaigns of the final run that ran on an A100-PCIe-40GB, the GPU of the runtime sweep (condor log of
# the final run: job 4 chemeleon on gefjon, job 2 morgan on fulla; chemprop_untrained ran on an L40S and is left out).
CHECK_REPRESENTATIONS = ["chemeleon", "morgan"]

N_SEEDS = 5
N_ROUNDS = 20
SURROGATES = [("tabpfn", "TabPFN (GPU)"), ("rf", "Random forest (CPU)"), ("gp", "RBF GP (CPU)"),
              ("gp_tanimoto", "Tanimoto GP (CPU, Morgan only)")]
NATIVE_DIST = {"tabpfn": "full", "rf": "gauss", "gp": "gauss", "gp_tanimoto": "gauss"}
MAIN_SURROGATES = ["tabpfn", "rf", "gp"]
SURROGATE_TEX = {"tabpfn": "TabPFN", "rf": "RF", "gp": "RBF GP"}
CRITERIA = ["greedy", "ei_yplus", "pi_yplus", "ei_ythresh", "pi_ythresh"]
CRITERION_TEX = {"greedy": "greedy", "ei_yplus": r"EI($y^{+}$)", "pi_yplus": r"PI($y^{+}$)",
                 "ei_ythresh": r"EI($y_{\text{thresh}}$)", "pi_ythresh": r"PI($y_{\text{thresh}}$)"}
SUBSAMPLING_TARGETS = ["usp7", "d2r"]
TARGET_TEX = {"usp7": "USP7", "d2r": "D2R"}
POOL_SIZE = {"usp7": 4535, "d2r": 2502}           # molecules (Table 3.1)


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


def tex(s, digits: int = 2) -> str:
    if s is None:
        return "---"
    m, se, won, n = s
    star = "^{*}" if abs(m) > 2 * se else ""
    return rf"\appcell{{{m:+.{digits}f} \pm {se:.{digits}f}{star}}}{{{won}/{n}}}"


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}~s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}~min"
    if seconds < 36 * 3600:
        return f"{seconds / 3600:.1f}~h"
    if seconds < 365 * 86400:
        return f"{seconds / 86400:.1f}~days"
    return f"{seconds / (365 * 86400):.1f}~years"


def tex_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def rng(lo: float, hi: float, digits: int = 2) -> str:
    return f"{lo:.{digits}f}--{hi:.{digits}f}"


class Doc:
    def __init__(self) -> None:
        self.tex: list[str] = []

    def table(self, label: str, caption: str, colspec: str, header: str, rows: list[str],
              size: str = r"\footnotesize") -> None:
        self.tex += [r"\begin{table}[!htbp]", r"\centering", size, r"\setlength{\tabcolsep}{3pt}",
                     rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                     rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} {colspec} @{{}}}}",
                     r"\toprule", header + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}",
                     r"\end{table}", ""]


# Indented row label under a group heading. The kern keeps the label on the first line (a \quad lets TeX break
# right after it, dropping a long label to the second line) and the hanging indent keeps wrapped lines indented.
SUB = r"\hangindent=0.5em\hangafter=1\leavevmode\kern0.5em "   # 0.5em: "Physicochemical" must fit a 2.7cm label column
def group(title: str, ncols: int) -> list[str]:
    return [rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{title}}}}} \\", r"\addlinespace[2pt]"]


# --------------------------------------------------------------------------- #
# Runtime                                                                     #
# --------------------------------------------------------------------------- #
def decompose() -> pd.DataFrame:
    """One row per (model, representation, labelled-set size): t(1,000), t(pool), a and b."""
    r = pd.read_csv(REPO_ROOT / RUNTIME_CSV)
    r = r[r.status == "ok"]
    sub = r[r.candidates == SUBSAMPLE].set_index(["model", "representation", "context_size"])
    pool = r[r.candidates != SUBSAMPLE].set_index(["model", "representation", "context_size"])
    j = sub[["median_seconds", "candidates"]].join(pool[["median_seconds", "candidates"]], lsuffix="_sub",
                                                    rsuffix="_pool", how="inner")
    j["b"] = (j.median_seconds_pool - j.median_seconds_sub) / (j.candidates_pool - j.candidates_sub)
    j["a"] = j.median_seconds_sub - j.candidates_sub * j.b
    print(f"runtime study: {len(r)} measurements, {len(j)} (model, representation, labelled set) settings")
    return j.reset_index()


def table_runtime(doc: Doc, j: pd.DataFrame) -> dict:
    print("\n=== TABLE tab:app_runtime — time per round and its two parts (S. aureus)")
    rows, curve = [], {}
    stats: dict[str, list] = {m: [] for m, _ in SURROGATES}
    sizes = sorted(j.context_size.unique())
    for m, name in SURROGATES:
        q = j[j.model == m]
        rows += group(name, 6)
        a_n, b_n = [], []
        for n in sizes:
            g = q[q.context_size == n]
            a_med, b_med = g.a.median(), g.b.median()
            a_n.append(a_med)
            b_n.append(b_med)
            share = g.a / g.median_seconds_sub            # fixed part as a share of a round with 1,000 candidates
            cross = g.a / g.b                             # candidates at which both parts cost the same
            stats[m].append((n, a_med, b_med, share.min(), share.max(), cross.median()))
            print(f"  {m:12s} n={n:>5}: 1,000 cand. {g.median_seconds_sub.median():6.2f} s "
                  f"({g.median_seconds_sub.min():.2f}-{g.median_seconds_sub.max():.2f}); pool {g.median_seconds_pool.median():6.2f} s "
                  f"({g.median_seconds_pool.min():.2f}-{g.median_seconds_pool.max():.2f}); a {a_med:6.2f} s; b {1e6 * b_med:6.0f} us; "
                  f"fixed share at 1,000 cand. {100 * share.min():.0f}-{100 * share.max():.0f} %; a/b {cross.median():,.0f} candidates")
            rows.append(SUB + rf"{tex_int(n)} & {g.median_seconds_sub.median():.2f} ({rng(g.median_seconds_sub.min(), g.median_seconds_sub.max())})"
                        rf" & {g.median_seconds_pool.median():.2f} ({rng(g.median_seconds_pool.min(), g.median_seconds_pool.max())})"
                        rf" & {a_med:.2f} & {tex_int(1e6 * b_med)} & {tex_int(cross.median())} \\")
        curve[m] = (np.array(sizes, float), np.array(a_n), np.array(b_n))
    print("  summary over the labelled-set sizes (medians over representations):")
    for m, _ in SURROGATES:
        s = stats[m]
        print(f"    {m:12s} a {min(x[1] for x in s):.2f}-{max(x[1] for x in s):.2f} s; b {1e3 * min(x[2] for x in s):.2f}-"
              f"{1e3 * max(x[2] for x in s):.2f} ms; fixed share at 1,000 candidates {100 * min(x[3] for x in s):.0f}-"
              f"{100 * max(x[4] for x in s):.0f} %; a/b {min(x[5] for x in s):,.0f}-{max(x[5] for x in s):,.0f} candidates")
    doc.table("tab:app_runtime",
              "Wall-clock time of one acquisition round on the \\emph{S.~aureus} pool, in seconds: the median over three "
              "representations (CheMeleon, Morgan fingerprints, random-init Chemprop) with the range in brackets, "
              "for $1{,}000$ candidates and for the whole remaining pool ($\\approx 31{,}000$). The columns $a$ and $b$ "
              "are the fixed part per round and the part per candidate (in $\\mu$s) of the cost model, time "
              "$= a + m\\,b$ for $m$ candidates; $a/b$ is the number of candidates at which the two cost the same. "
              "The Tanimoto-kernel Gaussian process was timed on Morgan fingerprints only. TabPFN ran on an NVIDIA "
              "A100 GPU and the other surrogates on the CPU.",
              "l l l r r r",
              r"\textbf{Labelled molecules} & \textbf{1,000 candidates (s)} & \textbf{Whole pool (s)} & "
              r"$\boldsymbol{a}$ \textbf{(s)} & $\boldsymbol{b}$ \textbf{($\mu$s)} & $\boldsymbol{a/b}$", rows)
    return curve


def table_capacity(doc: Doc) -> None:
    """One predict call as the candidates grow: peak GPU memory and time per candidate (runtime_capacity.py)."""
    print("\n=== TABLE tab:app_capacity — one TabPFN predict call: peak GPU memory (GB) and ms per candidate")
    c = pd.read_csv(REPO_ROOT / CAPACITY_CSV)
    c = c[c.status != "started"]
    total = float(c.gpu_total_gb.iloc[0])
    rows = []
    for n in CAPACITY_SIZES:
        cells, shares = [], []
        for rep, _ in CAPACITY_REPRESENTATIONS:
            mem, ms = [], []
            for ctx in CAPACITY_CONTEXTS:
                r = c[(c.representation == rep) & (c.context_size == ctx) & (c.candidates == n)]
                if len(r) and r.status.iloc[0] == "ok":
                    mem.append(f"{r.gpu_peak_allocated_gb.iloc[0]:.1f}")
                    ms.append(f"{r.ms_per_candidate.iloc[0]:.2f}")
                    shares.append(float(r.gpu_peak_share.iloc[0]))
                elif (rep, ctx, n) in CAPACITY_OUT_OF_MEMORY:
                    mem.append("out of memory")
                    ms.append("---")
                else:
                    mem.append("---")
                    ms.append("---")
            cells += [" / ".join(mem) if mem != ["---", "---"] else "---", " / ".join(ms) if ms != ["---", "---"] else "---"]
        share = f"${100 * min(shares):.0f}$--${100 * max(shares):.0f}\\%$" if shares else "---"
        print(f"  {n:>9,} candidates: " + "   ".join(cells) + f"   share of {total:.1f} GB: {share}")
        rows.append(f"{tex_int(n)} & " + " & ".join(cells) + f" & {share} \\\\")
    header = (r" & \multicolumn{2}{c}{\textbf{" + CAPACITY_REPRESENTATIONS[0][1] + r"}} & \multicolumn{2}{c}{\textbf{"
              + CAPACITY_REPRESENTATIONS[1][1] + r"}} & \\" + "\n" + r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}" + "\n"
              + r"\textbf{Candidates} & \textbf{Memory (GB)} & \textbf{ms / cand.} & \textbf{Memory (GB)} & "
              r"\textbf{ms / cand.} & \textbf{Share}")
    doc.table("tab:app_capacity",
              "One TabPFN prediction call on an NVIDIA A100 GPU with $" + f"{total:.1f}" + "$~GB of memory as the number of "
              "candidates grows, for CheMeleon and the randomly initialised Chemprop encoder (number of features in "
              "brackets): peak GPU memory and time per candidate (ms / cand.), each for $100$ / $1{,}100$ labelled "
              "molecules, and the share of the GPU's memory over the four settings. Candidate sets larger than the pool repeat its "
              "molecules. At $1{,}000$ candidates the time per candidate is dominated by the fixed cost of the call. "
              "A round of the benchmark makes two such calls. A dash marks a setting that was not run: CheMeleon above "
              "$200{,}000$ candidates because of the host memory of the job, and $1{,}100$ labelled molecules at "
              "$400{,}000$ because the run ended at its first failure.",
              "r l l l l l", header, rows)


def campaign_seconds(curve: dict, model: str, m: float, init: int, batch: int, rounds: int) -> float:
    """Sum over the rounds of a(n_t) + m b(n_t), each at that round's labelled-set size."""
    n, a, b = curve[model]
    sizes = init + batch * np.arange(rounds)
    return float(np.sum(np.interp(sizes, n, a) + m * np.interp(sizes, n, b)))


def table_extrapolation(doc: Doc, curve: dict, ratio: tuple[float, float, int]) -> None:
    print("\n=== TABLE tab:app_extrapolation — one campaign, scoring every candidate in each round")
    rows = group("Every candidate scored in each round", len(POOL_SIZES) + 1)
    for m, name in SURROGATES:
        cells = [human(campaign_seconds(curve, m, p, CAMPAIGN_INIT, CAMPAIGN_BATCH, CAMPAIGN_ROUNDS)) for p in POOL_SIZES]
        print(f"  {m:12s} " + "  ".join(f"{c:>12s}" for c in cells))
        rows.append(SUB + name + " & " + " & ".join(cells) + r" \\")
    rows += group(r"$1{,}000$ candidates scored in each round, whatever the size of the pool", len(POOL_SIZES) + 1)
    for m, name in SURROGATES:
        s = campaign_seconds(curve, m, SUBSAMPLE, CAMPAIGN_INIT, CAMPAIGN_BATCH, CAMPAIGN_ROUNDS)
        print(f"  {m:12s} with {SUBSAMPLE:,} candidates per round: {human(s)} ({s:.0f} s)")
        rows.append(SUB + name + rf" & \multicolumn{{{len(POOL_SIZES)}}}{{c}}{{{human(s)}}} \\")
    header = r"\textbf{Surrogate} & " + " & ".join(
        rf"$\mathbf{{10^{{{int(np.log10(p))}}}}}$" for p in POOL_SIZES)
    doc.table("tab:app_extrapolation",
              f"Time of one campaign of ${CAMPAIGN_ROUNDS}$ rounds (initial set ${CAMPAIGN_INIT}$, batch size "
              f"${CAMPAIGN_BATCH}$) as the number of candidates grows, extrapolated from the cost model of "
              "Table~\\ref{tab:app_runtime}. The upper rows score every candidate in each round, the lower rows "
              "$1{,}000$ per round. The per-candidate part is assumed constant, as measured up to $200{,}000$ candidates "
              "(Table~\\ref{tab:app_capacity}). TabPFN ran on an NVIDIA A100 GPU and the other surrogates on the CPU.",
              "l r r r r r", header, rows)


def check_cost_model(j: pd.DataFrame, df: pd.DataFrame) -> tuple[float, float, int]:
    """Compare the cost model with the rounds of the final run's S. aureus campaigns on the same GPU model: each
    campaign (surrogate, representation, configuration; greedy, 1,000 candidates; median over seeds per round)
    against the curve a(n), b(n) of its own representation."""
    print("\n=== CHECK — measured campaign rounds / cost model (S. aureus, greedy, 1,000 candidates, A100-PCIe campaigns)")
    rows = []
    q0 = df[(df.target == "saureus") & (df.criterion == "greedy") & (df.candidates == str(SUBSAMPLE))]
    for rep in CHECK_REPRESENTATIONS:
        for m in MAIN_SURROGATES:
            c = j[(j.model == m) & (j.representation == rep)].sort_values("context_size")
            for (init, batch), q in q0[(q0.model == m) & (q0.representation == rep)].groupby(["init_size", "batch_size"]):
                measured = q.groupby("round").seconds.median()
                n = init + batch * (measured.index.values - 1)
                predicted = np.interp(n, c.context_size, c.a) + SUBSAMPLE * np.interp(n, c.context_size, c.b)
                rows.append((rep, m, init, batch, measured.sum(), predicted.sum(), measured.sum() / predicted.sum(), n.max()))
                print(f"  {rep:10s} {m:7s} N_init {init:>3} K {batch:>2}: measured {measured.sum():7.1f} s, "
                      f"model {predicted.sum():7.1f} s, ratio {measured.sum() / predicted.sum():.2f}")
    r = pd.DataFrame(rows, columns=["rep", "model", "init", "batch", "measured", "predicted", "ratio", "n_max"])
    print(f"  all {len(r)}: median {r.ratio.median():.2f}, range {r.ratio.min():.2f}-{r.ratio.max():.2f}")
    print("  by surrogate: " + "; ".join(f"{m} {g.ratio.median():.2f} ({g.ratio.min():.2f}-{g.ratio.max():.2f})"
                                        for m, g in r.groupby("model")))
    small, large = r[r.n_max <= 300], r[r.n_max > 300]
    print(f"  labelled sets up to 300 molecules (batch 10): median {small.ratio.median():.2f}, range "
          f"{small.ratio.min():.2f}-{small.ratio.max():.2f} ({len(small)}); up to 1,060 (batch 50): median "
          f"{large.ratio.median():.2f}, range {large.ratio.min():.2f}-{large.ratio.max():.2f} ({len(large)})")
    return float(r.ratio.min()), float(r.ratio.max()), len(r)


# --------------------------------------------------------------------------- #
# Candidate subsampling                                                       #
# --------------------------------------------------------------------------- #
def read_final() -> pd.DataFrame:
    files = sorted((REPO_ROOT / FINAL_DIR).glob("final_*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"candidates": str, "tau": str, "dist": str}) for f in files],
                   ignore_index=True)
    df["criterion"] = np.where(df.acquisition == "greedy", "greedy", df.acquisition + "_" + df.tau)
    return df


def table_subsampling(doc: Doc, df: pd.DataFrame) -> None:
    print("\n=== TABLE tab:app_subsampling — scoring the whole pool minus 1,000 candidates per round")
    f = df[df["round"] == N_ROUNDS]
    key = ["target", "representation", "model", "criterion", "dist", "init_size", "batch_size", "candidates"]
    e = f.groupby(key).agg(cef=("cef", "mean"), best_y=("best_y", "mean"), n=("seed", "nunique")).reset_index()
    e = e[(e.n == N_SEEDS) & e.target.isin(SUBSAMPLING_TARGETS) & (e.representation == "chemeleon")]
    e = e[(e.criterion == "greedy") | (e.dist == e.model.map(NATIVE_DIST))]
    k = ["target", "model", "criterion", "init_size", "batch_size"]
    full = e[e.candidates == "all"].set_index(k)[["cef", "best_y"]]
    sub = e[e.candidates == str(SUBSAMPLE)].set_index(k)[["cef", "best_y"]]
    j = full.join(sub, lsuffix="_full", rsuffix="_1k", how="inner")
    j["d_cef"], j["d_best"] = j.cef_full - j.cef_1k, j.best_y_full - j.best_y_1k
    j = j[j.index.get_level_values("model").isin(MAIN_SURROGATES)]
    print(f"  {len(j)} matched pairs ({len(j) // len(MAIN_SURROGATES) // len(CRITERIA)} per surrogate and criterion)")

    def cells(d: pd.Series, digits: int = 2) -> list[str]:
        out = []
        for t in SUBSAMPLING_TARGETS:
            out.append(tex(stat(d.xs(t, level="target")), digits))
        out.append(tex(stat(d), digits))
        return out

    def show(label: str, d: pd.Series) -> None:
        print(f"  {label:34s} " + "  ".join(f"{txt(stat(d.xs(t, level='target'))):>24s}" for t in SUBSAMPLING_TARGETS)
              + f"  {txt(stat(d)):>24s}")

    rows = group("Final CEF, whole pool minus $1{,}000$ candidates, by surrogate", 4)
    for m in MAIN_SURROGATES:
        d = j.d_cef.xs(m, level="model", drop_level=False)
        show(f"CEF {SURROGATE_TEX[m]}", d)
        rows.append(SUB + rf"{SURROGATE_TEX[m]} & " + " & ".join(cells(d)) + r" \\[3pt]")
    show("CEF all three", j.d_cef)
    rows.append(SUB + r"All three & " + " & ".join(cells(j.d_cef)) + r" \\[3pt]")
    rows += group("Final CEF, by criterion (three surrogates)", 4)
    for c in CRITERIA:
        d = j.d_cef.xs(c, level="criterion", drop_level=False)
        show(f"CEF {c}", d)
        rows.append(SUB + rf"{CRITERION_TEX[c]} & " + " & ".join(cells(d)) + r" \\[3pt]")
    rows += group("Best activity found, whole pool minus $1{,}000$ candidates (activity units)", 4)
    for m in MAIN_SURROGATES:
        d = j.d_best.xs(m, level="model", drop_level=False)
        show(f"best_y {SURROGATE_TEX[m]}", d)
        rows.append(SUB + rf"{SURROGATE_TEX[m]} & " + " & ".join(cells(d, 3)) + r" \\[3pt]")
    show("best_y all three", j.d_best)
    rows.append(SUB + r"All three & " + " & ".join(cells(j.d_best, 3)) + r" \\[3pt]")

    ranks = {}
    for t in SUBSAMPLING_TARGETS:
        jt = j.xs(t, level="target")
        ranks[t] = jt.cef_full.rank().corr(jt.cef_1k.rank())
        print(f"  {t}: rank correlation of the AL experiments, whole pool vs 1,000 candidates: {ranks[t]:.2f} (n = {len(jt)})")
    share = ", ".join(f"{TARGET_TEX[t]} {100 * SUBSAMPLE / POOL_SIZE[t]:.0f}\\%" for t in SUBSAMPLING_TARGETS)
    rank_txt = ", ".join(f"{TARGET_TEX[t]} ${ranks[t]:.2f}$" for t in SUBSAMPLING_TARGETS)
    header = (r"\textbf{Comparison} & " + " & ".join(rf"\textbf{{{TARGET_TEX[t]}}}" for t in SUBSAMPLING_TARGETS)
              + r" & \textbf{Both}")
    doc.table("tab:app_subsampling",
              "Scoring every untested molecule instead of $1{,}000$ candidates per round, on the two smallest pools "
              "with CheMeleon: mean paired difference in final CEF (upper parts) and in the best activity found "
              "(lower part) $\\pm$ standard error and, beneath it, the pairs in which the whole pool ended higher; "
              "$^{*}$ marks an established difference. Pairs are AL experiments over surrogate, criterion and the "
              f"four configurations of a target. The subsample covers {share} of the pool. The rank correlation "
              f"between the final CEF of the AL experiments scored both ways is {rank_txt}.",
              "L{4.6cm} l l l", header, rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    doc = Doc()
    j = decompose()
    curve = table_runtime(doc, j)
    table_capacity(doc)
    df = read_final()
    ratio = check_cost_model(j, df)
    table_extrapolation(doc, curve, ratio)
    table_subsampling(doc, df)
    out = REPO_ROOT / OUT_TEX
    out.write_text("% Written by src/scripts/appendix_b_tables.py; do not edit by hand.\n"
                   r"\providecommand{\appcell}[2]{\begin{tabular}[t]{@{}l@{}}$#1$\\[-1pt]{\scriptsize (#2)}\end{tabular}}"
                   "\n\n" + "\n".join(doc.tex))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
