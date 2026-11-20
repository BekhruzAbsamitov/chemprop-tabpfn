"""scripts/summarise_extended.py — the results of the extended benchmark (E2-E6). Press Run.

Reads experiments/benchmark_extended/<experiment>_<target>_<representation>.csv, written by
benchmark_extended.py, and answers the follow-up questions of the 2026-09-29 supervision:

  RQ2  greedy vs EI and PI, anchored at the best value so far (y+) or at the hit threshold
       (y_thresh), from TabPFN's Gaussian approximation; per target and per surrogate.
  FULL EI and PI from TabPFN's full bar distribution instead of the Gaussian approximation.
  GP   the Tanimoto-kernel GP vs the RBF GP and TabPFN on Morgan fingerprints.
  MEAN the third metric, mean measured activity of the acquired molecules: does it rank the
       strategies the same way as CEF?
  SUB  candidate subsampling: 1,000 random candidates vs scoring the whole pool.

Comparisons are paired, final round, as in summarise_primary_benchmark.py: an AL experiment
is one (target, representation, surrogate, init size, batch size) combination, averaged over
its five seeds; a matched pair differs in the strategy only. Files that are still being
written are fine: only AL experiments with all five seeds at the final round are used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
RESULTS_DIR = "experiments/benchmark_extended"
TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
N_SEEDS = 5
EXPERIMENT_KEY = ["target", "representation", "model", "init_size", "batch_size"]


def paired(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "—"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    flag = "*" if abs(x.mean()) > 2 * se else " "
    return f"{x.mean():+.2f} ± {se:.2f}{flag}({(x > 0).sum()}/{len(x)})"


def load(experiment: str) -> pd.DataFrame:
    """Final-round rows of complete campaigns, averaged over seeds, one column per strategy."""
    files = sorted((REPO_ROOT / RESULTS_DIR).glob(f"{experiment}_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["strategy"] = np.where(df.acquisition == "greedy", "greedy",
                              df.acquisition + "_" + df.tau + "_" + df.dist)
    df["candidates"] = df["candidates"].astype(str)
    f = df[df["round"] == df["round"].max()]
    key = EXPERIMENT_KEY + ["candidates", "strategy"]
    g = f.groupby(key).agg(cef=("cef", "mean"), mean_y=("mean_y_acquired", "mean"),
                           n=("seed", "nunique")).reset_index()
    dropped = (g.n < N_SEEDS).sum()
    print(f"{experiment}: {len(files)} files, {len(g)} AL experiments × strategy "
          f"({dropped} incomplete, left out)")
    return g[g.n == N_SEEDS]


def wide(g: pd.DataFrame, value: str = "cef") -> pd.DataFrame:
    return g.pivot_table(index=EXPERIMENT_KEY + ["candidates"], columns="strategy", values=value)


def by_target(d: pd.Series, label: str) -> None:
    cells = [paired(d.xs(t, level="target")) if t in d.index.get_level_values("target") else "—"
             for t in TARGETS]
    print(f"  {label:34s} " + " ".join(f"{c:>21s}" for c in cells) + f" {paired(d):>21s}")


def header(title: str) -> None:
    print(f"\n{title}   (* = more than 2 SE from zero; pairs won in brackets)")
    print(f"  {'':34s} " + " ".join(f"{t:>21s}" for t in TARGETS) + f" {'all':>21s}")


def rq2(p: pd.DataFrame) -> None:
    header("RQ2 — strategy minus greedy, Gaussian approximation, all surrogates")
    for s in ["ei_yplus_gauss", "ei_ythresh_gauss", "pi_yplus_gauss", "pi_ythresh_gauss"]:
        by_target(p[s] - p.greedy, s)
    header("RQ2 — EI at y_thresh minus greedy, per surrogate")
    for m in ["tabpfn", "rf", "gp", "gp_tanimoto"]:
        q = p.xs(m, level="model", drop_level=False)
        by_target(q.ei_ythresh_gauss - q.greedy, m)
    header("RQ2 — y_thresh minus y+ (same acquisition)")
    by_target(p.ei_ythresh_gauss - p.ei_yplus_gauss, "EI")
    by_target(p.pi_ythresh_gauss - p.pi_yplus_gauss, "PI")
    header("RQ2 — EI at y_thresh minus greedy, per representation")
    for r in ["chemeleon", "morgan", "chemprop_untrained"]:
        q = p.xs(r, level="representation", drop_level=False)
        by_target(q.ei_ythresh_gauss - q.greedy, r)
    header("RQ2 — EI at y_thresh minus greedy, per batch size")
    for b in sorted(p.index.get_level_values("batch_size").unique()):
        q = p.xs(b, level="batch_size", drop_level=False)
        by_target(q.ei_ythresh_gauss - q.greedy, f"batch {b}")


def full_vs_gauss(p: pd.DataFrame) -> None:
    q = p.xs("tabpfn", level="model", drop_level=False)
    header("FULL — TabPFN, full bar distribution minus Gaussian approximation")
    for s in ["ei_yplus", "ei_ythresh", "pi_yplus", "pi_ythresh"]:
        by_target(q[f"{s}_full"] - q[f"{s}_gauss"], s)
    header("FULL — TabPFN, best full-distribution strategy minus greedy")
    for s in ["ei_ythresh_full", "pi_ythresh_full"]:
        by_target(q[s] - q.greedy, s)


def tanimoto(p: pd.DataFrame) -> None:
    m = p.xs("morgan", level="representation")
    gp_t, gp, tab = (m.xs(k, level="model") for k in ("gp_tanimoto", "gp", "tabpfn"))
    header("GP — Morgan fingerprints: Tanimoto GP minus RBF GP / minus TabPFN")
    for s in ["greedy", "ei_yplus_gauss", "ei_ythresh_gauss"]:
        by_target((gp_t[s] - gp[s]).dropna(), f"minus RBF GP, {s}")
    for s in ["greedy", "ei_yplus_gauss", "ei_ythresh_gauss"]:
        by_target((gp_t[s] - tab[s]).dropna(), f"minus TabPFN, {s}")


def mean_activity(g: pd.DataFrame, p: pd.DataFrame) -> None:
    y = wide(g, "mean_y")
    header("MEAN — mean activity of the acquired molecules, strategy minus greedy (activity units)")
    for s in ["ei_yplus_gauss", "ei_ythresh_gauss"]:
        by_target(y[s] - y.greedy, s)
    agree = []
    for s in ["ei_yplus_gauss", "ei_ythresh_gauss", "pi_ythresh_gauss"]:
        a = np.sign(p[s] - p.greedy) == np.sign(y[s] - y.greedy)
        agree.append(f"{s} {a.mean():.0%}")
    print("  pairs where mean activity and CEF agree on the sign: " + ", ".join(agree))


def subsampling(g: pd.DataFrame) -> None:
    p = g.pivot_table(index=EXPERIMENT_KEY + ["strategy"], columns="candidates", values="cef")
    if "all" not in p or "1000" not in p:
        print("  no matched pairs yet")
        return
    d = (p["all"] - p["1000"]).dropna()
    header("SUB — whole pool minus 1,000 candidates (CEF)")
    by_target(d, "all surrogates and strategies")
    for m in ["tabpfn", "rf", "gp"]:
        if m in d.index.get_level_values("model"):
            by_target(d.xs(m, level="model", drop_level=False), m)


def main() -> None:
    g = load("acquisition")
    p = wide(g)
    print("  complete AL experiments per target/representation:")
    print(g[g.strategy == "greedy"].groupby(["target", "representation"]).size().unstack().fillna(0)
          .astype(int).to_string())
    rq2(p)
    full_vs_gauss(p)
    tanimoto(p)
    mean_activity(g, p)
    print()
    subsampling(load("subsampling"))


if __name__ == "__main__":
    main()
