"""scripts/summarise_primary_benchmark.py — the headline numbers of the primary AL benchmark. Press Run.

Recomputes the paired comparisons behind RQ1-RQ3 (report §5e) from experiments/al_benchmark.csv,
after the S. aureus pool was corrected (2026-10-07). All comparisons are paired and use the
final round: an AL experiment is one (target, representation, surrogate, strategy, init size,
batch size) combination, stored as the mean over five seeds; a matched pair differs in one
factor only. Reported: mean paired difference in CEF, its standard error, and pairs won.

  RQ1  TabPFN vs RF and vs GP, per target; how often TabPFN is the weakest surrogate.
  RQ1b batch size at a MATCHED budget: TabPFN's margin over RF with the smaller batch (final
       round) vs the larger batch at the round where it has spent the same budget.
  RQ2  greedy vs Expected Improvement, per target and per surrogate.
  RQ3  sensitivity to the representation: the spread of CEF across the five representations
       that are not trained encoders, per surrogate.

Set COMPARE_OLD_SAUREUS to also print the S. aureus numbers of the uncorrected pool.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
BENCHMARK_CSV = "experiments/al_benchmark.csv"
OLD_SAUREUS_CSV = "experiments/al_benchmark_parts/old_label/saureus.csv"
COMPARE_OLD_SAUREUS = True
TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
RQ3_REPRESENTATIONS = ["morgan", "physchem", "mordred3d", "chemeleon", "chemprop_untrained"]
KEY = ["target", "representation", "strategy", "init_size", "batch_size"]


def paired(x: pd.Series) -> str:
    x = x.dropna()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return f"{x.mean():+.2f} ± {se:.2f}  ({(x > 0).sum()}/{len(x)})"


def final_round(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["round"] == df["round"].max()]


def rq1(f: pd.DataFrame) -> None:
    p = f.pivot_table(index=KEY, columns="model", values="mean_cef")
    print("RQ1 — TabPFN minus RF / minus GP (final CEF; mean ± SE, pairs won):")
    for t in TARGETS:
        q = p.xs(t, level="target")
        weakest = (q.idxmin(axis=1) == "tabpfn").sum()
        print(f"  {t:8s} vs RF {paired(q.tabpfn - q.rf):26s} vs GP {paired(q.tabpfn - q.gp):26s}"
              f" TabPFN weakest in {weakest}/{len(q)}")
    print(f"  {'all':8s} vs RF {paired(p.tabpfn - p.rf):26s} vs GP {paired(p.tabpfn - p.gp):26s}")


def rq1b(df: pd.DataFrame) -> None:
    print("\nRQ1b — TabPFN minus RF at a matched budget, smaller vs larger batch "
          "(mean over representations and strategies):")
    wins = n = 0
    for t in TARGETS:
        d = df[df.target == t]
        small, large = sorted(d.batch_size.unique())
        for init in sorted(d.init_size.unique()):
            r_large = 20 * small // large                      # round at which large has spent the same
            a = d[(d.init_size == init) & (d.batch_size == small) & (d["round"] == 20)]
            b = d[(d.init_size == init) & (d.batch_size == large) & (d["round"] == r_large)]
            ma = a.pivot_table(index=["representation", "strategy"], columns="model", values="mean_cef")
            mb = b.pivot_table(index=["representation", "strategy"], columns="model", values="mean_cef")
            gs, gl = (ma.tabpfn - ma.rf).mean(), (mb.tabpfn - mb.rf).mean()
            budget = int(a.budget.iloc[0])
            n += 1; wins += gs > gl
            print(f"  {t:8s} init {init:>3} budget {budget:>4}: batch {small:>2} {gs:+.2f}  vs  batch {large:>2} "
                  f"(round {r_large:>2}) {gl:+.2f}   {'smaller batch larger margin' if gs > gl else 'NOT'}")
    print(f"  -> the smaller batch gives TabPFN the larger margin in {wins} of {n} comparisons")


def rq2(f: pd.DataFrame) -> None:
    k = [c for c in KEY if c != "strategy"] + ["model"]
    p = f.pivot_table(index=k, columns="strategy", values="mean_cef")
    d = p["greedy"] - p["ei"]
    print("\nRQ2 — greedy minus EI (final CEF):")
    for t in TARGETS:
        print(f"  {t:8s} {paired(d.xs(t, level='target'))}")
    for m in ["tabpfn", "rf", "gp"]:
        print(f"  {m:8s} {paired(d.xs(m, level='model'))}")
    print(f"  {'all':8s} {paired(d)}")


def rq3(f: pd.DataFrame) -> None:
    g = f[f.representation.isin(RQ3_REPRESENTATIONS)]
    k = ["target", "model", "strategy", "init_size", "batch_size"]
    spread = g.groupby(k).mean_cef.agg(lambda x: x.max() - x.min()).unstack("model")
    print("\nRQ3 — spread of final CEF across the five representations (max − min), mean per surrogate:")
    for t in TARGETS:
        s = spread.xs(t, level="target").mean()
        print(f"  {t:8s} TabPFN {s.tabpfn:.2f}   RF {s.rf:.2f}   GP {s.gp:.2f}")
    s = spread.mean()
    print(f"  {'all':8s} TabPFN {s.tabpfn:.2f}   RF {s.rf:.2f}   GP {s.gp:.2f}")
    p = g.pivot_table(index=KEY, columns="model", values="mean_cef")
    m = p.xs("mordred3d", level="representation")
    print(f"  Mordred: GP minus TabPFN {paired(m.gp - m.tabpfn)}; GP final CEF by target "
          + ", ".join(f"{t} {m.xs(t, level='target').gp.mean():.2f}" for t in TARGETS))
    c = p.xs("chemeleon", level="representation")
    print(f"  Mordred vs CheMeleon under TabPFN, per target: "
          + ", ".join(f"{t} {paired(m.xs(t, level='target').tabpfn - c.xs(t, level='target').tabpfn)}" for t in TARGETS))


def main() -> None:
    df = pd.read_csv(REPO_ROOT / BENCHMARK_CSV)
    print(f"{BENCHMARK_CSV}: {len(df):,} rows; S. aureus base rate {df[df.target == 'saureus'].base_rate.iloc[0]:.4f}\n")
    f = final_round(df)
    rq1(f); rq1b(df); rq2(f); rq3(f)
    if COMPARE_OLD_SAUREUS and (REPO_ROOT / OLD_SAUREUS_CSV).exists():
        old = pd.read_csv(REPO_ROOT / OLD_SAUREUS_CSV)
        fo = final_round(old)
        print("\n=== S. aureus on the UNCORRECTED pool, for comparison ===")
        global TARGETS
        keep, TARGETS = TARGETS, ["saureus"]
        rq1(fo); rq1b(old); rq2(fo); rq3(fo)
        TARGETS = keep


if __name__ == "__main__":
    main()
