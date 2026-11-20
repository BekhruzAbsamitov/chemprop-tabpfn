"""scripts/summarise_transfer.py — do the trained encoders beat their own random initialisation? Press Run.

RQ4's transfer question (thesis Chapter 5.4). For every benchmark run, each trained Chemprop
encoder is compared with the random-initialisation Chemprop arm OF THE SAME RUN (identical
architecture, seed and starting weights), as paired final-round differences in CEF over
matched AL experiments: same target, surrogate, strategy, init size and batch size. CheMeleon,
present in every run, is reported as the strong reference.

Runs are never paired with each other (Methodology, reproducibility section): each run is
analysed only against its own control. S. aureus uses the corrected pool in every run
(re-run 2026-10-07); COMPARE_OLD_SAUREUS also prints the uncorrected S. aureus rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# SETTINGS                                                                    #
# --------------------------------------------------------------------------- #
# run name -> (merged CSV, old S. aureus part, trained arms to test)
RUNS = {
    "primary": ("experiments/al_benchmark.csv", "experiments/al_benchmark_parts/old_label/saureus.csv",
                ["chemprop_trained_full", "chemprop_trained_own", "chemprop_trained_own_long"]),
    "fullsp (100k steps)": ("experiments/al_benchmark_fullsp.csv",
                            "experiments/al_benchmark_parts_fullsp/old_label/saureus.csv",
                            ["chemprop_trained_full_sp", "chemprop_trained_full_sp_best"]),
    "10ep": ("experiments/al_benchmark_10ep.csv", "experiments/al_benchmark_parts_10ep/old_label/saureus.csv",
             ["chemprop_trained_full_sp_10ep", "chemprop_trained_full_sp_10ep_best", "chemprop_trained_full_sp"]),
    "50ep": ("experiments/al_benchmark_50ep.csv", "experiments/al_benchmark_parts_50ep/old_label/saureus.csv",
             ["chemprop_trained_full_sp_50ep_best", "chemprop_trained_full_sp_50ep_final",
              "chemprop_trained_full_sp_10ep"]),
}
CONTROL = "chemprop_untrained"
REFERENCE = "chemeleon"
COMPARE_OLD_SAUREUS = True
TARGETS = ["saureus", "tyk2", "usp7", "d2r"]
KEY = ["target", "model", "strategy", "init_size", "batch_size"]


def fmt(x: pd.Series) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "—"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    flag = "*" if abs(x.mean()) > 2 * se else " "
    return f"{x.mean():+.2f} ± {se:.2f}{flag}({(x > 0).sum()}/{len(x)})"


def normalise_own(df: pd.DataFrame) -> pd.DataFrame:
    """chemprop_trained_own_<target>[_long] -> chemprop_trained_own[_long], so targets pool."""
    df = df.copy()
    for t in TARGETS:
        df["representation"] = df["representation"].replace(
            {f"chemprop_trained_own_{t}": "chemprop_trained_own",
             f"chemprop_trained_own_{t}_long": "chemprop_trained_own_long"})
    return df


def summarise(name: str, df: pd.DataFrame, arms: list[str]) -> None:
    f = normalise_own(df[df["round"] == df["round"].max()])
    p = f.pivot_table(index=KEY, columns="representation", values="mean_cef")
    print(f"\n=== {name} ===   (* = more than 2 SE from zero; pairs won in brackets)")
    print(f"  {'arm minus random init':38s} " + " ".join(f"{t:>20s}" for t in TARGETS) + f" {'all':>20s}")
    for arm in arms + [REFERENCE]:
        if arm not in p:
            continue
        d = p[arm] - p[CONTROL]
        cells = [fmt(d.xs(t, level="target")) if t in d.index.get_level_values("target") else "—" for t in TARGETS]
        print(f"  {arm:38s} " + " ".join(f"{c:>20s}" for c in cells) + f" {fmt(d):>20s}")


def main() -> None:
    for name, (path, old, arms) in RUNS.items():
        df = pd.read_csv(REPO_ROOT / path)
        summarise(name, df, arms)
        if COMPARE_OLD_SAUREUS and (REPO_ROOT / old).exists():
            o = pd.read_csv(REPO_ROOT / old)
            summarise(f"{name}: S. aureus on the UNCORRECTED pool", o, arms)


if __name__ == "__main__":
    main()
