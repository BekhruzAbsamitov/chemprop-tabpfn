"""src/metrics.py — scoring functions shared across experiments.

Pure functions on tensors / plain lists (no models, no data loading), so they
are trivial to unit-test. Used by the static-ranking eval, the training-time
in-domain curve, and the active-learning loop.
"""

from __future__ import annotations

import torch
from scipy.stats import spearmanr


def score(predictions: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """Return (Spearman, R2). Spearman = ranking quality; R2 = calibration."""
    p = predictions.detach().cpu().numpy()
    t = truth.detach().cpu().numpy()
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - float(((t - p) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else float("nan")
    rho = float(spearmanr(p, t).statistic)
    return rho, r2


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and population standard deviation of a small list (NaNs dropped)."""
    vals = [v for v in values if v == v]  # drop NaNs
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var ** 0.5


def oracle_efficiency(mean_hits: list[float], budgets: list[int], total_hits: int) -> float:
    """Hit-finding efficiency vs the ideal oracle, in [0, 1].

    At a budget of b labels the best possible strategy finds min(b, total_hits)
    hits, so hits / min(b, total_hits) is the "fraction of the oracle" at that
    budget; averaging over the curve gives a single number where 1.0 = oracle and
    ~(hit base rate) = random. Reads like an enrichment / hit-rate.
    """
    if total_hits == 0:
        return float("nan")
    fractions = [h / min(b, total_hits) for h, b in zip(mean_hits, budgets)]
    return sum(fractions) / len(fractions)
