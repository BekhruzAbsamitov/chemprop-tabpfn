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


def cef(cumulative_hits: float, total_sampled: int, global_hit_rate: float) -> float:
    """Cumulative Enrichment Factor: the model's hit rate relative to random.

    CEF = (cumulative_hits / total_sampled) / global_hit_rate. 1.0 = no better than
    picking at random; > 1 = enriched (found hits faster than the base rate). This is
    the metric the chempfn AL benchmark reports, so the arms here are comparable to it.
    """
    if total_sampled == 0 or global_hit_rate <= 0:
        return float("nan")
    return (cumulative_hits / total_sampled) / global_hit_rate


def cumulative_hits_from_cef(cef_value: float, total_sampled: int,
                             global_hit_rate: float) -> float:
    """Exact inverse of `cef` — recover cumulative hits from a reported CEF.

    Benchmark runs before 2026-08-11 stored only `mean_new_hits` (hits the model
    acquired) and `mean_cef`. Because CEF is computed from the CUMULATIVE hit count —
    which also includes whatever hits landed in the initial random draw — the
    cumulative count is recoverable rather than lost:

        cef = (cumulative / total_sampled) / global_hit_rate
        =>  cumulative = cef * total_sampled * global_hit_rate

    The two counts differ by roughly `init * global_hit_rate` (~5 hits at init=100 and
    a 5 % base rate), so they are not interchangeable. Newer runs store both directly.
    """
    return cef_value * total_sampled * global_hit_rate


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
