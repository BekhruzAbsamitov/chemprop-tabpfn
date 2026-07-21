"""src/models/acquisition.py — active-learning acquisition functions.

In active learning we don't have labels for the whole pool; we can afford to
label a few molecules per round. An ACQUISITION function scores every unlabeled
molecule so we can pick the most useful ones to label next. Each score turns
TabPFN's prediction — a mean `mu` and an uncertainty `sigma` — into one number;
we then label the highest-scoring molecules.

Everything here is for MAXIMISATION: our target is pMIC, and higher pMIC = more
active, so "improvement" means beating the best activity seen so far (`best`).

The strategies:
    * Expected Improvement (EI): how much we expect to beat `best`, averaged over
      the prediction's uncertainty. Balances "high mean" against "high
      uncertainty" automatically — the standard Bayesian-optimisation choice.
    * Probability of Improvement (PI): just the probability of beating `best`.
      Greedier than EI (ignores HOW MUCH we'd improve).
    * Upper Confidence Bound (UCB): mu + kappa * sigma. Explicit knob `kappa`
      trades off exploitation (mean) vs exploration (uncertainty).
    * Greedy: mu alone. Pure exploitation, ignores uncertainty.
    * (Random has no score — the loop just picks at random. It's the floor every
      real strategy must beat.)

These are pure functions of tensors — no model, no data loading — so they are
easy to unit-test against hand-computed values (see tests/test_acquisition.py).
Press Run to see the four strategies rank the same toy molecules differently.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _standard_normal_cdf(z: Tensor) -> Tensor:
    """P(Z <= z) for a standard normal Z, via the error function."""
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def _standard_normal_pdf(z: Tensor) -> Tensor:
    """Standard-normal density at z."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * z * z)


def expected_improvement(mu: Tensor, sigma: Tensor, best: float, xi: float = 0.0) -> Tensor:
    """Expected amount by which each candidate beats `best` (for maximisation).

    EI(x) = (mu - t) * Phi(z) + sigma * phi(z),  where t = best + xi,  z = (mu - t)/sigma.
    `xi` >= 0 demands a minimum improvement margin (more exploration). Where
    sigma == 0 the closed form is undefined, so EI collapses to max(mu - t, 0).
    """
    mu = mu.float()
    sigma = sigma.float()
    t = best + xi
    positive_sigma = sigma > 0
    safe_sigma = torch.where(positive_sigma, sigma, torch.ones_like(sigma))  # avoid /0
    z = (mu - t) / safe_sigma
    ei = (mu - t) * _standard_normal_cdf(z) + safe_sigma * _standard_normal_pdf(z)
    ei = torch.where(positive_sigma, ei, torch.clamp(mu - t, min=0.0))
    return torch.clamp(ei, min=0.0)


def probability_of_improvement(mu: Tensor, sigma: Tensor, best: float, xi: float = 0.0) -> Tensor:
    """Probability each candidate beats `best` (for maximisation).

    PI(x) = Phi((mu - best - xi) / sigma). Where sigma == 0 it is 1 if mu > t else 0.
    """
    mu = mu.float()
    sigma = sigma.float()
    t = best + xi
    positive_sigma = sigma > 0
    safe_sigma = torch.where(positive_sigma, sigma, torch.ones_like(sigma))
    z = (mu - t) / safe_sigma
    pi = _standard_normal_cdf(z)
    return torch.where(positive_sigma, pi, (mu > t).float())


def upper_confidence_bound(mu: Tensor, sigma: Tensor, kappa: float = 1.0) -> Tensor:
    """mu + kappa * sigma — optimism in the face of uncertainty."""
    return mu.float() + kappa * sigma.float()


def acquisition_scores(
    strategy: str,
    mu: Tensor,
    sigma: Tensor,
    best: float,
    *,
    xi: float = 0.0,
    kappa: float = 1.0,
) -> Tensor:
    """Score every candidate under `strategy`; higher score = label sooner.

    Handles the score-based strategies ("ei", "pi", "ucb", "greedy"). "random"
    has no score and is handled by the caller (it just samples uniformly).
    """
    strategy = strategy.lower()
    if strategy == "ei":
        return expected_improvement(mu, sigma, best, xi)
    if strategy == "pi":
        return probability_of_improvement(mu, sigma, best, xi)
    if strategy == "ucb":
        return upper_confidence_bound(mu, sigma, kappa)
    if strategy == "greedy":
        return mu.float()
    raise ValueError(
        f"unknown scored strategy {strategy!r} "
        "(expected 'ei', 'pi', 'ucb', or 'greedy'; 'random' is handled by the loop)"
    )


# --------------------------------------------------------------------------- #
# Press Run to watch the strategies rank the same toy candidates.             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Three candidates: A = high mean/low uncertainty, B = medium/medium,
    # C = low mean but very uncertain. best-so-far = 1.0.
    mu = torch.tensor([1.2, 1.0, 0.6])
    sigma = torch.tensor([0.1, 0.4, 1.5])
    best = 1.0

    for name in ("greedy", "pi", "ei", "ucb"):
        scores = acquisition_scores(name, mu, sigma, best)
        pick = int(scores.argmax())
        print(f"{name:>7s}: scores {scores.numpy().round(3)}  -> picks candidate {'ABC'[pick]}")
