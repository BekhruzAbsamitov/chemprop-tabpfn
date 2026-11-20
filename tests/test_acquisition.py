"""Tests for src/models/acquisition.py — the acquisition MATH, checked against
hand/SciPy-computed values, plus the ranking behaviour each strategy promises."""

from __future__ import annotations

import math

import torch

from models.acquisition import (
    acquisition_scores,
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def test_expected_improvement_matches_closed_form():
    mu = torch.tensor([1.5])
    sigma = torch.tensor([0.5])
    best = 1.0
    z = (1.5 - 1.0) / 0.5  # = 1.0
    expected = (1.5 - 1.0) * _Phi(z) + 0.5 * _phi(z)
    got = expected_improvement(mu, sigma, best).item()
    assert math.isclose(got, expected, rel_tol=1e-5)


def test_expected_improvement_zero_sigma_is_hinge():
    # sigma == 0: EI collapses to max(mu - best, 0)
    mu = torch.tensor([2.0, 0.5])
    sigma = torch.tensor([0.0, 0.0])
    ei = expected_improvement(mu, sigma, best=1.0)
    assert torch.allclose(ei, torch.tensor([1.0, 0.0]))


def test_expected_improvement_is_nonnegative():
    torch.manual_seed(0)
    mu = torch.randn(50)
    sigma = torch.rand(50)  # >= 0
    ei = expected_improvement(mu, sigma, best=0.3, xi=0.05)
    assert (ei >= 0).all()


def test_probability_of_improvement_matches_normal_cdf():
    mu = torch.tensor([1.2])
    sigma = torch.tensor([0.4])
    best = 1.0
    expected = _Phi((1.2 - 1.0) / 0.4)
    got = probability_of_improvement(mu, sigma, best).item()
    assert math.isclose(got, expected, rel_tol=1e-5)


def test_probability_of_improvement_zero_sigma_is_step():
    mu = torch.tensor([1.5, 0.5])
    sigma = torch.tensor([0.0, 0.0])
    pi = probability_of_improvement(mu, sigma, best=1.0)
    assert torch.allclose(pi, torch.tensor([1.0, 0.0]))


def test_xi_makes_ei_and_pi_more_conservative():
    # A larger improvement margin xi lowers both EI and PI (harder to "improve").
    mu, sigma = torch.tensor([1.2]), torch.tensor([0.5])
    assert expected_improvement(mu, sigma, 1.0, xi=0.3).item() < expected_improvement(mu, sigma, 1.0).item()
    assert probability_of_improvement(mu, sigma, 1.0, xi=0.3).item() < probability_of_improvement(mu, sigma, 1.0).item()


def test_ucb_trades_off_mean_and_uncertainty():
    mu = torch.tensor([1.0, 1.0])
    sigma = torch.tensor([0.1, 0.9])
    ucb = upper_confidence_bound(mu, sigma, kappa=2.0)
    assert torch.allclose(ucb, torch.tensor([1.2, 2.8]))
    # with kappa>0 the more-uncertain candidate wins
    assert ucb[1] > ucb[0]


def test_greedy_ignores_uncertainty():
    mu = torch.tensor([0.9, 0.4])
    sigma = torch.tensor([0.01, 5.0])
    scores = acquisition_scores("greedy", mu, sigma, best=0.0)
    assert torch.allclose(scores, mu)  # sigma had no effect


def test_ei_prefers_uncertainty_when_means_tie():
    # Two candidates with equal mean at the frontier: EI should prefer the more
    # uncertain one (more upside). This is the core AL behaviour.
    mu = torch.tensor([1.0, 1.0])
    sigma = torch.tensor([0.2, 0.8])
    ei = acquisition_scores("ei", mu, sigma, best=1.0)
    assert ei[1] > ei[0]


def test_unknown_strategy_raises():
    try:
        acquisition_scores("banana", torch.zeros(3), torch.ones(3), best=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown strategy")
