"""Tests for src/metrics.py — score, mean_std, oracle_efficiency.

Pure functions on tensors / lists, so no model or data is needed.
"""

from __future__ import annotations

import pytest
import torch

from metrics import mean_std, oracle_efficiency, score


# --------------------------------------------------------------------------- #
# score                                                                       #
# --------------------------------------------------------------------------- #
def test_score_perfect_prediction():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rho, r2 = score(p, p.clone())
    assert rho == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_score_reversed_ranking_is_minus_one():
    p = torch.tensor([4.0, 3.0, 2.0, 1.0])
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rho, _ = score(p, t)
    assert rho == pytest.approx(-1.0)


def test_score_predicting_the_mean_gives_zero_r2():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = torch.full((4,), float(t.mean()))
    _, r2 = score(p, t)
    assert r2 == pytest.approx(0.0, abs=1e-6)


def test_score_r2_hand_computed():
    # t=[1,2,3], p=[1,2,5]: ss_res=4, ss_tot=2 -> r2 = 1 - 4/2 = -1
    t = torch.tensor([1.0, 2.0, 3.0])
    p = torch.tensor([1.0, 2.0, 5.0])
    _, r2 = score(p, t)
    assert r2 == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# mean_std                                                                    #
# --------------------------------------------------------------------------- #
def test_mean_std_basic():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert s == pytest.approx((2.0 / 3.0) ** 0.5)  # population std


def test_mean_std_drops_nans():
    m, _ = mean_std([1.0, float("nan"), 3.0])
    assert m == pytest.approx(2.0)


def test_mean_std_all_nan_returns_nan():
    m, s = mean_std([float("nan")])
    assert m != m and s != s  # both NaN


# --------------------------------------------------------------------------- #
# oracle_efficiency                                                           #
# --------------------------------------------------------------------------- #
def test_oracle_efficiency_perfect_is_one():
    # If every label spent finds a hit (hits == budget), efficiency == 1.
    eff = oracle_efficiency(mean_hits=[10.0, 20.0, 30.0], budgets=[10, 20, 30], total_hits=100)
    assert eff == pytest.approx(1.0)


def test_oracle_efficiency_zero_hits_is_zero():
    eff = oracle_efficiency(mean_hits=[0.0, 0.0], budgets=[10, 20], total_hits=100)
    assert eff == pytest.approx(0.0)


def test_oracle_efficiency_no_hits_in_pool_is_nan():
    eff = oracle_efficiency(mean_hits=[0.0], budgets=[10], total_hits=0)
    assert eff != eff  # NaN
