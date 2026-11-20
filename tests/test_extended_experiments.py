"""Tests for the follow-up experiments of the 2026-09-29 supervision (E1-E7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "src" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import benchmark_extended as be  # noqa: E402
import calibration_study as cs  # noqa: E402
from models.baselines import GaussianProcessTanimotoHead, TanimotoKernel  # noqa: E402


def test_tanimoto_kernel_matches_its_definition():
    a = np.array([[1, 1, 0, 0]])
    b = np.array([[1, 0, 1, 0]])
    k = TanimotoKernel()
    assert k(a, b)[0, 0] == pytest.approx(1 / 3)          # 1 shared bit of 3 set in total
    assert k(a)[0, 0] == pytest.approx(1.0)
    assert np.allclose(k.diag(np.vstack([a, b])), 1.0)
    assert k(np.zeros((1, 4)))[0, 0] == pytest.approx(1.0)  # empty fingerprints: defined as 1


def test_tanimoto_gp_returns_mean_and_positive_std():
    rng = np.random.default_rng(0)
    X = (rng.random((50, 64)) < 0.2).astype("float32")
    y = X[:, :5].sum(1).astype("float32")
    mu, sd = GaussianProcessTanimotoHead().predict_dist(
        torch.tensor(X[:40]), torch.tensor(y[:40]), torch.tensor(X[40:]))
    assert mu.shape == sd.shape == (10,)
    assert bool((sd > 0).all())


@pytest.mark.parametrize("model,rep,strategy,ok", [
    ("gp_tanimoto", "morgan", be.GREEDY, True),
    ("gp_tanimoto", "chemeleon", be.GREEDY, False),          # Tanimoto needs binary bits
    ("tabpfn", "chemeleon", ("ei", "yplus", "full"), True),
    ("rf", "chemeleon", ("ei", "yplus", "full"), False),     # no full posterior for RF
    ("gp", "morgan", ("pi", "ythresh", "gauss"), True),
])
def test_arm_validity_rules(model, rep, strategy, ok):
    assert be.arm_is_valid(model, rep, strategy) is ok


def test_hit_threshold_is_the_weakest_hit():
    y = torch.tensor([1.0, 5.0, 3.0, 9.0, 7.0])
    assert be.hit_threshold({"hit_top_frac": 0.4}, y, {3, 4}) == pytest.approx(7.0)
    assert be.hit_threshold({"hit_threshold": 2.41}, y, {3}) == pytest.approx(2.41)


def test_campaign_loop_with_a_perfect_oracle_model():
    """A model that 'predicts' the true values must acquire hits greedily and log the
    mean activity of what it acquired."""
    n = 300
    y = torch.linspace(0, 1, n)
    X = y[:, None].repeat(1, 3)
    hit_set = set(range(n - 15, n))                          # top 5 %

    class Oracle:
        def predict_dist(self, xc, yc, xq):
            return xq[:, 0].clone(), torch.full((xq.shape[0],), 0.1)

    be.N_ROUNDS = 3
    rows = be.run_campaign(Oracle(), be.GREEDY, X, y, hit_set, y_thresh=float(y[n - 15]),
                           init=10, batch=5, seed=0, n_cand=0, base_rate=15 / n)
    assert len(rows) == 3
    assert rows[-1]["cum_hits"] >= 15 - 0                    # all 15 acquired in 3 rounds
    assert float(rows[-1]["mean_y_acquired"]) > float(y[n - 16])
    for strategy in [("ei", "yplus", "gauss"), ("pi", "ythresh", "gauss")]:
        rows = be.run_campaign(Oracle(), strategy, X, y, hit_set, float(y[n - 15]),
                               10, 5, 0, 0, 15 / n)
        assert rows[-1]["budget"] == 10 + 3 * 5


def test_ece_is_zero_for_a_calibrated_gaussian_and_signed_for_overconfidence():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=40000)
    sd = np.full_like(mu, 0.5)
    y = mu + sd * rng.normal(size=mu.shape)
    calibrated = cs.gaussian_rows(y, mu, sd)
    over = cs.gaussian_rows(y, mu, sd / 2)
    assert calibrated["ece"] < 0.01
    assert over["ece"] > 0.15 and over["signed_calibration_error"] < 0     # too narrow
    under = cs.gaussian_rows(y, mu, sd * 2)
    assert under["signed_calibration_error"] > 0                           # too wide


def test_nll_summaries_survive_a_single_catastrophic_molecule():
    """One molecule with a tiny sigma and a miss must not make the robust NLLs explode."""
    rng = np.random.default_rng(2)
    mu = rng.normal(size=1000)
    sd = np.full_like(mu, 0.5)
    y = mu + sd * rng.normal(size=mu.shape)
    sd[0], y[0] = 1e-6, mu[0] + 1.0                      # a spike prediction that misses
    r = cs.gaussian_rows(y, mu, sd)
    assert r["nll"] > 100                                 # the plain mean is dominated by it
    assert r["nll_median"] < 1.5
    assert r["nll_capped"] < 1.5
