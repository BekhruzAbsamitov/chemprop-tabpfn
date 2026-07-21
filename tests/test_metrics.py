"""Tests for the metric + evaluation logic in src/scripts/run_experiment.py.

`score` and `evaluate_on_target` are the code that turns predictions into the
numbers we report, so their correctness matters most. These use a dummy model,
so no TabPFN is needed.
"""

from __future__ import annotations

import pytest
import torch

from scripts import run_experiment


def test_score_perfect_prediction():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rho, r2 = run_experiment.score(p, p.clone())
    assert rho == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_score_reversed_ranking_is_minus_one():
    p = torch.tensor([4.0, 3.0, 2.0, 1.0])
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rho, _ = run_experiment.score(p, t)
    assert rho == pytest.approx(-1.0)


def test_score_predicting_the_mean_gives_zero_r2():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    p = torch.full((4,), float(t.mean()))
    _, r2 = run_experiment.score(p, t)
    assert r2 == pytest.approx(0.0, abs=1e-6)


def test_score_r2_hand_computed():
    # t=[1,2,3], p=[1,2,5]: ss_res=4, ss_tot=2 -> r2 = 1 - 4/2 = -1
    t = torch.tensor([1.0, 2.0, 3.0])
    p = torch.tensor([1.0, 2.0, 5.0])
    _, r2 = run_experiment.score(p, t)
    assert r2 == pytest.approx(-1.0)


class _DummyHead:
    """Stand-in for TabPFNHead: predicts the query's first feature column."""

    device = "cpu"

    def predict(self, x_context, y_context, x_query):
        return x_query[:, 0]


def test_evaluate_on_target_perfect_model(monkeypatch):
    # small, deterministic settings
    monkeypatch.setattr(run_experiment, "N_CONTEXT", 30)
    monkeypatch.setattr(run_experiment, "N_QUERY", 30)
    monkeypatch.setattr(run_experiment, "N_SPLITS", 3)
    monkeypatch.setattr(run_experiment, "SEED", 0)

    # molecule "i" has activity i; featurize turns it into the number i, and the
    # dummy head returns that number -> perfect predictions -> Spearman/R2 == 1.
    smiles = [str(i) for i in range(100)]
    y = torch.arange(100.0)

    def featurize(smis):
        return torch.tensor([[float(s)] for s in smis])

    rho, r2 = run_experiment.evaluate_on_target(featurize, _DummyHead(), smiles, y)
    assert rho == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_evaluate_on_target_averages_over_all_splits(monkeypatch):
    calls = {"n": 0}

    monkeypatch.setattr(run_experiment, "N_CONTEXT", 10)
    monkeypatch.setattr(run_experiment, "N_QUERY", 10)
    monkeypatch.setattr(run_experiment, "N_SPLITS", 4)
    monkeypatch.setattr(run_experiment, "SEED", 0)

    smiles = [str(i) for i in range(50)]
    y = torch.arange(50.0)

    def featurize(smis):
        calls["n"] += 1
        return torch.tensor([[float(s)] for s in smis])

    run_experiment.evaluate_on_target(featurize, _DummyHead(), smiles, y)
    # featurize is called twice per split (context + query) across 4 splits
    assert calls["n"] == 8
