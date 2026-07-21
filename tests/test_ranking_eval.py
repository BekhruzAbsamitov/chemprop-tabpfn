"""Tests for src/ranking_eval.py — the static-ranking split + scoring logic.

Uses a dummy head (no TabPFN), so these are fast.
"""

from __future__ import annotations

import random

import pytest
import torch

import ranking_eval


def test_make_splits_reproducible_sizes_and_disjoint():
    a = ranking_eval.make_splits(100, random.Random(0), n_context=30, n_query=30, n_samples=3)
    b = ranking_eval.make_splits(100, random.Random(0), n_context=30, n_query=30, n_samples=3)
    assert a == b  # deterministic for a fixed rng seed
    assert len(a) == 3
    for ctx, qry in a:
        assert len(ctx) == 30 and len(qry) == 30
        assert set(ctx).isdisjoint(qry)  # no molecule in both context and query


class _DummyHead:
    """Stand-in for TabPFNHead: predicts the query's first feature column."""

    device = "cpu"

    def predict(self, x_context, y_context, x_query):
        return x_query[:, 0]


def test_evaluate_arm_perfect_model():
    """Molecule 'i' has activity i; featurize -> [[i]]; dummy head returns i ->
    perfect predictions -> Spearman/R2 == 1 on every split."""
    smiles = [str(i) for i in range(100)]
    y = torch.arange(100.0)
    splits = ranking_eval.make_splits(100, random.Random(0), n_context=30, n_query=30, n_samples=3)

    def featurize(smis):
        return torch.tensor([[float(s)] for s in smis])

    rhos, r2s = ranking_eval._evaluate_arm(featurize, _DummyHead(), smiles, y, splits)
    assert len(rhos) == 3
    assert all(r == pytest.approx(1.0) for r in rhos)
    assert all(r == pytest.approx(1.0) for r in r2s)
