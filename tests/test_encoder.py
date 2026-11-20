"""Tests for src/models/encoder.py — the Chemprop encoder (needs chemprop, no network)."""

from __future__ import annotations

import pytest
import torch

from models.encoder import MoleculeEncoder


@pytest.fixture(scope="module")
def encoder():
    return MoleculeEncoder(hidden_size=64, depth=2)


def test_output_shape(encoder):
    out = encoder(["CCO", "c1ccccc1", "CCN"])
    assert out.shape == (3, encoder.output_dim)


def test_output_dim_equals_hidden_size(encoder):
    assert encoder.output_dim == 64


def test_same_molecule_gives_same_embedding(encoder):
    encoder.eval()
    assert torch.allclose(encoder(["CCO"]), encoder(["CCO"]))


def test_different_molecules_give_different_embeddings(encoder):
    encoder.eval()
    out = encoder(["CCO", "c1ccccc1"])
    assert not torch.allclose(out[0], out[1])


def test_embedding_is_differentiable(encoder):
    out = encoder(["CCO", "CCN"])
    assert out.requires_grad
    out.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())


def test_empty_input_raises(encoder):
    with pytest.raises(ValueError):
        encoder([])


def test_layernorm_output_is_roughly_standardized(encoder):
    # normalize=True applies LayerNorm, so each row should be ~zero-mean / unit-scale
    out = encoder(["CC(=O)Oc1ccccc1C(=O)O"]).detach()
    assert abs(float(out.mean())) < 0.5
    assert 0.5 < float(out.std()) < 2.0
