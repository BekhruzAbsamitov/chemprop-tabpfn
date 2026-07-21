"""Tests for src/models/morgan.py — the Morgan-fingerprint baseline."""

from __future__ import annotations

import torch

from models.morgan import morgan_features


def test_shape_matches_n_bits():
    fp = morgan_features(["CCO", "c1ccccc1"], n_bits=512)
    assert fp.shape == (2, 512)


def test_values_are_binary():
    fp = morgan_features(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    assert set(fp.unique().tolist()) <= {0.0, 1.0}


def test_deterministic():
    assert torch.equal(morgan_features(["CCO"]), morgan_features(["CCO"]))


def test_real_molecule_sets_some_bits():
    assert morgan_features(["CCO"]).sum() > 0


def test_unparseable_smiles_gives_zero_row():
    fp = morgan_features(["this_is_not_a_molecule"])
    assert fp.shape[0] == 1
    assert fp.sum() == 0


def test_different_molecules_have_different_fingerprints():
    fp = morgan_features(["CCO", "c1ccccc1"])
    assert not torch.equal(fp[0], fp[1])
