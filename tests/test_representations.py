"""Tests for the static molecular representations — Morgan, physchem, Mordred 3D.

These are the classical baselines the learned encoders are compared against, so the
contract that matters is shared: one row per input SMILES, all-zero for anything
unparseable, and finite everywhere (TabPFN's preprocessing indexes on feature values,
so a NaN crashes it rather than degrading gracefully).
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from rdkit import RDLogger

from models.mordred3d import mordred3d_features
from models.morgan import morgan_features
from models.physchem import physchem_features

RDLogger.DisableLog("rdApp.*")  # unparseable SMILES are deliberate here

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
WITH_BAD = SMILES + ["not-a-molecule"]

FEATURIZERS = [
    pytest.param(morgan_features, 2048, id="morgan"),
    pytest.param(physchem_features, 200, id="physchem"),
    pytest.param(lambda s: mordred3d_features(s, cache_dir=None, verbose=False),
                 1826, id="mordred3d"),
]


@pytest.mark.parametrize("featurize,width", FEATURIZERS)
def test_shape_is_one_row_per_molecule(featurize, width):
    x = featurize(SMILES)
    assert x.shape == (len(SMILES), width)
    assert x.dtype == torch.float32


@pytest.mark.parametrize("featurize,width", FEATURIZERS)
def test_unparseable_smiles_becomes_a_zero_row(featurize, width):
    """A bad SMILES must not shift or drop rows — downstream code indexes by position."""
    x = featurize(WITH_BAD)
    assert x.shape[0] == len(WITH_BAD)
    assert bool((x[-1] == 0).all())


@pytest.mark.parametrize("featurize,width", FEATURIZERS)
def test_all_values_finite(featurize, width):
    """NaN/Inf reach TabPFN as out-of-bounds indices and crash it — never emit them."""
    assert bool(torch.isfinite(featurize(WITH_BAD)).all())


@pytest.mark.parametrize("featurize,width", FEATURIZERS)
def test_deterministic(featurize, width):
    """Mordred embeds a conformer, so it is only reproducible via its fixed seed."""
    assert torch.equal(featurize(SMILES), featurize(SMILES))


def test_physchem_normalized_lies_in_unit_interval():
    x = physchem_features(SMILES, normalized=True)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_mordred_cache_round_trips_and_keys_on_the_smiles_list():
    with tempfile.TemporaryDirectory() as d:
        a = mordred3d_features(SMILES, cache_dir=d, verbose=False)
        b = mordred3d_features(SMILES, cache_dir=d, verbose=False)   # cache hit
        assert torch.equal(a, b)
        # A different list must not collide with the cached entry.
        c = mordred3d_features(SMILES[:2], cache_dir=d, verbose=False)
        assert c.shape[0] == 2


# --------------------------------------------------------------------------- #
# The supervisor's pretrained RIGR encoder (2026-08-20). Not part of the shared  #
# contract above: like the other NEURAL arms it raises on unparseable SMILES     #
# rather than emitting a zero row, so it gets its own checks.                    #
# --------------------------------------------------------------------------- #
from models.rigr_encoder import N_HIDDEN, RIGREncoder  # noqa: E402


def test_rigr_pretrained_weights_load_and_shape():
    """If the checkpoint is present it must load STRICTLY — a silent fallback to random
    init would benchmark the wrong thing under the name 'pretrained'."""
    enc = RIGREncoder(verbose=False)
    if not enc.pretrained:
        pytest.skip("checkpoint data/weights/chemicl_mpnn.pt not present")
    x = enc(SMILES)
    assert x.shape == (len(SMILES), N_HIDDEN)
    assert bool(torch.isfinite(x).all())


def test_rigr_embedding_is_the_full_forward_pass():
    """n_tasks=N_HIDDEN means the predictor's output IS the embedding. `.fingerprint()`
    would return the N_HIDDEN//2 intermediate instead — the easy, silent mistake."""
    enc = RIGREncoder(weights_path=None, verbose=False)
    assert enc(SMILES).shape[1] == N_HIDDEN
    assert N_HIDDEN // 2 != N_HIDDEN  # guards the test itself against a config change


def test_rigr_is_resonance_invariant():
    """The point of RIGR: benzene written aromatic vs Kekule must embed identically."""
    enc = RIGREncoder(weights_path=None, verbose=False)
    pair = enc(["c1ccccc1", "C1=CC=CC=C1"])
    assert torch.allclose(pair[0], pair[1], atol=1e-6)


def test_rigr_is_frozen():
    """It is a fixed yardstick like Chemeleon — nothing here may collect gradients."""
    enc = RIGREncoder(weights_path=None, verbose=False)
    assert not any(p.requires_grad for p in enc.parameters())
    assert not enc.training


def test_rigr_missing_checkpoint_does_not_masquerade_as_pretrained():
    """A wrong path must fall back visibly, never report itself as pretrained."""
    enc = RIGREncoder(weights_path="data/weights/does_not_exist.pt", verbose=False)
    assert enc.pretrained is False
