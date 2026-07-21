"""Tests for the training pipeline (src/training.py + scripts/run_experiment.py).

Two tiers (mirrors tests/test_tabpfn_head.py):
  * Fast tests — pure helpers (stable_seed, build_val_episodes); no TabPFN.
  * Integration — a tiny end-to-end TRAINING smoke that needs the real TabPFN
    checkpoint; skips cleanly if it/HF_TOKEN is unavailable.
"""

from __future__ import annotations

import math
import random
import zlib

import pytest

from data_utils import data
from scripts import run_experiment as exp
import training


# --------------------------------------------------------------------------- #
# Fast tests (no TabPFN model needed)                                         #
# --------------------------------------------------------------------------- #
def test_stable_seed_is_deterministic_and_process_independent():
    """crc32-based seed: same across processes (unlike Python's hash())."""
    assert training.stable_seed("CHEMBL1_Ki_nM_9") == training.stable_seed("CHEMBL1_Ki_nM_9")
    assert training.stable_seed("a") != training.stable_seed("b")
    assert training.stable_seed("abc") == (zlib.crc32(b"abc") & 0xFFFFFFFF)


def test_build_val_episodes_reproducible_min_context():
    """Fixed val set: reproducible across calls, every context has >= 5 mols."""
    test_assays = data.load_chembl_assays("test", use_full=False)
    kw = dict(n_episodes=5, max_context=16, max_query=16)
    a = training.build_val_episodes(test_assays, rng=random.Random(0), **kw)
    b = training.build_val_episodes(test_assays, rng=random.Random(0), **kw)

    assert 0 < len(a) <= 5
    assert [e.context_smiles for e in a] == [e.context_smiles for e in b]  # deterministic
    for e in a:
        assert len(e.context_smiles) >= 5      # min-context floor
        assert len(e.context_smiles) <= 16     # cap respected
        assert e.context_y.numel() == len(e.context_smiles)


# --------------------------------------------------------------------------- #
# Integration: tiny end-to-end training smoke (needs the TabPFN checkpoint)    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tabpfn_available():
    exp._load_hf_token()
    try:
        exp.TabPFNHead(n_estimators=1, loss_fn="nll")
    except Exception as e:  # noqa: BLE001 — any load/network/token failure => skip
        pytest.skip(f"TabPFN checkpoint unavailable: {e}")


def test_train_smoke_writes_curve_and_checkpoint(tmp_path, monkeypatch, tabpfn_available):
    """run_experiment.main(): trains a few steps on dev, logs the in-domain val
    curve, and checkpoints the encoder."""
    monkeypatch.setattr(exp, "USE_FULL_DATA", False)
    monkeypatch.setattr(exp, "HIDDEN_SIZE", 32)
    monkeypatch.setattr(exp, "N_ESTIMATORS", 1)
    monkeypatch.setattr(exp, "N_TRAIN_STEPS", 6)
    monkeypatch.setattr(exp, "PRINT_EVERY", 3)
    monkeypatch.setattr(exp, "EVAL_EVERY", 3)
    monkeypatch.setattr(exp, "N_VAL_EPISODES", 3)

    curve = tmp_path / "curve.csv"
    ckpt = tmp_path / "encoder.pt"
    monkeypatch.setattr(exp, "VAL_CURVE_CSV", str(curve))
    monkeypatch.setattr(exp, "SAVE_ENCODER_TO", str(ckpt))

    exp.main()

    assert curve.exists()
    rows = curve.read_text().strip().splitlines()
    assert rows[0] == "step,train_loss,val_loss,val_spearman,seconds"
    assert len(rows) >= 3  # header + step-0 baseline + eval rows
    for line in rows[1:]:
        rho = float(line.split(",")[3])
        assert -1.0 <= rho <= 1.0

    import torch
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["hidden_size"] == 32
    assert "state_dict" in payload


def test_evaluate_val_returns_finite_metrics(monkeypatch, tabpfn_available):
    """training.evaluate_val yields a finite mean NLL and a Spearman in [-1, 1]."""
    head = exp.TabPFNHead(n_estimators=1, loss_fn="nll")
    encoder = exp.MoleculeEncoder(hidden_size=32).to(head.device).eval()
    test_assays = data.load_chembl_assays("test", use_full=False)
    val = training.build_val_episodes(
        test_assays, n_episodes=3, max_context=16, max_query=16, rng=random.Random(0)
    )
    loss, rho = training.evaluate_val(encoder, head, val)
    assert math.isfinite(loss)
    assert -1.0 <= rho <= 1.0
