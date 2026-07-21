"""Tests for scripts/run_experiment.py.

Two tiers (mirrors tests/test_tabpfn_head.py):
  * Fast tests — pure helpers (score, _stable_seed, build_val_episodes); no TabPFN.
  * Integration — a tiny end-to-end TRAINING smoke (EVAL_ON_TARGET=False) that
    needs the real TabPFN checkpoint; skips cleanly if it/HF_TOKEN is unavailable.
"""

from __future__ import annotations

import random
import zlib

import pytest
import torch

from data_utils import data
from scripts import run_experiment as exp


# --------------------------------------------------------------------------- #
# Fast tests (no TabPFN model needed)                                         #
# --------------------------------------------------------------------------- #
def test_score_perfect_prediction():
    """Identical preds/truth -> Spearman 1.0 and R2 1.0."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rho, r2 = exp.score(x, x)
    assert rho == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_score_monotonic_but_scaled():
    """Perfect ranking but wrong scale -> Spearman 1.0, R2 < 1 (calibration off)."""
    preds = torch.tensor([10.0, 20.0, 30.0])
    truth = torch.tensor([1.0, 2.0, 3.0])
    rho, r2 = exp.score(preds, truth)
    assert rho == pytest.approx(1.0)
    assert r2 < 1.0


def test_stable_seed_is_deterministic_and_process_independent():
    """crc32-based seed: same across processes (unlike Python's hash())."""
    assert exp._stable_seed("CHEMBL1_Ki_nM_9") == exp._stable_seed("CHEMBL1_Ki_nM_9")
    assert exp._stable_seed("a") != exp._stable_seed("b")
    # A fixed, known value proves it is NOT the randomized built-in hash().
    assert exp._stable_seed("abc") == (zlib.crc32(b"abc") & 0xFFFFFFFF)


def test_build_val_episodes_reproducible_min_context(monkeypatch):
    """Fixed val set: reproducible across calls, and every context has >= 5 mols."""
    monkeypatch.setattr(exp, "N_VAL_EPISODES", 5)
    monkeypatch.setattr(exp, "MAX_TRAIN_CONTEXT", 16)
    monkeypatch.setattr(exp, "MAX_TRAIN_QUERY", 16)

    test_assays = data.load_chembl_assays("test", use_full=False)
    a = exp.build_val_episodes(test_assays, random.Random(0))
    b = exp.build_val_episodes(test_assays, random.Random(0))

    assert 0 < len(a) <= 5
    # Deterministic seeding -> byte-identical episodes on a repeat call.
    assert [e.context_smiles for e in a] == [e.context_smiles for e in b]
    assert [e.query_smiles for e in a] == [e.query_smiles for e in b]
    for e in a:
        assert len(e.context_smiles) >= 5          # min-context floor
        assert len(e.query_smiles) >= 1
        assert len(e.context_smiles) <= 16         # cap respected
        assert e.context_y.numel() == len(e.context_smiles)


# --------------------------------------------------------------------------- #
# Integration: a tiny end-to-end TRAINING smoke (needs the TabPFN checkpoint)  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tabpfn_available():
    exp._load_hf_token()
    try:
        exp.TabPFNHead(n_estimators=1, loss_fn="nll")
    except Exception as e:  # noqa: BLE001 — any load/network/token failure => skip
        pytest.skip(f"TabPFN checkpoint unavailable: {e}")


def test_train_only_smoke_writes_curve_and_checkpoint(tmp_path, monkeypatch, tabpfn_available):
    """EVAL_ON_TARGET=False path: trains a few steps on dev, logs the in-domain
    val curve, and checkpoints the encoder — without touching S. aureus/Morgan."""
    monkeypatch.setattr(exp, "USE_FULL_DATA", False)
    monkeypatch.setattr(exp, "EVAL_ON_TARGET", False)
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

    # Val curve written with the right header + baseline + at least one eval row.
    assert curve.exists()
    rows = curve.read_text().strip().splitlines()
    assert rows[0] == "step,train_loss,val_loss,val_spearman,seconds"
    assert len(rows) >= 3  # header + step-0 baseline + step-3/6 evals
    # val_spearman column stays in [-1, 1].
    for line in rows[1:]:
        rho = float(line.split(",")[3])
        assert -1.0 <= rho <= 1.0

    # Encoder checkpoint saved and reloadable with its width.
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["hidden_size"] == 32
    assert "state_dict" in payload


def test_evaluate_val_returns_finite_metrics(tmp_path, monkeypatch, tabpfn_available):
    """evaluate_val yields a finite mean NLL and a Spearman in [-1, 1]."""
    monkeypatch.setattr(exp, "N_VAL_EPISODES", 3)
    monkeypatch.setattr(exp, "MAX_TRAIN_CONTEXT", 16)
    monkeypatch.setattr(exp, "MAX_TRAIN_QUERY", 16)

    import math

    head = exp.TabPFNHead(n_estimators=1, loss_fn="nll")
    encoder = exp.MoleculeEncoder(hidden_size=32).to(head.device).eval()
    test_assays = data.load_chembl_assays("test", use_full=False)
    val = exp.build_val_episodes(test_assays, random.Random(0))

    loss, rho = exp.evaluate_val(encoder, head, val)
    assert math.isfinite(loss)
    assert -1.0 <= rho <= 1.0
