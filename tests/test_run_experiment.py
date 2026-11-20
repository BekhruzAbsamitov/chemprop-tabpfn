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
from pathlib import Path

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


def test_best_val_checkpoint_captures_the_peak_not_the_final_step(tmp_path, monkeypatch):
    """A run that peaks mid-way and decays must keep the PEAK weights in *_best.pt.

    Stubs the encoder/head (no TabPFN) and scripts val_spearman as
    0.1 -> 0.5 (peak) -> 0.2, so the best step is unambiguous.
    """
    import torch

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4))

        def forward(self, smiles):
            return self.w.expand(len(smiles), 4) + torch.zeros(len(smiles), 4)

    class DummyHead:
        def loss(self, ctx_emb, ctx_y, q_emb, q_y):
            return (ctx_emb.sum() + q_emb.sum()) ** 2

    rhos = iter([0.1, 0.5, 0.2])  # step 0, step 1, step 2
    monkeypatch.setattr(training, "evaluate_val", lambda *a, **k: (1.0, next(rhos)))

    saved: list[str] = []
    monkeypatch.setattr(training, "save_encoder",
                        lambda enc, path, hs: saved.append(Path(path).name))

    assays = data.load_chembl_assays("train", use_full=False)[:4]
    val = training.build_val_episodes(
        assays, n_episodes=1, max_context=8, max_query=8, rng=random.Random(0))
    cfg = training.TrainConfig(
        n_steps=2, lr=1e-3, max_context=8, max_query=8, hidden_size=4, seed=0,
        eval_every=1, print_every=1, val_curve_csv="",
        save_encoder_to=str(tmp_path / "enc.pt"))

    training.train_encoder(DummyEncoder(), DummyHead(), assays, val, cfg, 0.0,
                           repo_root=tmp_path)

    # Best written at step 0 (0.1) and step 1 (0.5), but NOT step 2 (0.2 < 0.5).
    assert saved.count("enc_best.pt") == 2
    assert saved[-1] == "enc.pt"  # the final save is the latest-weights one


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
    curve, and checkpoints the encoder. Configures a tiny throwaway Preset."""
    curve = tmp_path / "curve.csv"
    ckpt = tmp_path / "encoder.pt"
    preset = exp.Preset(
        use_full_data=False, hidden_size=32, n_estimators=1, n_train_steps=6,
        print_every=3, eval_every=3, n_val_episodes=3,
        val_curve_csv=str(curve), save_encoder_to=str(ckpt),
    )
    monkeypatch.setattr(exp, "resolve_preset", lambda: ("test", preset))

    exp.main()

    assert curve.exists()
    rows = curve.read_text().strip().splitlines()
    assert rows[0] == ("step,train_loss,val_loss,val_spearman,seconds,"
                       "grad_mean,grad_std,grad_min,grad_max,grad_clip_frac,lr")
    assert len(rows) >= 3  # header + step-0 baseline + eval rows
    for line in rows[1:]:
        rho = float(line.split(",")[3])
        assert -1.0 <= rho <= 1.0
    # Grad norms are recorded (finite and >= 0) from the first real eval onward.
    g_mean = float(rows[2].split(",")[5])
    assert g_mean == g_mean and g_mean >= 0.0

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


# --------------------------------------------------------------------------- #
# Epoch mode (2026-09-11 supervision): full validation, early stopping         #
# --------------------------------------------------------------------------- #
def _dummy_encoder_and_head():
    import torch

    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4))

        def forward(self, smiles):
            return self.w.expand(len(smiles), 4) + torch.zeros(len(smiles), 4)

    class DummyHead:
        def loss(self, ctx_emb, ctx_y, q_emb, q_y):
            return (ctx_emb.sum() + q_emb.sum()) ** 2

    return DummyEncoder(), DummyHead()


def _run_epochs(tmp_path, monkeypatch, val_losses, **cfg_kw):
    """Run train_encoder_epochs with scripted validation losses; return (summary, rows)."""
    import csv as _csv

    losses = iter(val_losses)
    monkeypatch.setattr(training, "evaluate_val_full",
                        lambda *a, **k: (next(losses), 0.3, 100))
    saved: list[str] = []
    monkeypatch.setattr(training, "save_encoder",
                        lambda enc, path, hs: saved.append(Path(path).name))
    assays = data.load_chembl_assays("train", use_full=False)[:4]
    val = training.build_val_episodes(
        assays, n_episodes=1, max_context=8, max_query=8, rng=random.Random(0))
    curve = tmp_path / "curve.csv"
    cfg = training.EpochConfig(
        lr=1e-3, max_context=8, max_query=8, hidden_size=4, seed=0, print_every=0,
        val_curve_csv=str(curve), save_encoder_to=str(tmp_path / "enc.pt"), **cfg_kw)
    enc, head = _dummy_encoder_and_head()
    summary = training.train_encoder_epochs(enc, head, assays, val, cfg, 0.0,
                                            repo_root=tmp_path)
    rows = list(_csv.DictReader(curve.open()))
    return summary, rows, saved


def test_early_stopping_fires_after_patience_epochs_without_improvement(tmp_path, monkeypatch):
    # epoch 0 = 2.00, epoch 1 = 1.90 (best), then 3 epochs that do not beat 1.90.
    summary, rows, saved = _run_epochs(
        tmp_path, monkeypatch, [2.00, 1.90, 1.95, 1.91, 1.93, 1.50],
        n_epochs=10, patience=3)
    assert summary["best_epoch"] == 1
    assert summary["epochs_run"] == 4                 # stopped; epoch 5's 1.50 never ran
    assert "early stopping" in summary["stop_reason"]
    assert [r["epoch"] for r in rows] == ["0", "1", "2", "3", "4"]
    assert rows[-1]["event"] == "early_stop"
    assert rows[-1]["epochs_without_improvement"] == "3"
    # best written for epoch 0 (baseline) and epoch 1 only; one checkpoint per epoch kept
    assert saved.count("enc_best.pt") == 2
    assert {f"epoch_{i:03d}.pt" for i in range(5)} <= set(saved)


def test_untrained_baseline_can_remain_best(tmp_path, monkeypatch):
    """If no epoch beats epoch 0, _best is the untrained encoder and the run says so."""
    summary, rows, saved = _run_epochs(
        tmp_path, monkeypatch, [1.80, 1.85, 1.90], n_epochs=10, patience=2)
    assert summary["best_epoch"] == 0
    assert summary["epochs_run"] == 2
    assert saved.count("enc_best.pt") == 1


def test_deadline_already_passed_runs_nothing_and_writes_nothing(tmp_path, monkeypatch):
    """A requeued job that starts after its deadline must exit without touching files.

    Previously it ran epoch 1 first (the predictive check needs one epoch's duration),
    which overwrote a finished run's curve and checkpoints on 2026-10-01.
    """
    calls = []
    monkeypatch.setattr(training, "evaluate_val_full",
                        lambda *a, **k: calls.append(1) or (2.0, 0.3, 100))
    saved: list[str] = []
    monkeypatch.setattr(training, "save_encoder",
                        lambda enc, path, hs: saved.append(Path(path).name))
    assays = data.load_chembl_assays("train", use_full=False)[:4]
    val = training.build_val_episodes(
        assays, n_episodes=1, max_context=8, max_query=8, rng=random.Random(0))
    curve = tmp_path / "curve.csv"
    cfg = training.EpochConfig(
        n_epochs=10, patience=5, lr=1e-3, max_context=8, max_query=8, hidden_size=4,
        print_every=0, val_curve_csv=str(curve), save_encoder_to=str(tmp_path / "enc.pt"),
        deadline_utc="2000-01-01T00:00:00+00:00")     # already passed
    enc, head = _dummy_encoder_and_head()
    summary = training.train_encoder_epochs(enc, head, assays, val, cfg, 0.0,
                                            repo_root=tmp_path)
    assert summary["epochs_run"] == 0
    assert "already passed" in summary["stop_reason"]
    assert not curve.exists() and saved == [] and calls == []   # not even the baseline


def test_future_deadline_does_not_stop_a_short_run(tmp_path, monkeypatch):
    summary, rows, _ = _run_epochs(
        tmp_path, monkeypatch, [2.0, 1.9, 1.8], n_epochs=2, patience=5,
        deadline_utc="2999-01-01T00:00:00+00:00")
    assert summary["epochs_run"] == 2
    assert "completed" in summary["stop_reason"]


@pytest.mark.parametrize("existing", ["curve.csv", "enc.pt", "enc_best.pt", "enc_epochs/epoch_000.pt"])
def test_refuses_to_overwrite_a_previous_run(tmp_path, monkeypatch, existing):
    target = tmp_path / existing
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("previous run")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_epochs(tmp_path, monkeypatch, [2.0, 1.9], n_epochs=1, patience=5)
    assert target.read_text() == "previous run"                 # untouched


def test_overwrite_true_replaces_a_previous_run(tmp_path, monkeypatch):
    (tmp_path / "curve.csv").write_text("previous run")
    summary, rows, _ = _run_epochs(tmp_path, monkeypatch, [2.0, 1.9], n_epochs=1,
                                   patience=5, overwrite=True)
    assert summary["epochs_run"] == 1
    assert [r["epoch"] for r in rows] == ["0", "1"]


def test_full_validation_loss_is_weighted_by_data_point():
    """A 3-point episode must count three times as much as a 1-point one."""
    import torch

    class Ep:
        def __init__(self, n_query, loss):
            self.context_smiles = ["C"] * 5
            self.context_y = torch.ones(5)
            self.query_smiles = ["C"] * n_query
            self.query_y = torch.arange(float(n_query))
            self.loss = loss

    episodes = [Ep(1, 1.0), Ep(3, 3.0)]
    by_size = {1: 1.0, 3: 3.0}

    class Head:
        def loss(self, ctx_emb, ctx_y, q_emb, q_y):
            return torch.tensor(by_size[len(q_y)])

        def predict(self, ctx_emb, ctx_y, q_emb):
            return torch.arange(float(q_emb.shape[0]))

    enc, _ = _dummy_encoder_and_head()
    loss, rho, n_points = training.evaluate_val_full(enc, Head(), episodes)
    assert n_points == 4
    assert loss == pytest.approx((1 * 1.0 + 3 * 3.0) / 4)   # 2.5, not the episode mean 2.0
    assert rho == pytest.approx(1.0)   # only the 3-point episode has a Spearman


def test_epoch_mode_smoke_with_real_tabpfn(tmp_path, monkeypatch, tabpfn_available):
    """run_experiment.main() in epoch mode on a handful of dev assays."""
    real_load = data.load_chembl_assays
    monkeypatch.setattr(exp.data, "load_chembl_assays",
                        lambda split, **kw: real_load(split, **kw)[:6])
    curve = tmp_path / "curve.csv"
    preset = exp.Preset(
        use_full_data=False, hidden_size=32, n_estimators=1, n_epochs=2, patience=5,
        print_every=0, val_curve_csv=str(curve), save_encoder_to=str(tmp_path / "e.pt"))
    monkeypatch.setattr(exp, "resolve_preset", lambda: ("epoch-smoke", preset))
    exp.main()
    import csv as _csv
    rows = list(_csv.DictReader(curve.open()))
    assert [r["epoch"] for r in rows] == ["0", "1", "2"]
    for r in rows:
        assert math.isfinite(float(r["val_loss"]))
        assert int(r["val_points"]) > 0
    assert (tmp_path / "e_epochs" / "epoch_002.pt").exists()
