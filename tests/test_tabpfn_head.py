"""Tests for src/models/tabpfn_regressor.py.

Split in two:
  * Fast tests that need no model — validation + the mean-decoding MATH, verified
    against a hand-built fake distribution (this is the core "are the values
    computed correctly" check).
  * Integration tests that need the real TabPFN checkpoint; they skip cleanly if
    it (or HF_TOKEN / network) is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from models.tabpfn_regressor import TabPFNHead, _load_hf_token


# --------------------------------------------------------------------------- #
# Fast tests (no TabPFN model needed)                                         #
# --------------------------------------------------------------------------- #
def test_invalid_loss_fn_raises():
    with pytest.raises(ValueError):
        TabPFNHead(loss_fn="banana")


def test_predict_mean_decodes_distribution_mean_and_denormalizes():
    """_predict_mean must compute sum(prob * bin_center) then undo z-normalization.

    We feed a hand-built distribution: sample 0's mass is on the first bin
    (center 1), sample 1's mass is on the last bin (center 5). With the stored
    normalization mean=2, std=3, the expected outputs are 1*3+2=5 and 5*3+2=17.
    """
    head = TabPFNHead.__new__(TabPFNHead)  # bypass the heavy __init__

    class FakeReg:
        y_train_mean_ = 2.0
        y_train_std_ = 3.0

        def forward(self, x, use_inference_mode=True):
            # averaged_logits is [n_bins, n_samples]; big value = all mass there
            averaged_logits = torch.tensor([[100.0, 0.0], [0.0, 0.0], [0.0, 100.0]])
            borders = [torch.tensor([0.0, 2.0, 4.0, 6.0])]  # centers -> [1, 3, 5]
            return averaged_logits, None, borders

    head.reg = FakeReg()
    out = head._predict_mean(torch.zeros(2, 4))
    assert torch.allclose(out, torch.tensor([5.0, 17.0]), atol=1e-2)


def test_predict_mean_std_computes_distribution_std_and_scales_it():
    """_predict_mean_std must return sqrt(E[c^2]-E[c]^2) in z-space, scaled by std.

    Distribution: mass split 50/50 between bin-center 1 and bin-center 5.
        mean_z = 3, var_z = 0.5*1 + 0.5*25 - 9 = 4, std_z = 2.
    With stored mean=2, std=3: mean = 3*3+2 = 11, std = 2*3 = 6.
    """
    head = TabPFNHead.__new__(TabPFNHead)

    class FakeReg:
        y_train_mean_ = 2.0
        y_train_std_ = 3.0

        def forward(self, x, use_inference_mode=True):
            # logits softmax to ~[0.5, 0, 0.5] over centers [1, 3, 5]
            averaged_logits = torch.tensor([[20.0], [-20.0], [20.0]])
            borders = [torch.tensor([0.0, 2.0, 4.0, 6.0])]
            return averaged_logits, None, borders

    head.reg = FakeReg()
    mean, std = head._predict_mean_std(torch.zeros(1, 4))
    assert torch.allclose(mean, torch.tensor([11.0]), atol=1e-2)
    assert torch.allclose(std, torch.tensor([6.0]), atol=1e-2)


# --------------------------------------------------------------------------- #
# Integration tests (need the real TabPFN checkpoint)                         #
# --------------------------------------------------------------------------- #
def _make_head(loss_fn: str = "nll") -> TabPFNHead:
    _load_hf_token()
    try:
        return TabPFNHead(n_estimators=1, loss_fn=loss_fn)
    except Exception as exc:  # no token / no network / no cached checkpoint
        pytest.skip(f"TabPFN checkpoint unavailable: {exc}")


@pytest.fixture(scope="module")
def head():
    return _make_head("nll")


def test_predict_returns_finite_prediction_per_query(head):
    torch.manual_seed(0)
    x_context, y_context = torch.randn(15, 8), torch.randn(15)
    x_query = torch.randn(6, 8)
    out = head.predict(x_context, y_context, x_query)
    assert out.shape == (6,)
    assert torch.isfinite(out).all()


def test_predict_dist_returns_finite_positive_uncertainty(head):
    """predict_dist must give a mean AND a positive, finite std per query — the
    uncertainty that EI/PI acquisition consumes."""
    torch.manual_seed(0)
    x_context, y_context = torch.randn(15, 8), torch.randn(15)
    x_query = torch.randn(6, 8)
    mean, std = head.predict_dist(x_context, y_context, x_query)
    assert mean.shape == (6,) and std.shape == (6,)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert (std > 0).all()


@pytest.mark.parametrize("loss_fn", ["nll", "mse", "huber"])
def test_loss_is_finite_scalar_and_flows_gradient_to_inputs(loss_fn):
    head = _make_head(loss_fn)
    torch.manual_seed(0)
    x_context = torch.randn(15, 8, requires_grad=True)
    y_context = torch.randn(15)
    x_query = torch.randn(6, 8, requires_grad=True)
    y_query = torch.randn(6)

    loss = head.loss(x_context, y_context, x_query, y_query)
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    assert x_context.grad is not None and x_query.grad is not None
    assert torch.isfinite(x_context.grad).all() and torch.isfinite(x_query.grad).all()


def test_spike_fix_keeps_loss_bounded_for_outlier_query(head):
    """A near-constant context + an extreme query would blow the loss up to the
    thousands without the target-z clip; with it, the loss stays bounded."""
    torch.manual_seed(0)
    x_context = torch.randn(20, 8, requires_grad=True)
    y_context = torch.full((20,), 5.0) + torch.randn(20) * 1e-3  # tiny spread
    x_query = torch.randn(4, 8, requires_grad=True)
    y_query = torch.tensor([5.0, 5.0, 5.0, 200.0])  # one wild outlier

    loss = head.loss(x_context, y_context, x_query, y_query)
    assert torch.isfinite(loss)
    assert loss.item() < 100.0
