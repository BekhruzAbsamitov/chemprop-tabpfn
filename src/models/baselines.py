"""src/models/baselines.py — the AL-benchmark model zoo (inference only).

Each model exposes the SAME interface — `predict_dist(x_context, y_context, x_query)
-> (mu, sigma)` — so the active-learning loop drives TabPFN, RandomForest, and
GaussianProcess through identical code. Each refits on the given context every call
(the context grows every AL round).

  * InferenceTabPFN  — the STANDARD (non-differentiable) TabPFN inference path. NOT
    TabPFNHead: that one uses differentiable_input=True + float32 for TRAINING, which
    is far too memory-hungry for a big inference sweep (OOMs a 40 GB GPU). Here we
    read mean + an uncertainty (from the 16/84 quantile spread) via the public API.
  * RandomForest     — mu/sigma from the spread across trees.
  * GaussianProcess  — mu/sigma from the posterior (RBF kernel), the classic BO model.
  * GaussianProcessTanimoto — the same with a Tanimoto kernel, the standard GP kernel for
    binary fingerprints (2026-09-29 supervision; kernel and settings as suggested there).

InferenceTabPFN.predict_full additionally returns TabPFN's whole predictive distribution
(the bar distribution and its logits, in raw activity units), so acquisition functions
can be computed from the full posterior instead of a Gaussian approximation.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


class InferenceTabPFN:
    """Standard TabPFN regressor for inference. mu = predicted mean; sigma from the
    16th/84th-percentile spread (≈ 1 std for a roughly-Gaussian predictive)."""

    name = "tabpfn"

    def __init__(self, n_estimators: int = 4, device: str | None = None):
        from tabpfn import TabPFNRegressor

        from models.tabpfn_regressor import resolve_tabpfn_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reg = TabPFNRegressor(
            model_path=resolve_tabpfn_checkpoint(),
            n_estimators=n_estimators,
            device=self.device,
            ignore_pretraining_limits=True,
        )

    def predict_dist(self, x_context, y_context, x_query) -> tuple[torch.Tensor, torch.Tensor]:
        self.reg.fit(_np(x_context), _np(y_context).ravel())
        xq = _np(x_query)
        mu = np.asarray(self.reg.predict(xq, output_type="mean")).ravel()
        q = self.reg.predict(xq, output_type="quantiles", quantiles=[0.1587, 0.8413])
        lo, hi = np.asarray(q[0]).ravel(), np.asarray(q[1]).ravel()
        sigma = np.clip((hi - lo) / 2.0, 1e-6, None)
        return (torch.tensor(mu, dtype=torch.float32),
                torch.tensor(sigma, dtype=torch.float32))

    def predict_full(self, x_context, y_context, x_query):
        """(mu, sigma, criterion, logits) from ONE fit and ONE prediction.

        mu and sigma are exactly those of predict_dist. `criterion` is TabPFN's bar
        distribution in raw activity units and `logits` its log-probabilities per query
        (shape [n_query, n_bars]), so criterion.ei(logits, tau) / criterion.pi(logits, tau)
        give EI / PI under the full predictive distribution, with no Gaussian assumption.
        """
        self.reg.fit(_np(x_context), _np(y_context).ravel())
        out = self.reg.predict(_np(x_query), output_type="full")
        q = self.reg.predict(_np(x_query), output_type="quantiles", quantiles=[0.1587, 0.8413])
        lo, hi = np.asarray(q[0]).ravel(), np.asarray(q[1]).ravel()
        mu = torch.tensor(np.asarray(out["mean"]).ravel(), dtype=torch.float32)
        sigma = torch.tensor(np.clip((hi - lo) / 2.0, 1e-6, None), dtype=torch.float32)
        criterion = out["criterion"]
        logits = out["logits"].float().to(criterion.borders.device)
        return mu, sigma, criterion, logits


class RandomForestHead:
    """RandomForest baseline. Uncertainty = std of the per-tree predictions."""

    name = "rf"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def predict_dist(self, x_context, y_context, x_query) -> tuple[torch.Tensor, torch.Tensor]:
        model = RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
        model.fit(_np(x_context), _np(y_context).ravel())
        xq = _np(x_query)
        preds = np.stack([t.predict(xq) for t in model.estimators_])  # (n_trees, n_query)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)
        return (torch.tensor(mu, dtype=torch.float32),
                torch.tensor(sigma, dtype=torch.float32))


class GaussianProcessHead:
    """GaussianProcess baseline (RBF kernel). Uncertainty = posterior std."""

    name = "gp"

    def __init__(self, random_state: int = 42, n_restarts: int = 2, alpha: float = 0.1):
        self.random_state = random_state
        self.n_restarts = n_restarts
        self.alpha = alpha

    def predict_dist(self, x_context, y_context, x_query) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts, random_state=self.random_state)
        model.fit(_np(x_context), _np(y_context).ravel())
        mu, sigma = model.predict(_np(x_query), return_std=True)
        return (torch.tensor(mu, dtype=torch.float32),
                torch.tensor(sigma, dtype=torch.float32))


class TanimotoKernel(Kernel):
    """Tanimoto similarity <x, y> / (|x|^2 + |y|^2 - <x, y>), the convention for binary
    fingerprints. Also a valid (positive semi-definite) kernel on any non-zero real vectors,
    which is how the final benchmark run uses it on the other representations.

    No hyperparameters; scale and noise come from the kernels it is combined with.
    Code as suggested in the 2026-09-29 supervision.
    """

    def __init__(self):
        pass  # explicit __init__ is required, or sklearn's clone() fails

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X).astype(float)
        Y = X if Y is None else np.atleast_2d(Y).astype(float)
        dot = X @ Y.T
        x2 = np.einsum("ij,ij->i", X, X)[:, None]
        y2 = np.einsum("ij,ij->i", Y, Y)[None, :]
        denom = x2 + y2 - dot
        K = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)
        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        return np.ones(np.atleast_2d(X).shape[0])

    def is_stationary(self):
        return False


class GaussianProcessTanimotoHead:
    """Gaussian process with a Tanimoto kernel (conventionally on binary fingerprints).

    Configuration as suggested in the 2026-09-29 supervision:
    ConstantKernel(1.0) * Tanimoto() + WhiteKernel(1e-2), normalize_y=True.
    Note that this differs from GaussianProcessHead (RBF) not only in the kernel but
    also in fitting the noise level and normalising the targets.
    """

    name = "gp_tanimoto"

    def __init__(self, random_state: int = 42, n_restarts: int = 2):
        self.random_state = random_state
        self.n_restarts = n_restarts

    def predict_dist(self, x_context, y_context, x_query) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = C(1.0) * TanimotoKernel() + WhiteKernel(1e-2)
        model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True,
            n_restarts_optimizer=self.n_restarts, random_state=self.random_state)
        model.fit(_np(x_context), _np(y_context).ravel())
        mu, sigma = model.predict(_np(x_query), return_std=True)
        return (torch.tensor(mu, dtype=torch.float32),
                torch.tensor(sigma, dtype=torch.float32))
