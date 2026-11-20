from __future__ import annotations

import os
from pathlib import Path

import torch
from tabpfn import TabPFNRegressor
from torch import Tensor, nn

# --------------------------------------------------------------------------- #
# SETTINGS                                                                     #
# --------------------------------------------------------------------------- #
N_ESTIMATORS = 1  # how many TabPFN passes to average (1 = fast; 8+ = stronger)
TARGET_Z_CLIP = 5.0  # clip rescaled query labels to +/- this (the "spike fix")

# Which training loss to minimize:
#   "nll"   — TabPFN's own distributional loss (negative log-likelihood). Trains
#             calibrated UNCERTAINTY, which active-learning acquisition (EI/PI)
#             needs. Recommended for this thesis.
#   "mse"   — squared error on the single predicted number (point estimate only).
#   "huber" — like MSE but gentler on outliers (point estimate only).
LOSS_FN = "nll"

# TabPFN's pre-trained weights live in a gated Hugging Face repo. We download the
# checkpoint file and hand its path to TabPFN directly, which also sidesteps the
# separate PriorLabs license prompt — so you only need a Hugging Face token.
TABPFN_REPO = "Prior-Labs/tabpfn_3"
TABPFN_CHECKPOINT = "tabpfn-v3-regressor-v3_default.ckpt"

# This file is at src/models/, so the repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_device() -> str:
    """Use the GPU if one is available, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_hf_token() -> None:
    """Read HF_TOKEN from a .env file at the repo root (if not already set)."""
    if os.environ.get("HF_TOKEN"):
        return
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN=") and "=" in line:
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()


def resolve_tabpfn_checkpoint() -> str:
    """Download the TabPFN checkpoint (cached after the first time) and return its path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(TABPFN_REPO, TABPFN_CHECKPOINT)


class TabPFNHead(nn.Module):
    """A frozen TabPFN regressor used as a differentiable prediction head."""

    def __init__(
            self,
            n_estimators: int = N_ESTIMATORS,
            loss_fn: str = LOSS_FN,
            random_state: int = 0,
    ):
        super().__init__()
        if loss_fn not in ("nll", "mse", "huber"):
            raise ValueError(f"loss_fn must be 'nll', 'mse', or 'huber', got {loss_fn!r}")
        self.loss_fn = loss_fn
        self.device = _resolve_device()
        # differentiable_input=True is what lets gradients flow back to the inputs.
        self.reg = TabPFNRegressor(
            model_path=resolve_tabpfn_checkpoint(),
            differentiable_input=True,
            n_estimators=n_estimators,
            device=self.device,
            random_state=random_state,
            fit_mode="fit_preprocessors",
            inference_precision=torch.float32,  # keep gradients in full precision
            ignore_pretraining_limits=True,
        )

    def _predict_mean_std(self, x_query: Tensor) -> tuple[Tensor, Tensor]:
        """Decode TabPFN's output into a (mean, std) per query molecule.

        TabPFN predicts a probability distribution over value-bins, not a single
        number. We read off both moments of that distribution:
            mean = sum(prob * bin_center)
            std  = sqrt(sum(prob * bin_center^2) - mean^2)
        then undo the rescaling TabPFN applied to the context labels. The std is
        TabPFN's own predictive UNCERTAINTY — active-learning acquisition (EI/PI)
        needs it. Assumes the context has already been fitted.
        """
        averaged_logits, _per_estimator, borders = self.reg.forward(x_query, use_inference_mode=True)
        per_sample_logits = averaged_logits.transpose(0, 1)  # -> [n_query, n_bins]

        edges = torch.as_tensor(
            borders[0],
            device=per_sample_logits.device,
            dtype=per_sample_logits.dtype
        )

        bin_centers = (edges[:-1] + edges[1:]) / 2.0 if edges.numel() == per_sample_logits.shape[-1] + 1 else edges

        probs = torch.softmax(per_sample_logits.float(), dim=-1)
        mean_in_z_space = (probs * bin_centers).sum(dim=-1)
        var_in_z_space = (probs * bin_centers.pow(2)).sum(dim=-1) - mean_in_z_space.pow(2)
        std_in_z_space = var_in_z_space.clamp(min=0.0).sqrt()

        # undo the context rescaling to get back to real activity units.
        # A location+scale shift moves the mean but scales the std by |std|.
        scale = float(self.reg.y_train_std_)
        mean = mean_in_z_space * scale + float(self.reg.y_train_mean_)
        std = std_in_z_space * abs(scale)
        return mean, std

    def _predict_mean(self, x_query: Tensor) -> Tensor:
        """The predicted number per query molecule (the distribution's mean)."""
        return self._predict_mean_std(x_query)[0]

    def loss(self, x_context: Tensor, y_context: Tensor, x_query: Tensor, y_query: Tensor) -> Tensor:
        """How wrong TabPFN is on the query set — the number training minimizes.

        Gradients flow from this loss, through frozen TabPFN, back into x_context
        and x_query (the embeddings), and therefore into the Chemprop encoder.
        """
        y_context = y_context.to(x_context.device)
        y_query = y_query.to(x_query.device)

        # Show TabPFN the labeled context (this fits its in-context predictor).
        self.reg.fit_with_differentiable_input(x_context, y_context.flatten())

        # Compare predictions to the truth in TabPFN's own rescaled ("z") space.
        mean, std = float(self.reg.y_train_mean_), float(self.reg.y_train_std_)
        target_z = (y_query.flatten() - mean) / std
        target_z = target_z.clamp(-TARGET_Z_CLIP, TARGET_Z_CLIP)  # the spike fix

        if self.loss_fn == "nll":
            # Distributional loss: how unlikely the true value is under TabPFN's
            # predicted distribution. Uses TabPFN's native bar-distribution, so it
            # trains the whole distribution (including uncertainty), not just a point.
            averaged_logits, _per_estimator, _borders = self.reg.forward(x_query, use_inference_mode=True)
            logits = averaged_logits.transpose(0, 1)  # -> [n_query, n_bins]
            # znorm_space_bardist_ scores the distribution in the same rescaled
            # ("z") space our target_z lives in. (Replaces the deprecated bardist_.)
            # Its bin borders can sit on a different device (CPU) than our logits
            # on GPU, so align all three to the logits' device before scoring.
            bardist = self.reg.znorm_space_bardist_.to(logits.device)
            return bardist(logits, target_z.to(logits.device)).mean()

        # Point losses: compare the single predicted number to the truth.
        preds_z = (self._predict_mean(x_query) - mean) / std
        if self.loss_fn == "mse":
            return torch.nn.functional.mse_loss(preds_z, target_z)
        return torch.nn.functional.smooth_l1_loss(preds_z, target_z)  # "huber"

    @torch.no_grad()
    def predict(self, x_context: Tensor, y_context: Tensor, x_query: Tensor) -> Tensor:
        """Predicted activity for the query molecules (no training, real units)."""
        y_context = y_context.to(x_context.device)
        self.reg.fit_with_differentiable_input(x_context, y_context.flatten())
        return self._predict_mean(x_query)

    @torch.no_grad()
    def predict_dist(
        self, x_context: Tensor, y_context: Tensor, x_query: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predicted (mean, std) for the query molecules — the std is TabPFN's
        uncertainty, which active-learning acquisition (EI/PI) needs."""
        y_context = y_context.to(x_context.device)
        self.reg.fit_with_differentiable_input(x_context, y_context.flatten())
        return self._predict_mean_std(x_query)


# --------------------------------------------------------------------------- #
# Press Run to (1) predict from example numbers and (2) prove the training     #
# signal flows back to the inputs. Requires HF_TOKEN in your .env.             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _load_hf_token()
    torch.manual_seed(0)

    n_features = 16  # stand-in for encoder embeddings (real ones are 128-long)
    # Context = 20 molecules WITH labels; query = 8 molecules to predict.
    ctx_x = torch.randn(20, n_features, requires_grad=True)
    ctx_y = torch.randn(20)
    qry_x = torch.randn(8, n_features, requires_grad=True)
    qry_y = torch.randn(8)

    head = TabPFNHead(n_estimators=8)

    predictions = head.predict(ctx_x, ctx_y, qry_x)
    print(f"predicted {tuple(predictions.shape)} activities for the query molecules:")
    print(predictions)

    loss = head.loss(ctx_x, ctx_y, qry_x, qry_y)
    loss.backward()
    reached_inputs = qry_x.grad is not None and ctx_x.grad is not None
    print(f"\nloss function: {head.loss_fn}   training loss: {loss.item():.4f}")
    print(f"gradient reached the embeddings (so it will reach Chemprop): {reached_inputs}")
