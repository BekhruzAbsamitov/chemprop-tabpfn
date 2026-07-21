"""src/models/checkpoints.py — save/load a trained MoleculeEncoder.

One place for the encoder-persistence format so training and evaluation agree on
it. A checkpoint stores the encoder weights AND its hidden size, so the reloader
can rebuild the right architecture without the caller having to remember it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.encoder import MoleculeEncoder


def save_encoder(encoder: MoleculeEncoder, path: str | Path, hidden_size: int) -> None:
    """Persist the encoder's weights + width to `path` (parent dirs created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder.state_dict(), "hidden_size": hidden_size}, out)


def load_encoder(path: str | Path, device: str = "cpu") -> MoleculeEncoder:
    """Rebuild a MoleculeEncoder from a checkpoint and load its weights (eval mode)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    encoder = MoleculeEncoder(hidden_size=ckpt["hidden_size"]).to(device)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    return encoder
