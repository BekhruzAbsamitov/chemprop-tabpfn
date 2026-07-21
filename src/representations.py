"""src/representations.py — the molecular-representation "arms" to compare.

An arm is just a name + a featurize function `list[str] -> tensor`. This builds
the standard set the experiments compare — a random-init Chemprop encoder, the
ChEMBL-trained encoder, Chemeleon, and Morgan — so every script draws them the
same way instead of re-implementing the loading logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from models.chemeleon import CheMeleonFingerprint
from models.checkpoints import load_encoder
from models.encoder import MoleculeEncoder
from models.morgan import morgan_features

Featurize = Callable[[list[str]], torch.Tensor]
Arm = tuple[str, Featurize]


def to_tensor(x) -> torch.Tensor:
    """Chemprop returns a tensor; Morgan/Chemeleon may return numpy. Normalize."""
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


def build_arms(
    device: str,
    *,
    random_hidden_size: int = 128,
    trained_path: str | Path | None = None,
    include_chemeleon: bool = True,
    include_morgan: bool = False,
    repo_root: Path | None = None,
) -> list[Arm]:
    """Build the requested representation arms as (name, featurize_fn) pairs.

    `trained_path` (if given and present) adds the ChEMBL-trained encoder, whose
    width is read from the checkpoint. A missing trained checkpoint is skipped
    with a warning rather than erroring, so evaluation can run before training
    finishes.
    """
    arms: list[Arm] = []

    random_encoder = MoleculeEncoder(hidden_size=random_hidden_size).to(device).eval()
    arms.append(("Chemprop-random", lambda s: random_encoder(s)))

    if trained_path is not None:
        path = Path(trained_path)
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        if path.exists():
            trained_encoder = load_encoder(path, device)
            arms.append(("Chemprop-trained", lambda s: trained_encoder(s)))
        else:
            print(f"  WARNING: trained encoder not found at {path} — skipping that arm.")

    if include_chemeleon:
        fingerprinter = CheMeleonFingerprint(device=device)
        arms.append(("Chemeleon", lambda s: fingerprinter(s)))

    if include_morgan:
        arms.append(("Morgan", lambda s: morgan_features(s)))

    return arms
