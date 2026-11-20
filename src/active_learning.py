"""src/active_learning.py — shared helpers for the active-learning benchmarks.

The frozen-representation loop (run_active_learning.py) and the fine-tuning loop
(run_active_learning_finetune.py) have different acquisition loops, but share how
they turn molecules into a scrubbed feature matrix, and how they persist the
hits-vs-budget curves. The oracle-efficiency metric lives in
metrics.oracle_efficiency.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def batched_encode(featurize, smiles: list[str], device: torch.device,
                   *, batch: int = 256, warn: bool = True) -> torch.Tensor:
    """Featurize all SMILES in chunks -> [N, d], with non-finite rows zeroed.

    Some molecules encode to NaN/Inf (e.g. a SMILES that yields an empty graph ->
    mean over 0 atoms). TabPFN's preprocessing indexes on feature values, so a
    NaN/Inf sends an index out of bounds and crashes CUDA — we scrub them the same
    way Morgan handles an unparseable molecule. `featurize` may be an encoder or
    any callable returning a tensor/array per batch of SMILES.
    """
    rows = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch):
            out = featurize(smiles[start:start + batch])
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=torch.float32)
            rows.append(out.to(device))
    X = torch.cat(rows, dim=0)
    n_bad = int((~torch.isfinite(X)).any(dim=1).sum())
    if warn and n_bad:
        print(f"    [warn] {n_bad} molecules had non-finite features -> zeroed")
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def save_hit_curves(curves: dict[str, list[float]], budgets: list[int],
                    total_hits: int, path: Path) -> Path:
    """Write averaged hits-vs-budget curves to CSV. `curves` maps a label
    (e.g. arm or 'arm/strategy') to its mean-hits list aligned with `budgets`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "budget", "mean_hits", "recall", "total_hits"])
        for label, mean_hits in curves.items():
            for budget, h in zip(budgets, mean_hits):
                writer.writerow([label, budget, f"{h:.3f}", f"{h / total_hits:.4f}", total_hits])
    return path
