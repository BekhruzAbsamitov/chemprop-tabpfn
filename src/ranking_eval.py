"""src/ranking_eval.py — static one-shot ranking evaluation (Experiment A).

Given representation arms and target datasets, measure each arm's ranking quality
(Spearman) with a frozen TabPFN: draw N_SAMPLES random context/query splits per
dataset — the SAME splits for every arm (a fair comparison) — score each, and
report mean ± std. "Static" = no fine-tuning; each arm just turns molecules into
numbers and TabPFN ranks the query set in one shot from the labeled context.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from data_utils import data
from metrics import mean_std, score
from models.tabpfn_regressor import TabPFNHead
from representations import Arm, to_tensor

# A dataset entry: (display name, csv path rel. to repo root, smiles col, target col).
Dataset = tuple[str, str, str, str]


def make_splits(n: int, rng: random.Random, *, n_context: int, n_query: int,
                n_samples: int) -> list[tuple[list[int], list[int]]]:
    """n_samples fixed (context, query) index splits, shared across all arms."""
    idx = list(range(n))
    splits = []
    for _ in range(n_samples):
        rng.shuffle(idx)
        splits.append((list(idx[:n_context]), list(idx[n_context: n_context + n_query])))
    return splits


def _evaluate_arm(featurize, head, smiles, y, splits) -> tuple[list[float], list[float]]:
    """Spearman & R2 for each split (static one-shot ranking)."""
    rhos, r2s = [], []
    for ctx, qry in splits:
        with torch.no_grad():
            x_ctx = to_tensor(featurize([smiles[i] for i in ctx])).to(head.device)
            x_qry = to_tensor(featurize([smiles[i] for i in qry])).to(head.device)
            preds = head.predict(x_ctx, y[ctx], x_qry)
        rho, r2 = score(preds, y[qry])
        rhos.append(rho)
        r2s.append(r2)
    return rhos, r2s


def _print_table(title: str, table: dict, arms: list[Arm], ds_names: list[str]) -> None:
    width = 22 + 16 * len(ds_names) + 12
    print("=" * width)
    print(title)
    print("=" * width)
    print(f"  {'arm':18s}" + "".join(f"{d:>15s} " for d in ds_names) + f"{'overall':>10s}")
    for arm_name, _ in arms:
        row = f"  {arm_name:18s}"
        per_ds_means = []
        for d in ds_names:
            if d in table[arm_name]:
                m, s = table[arm_name][d]
                row += f"{m:7.3f} +/-{s:5.3f} "
                per_ds_means.append(m)
            else:
                row += f"{'-':>15s} "
        overall, _ = mean_std(per_ds_means)
        row += f"{overall:>10.3f}"
        print(row)
    print()


def run_static_ranking(
    arms: list[Arm],
    datasets: list[Dataset],
    *,
    n_context: int,
    n_query: int,
    n_samples: int,
    n_estimators: int,
    loss_fn: str,
    seed: int,
    repo_root: Path,
) -> dict:
    """Evaluate every arm on every dataset; print Spearman + R2 tables; return the
    Spearman table {arm: {dataset: (mean, std)}}."""
    t0 = time.time()
    # One frozen TabPFN head per arm (arms live in different feature spaces).
    heads = {name: TabPFNHead(n_estimators=n_estimators, loss_fn=loss_fn) for name, _ in arms}
    print(f"arms: {[n for n, _ in arms]} | {n_samples} samples x "
          f"{n_context} ctx / {n_query} qry\n")

    rho_table: dict = {n: {} for n, _ in arms}
    r2_table: dict = {n: {} for n, _ in arms}

    for ds_name, rel_path, smiles_col, target_col in datasets:
        print(f"=== {ds_name} ===")
        smiles, y = data.load_target(
            repo_root / rel_path, smiles_col=smiles_col, target_col=target_col
        )
        print(f"  {len(smiles)} unique molecules | target '{target_col}' mean {float(y.mean()):.2f}")
        if len(smiles) < n_context + n_query:
            print(f"  WARNING: only {len(smiles)} molecules (< {n_context + n_query}); skipping.")
            continue

        splits = make_splits(len(smiles), random.Random(seed),
                             n_context=n_context, n_query=n_query, n_samples=n_samples)
        for arm_name, featurize in arms:
            rhos, r2s = _evaluate_arm(featurize, heads[arm_name], smiles, y, splits)
            rho_table[arm_name][ds_name] = mean_std(rhos)
            r2_table[arm_name][ds_name] = mean_std(r2s)
            m, s = rho_table[arm_name][ds_name]
            print(f"  {arm_name:18s} Spearman {m:6.3f} +/- {s:.3f}  ({time.time() - t0:.0f}s)")
        print()

    ds_names = [d[0] for d in datasets]
    print()
    _print_table("Spearman (mean +/- std over samples) — higher is better", rho_table, arms, ds_names)
    _print_table("R2 (mean +/- std over samples) — higher is better", r2_table, arms, ds_names)
    print(f"total time: {time.time() - t0:.0f}s")
    return rho_table
