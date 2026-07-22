from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import zlib
from scipy.stats import spearmanr

from data_utils import data
from models.checkpoints import save_encoder


def stable_seed(text: str) -> int:
    """Deterministic per-assay seed (Python's hash() is randomized per process)."""
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


def build_val_episodes(
        assays: list[data.Assay],
        *,
        n_episodes: int,
        max_context: int,
        max_query: int,
        rng: random.Random,
) -> list[data.Episode]:
    """Freeze `n_episodes` episodes from held-out assays.

    Each assay's context/query split is seeded deterministically (crc32 of the
    assay id) so the val set is identical across runs — a reproducible curve.
    """
    order = list(assays)
    rng.shuffle(order)
    episodes: list[data.Episode] = []
    for assay in order:
        ep = data.make_episode(
            assay, max_context=max_context, max_query=max_query,
            rng=random.Random(stable_seed(assay.assay_id)),
        )
        if ep is not None:
            episodes.append(ep)
        if len(episodes) == n_episodes:
            break
    return episodes


@torch.no_grad()
def evaluate_val(encoder, head, val_episodes) -> tuple[float, float]:
    """Mean NLL and mean Spearman over the fixed validation episodes."""
    was_training = encoder.training
    encoder.eval()
    losses, rhos = [], []
    for ep in val_episodes:
        emb_c = encoder(ep.context_smiles)
        emb_q = encoder(ep.query_smiles)
        losses.append(float(head.loss(emb_c, ep.context_y, emb_q, ep.query_y)))
        preds = head.predict(emb_c, ep.context_y, emb_q)
        p, t = preds.cpu().numpy(), ep.query_y.flatten().cpu().numpy()
        if len(p) > 1:
            rho = float(spearmanr(p, t).statistic)
            if rho == rho:  # drop NaN (an episode whose predictions came out constant)
                rhos.append(rho)
    if was_training:
        encoder.train()
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return mean(losses), mean(rhos)


@dataclass
class TrainConfig:
    """Everything train_encoder needs; a script fills this from its constants."""
    n_steps: int
    lr: float
    max_context: int
    max_query: int
    hidden_size: int
    seed: int = 0
    eval_every: int = 0  # 0 disables the in-domain curve
    print_every: int = 100
    val_curve_csv: str = ""  # "" = don't write the curve CSV
    save_encoder_to: str = ""  # "" = don't checkpoint


def train_encoder(encoder, head, assays, val_episodes, cfg: TrainConfig, t0: float,
                  *, repo_root: Path) -> None:
    """Train `encoder` so TabPFN predicts the ChEMBL episodes better.

    Logs the running train loss and, if `val_episodes` is non-empty, the fixed
    in-domain val loss/Spearman every cfg.eval_every steps (also checkpointing
    the encoder so a killed long job keeps its latest weights).
    """
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr)
    rng = random.Random(cfg.seed)
    encoder.train()
    recent_losses: list[float] = []

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    curve_path = None
    if cfg.val_curve_csv and val_episodes:
        curve_path = _resolve(cfg.val_curve_csv)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        with curve_path.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_spearman", "seconds"])

    def checkpoint() -> None:
        if cfg.save_encoder_to:
            save_encoder(encoder, _resolve(cfg.save_encoder_to), cfg.hidden_size)

    def log_val(step: int, train_avg: float) -> None:
        v_loss, v_rho = evaluate_val(encoder, head, val_episodes)
        secs = time.time() - t0
        print(f"  step {step:>6}/{cfg.n_steps} | train {train_avg:6.3f} | "
              f"val_loss {v_loss:6.3f} | val_spearman {v_rho:+.3f}  ({secs:.0f}s)")
        if curve_path:
            with curve_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{train_avg:.4f}", f"{v_loss:.4f}", f"{v_rho:.4f}", f"{secs:.0f}"])
        checkpoint()

    if val_episodes:
        log_val(0, float("nan"))  # baseline (untrained) point

    # Without-replacement episode stream (shuffled epochs): no redundant repeats.
    episodes = data.iter_episodes(
        assays, max_context=cfg.max_context, max_query=cfg.max_query, rng=rng
    )
    for step in range(1, cfg.n_steps + 1):
        episode = next(episodes)
        optimizer.zero_grad()
        emb_context = encoder(episode.context_smiles)
        emb_query = encoder(episode.query_smiles)
        loss = head.loss(emb_context, episode.context_y, emb_query, episode.query_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()

        recent_losses.append(loss.item())
        if step % cfg.print_every == 0:
            avg = sum(recent_losses) / len(recent_losses)
            recent_losses.clear()
            if val_episodes and cfg.eval_every and step % cfg.eval_every == 0:
                log_val(step, avg)
            else:
                print(f"  step {step:>6}/{cfg.n_steps} | avg loss {avg:6.3f}  "
                      f"({time.time() - t0:.0f}s)")
