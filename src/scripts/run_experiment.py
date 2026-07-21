from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import torch
import zlib
from scipy.stats import spearmanr

from data_utils import data
from models.encoder import MoleculeEncoder  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402

# models.morgan is imported lazily inside main() — only the target evaluation
# needs it, so a pure training run (EVAL_ON_TARGET=False) doesn't require it.

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
USE_FULL_DATA = False  # False = small dev ChEMBL (fast); True = full 4.5M-row file
HIDDEN_SIZE = 128  # encoder width (fast local = 128; thesis/GPU scale = 300)
N_TRAIN_STEPS = 500  # how many training episodes to run
LEARNING_RATE = 1e-4  # training step size
N_ESTIMATORS = 1  # TabPFN passes to average (1 = fast local; 8+ on a GPU)
LOSS_FN = "nll"  # "nll" (recommended) | "mse" | "huber"

# Transfer evaluation on the S. aureus target (untrained/trained/Morgan). Set
# False to ONLY train the encoder (+ track the in-domain val curve) and skip
# loading S. aureus entirely — evaluate the saved encoder separately later.
EVAL_ON_TARGET = True

N_CONTEXT = 200  # labeled molecules TabPFN learns from, per evaluation split
N_QUERY = 200  # molecules it predicts, per split
N_SPLITS = 3  # random context/query draws, averaged (more = steadier)

# Cap on training-episode size. Full ChEMBL has assays with thousands of
# molecules; a differentiable TabPFN backward over a whole giant assay exhausts
# even an 80 GB GPU. Capping keeps each step's memory bounded (the split is
# already shuffled, so this is a uniform random subsample of the assay).
MAX_TRAIN_CONTEXT = 128
MAX_TRAIN_QUERY = 128

SEED = 0
PRINT_EVERY = 100  # print training progress every N steps

# IN-DOMAIN LEARNING CURVE (the honest signal). The per-step training loss is
# noisy (a fresh assay each step), so we ALSO evaluate a FIXED held-out set of
# ChEMBL test-split episodes every EVAL_EVERY steps: mean NLL + mean Spearman,
# the same episodes every time. A decreasing val_loss / rising val_spearman is
# real learning; flat means the encoder isn't learning (regardless of the noisy
# train loss). EVAL_EVERY must be a multiple of PRINT_EVERY. 0 disables it.
EVAL_EVERY = 500
N_VAL_EPISODES = 32
VAL_CURVE_CSV = "experiments/train_val_curve.csv"  # incremental; "" = don't save

# Where to save the trained encoder (so it can be reloaded / reused later).
# Empty string = don't save. Relative paths are taken from the repo root.
# On a long run the encoder is ALSO re-saved at every eval, so a killed job
# still leaves the latest weights on disk.
SAVE_ENCODER_TO = ""


def score(predictions: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """Return (Spearman, R2). Spearman = ranking quality; R2 = calibration."""
    p = predictions.detach().cpu().numpy()
    t = truth.detach().cpu().numpy()
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - float(((t - p) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else float("nan")
    rho = float(spearmanr(p, t).statistic)
    return rho, r2


def evaluate_on_target(featurize, head, smiles, y) -> tuple[float, float]:
    """Average (Spearman, R2) over N_SPLITS random context/query draws of the target.

    `featurize` turns a list of SMILES into a tensor of numbers (either the
    Chemprop encoder or Morgan). `head` is the frozen TabPFN.
    """
    rng = random.Random(SEED)
    idx = list(range(len(smiles)))
    rhos, r2s = [], []
    for _ in range(N_SPLITS):
        rng.shuffle(idx)
        ctx, qry = idx[:N_CONTEXT], idx[N_CONTEXT: N_CONTEXT + N_QUERY]
        with torch.no_grad():
            x_ctx = featurize([smiles[i] for i in ctx]).to(head.device)
            x_qry = featurize([smiles[i] for i in qry]).to(head.device)
            preds = head.predict(x_ctx, y[ctx], x_qry)
        rho, r2 = score(preds, y[qry])
        rhos.append(rho)
        r2s.append(r2)
    return sum(rhos) / len(rhos), sum(r2s) / len(r2s)


# --------------------------------------------------------------------------- #
# In-domain learning curve (fixed held-out ChEMBL episodes)                    #
# --------------------------------------------------------------------------- #
def _stable_seed(text: str) -> int:
    """Deterministic per-assay seed (Python's hash() is randomized per process)."""
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


def build_val_episodes(assays, rng) -> list:
    """Freeze N_VAL_EPISODES episodes from the held-out ChEMBL test assays.

    Each assay's context/query split is seeded deterministically (crc32 of the
    assay id) so the val set is identical across runs — a reproducible curve.
    """
    order = list(assays)
    rng.shuffle(order)
    episodes = []
    for assay in order:
        ep = data.make_episode(
            assay, max_context=MAX_TRAIN_CONTEXT, max_query=MAX_TRAIN_QUERY,
            rng=random.Random(_stable_seed(assay.assay_id)),
        )
        if ep is not None:
            episodes.append(ep)
        if len(episodes) == N_VAL_EPISODES:
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
            rhos.append(float(spearmanr(p, t).statistic))
    if was_training:
        encoder.train()
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return mean(losses), mean(rhos)


def _save_encoder(encoder) -> None:
    """Persist the encoder (overwrite) so a killed long job keeps its weights."""
    if not SAVE_ENCODER_TO:
        return
    path = Path(SAVE_ENCODER_TO)
    path = path if path.is_absolute() else data.REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder.state_dict(), "hidden_size": HIDDEN_SIZE}, path)


def train_encoder(encoder, head, assays, val_episodes, t0) -> None:
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    rng = random.Random(SEED)
    encoder.train()
    recent_losses: list[float] = []

    curve_path = None
    if VAL_CURVE_CSV and val_episodes:
        curve_path = Path(VAL_CURVE_CSV)
        curve_path = curve_path if curve_path.is_absolute() else data.REPO_ROOT / curve_path
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        with curve_path.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_spearman", "seconds"])

    def log_val(step: int, train_avg: float) -> None:
        v_loss, v_rho = evaluate_val(encoder, head, val_episodes)
        secs = time.time() - t0
        print(f"  step {step:>6}/{N_TRAIN_STEPS} | train {train_avg:6.3f} | "
              f"val_loss {v_loss:6.3f} | val_spearman {v_rho:+.3f}  ({secs:.0f}s)")
        if curve_path:
            with curve_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{train_avg:.4f}", f"{v_loss:.4f}", f"{v_rho:.4f}", f"{secs:.0f}"])
        _save_encoder(encoder)  # checkpoint the latest weights

    if val_episodes:
        log_val(0, float("nan"))

    episodes = data.iter_episodes(
        assays, max_context=MAX_TRAIN_CONTEXT, max_query=MAX_TRAIN_QUERY, rng=rng
    )
    for step in range(1, N_TRAIN_STEPS + 1):
        episode = next(episodes)

        optimizer.zero_grad()
        emb_context = encoder(episode.context_smiles)  # gradients flow through these
        emb_query = encoder(episode.query_smiles)
        loss = head.loss(emb_context, episode.context_y, emb_query, episode.query_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)  # keep steps sane
        optimizer.step()

        recent_losses.append(loss.item())
        if step % PRINT_EVERY == 0:
            avg = sum(recent_losses) / len(recent_losses)
            recent_losses.clear()
            if val_episodes and EVAL_EVERY and step % EVAL_EVERY == 0:
                log_val(step, avg)  # train avg + fixed-val loss/spearman
            else:
                print(f"  step {step:>6}/{N_TRAIN_STEPS} | avg loss {avg:6.3f}  "
                      f"({time.time() - t0:.0f}s)")


def main() -> None:
    _load_hf_token()
    torch.manual_seed(SEED)
    t0 = time.time()

    # 1. Data — training ChEMBL + in-domain val; the S. aureus target only if
    #    we're going to evaluate transfer (EVAL_ON_TARGET).
    train_assays = data.load_chembl_assays("train", use_full=USE_FULL_DATA)
    print(f"loaded {len(train_assays)} ChEMBL training assays  ({time.time() - t0:.0f}s)")
    val_episodes = []
    if EVAL_EVERY and N_VAL_EPISODES:
        test_assays = data.load_chembl_assays("test", use_full=USE_FULL_DATA)
        val_episodes = build_val_episodes(test_assays, random.Random(SEED))
        print(f"  {len(val_episodes)} fixed in-domain val episodes (ChEMBL test held-out)")
    smiles = y = None
    if EVAL_ON_TARGET:
        print("loading S. aureus target ...")
        smiles, y = data.load_target()
        print(f"  {len(smiles)} molecules | pMIC mean {y.mean():.2f}")

    # 2. Models
    encoder = MoleculeEncoder(hidden_size=HIDDEN_SIZE)
    head = TabPFNHead(n_estimators=N_ESTIMATORS, loss_fn=LOSS_FN)
    encoder.to(head.device)
    print(f"  device: {head.device} | loss: {LOSS_FN} | hidden_size: {HIDDEN_SIZE} | "
          f"n_estimators: {N_ESTIMATORS} | full_data: {USE_FULL_DATA}")

    def chemprop_featurize(smiles_list: list[str]) -> torch.Tensor:
        return encoder(smiles_list)

    # 3. Untrained baseline (only when evaluating transfer)
    rho_untrained = r2_untrained = None
    if EVAL_ON_TARGET:
        print("\nmeasuring UNTRAINED encoder on S. aureus ...")
        rho_untrained, r2_untrained = evaluate_on_target(chemprop_featurize, head, smiles, y)

    # 4. Train
    print(f"\ntraining encoder for {N_TRAIN_STEPS} steps ...")
    train_encoder(encoder, head, train_assays, val_episodes, t0)

    # 4b. Save the trained encoder (also checkpointed at every eval during training).
    if SAVE_ENCODER_TO:
        _save_encoder(encoder)
        print(f"  saved trained encoder -> {SAVE_ENCODER_TO}")

    # 5-7. Transfer evaluation on S. aureus (skipped when EVAL_ON_TARGET=False —
    #      then this file just TRAINS the encoder and tracks the in-domain curve).
    print("\ntraining complete — encoder trained and saved. Target evaluation "
          "skipped (EVAL_ON_TARGET=False);")
    print(f"  in-domain learning curve -> {VAL_CURVE_CSV or 'stdout above'}")

    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
