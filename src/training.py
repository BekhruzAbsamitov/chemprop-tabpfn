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
from metrics import mean_std
from models.checkpoints import save_encoder

# Max gradient norm the encoder is clipped to each step. Also the threshold the
# logged `grad_clip_frac` counts against (fraction of steps that hit the clip).
GRAD_CLIP_NORM = 1.0


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
    grad_clip_norm: float = GRAD_CLIP_NORM  # max encoder grad norm per step
    # Also keep a step-tagged copy every N steps, in a `<name>_steps/` subdirectory
    # (0 = off). Needed for the CKA-across-training study: CKA(E_initial, E_step_k)
    # is only computable if the intermediate encoders were kept. step_000000 is the
    # untrained encoder the run started from — i.e. E_initial itself.
    checkpoint_every: int = 0
    # Learning-rate schedule: "none" keeps `lr` constant for the whole run (what every
    # run before 2026-09-03 did), "cosine" anneals it from `lr` to ~0 over n_steps.
    # Motivation: gradients are clipped on 100 % of steps, so the update magnitude is
    # pinned at lr x grad_clip_norm and NEVER decays — the two mechanisms by which
    # training normally settles (shrinking gradients, an annealed rate) are both absent,
    # which is why the validation curve fluctuates undiminished to the last step.
    lr_schedule: str = "none"


def train_encoder(encoder, head, assays, val_episodes, cfg: TrainConfig, t0: float,
                  *, repo_root: Path) -> None:
    """Train `encoder` so TabPFN predicts the ChEMBL episodes better.

    Logs the running train loss and, if `val_episodes` is non-empty, the fixed
    in-domain val loss/Spearman every cfg.eval_every steps.

    Checkpoints TWO files (when cfg.save_encoder_to is set): the latest weights, so
    a killed long job keeps its progress, and `<name>_best.pt` at the highest
    val_spearman, so a run that peaks and then decays still yields a usable encoder.
    """
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr)
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_steps)
    elif cfg.lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r}; use 'none' or 'cosine'")
    rng = random.Random(cfg.seed)
    encoder.train()

    recent_losses: list[float] = []
    # Gradient diagnostics (2026-07-30 TODO 1): is the encoder getting a usable
    # signal at all? clip_grad_norm_ returns the PRE-clip total norm, so tracking
    # it is free. Vanishing norms => no signal; huge/high-variance norms => the
    # step is unstable; moderate + steady norms with a flat loss => the signal is
    # real but uninformative (the fundamental-negative case).
    recent_gnorms: list[float] = []
    all_gnorms: list[float] = []  # whole-run summary

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    def grad_stats(vals: list[float]) -> tuple[float, float, float, float, float]:
        """(mean, std, min, max, clipped-fraction) of a window of grad norms."""
        if not vals:
            nan = float("nan")
            return nan, nan, nan, nan, nan
        m, s = mean_std(vals)
        clipped = sum(1 for v in vals if v > cfg.grad_clip_norm) / len(vals)
        return m, s, min(vals), max(vals), clipped

    curve_path = None
    if cfg.val_curve_csv and val_episodes:
        curve_path = _resolve(cfg.val_curve_csv)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        with curve_path.open("w", newline="") as f:
            # Grad columns are APPENDED so the original 5 stay positionally
            # identical — older curve CSVs and any plotting of them still work.
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_spearman", "seconds",
                                    "grad_mean", "grad_std", "grad_min", "grad_max",
                                    "grad_clip_frac", "lr"])

    # Best-val tracking: the FINAL weights are not necessarily the best ones — a
    # run can peak and then decay (the 5k USP7 run peaked at step 3750 and fell
    # back by 5000). So we keep two checkpoints: the latest (so a killed job keeps
    # its progress) and a separate `<name>_best.pt` at the highest val_spearman.
    # The step-0 baseline competes too: if it wins, "training never beat random
    # init" is itself the result, and _best.pt honestly holds the untrained encoder.
    best = {"rho": float("-inf"), "step": -1}

    def checkpoint(step: int, v_rho: float) -> None:
        if not cfg.save_encoder_to:
            return
        dest = _resolve(cfg.save_encoder_to)
        save_encoder(encoder, dest, cfg.hidden_size)
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            steps_dir = dest.with_name(f"{dest.stem}_steps")
            steps_dir.mkdir(parents=True, exist_ok=True)
            save_encoder(encoder, steps_dir / f"step_{step:06d}{dest.suffix}", cfg.hidden_size)
        if v_rho == v_rho and v_rho > best["rho"]:  # v_rho == v_rho filters NaN
            best.update(rho=v_rho, step=step)
            save_encoder(encoder, dest.with_name(f"{dest.stem}_best{dest.suffix}"),
                         cfg.hidden_size)
            print(f"      new best val_spearman {v_rho:+.3f} (step {step}) "
                  f"-> {dest.stem}_best{dest.suffix}")

    def log_val(step: int, train_avg: float, gwin: list[float]) -> None:
        v_loss, v_rho = evaluate_val(encoder, head, val_episodes)
        secs = time.time() - t0
        g_mean, g_std, g_min, g_max, g_clip = grad_stats(gwin)
        print(f"  step {step:>6}/{cfg.n_steps} | train {train_avg:6.3f} | "
              f"val_loss {v_loss:6.3f} | val_spearman {v_rho:+.3f}  ({secs:.0f}s)")
        if curve_path:
            with curve_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [step, f"{train_avg:.4f}", f"{v_loss:.4f}", f"{v_rho:.4f}", f"{secs:.0f}",
                     f"{g_mean:.6g}", f"{g_std:.6g}", f"{g_min:.6g}", f"{g_max:.6g}",
                     f"{g_clip:.3f}", f"{optimizer.param_groups[0]['lr']:.3e}"])
        checkpoint(step, v_rho)

    if val_episodes:
        log_val(0, float("nan"), [])  # baseline (untrained) point — no grads yet

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
        # Returns the total encoder grad norm BEFORE clipping — the diagnostic.
        gnorm = float(torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        recent_losses.append(loss.item())
        recent_gnorms.append(gnorm)
        all_gnorms.append(gnorm)
        # Evaluation is checked INDEPENDENTLY of print_every. It used to be nested
        # inside it, which silently required `step` to be a multiple of BOTH — fine
        # while eval_every was 2000 and print_every 500, but an epoch of the curated
        # set is 31,384 steps = 2^3 x 3923 with 3923 prime, so an epoch-aligned
        # eval_every shares no factor with print_every and NO evaluation would ever
        # have fired. Epoch-level monitoring (2026-08-27 supervision) needs this.
        do_eval = bool(val_episodes) and cfg.eval_every and step % cfg.eval_every == 0
        if step % cfg.print_every == 0 or do_eval:
            avg = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
            gwin = list(recent_gnorms)
            g_mean, g_std, g_min, g_max, g_clip = grad_stats(gwin)
            recent_losses.clear()
            recent_gnorms.clear()
            if do_eval:
                log_val(step, avg, gwin)
            else:
                print(f"  step {step:>6}/{cfg.n_steps} | avg loss {avg:6.3f}  "
                      f"({time.time() - t0:.0f}s)")
            print(f"        grad |g| mean {g_mean:.3e} std {g_std:.3e} "
                  f"min {g_min:.3e} max {g_max:.3e} | clipped {g_clip:.0%}")
            # CUDA memory, to settle why the 2026-08-22 full_sp run died of OOM at
            # step ~12k on a 40GB card after 12k healthy steps (report §7 / runlogs).
            # `allocated` is live tensors, `reserved` is what the caching allocator
            # holds from the OS. A rising `allocated` means a genuine leak; flat
            # `allocated` under rising `reserved` means fragmentation instead. `peak`
            # is the high-water mark since the last reset, which catches a single
            # pathological episode that a point-in-time reading would miss.
            if torch.cuda.is_available():
                mb = 1024 ** 2
                print(f"        cuda mem allocated {torch.cuda.memory_allocated() / mb:8.1f} MiB"
                      f" | reserved {torch.cuda.memory_reserved() / mb:8.1f} MiB"
                      f" | peak {torch.cuda.max_memory_allocated() / mb:8.1f} MiB")
                torch.cuda.reset_peak_memory_stats()  # peak is per-window, not cumulative

    # Whole-run gradient summary + a first-pass reading of it. The verdict below is
    # a HINT to speed up triage, not a conclusion: "healthy" only means the encoder
    # received a usable, stable signal — if the loss is still flat, that points at
    # the fundamental-negative case rather than an optimization bug.
    if all_gnorms:
        g_mean, g_std, g_min, g_max, g_clip = grad_stats(all_gnorms)
        if g_mean < 1e-5:
            verdict = "VANISHING — the encoder is getting almost no gradient signal"
        elif g_clip > 0.5:
            verdict = f"LARGE — {g_clip:.0%} of steps hit the {cfg.grad_clip_norm} clip"
        elif g_std > 5 * g_mean:
            verdict = "HIGH-VARIANCE — some batches dominate the updates"
        else:
            verdict = "healthy magnitude/variance (so a flat loss is not a grad problem)"
        print(f"\ngradients over {len(all_gnorms)} steps: mean {g_mean:.3e} | std {g_std:.3e} "
              f"| min {g_min:.3e} | max {g_max:.3e} | clipped {g_clip:.1%}\n  -> {verdict}")

    # Which checkpoint should be evaluated? Say it explicitly — if the best step is
    # 0, training never beat the untrained encoder; if it is well before the end,
    # the run peaked and decayed, so the FINAL weights understate it.
    if cfg.save_encoder_to and best["step"] >= 0:
        where = "the untrained baseline" if best["step"] == 0 else f"step {best['step']}"
        print(f"\nbest val_spearman {best['rho']:+.3f} at {where} "
              f"(final step {cfg.n_steps}); best weights kept separately as *_best.pt")


# =========================================================================== #
# EPOCH MODE — the 2026-09-11 supervision's training protocol.                 #
#                                                                             #
#   one validation epoch  = evaluate the model on ALL validation data          #
#   validation loss       = loss over all validation data POINTS              #
#   validation loss curve = that loss for consecutive epochs                  #
#   up to 50 epochs, lower learning rate, stop when validation loss has not    #
#   improved for 5 epochs in a row (early stopping)                           #
#                                                                             #
# Kept as a separate function rather than folded into train_encoder, so every  #
# step-based run already in report.md stays exactly reproducible.             #
# =========================================================================== #
@torch.no_grad()
def evaluate_val_full(encoder, head, val_episodes) -> tuple[float, float, int]:
    """(point-weighted val loss, mean Spearman, number of validation points).

    The loss is weighted by data point, not by episode: head.loss returns the mean
    over an episode's query molecules, so multiplying by the query count and dividing
    by the grand total gives the mean over every validation point. Averaging episode
    means instead would let a 2-molecule assay count as much as a 300-molecule one.

    Spearman is a rank correlation within one assay and has no per-point
    decomposition, so it stays a mean over episodes; episodes with a single query
    molecule contribute to the loss but cannot have a Spearman.
    """
    was_training = encoder.training
    encoder.eval()
    loss_sum, n_points, rhos = 0.0, 0, []
    for ep in val_episodes:
        emb_c = encoder(ep.context_smiles)
        emb_q = encoder(ep.query_smiles)
        n_q = len(ep.query_smiles)
        loss_sum += float(head.loss(emb_c, ep.context_y, emb_q, ep.query_y)) * n_q
        n_points += n_q
        if n_q > 1:
            preds = head.predict(emb_c, ep.context_y, emb_q)
            rho = float(spearmanr(preds.cpu().numpy(), ep.query_y.flatten().cpu().numpy()).statistic)
            if rho == rho:
                rhos.append(rho)
    if was_training:
        encoder.train()
    loss = loss_sum / n_points if n_points else float("nan")
    rho = sum(rhos) / len(rhos) if rhos else float("nan")
    return loss, rho, n_points


@dataclass
class EpochConfig:
    """Settings for train_encoder_epochs."""
    n_epochs: int
    patience: int                 # epochs without val-loss improvement before stopping
    lr: float
    max_context: int              # TRAINING episode caps (validation has its own)
    max_query: int
    hidden_size: int
    seed: int = 0
    print_every: int = 1000       # progress lines inside an epoch, in steps
    grad_clip_norm: float = GRAD_CLIP_NORM
    val_curve_csv: str = ""
    save_encoder_to: str = ""
    # Stop cleanly before this UTC time (ISO 8601, e.g. "2026-10-01T00:00:00+00:00").
    # The check is PREDICTIVE: the run stops when the next epoch would not finish in
    # time, so the last completed epoch lands before the deadline, not after it.
    deadline_utc: str = ""
    min_delta: float = 0.0        # an improvement must beat the best by more than this
    # Refuse to start if this run's curve CSV or checkpoints already exist. A job that the
    # cluster requeues (e.g. after maintenance) restarts from scratch, and without this
    # guard it silently overwrites the finished run's results — which is how the 27-epoch
    # full_sp_50ep run lost its curve and its _best.pt on 2026-10-01. Set True only to
    # deliberately replace a previous run.
    overwrite: bool = False


EPOCH_CURVE_COLUMNS = [
    "epoch", "steps", "train_loss", "val_loss", "val_spearman", "val_points", "seconds",
    "epoch_seconds", "grad_mean", "grad_clip_frac", "lr", "best_val_loss",
    "epochs_without_improvement", "event",
]


def train_encoder_epochs(encoder, head, assays, val_episodes, cfg: EpochConfig, t0: float,
                         *, repo_root: Path) -> dict:
    """Train in whole epochs, validate on the full set after each, stop early.

    Epoch 0 is the untrained encoder and competes for "best" like any other epoch: if
    no epoch beats it within `patience`, training stops and *_best.pt honestly holds
    the untrained weights.

    Writes one CSV row per epoch and three kinds of checkpoint:
      <name>.pt                latest weights (overwritten each epoch)
      <name>_best.pt           lowest validation loss so far
      <name>_epochs/epoch_NNN.pt   every epoch, for later analysis (e.g. feature_stats)

    Returns a summary dict: epochs_run, best_epoch, best_val_loss, stop_reason.
    """
    from datetime import datetime, timedelta, timezone

    if not val_episodes:
        raise ValueError("epoch mode needs validation episodes — early stopping has nothing to watch")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr)
    rng = random.Random(cfg.seed)
    deadline = datetime.fromisoformat(cfg.deadline_utc) if cfg.deadline_utc else None
    encoder.train()

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    curve_path = _resolve(cfg.val_curve_csv) if cfg.val_curve_csv else None
    dest = _resolve(cfg.save_encoder_to) if cfg.save_encoder_to else None

    # ---- guards: run BEFORE anything is written ------------------------------
    # 1. A deadline that has already passed means there is no time for even one epoch.
    #    Without this, a requeued job that starts after its deadline still runs epoch 1
    #    (the predictive check below needs one epoch's duration to estimate), then stops —
    #    having overwritten everything on the way.
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        reason = f"deadline {cfg.deadline_utc} had already passed at start; nothing was run"
        print(f"\n{reason}")
        return {"epochs_run": 0, "best_epoch": None, "best_val_loss": float("nan"),
                "stop_reason": reason}
    # 2. Never overwrite a previous run's results unless asked to.
    if not cfg.overwrite:
        existing = [p for p in (curve_path, dest,
                                dest.with_name(f"{dest.stem}_best{dest.suffix}") if dest else None)
                    if p is not None and p.exists()]
        epochs_dir = dest.with_name(f"{dest.stem}_epochs") if dest else None
        if epochs_dir is not None and epochs_dir.is_dir() and any(epochs_dir.iterdir()):
            existing.append(epochs_dir)
        if existing:
            raise FileExistsError(
                "refusing to overwrite the results of a previous run:\n  "
                + "\n  ".join(str(p) for p in existing)
                + "\nUse new output names, move the old files away, or set overwrite=True.")

    if curve_path:
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        with curve_path.open("w", newline="") as f:
            csv.writer(f).writerow(EPOCH_CURVE_COLUMNS)

    def save(epoch: int, is_best: bool) -> None:
        if dest is None:
            return
        save_encoder(encoder, dest, cfg.hidden_size)
        epochs_dir = dest.with_name(f"{dest.stem}_epochs")
        epochs_dir.mkdir(parents=True, exist_ok=True)
        save_encoder(encoder, epochs_dir / f"epoch_{epoch:03d}{dest.suffix}", cfg.hidden_size)
        if is_best:
            save_encoder(encoder, dest.with_name(f"{dest.stem}_best{dest.suffix}"), cfg.hidden_size)

    def write_row(epoch, steps, train_loss, v_loss, v_rho, n_pts, epoch_secs, gnorms, event):
        g_mean = sum(gnorms) / len(gnorms) if gnorms else float("nan")
        g_clip = (sum(1 for g in gnorms if g > cfg.grad_clip_norm) / len(gnorms)
                  if gnorms else float("nan"))
        print(f"  epoch {epoch:>3}/{cfg.n_epochs} | train {train_loss:6.3f} | "
              f"val_loss {v_loss:.4f} | val_spearman {v_rho:+.4f} | "
              f"best {best['loss']:.4f} @ epoch {best['epoch']} | "
              f"no improvement {stale}/{cfg.patience} | {epoch_secs / 3600:.2f} h"
              f"{'  <- ' + event if event else ''}")
        if curve_path:
            with curve_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, steps, f"{train_loss:.4f}", f"{v_loss:.4f}", f"{v_rho:.4f}", n_pts,
                    f"{time.time() - t0:.0f}", f"{epoch_secs:.0f}", f"{g_mean:.6g}",
                    f"{g_clip:.3f}", f"{optimizer.param_groups[0]['lr']:.3e}",
                    f"{best['loss']:.4f}", stale, event])

    # ---- epoch 0: the untrained baseline -------------------------------------
    v0 = time.time()
    v_loss, v_rho, n_pts = evaluate_val_full(encoder, head, val_episodes)
    best = {"loss": v_loss, "epoch": 0}
    stale = 0
    print(f"validation set: {len(val_episodes)} episodes, {n_pts:,} data points "
          f"(one full pass took {(time.time() - v0) / 60:.1f} min)")
    save(0, is_best=True)
    write_row(0, 0, float("nan"), v_loss, v_rho, n_pts, 0.0, [], "baseline")

    steps = 0
    epoch_secs_hist: list[float] = []
    stop_reason = f"completed all {cfg.n_epochs} epochs"
    epoch = 0
    for epoch in range(1, cfg.n_epochs + 1):
        e0 = time.time()
        order = list(assays)
        rng.shuffle(order)                 # one pass over EVERY training assay
        losses: list[float] = []
        gnorms: list[float] = []
        for assay in order:
            episode = data.make_episode(assay, max_context=cfg.max_context,
                                        max_query=cfg.max_query, rng=rng)
            if episode is None:            # too small, or labels near-constant
                continue
            optimizer.zero_grad()
            loss = head.loss(encoder(episode.context_smiles), episode.context_y,
                             encoder(episode.query_smiles), episode.query_y)
            loss.backward()
            gnorms.append(float(torch.nn.utils.clip_grad_norm_(encoder.parameters(),
                                                               cfg.grad_clip_norm)))
            optimizer.step()
            losses.append(loss.item())
            steps += 1
            if cfg.print_every and steps % cfg.print_every == 0:
                mem = ""
                if torch.cuda.is_available():
                    mem = (f" | cuda peak {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
                    torch.cuda.reset_peak_memory_stats()
                recent = losses[-cfg.print_every:]
                print(f"    epoch {epoch} step {steps:>7} | avg loss "
                      f"{sum(recent) / len(recent):6.3f}{mem}  ({time.time() - t0:.0f}s)")

        v_loss, v_rho, n_pts = evaluate_val_full(encoder, head, val_episodes)
        improved = v_loss < best["loss"] - cfg.min_delta
        if improved:
            best = {"loss": v_loss, "epoch": epoch}
            stale = 0
        else:
            stale += 1
        save(epoch, is_best=improved)
        epoch_secs = time.time() - e0
        epoch_secs_hist.append(epoch_secs)
        train_loss = sum(losses) / len(losses) if losses else float("nan")

        event = "best" if improved else ""
        stop = False
        if stale >= cfg.patience:
            stop_reason = (f"early stopping: no improvement in validation loss for "
                           f"{cfg.patience} epochs (best at epoch {best['epoch']})")
            event, stop = "early_stop", True
        elif deadline is not None:
            mean_epoch = sum(epoch_secs_hist) / len(epoch_secs_hist)
            if datetime.now(timezone.utc) + timedelta(seconds=mean_epoch) > deadline:
                stop_reason = (f"deadline: another epoch (~{mean_epoch / 3600:.1f} h) would "
                               f"not finish before {cfg.deadline_utc}")
                event, stop = "deadline", True
        write_row(epoch, steps, train_loss, v_loss, v_rho, n_pts, epoch_secs, gnorms, event)
        if stop:
            break

    print(f"\nstopped after epoch {epoch}: {stop_reason}")
    print(f"best validation loss {best['loss']:.4f} at epoch {best['epoch']}"
          + (" — the UNTRAINED encoder; no epoch improved on it" if best["epoch"] == 0 else ""))
    return {"epochs_run": epoch, "best_epoch": best["epoch"], "best_val_loss": best["loss"],
            "stop_reason": stop_reason}
