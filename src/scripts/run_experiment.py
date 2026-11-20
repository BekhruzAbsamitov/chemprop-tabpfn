"""scripts/run_experiment.py — train the Chemprop encoder on ChEMBL. Press Run.

ONE configurable training script for every run we do. Instead of a separate file
per target/scale (the old run_experiment_{gpu,d2r,d2r_long,usp7,saureus}.py), pick a
named PRESET below and press Run. Each preset is just a bundle of the settings that
differ; everything else inherits the `Preset` defaults (the fast local "dev" run).

  dev            fast local smoke test on dev ChEMBL (the defaults)
  full           full cleaned ChEMBL, thesis scale (§5b) — the all-of-ChEMBL firehose
  d2r            D2R adaptive, domain-matched GPCR set, 20k steps (§5c)
  usp7           USP7 adaptive, small protease set, eval often (§5c)
  saureus        S. aureus adaptive, organism MIC set, 8k steps (§5c)
  d2r_long       D2R adaptive, 100k steps (~32 passes)   } the "train it long" test
  usp7_long      USP7 adaptive, 100k steps (~719 passes) } (§7) — rules out
  saureus_long   S. aureus adaptive, 100k (~125 passes)  } "it needs more steps"

Trains the encoder differentiably (gradients flow through the frozen TabPFN) and
tracks an HONEST in-domain learning curve: a fixed held-out set of ChEMBL/adaptive
test episodes is scored every `eval_every` steps (val_loss + val_spearman), because
the per-step training loss is too noisy to read (a different assay each step).

Evaluation on the target datasets (S. aureus / D2R / USP7) is a SEPARATE step —
run scripts/model_evaluation.py (or evaluate_adaptive.py) on the saved encoder.

To run: set EXPERIMENT below and press Run in the IDE. (On the cluster you may pass
a preset name as the first argument — `python run_experiment.py d2r_long` — which
overrides EXPERIMENT; the IDE Run button passes no argument and uses the constant.)
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402
from models.checkpoints import save_encoder  # noqa: E402
from models.encoder import MoleculeEncoder  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402
from training import (EpochConfig, TrainConfig, build_val_episodes,  # noqa: E402
                      train_encoder, train_encoder_epochs)


# --------------------------------------------------------------------------- #
# A run's settings. Defaults = the fast local "dev" run; presets override only  #
# what differs. Cap on training-episode size (max_train_*): a differentiable    #
# TabPFN backward over a giant assay exhausts even an 80 GB GPU; rarely binds.  #
# --------------------------------------------------------------------------- #
@dataclass
class Preset:
    # Data. use_full_data: False = small dev ChEMBL (fast), True = full cleaned file.
    # train_data_file overrides both with a specific parquet (same 4-col schema, its
    # own train/test split) — e.g. an adaptive set; "" = dev/full by use_full_data.
    use_full_data: bool = False
    train_data_file: str = ""
    # Model / optimization.
    hidden_size: int = 128         # encoder width (local 128; thesis/GPU 300)
    n_estimators: int = 1          # TabPFN passes to average (1 fast local; 8+ GPU)
    n_train_steps: int = 500       # how many training episodes to run
    learning_rate: float = 1e-4    # 1e-3 thrashed on dev; 1e-4 is stable
    loss_fn: str = "nll"           # "nll" (recommended) | "mse" | "huber"
    # Max encoder grad norm per step. Measured norms are ~150, so the historical
    # 1.0 clips EVERY step (see report §7 branch 1) — raise it to let magnitude
    # differences between batches through.
    grad_clip_norm: float = 1.0
    max_train_context: int = 128
    max_train_query: int = 128
    # In-domain learning curve: fixed held-out test episodes scored every
    # eval_every steps (a multiple of print_every). 0 disables it.
    eval_every: int = 500
    n_val_episodes: int = 32
    val_curve_csv: str = "experiments/train_val_curve.csv"  # incremental; "" = off
    # Output. save_encoder_to: also checkpointed at every eval; "" = don't save.
    save_encoder_to: str = ""
    # Keep a step-tagged encoder every N steps in `<name>_steps/` (0 = off) — the
    # trajectory the CKA-across-training study needs. step_000000 is E_initial.
    checkpoint_every: int = 0
    # "none" (constant lr) or "cosine" (anneal lr -> ~0 across the run).
    lr_schedule: str = "none"
    # EPOCH MODE (2026-09-11 supervision). n_epochs > 0 switches training from a fixed
    # step count to whole epochs, with validation on the WHOLE held-out set after each
    # epoch and early stopping; n_train_steps / eval_every / n_val_episodes are then
    # unused. patience = epochs without val-loss improvement before stopping.
    n_epochs: int = 0
    patience: int = 5
    deadline_utc: str = ""        # stop cleanly before this UTC time; "" = no deadline
    # Validation episode caps in epoch mode. None = uncapped, so the validation loss
    # really covers every held-out data point. Safe memory-wise because validation runs
    # without gradients; the untrained epoch-0 pass exercises the largest episode first,
    # so a problem would surface within the first hour rather than days in.
    val_max_context: int | None = None
    val_max_query: int | None = None
    seed: int = 0
    print_every: int = 100


PRESETS: dict[str, Preset] = {
    # Fast local smoke test on dev ChEMBL — the defaults. Press Run to use it.
    "dev": Preset(),

    # Full cleaned ChEMBL, thesis scale (§5b). ~28% of 180k assays once; the fixed
    # val curve reveals learning (or its absence) long before the last step.
    "full": Preset(
        use_full_data=True, hidden_size=300, n_estimators=8,
        n_train_steps=50000, eval_every=2000, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve.csv",
        save_encoder_to="experiments/trained_encoder_full.pt", print_every=500),

    # Single-protein curated set (2026-08-20 supervision). Stricter ChEMBL 36 cut:
    # binding assays, single-protein targets, confidence 9, IC50/Ki/Kd, >=5 measurements.
    # 34.2k train assays — 5x FEWER than `full`, but far cleaner and, decisively, nearly
    # leakage-free against the AL pools (S. aureus 77%->5%, D2R 91%->1%). Retraining here
    # is what removes the leakage caveat qualifying two targets in §5e.
    # 100k steps is ~2.9 passes; step checkpoints kept so §8b/§9 can be redone on it.
    # Episode caps are 64, not the usual 128. The first attempt (2026-08-22, cluster
    # 182401) OOM'd at step ~12k on a 40GB card. The memory instrumentation showed why:
    # live memory was a flat 465 MiB — no leak — but a single step PEAKED at 38.6 GiB of
    # a 39.4 GiB card, i.e. 98 % of capacity, so the first episode larger than anything
    # seen so far tipped it over. Attention is quadratic in episode length, so halving
    # the cap cuts the peak to ~0.25x. The scientific cost is negligible: it truncates
    # 1.9 % of episodes instead of 0.7 % (still a uniform random subset, since indices
    # are pre-shuffled) and leaves the MEDIAN episode — 15 molecules — untouched.
    "full_sp": Preset(
        train_data_file="data/curated/chembl36_single_protein_clean.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=100000,
        max_train_context=64, max_train_query=64,
        eval_every=2000, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_full_sp.csv",
        save_encoder_to="experiments/trained_encoder_full_sp.pt", print_every=500,
        checkpoint_every=2000),

    # 10 EPOCHS on the curated single-protein set (2026-08-27 supervision: "train for
    # at least 10 epochs, monitor validation loss and spearman on epoch-level, plot the
    # first 10 epochs, is there a trend to convergence?").
    #
    # 1 epoch = 31,384 usable episodes (34,192 assays, 2,808 of which cannot form an
    # episode), and one step consumes one episode -> 10 epochs = 313,840 steps. At the
    # measured 1,423 s / 1,000 steps that is ~124 h, about 5.2 days.
    #
    # eval_every = 3,923 = 1/8 epoch. Finer than "epoch-level" so the curve plots
    # smoothly (80 points), while every 8th evaluation lands EXACTLY on an epoch
    # boundary because 8 x 3,923 = 31,384. NOTE this only works because the eval check
    # is now independent of print_every: an epoch is 2^3 x 3,923 with 3,923 prime, so
    # under the old nested check (step % print_every == 0 AND step % eval_every == 0)
    # an epoch-aligned eval_every would have produced NO evaluations at all.
    #
    # NO resume: a crash loses the ability to continue, but not the results — the curve
    # CSV is written incrementally and step-tagged encoders land every 3,923 steps.
    "full_sp_10ep": Preset(
        train_data_file="data/curated/chembl36_single_protein_clean.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=313840,
        max_train_context=64, max_train_query=64,
        eval_every=3923, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_full_sp_10ep.csv",
        save_encoder_to="experiments/trained_encoder_full_sp_10ep.pt", print_every=500,
        checkpoint_every=3923),

    # ---- 2026-09-03 supervision: the validation curve fluctuates undiminished to the
    # last step, and a lower learning rate was suggested. WHY it cannot settle on its
    # own: gradients (33-75) exceed grad_clip_norm=1.0 on 100 % of steps, so every
    # update is rescaled to exactly lr x 1.0 and the magnitude never shrinks; there was
    # also no schedule. Both mechanisms that normally quiet a run are therefore absent.
    #
    # Two variants, because "lower learning rate" and "decay the learning rate" are
    # different fixes and the observation does not distinguish them. Everything else is
    # identical to full_sp_10ep, so all three runs are directly comparable.

    # (a) The literal suggestion: same schedule (none), lr lowered 3.3x.
    "full_sp_10ep_lowlr": Preset(
        train_data_file="data/curated/chembl36_single_protein_clean.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=313840,
        max_train_context=64, max_train_query=64, learning_rate=3e-5,
        eval_every=3923, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_full_sp_10ep_lowlr.csv",
        save_encoder_to="experiments/trained_encoder_full_sp_10ep_lowlr.pt",
        print_every=500, checkpoint_every=3923),

    # (b) Cosine decay from the rate that demonstrably learned (1e-4) to ~0. Keeps the
    # early phase — where the previous run showed its only real progress, val_loss
    # -0.0335/epoch at t=-7.00 — and quiets only the plateau.
    "full_sp_10ep_cosine": Preset(
        train_data_file="data/curated/chembl36_single_protein_clean.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=313840,
        max_train_context=64, max_train_query=64,
        learning_rate=1e-4, lr_schedule="cosine",
        eval_every=3923, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_full_sp_10ep_cosine.csv",
        save_encoder_to="experiments/trained_encoder_full_sp_10ep_cosine.pt",
        print_every=500, checkpoint_every=3923),

    # ---- 2026-09-11 supervision: the training protocol, done properly ----------
    #   one validation epoch = evaluate on ALL validation data
    #   validation loss      = loss over all validation data points
    #   train longer than 10 epochs with a smaller learning rate
    #   50 epochs, stop if validation loss does not improve for 5 epochs in a row
    # 3e-5 is the smaller rate already tested: at 10 epochs its validation loss was still
    # falling (slope -0.0071/epoch, t=-2.92), so it had room to go. Each epoch is ~12.4 h
    # of training plus ~1 h of full validation, so the 1 October deadline caps the run at
    # roughly 30 epochs if early stopping has not fired by then.
    "full_sp_50ep": Preset(
        train_data_file="data/curated/chembl36_single_protein_clean.parquet",
        hidden_size=300, n_estimators=8,
        max_train_context=64, max_train_query=64,
        learning_rate=3e-5, n_epochs=50, patience=5,
        deadline_utc="2026-10-01T00:00:00+00:00",
        val_curve_csv="experiments/val_curve_full_sp_50ep.csv",
        save_encoder_to="experiments/trained_encoder_full_sp_50ep.pt",
        print_every=2000),

    # Independent REPLICATE of full_sp_50ep (2026-10-05): identical except seed=1, which
    # changes the encoder's initialisation, the training order and the training episodes'
    # support/query splits. The VALIDATION set is unchanged: build_val_episodes seeds each
    # assay's split by its assay id, and with all held-out assays used the run seed only
    # reorders them — so the two curves are measured on the same 3,497 episodes and can be
    # compared directly. Own output names, so it can never touch the seed-0 run's files
    # (and train_encoder_epochs now refuses to overwrite in any case).
    "full_sp_50ep_seed1": Preset(
        train_data_file="data/curated/chembl36_single_protein_clean.parquet",
        hidden_size=300, n_estimators=8,
        max_train_context=64, max_train_query=64,
        learning_rate=3e-5, n_epochs=50, patience=5, seed=1,
        deadline_utc="2026-10-26T00:00:00+00:00",
        val_curve_csv="experiments/val_curve_full_sp_50ep_seed1.csv",
        save_encoder_to="experiments/trained_encoder_full_sp_50ep_seed1.pt",
        print_every=2000),

    # D2R adaptive (§5c) — domain-matched GPCR set (~3.1k train assays); 20k ~6 passes.
    "d2r": Preset(
        train_data_file="data/curated/train_adaptive_D2R.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=20000,
        eval_every=1000, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_d2r.csv",
        save_encoder_to="experiments/trained_encoder_d2r.pt", print_every=200),

    # D2R long (§7) — same set, 100k steps (~32 passes): the "does it just need more
    # training?" test. If val stays flat across 32 epochs, "too few steps" is ruled out.
    "d2r_long": Preset(
        train_data_file="data/curated/train_adaptive_D2R.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=100000,
        eval_every=2000, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_d2r_long.csv",
        save_encoder_to="experiments/trained_encoder_d2r_long.pt", print_every=500,
        checkpoint_every=2000),  # 50 step-tagged encoders for the CKA trajectory

    # USP7 adaptive (§5c) — SMALL protease set (~139 train assays). Overfitting the
    # meta-train distribution is a real risk, so eval often and read the val PEAK.
    "usp7": Preset(
        train_data_file="data/curated/train_adaptive_USP7.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=5000,
        eval_every=250, n_val_episodes=16,
        val_curve_csv="experiments/train_val_curve_usp7.csv",
        save_encoder_to="experiments/trained_encoder_usp7.pt", print_every=250),

    # USP7 long (§7) — 100k steps over only ~139 train assays is ~719 passes, far
    # into memorization territory: if the encoder CAN fit this meta-train set, train
    # loss must fall here (and val will show the overfit gap). A flat train loss even
    # at 719 passes is the strongest possible "no signal" evidence. Only 16 val
    # episodes exist, so single-point val_spearman swings +-0.1 — read the trend.
    "usp7_long": Preset(
        train_data_file="data/curated/train_adaptive_USP7.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=100000,
        eval_every=2000, n_val_episodes=16,
        val_curve_csv="experiments/train_val_curve_usp7_long.csv",
        save_encoder_to="experiments/trained_encoder_usp7_long.pt", print_every=500,
        checkpoint_every=2000),  # 50 step-tagged encoders for the CKA trajectory

    # S. aureus adaptive (§5c) — organism MIC set (~802 train assays); 8k ~10 passes.
    "saureus": Preset(
        train_data_file="data/curated/train_adaptive_S_aureus.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=8000,
        eval_every=500, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_saureus.csv",
        save_encoder_to="experiments/trained_encoder_saureus.pt", print_every=250),

    # S. aureus long (§7) — same organism set, 100k steps (~125 passes). The 8k run
    # ended BELOW its step-0 baseline (0.203 vs 0.248); this checks whether that is
    # drift/noise or a real downward trend given 12x the training.
    "saureus_long": Preset(
        train_data_file="data/curated/train_adaptive_S_aureus.parquet",
        hidden_size=300, n_estimators=8, n_train_steps=100000,
        eval_every=2000, n_val_episodes=64,
        val_curve_csv="experiments/train_val_curve_saureus_long.csv",
        save_encoder_to="experiments/trained_encoder_saureus_long.pt", print_every=500,
        checkpoint_every=2000),  # 50 step-tagged encoders for the CKA trajectory
}

# Which preset to run. Edit this, then press Run. (Cluster: pass a preset name as
# the first CLI arg to override, e.g. `python run_experiment.py d2r_long`.)
EXPERIMENT = "dev"


def resolve_preset() -> tuple[str, Preset]:
    """The preset to run: CLI arg (cluster convenience) if given, else EXPERIMENT."""
    name = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return name, PRESETS[name]


def main() -> None:
    _load_hf_token()
    name, p = resolve_preset()
    torch.manual_seed(p.seed)
    t0 = time.time()
    print(f"=== experiment preset: {name} ===")

    # 1. Data — training ChEMBL/adaptive set + a fixed in-domain val set.
    train_path = (REPO_ROOT / p.train_data_file) if p.train_data_file else None
    if train_path:
        print(f"training file: {p.train_data_file}")
    train_assays = data.load_chembl_assays("train", use_full=p.use_full_data, path=train_path)
    print(f"loaded {len(train_assays)} training assays  ({time.time() - t0:.0f}s)")
    val_episodes = []
    if p.n_epochs:
        test_assays = data.load_chembl_assays("test", use_full=p.use_full_data, path=train_path)
        val_episodes = build_val_episodes(
            test_assays, n_episodes=len(test_assays),          # ALL of them
            max_context=p.val_max_context, max_query=p.val_max_query,
            rng=random.Random(p.seed),
        )
        print(f"  {len(val_episodes)} validation episodes = the whole held-out set "
              f"({len(test_assays)} assays; the rest are too small to form an episode)")
    elif p.eval_every and p.n_val_episodes:
        test_assays = data.load_chembl_assays("test", use_full=p.use_full_data, path=train_path)
        val_episodes = build_val_episodes(
            test_assays, n_episodes=p.n_val_episodes,
            max_context=p.max_train_context, max_query=p.max_train_query,
            rng=random.Random(p.seed),
        )
        print(f"  {len(val_episodes)} fixed in-domain val episodes (held-out test)")

    # 2. Models.
    encoder = MoleculeEncoder(hidden_size=p.hidden_size)
    head = TabPFNHead(n_estimators=p.n_estimators, loss_fn=p.loss_fn)
    encoder.to(head.device)
    print(f"  device: {head.device} | loss: {p.loss_fn} | hidden_size: {p.hidden_size} | "
          f"n_estimators: {p.n_estimators} | full_data: {p.use_full_data}")

    # 3. Train (logs the in-domain curve + checkpoints along the way).
    if p.n_epochs:
        ecfg = EpochConfig(
            n_epochs=p.n_epochs, patience=p.patience, lr=p.learning_rate,
            max_context=p.max_train_context, max_query=p.max_train_query,
            hidden_size=p.hidden_size, seed=p.seed, print_every=p.print_every,
            grad_clip_norm=p.grad_clip_norm, val_curve_csv=p.val_curve_csv,
            save_encoder_to=p.save_encoder_to, deadline_utc=p.deadline_utc)
        print(f"\ntraining for up to {p.n_epochs} epochs (patience {p.patience}, "
              f"lr {p.learning_rate:g}{', deadline ' + p.deadline_utc if p.deadline_utc else ''}) ...")
        summary = train_encoder_epochs(encoder, head, train_assays, val_episodes, ecfg, t0,
                                       repo_root=REPO_ROOT)
        print(f"\n{summary}")
        print(f"total time: {time.time() - t0:.0f}s")
        return

    cfg = TrainConfig(
        n_steps=p.n_train_steps, lr=p.learning_rate,
        max_context=p.max_train_context, max_query=p.max_train_query, hidden_size=p.hidden_size,
        seed=p.seed, eval_every=p.eval_every, print_every=p.print_every,
        val_curve_csv=p.val_curve_csv, save_encoder_to=p.save_encoder_to,
        grad_clip_norm=p.grad_clip_norm, checkpoint_every=p.checkpoint_every,
        lr_schedule=p.lr_schedule,
    )
    print(f"\ntraining encoder for {p.n_train_steps} steps ...")
    train_encoder(encoder, head, train_assays, val_episodes, cfg, t0, repo_root=REPO_ROOT)

    # 4. Final save (also checkpointed during training when val runs).
    if p.save_encoder_to:
        path = Path(p.save_encoder_to)
        save_encoder(encoder, path if path.is_absolute() else REPO_ROOT / path, p.hidden_size)
        print(f"  saved trained encoder -> {p.save_encoder_to}")

    print(f"\ntraining complete. in-domain curve -> {p.val_curve_csv or 'stdout above'}")
    print("evaluate the saved encoder with scripts/model_evaluation.py")
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
