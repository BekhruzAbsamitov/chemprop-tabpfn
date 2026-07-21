"""scripts/run_experiment.py — train the Chemprop encoder on ChEMBL. Press Run.

Trains the encoder differentiably (gradients flow through the frozen TabPFN) and
tracks an HONEST in-domain learning curve: a fixed held-out set of ChEMBL test
episodes is scored every EVAL_EVERY steps (val_loss + val_spearman), because the
per-step training loss is too noisy to read (a different assay each step).

Evaluation on the target datasets (S. aureus / D2R / USP7) is a SEPARATE step —
run scripts/model_evaluation.py on the saved encoder afterwards.

The loop and helpers live in src/training.py; this file is just the settings and
wiring. No flags / env vars — open and press Run.
"""

from __future__ import annotations

import random
import sys
import time
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
from training import TrainConfig, build_val_episodes, train_encoder  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
USE_FULL_DATA = False  # False = small dev ChEMBL (fast); True = full cleaned file
HIDDEN_SIZE = 128      # encoder width (fast local = 128; thesis/GPU scale = 300)
N_TRAIN_STEPS = 500    # how many training episodes to run
LEARNING_RATE = 1e-4   # training step size
N_ESTIMATORS = 1       # TabPFN passes to average (1 = fast local; 8+ on a GPU)
LOSS_FN = "nll"        # "nll" (recommended) | "mse" | "huber"

# Cap on training-episode size — a differentiable TabPFN backward over a giant
# assay exhausts even an 80 GB GPU. Rarely binds (most assays are small).
MAX_TRAIN_CONTEXT = 128
MAX_TRAIN_QUERY = 128

SEED = 0
PRINT_EVERY = 100      # print training progress every N steps

# In-domain learning curve: fixed held-out ChEMBL test episodes, scored every
# EVAL_EVERY steps. EVAL_EVERY must be a multiple of PRINT_EVERY. 0 disables it.
EVAL_EVERY = 500
N_VAL_EPISODES = 32
VAL_CURVE_CSV = "experiments/train_val_curve.csv"  # incremental; "" = don't save

# Where to save the trained encoder (also checkpointed at every eval). "" = don't.
SAVE_ENCODER_TO = ""


def main() -> None:
    _load_hf_token()
    torch.manual_seed(SEED)
    t0 = time.time()

    # 1. Data — training ChEMBL + a fixed in-domain val set.
    train_assays = data.load_chembl_assays("train", use_full=USE_FULL_DATA)
    print(f"loaded {len(train_assays)} ChEMBL training assays  ({time.time() - t0:.0f}s)")
    val_episodes = []
    if EVAL_EVERY and N_VAL_EPISODES:
        test_assays = data.load_chembl_assays("test", use_full=USE_FULL_DATA)
        val_episodes = build_val_episodes(
            test_assays, n_episodes=N_VAL_EPISODES,
            max_context=MAX_TRAIN_CONTEXT, max_query=MAX_TRAIN_QUERY, rng=random.Random(SEED),
        )
        print(f"  {len(val_episodes)} fixed in-domain val episodes (ChEMBL test held-out)")

    # 2. Models.
    encoder = MoleculeEncoder(hidden_size=HIDDEN_SIZE)
    head = TabPFNHead(n_estimators=N_ESTIMATORS, loss_fn=LOSS_FN)
    encoder.to(head.device)
    print(f"  device: {head.device} | loss: {LOSS_FN} | hidden_size: {HIDDEN_SIZE} | "
          f"n_estimators: {N_ESTIMATORS} | full_data: {USE_FULL_DATA}")

    # 3. Train (logs the in-domain curve + checkpoints along the way).
    cfg = TrainConfig(
        n_steps=N_TRAIN_STEPS, lr=LEARNING_RATE,
        max_context=MAX_TRAIN_CONTEXT, max_query=MAX_TRAIN_QUERY, hidden_size=HIDDEN_SIZE,
        seed=SEED, eval_every=EVAL_EVERY, print_every=PRINT_EVERY,
        val_curve_csv=VAL_CURVE_CSV, save_encoder_to=SAVE_ENCODER_TO,
    )
    print(f"\ntraining encoder for {N_TRAIN_STEPS} steps ...")
    train_encoder(encoder, head, train_assays, val_episodes, cfg, t0, repo_root=REPO_ROOT)

    # 4. Final save (also checkpointed during training when val runs).
    if SAVE_ENCODER_TO:
        path = Path(SAVE_ENCODER_TO)
        save_encoder(encoder, path if path.is_absolute() else REPO_ROOT / path, HIDDEN_SIZE)
        print(f"  saved trained encoder -> {SAVE_ENCODER_TO}")

    print(f"\ntraining complete. in-domain curve -> {VAL_CURVE_CSV or 'stdout above'}")
    print("evaluate the saved encoder with scripts/model_evaluation.py")
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
