"""src/scripts/run_experiment_gpu.py — the FULL-SCALE version, for a GPU.

Exactly the same experiment as run_experiment.py, just with the big settings the
thesis actually claims. This is the fair test of RQ4: train on the FULL ChEMBL
dataset (all 248k assays, not the 1% dev subset), with a wider encoder and a
stronger TabPFN ensemble, then compare untrained / trained / Morgan on S. aureus.

You don't edit run_experiment.py — this file just overrides its settings and
runs it. Submit on the cluster with cluster/run_gpu.slurm, or, if you already
have a GPU, press Run.
"""

from __future__ import annotations

from scripts import run_experiment as exp

# --------------------------------------------------------------------------- #
# FULL-SCALE SETTINGS (GPU). Edit here — run_experiment.py stays untouched.    #
# --------------------------------------------------------------------------- #
exp.USE_FULL_DATA = True      # the full cleaned ChEMBL (180k train assays)
exp.HIDDEN_SIZE = 300         # thesis-scale encoder width (local used 128)
exp.N_ESTIMATORS = 8          # stronger TabPFN ensemble (local used 1)
exp.N_TRAIN_STEPS = 50000     # substantial coverage (~28% of 180k assays once);
                              # ~1.4s/step on an A100 => ~19h. The fixed-val curve
                              # below reveals learning (or its absence) long before
                              # then — watch val_loss / val_spearman, not train loss.
exp.LEARNING_RATE = 1e-4      # 1e-3 thrashed on dev; 1e-4 is stable
exp.LOSS_FN = "nll"           # "nll" | "mse" | "huber"

# Just TRAIN the encoder now (+ in-domain val curve). We evaluate the saved
# encoder on the targets separately afterwards, so don't load S. aureus here.
exp.EVAL_ON_TARGET = False

# IN-DOMAIN LEARNING CURVE — the honest signal the previous full run lacked.
exp.EVAL_EVERY = 2000         # eval the fixed ChEMBL-test val set every N steps
exp.N_VAL_EPISODES = 64       # steadier in-domain estimate
exp.VAL_CURVE_CSV = "experiments/train_val_curve.csv"

exp.N_CONTEXT = 500           # bigger context = TabPFN has more to learn from
exp.N_QUERY = 500
exp.N_SPLITS = 10             # steadier, more trustworthy transfer metrics

exp.SEED = 0
exp.PRINT_EVERY = 500
exp.SAVE_ENCODER_TO = "experiments/trained_encoder_full.pt"  # keep the result!


if __name__ == "__main__":
    exp.main()
