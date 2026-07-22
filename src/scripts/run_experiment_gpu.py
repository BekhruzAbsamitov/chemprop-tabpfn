from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import run_experiment as exp  # noqa: E402


exp.USE_FULL_DATA = True      # the full cleaned ChEMBL (180k train assays)
exp.HIDDEN_SIZE = 300         # thesis-scale encoder width (local used 128)
exp.N_ESTIMATORS = 8          # stronger TabPFN ensemble (local used 1)
exp.N_TRAIN_STEPS = 50000     # substantial coverage (~28% of 180k assays once);
                              # ~1.4s/step on an A100 => ~19h. The fixed-val curve
                              # reveals learning (or its absence) long before then
                              # — watch val_loss / val_spearman, not train loss.
exp.LEARNING_RATE = 1e-4      # 1e-3 thrashed on dev; 1e-4 is stable
exp.LOSS_FN = "nll"           # "nll" | "mse" | "huber"

# IN-DOMAIN LEARNING CURVE — the honest signal the previous full run lacked.
exp.EVAL_EVERY = 2000         # eval the fixed ChEMBL-test val set every N steps
exp.N_VAL_EPISODES = 64       # steadier in-domain estimate
exp.VAL_CURVE_CSV = "experiments/train_val_curve.csv"

exp.SEED = 0
exp.PRINT_EVERY = 500
exp.SAVE_ENCODER_TO = "experiments/trained_encoder_full.pt"  # keep the result!


if __name__ == "__main__":
    exp.main()
