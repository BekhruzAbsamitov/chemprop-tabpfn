"""scripts/run_active_learning.py — THE active-learning experiment. Press Run.

This is the experiment the thesis actually hinges on. The earlier
`run_experiment.py` asked a STATIC question (one-shot: does a trained encoder
rank a fixed test set better?). This asks the ACTIVE-LEARNING question:

    Starting from just a handful of labels, and allowed to label a few more
    molecules each round, which representation finds the active molecules
    FASTEST — Morgan, an untrained Chemprop encoder, or the trained one?

That is what a real drug-discovery campaign does: you can only afford a few
assays per round, so you want each round to spend its budget on the molecules
most likely to be active. TabPFN gives us a mean AND an uncertainty per
molecule; the acquisition function (EI/PI/…) turns those into "label this next".

How it works (a retrospective AL benchmark):
    * We DO know every S. aureus label, but we HIDE them.
    * "Active" (a hit) = a molecule in the top ACTIVE_QUANTILE of true pMIC.
    * Round 0: reveal a small random SEED_SIZE of labels.
    * Each round: TabPFN (fit on revealed labels) scores the hidden pool; we
      reveal the BATCH_SIZE highest-scoring molecules.
    * We track how many hits we've uncovered vs how many labels we've spent.
    * Repeat for several random seeds and average.

A representation is GOOD if its curve rises fast — many hits for few labels —
and clearly above the `random` acquisition floor.

All settings are constants below. No flags, no environment variables — open in
your IDE and press Run. (Needs HF_TOKEN in .env for TabPFN, same as the others.)
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from active_learning import batched_encode  # noqa: E402
from data_utils import data  # noqa: E402
from metrics import oracle_efficiency  # noqa: E402
from models.acquisition import acquisition_scores  # noqa: E402
from models.checkpoints import load_encoder  # noqa: E402
from models.encoder import MoleculeEncoder  # noqa: E402
from models.morgan import morgan_features  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Which representations to compare. "trained_chemprop" loads the encoder saved
# by run_experiment_gpu.py; it is skipped (with a note) if the file is missing.
ARMS = ["morgan", "untrained_chemprop", "trained_chemprop"]

# Which acquisition strategies to compare. "random" is the floor every real
# strategy must beat; "ei" is the headline Bayesian-optimisation choice.
STRATEGIES = ["ei", "greedy", "random"]

ACTIVE_QUANTILE = 0.05  # top 5% of true pMIC counts as a "hit" (an active)
SEED_SIZE = 10          # labels revealed for free at the start of each run
BATCH_SIZE = 10         # labels acquired per round
N_ROUNDS = 15           # acquisition rounds (final budget = SEED_SIZE + N_ROUNDS*BATCH_SIZE)
N_SEEDS = 3             # repeat with different random starts, then average

N_ESTIMATORS = 4        # TabPFN passes to average (higher = steadier, slower)
# The S. aureus pool has ~40k molecules; scoring ALL of them every round is huge
# (and unrealistic — real campaigns screen a subset). Each round we instead score
# a fresh random CANDIDATE_POOL_SIZE draw of the unlabeled pool and pick the batch
# from those. Set to 0 to score the whole pool (only feasible on a big GPU).
CANDIDATE_POOL_SIZE = 1000
XI = 0.0                # EI/PI exploration margin (demand this much improvement)
KAPPA = 1.0             # UCB exploration weight (unused unless "ucb" is in STRATEGIES)

HIDDEN_SIZE = 300       # untrained-encoder width (the trained one uses ITS saved width)
TRAINED_ENCODER_PATH = "experiments/trained_encoder_full.pt"

SEED = 0
SAVE_CURVES_TO = "experiments/al_curves.csv"  # per-arm/strategy averaged curves; "" = don't save


# --------------------------------------------------------------------------- #
# Featurization — turn the whole pool into numbers ONCE per arm.               #
# batched_encode chunks the pool and zeros non-finite rows (src/active_learning).
# --------------------------------------------------------------------------- #
def build_arm_features(arm: str, smiles: list[str], device: torch.device) -> torch.Tensor | None:
    """Precompute the pool's feature matrix for one representation (or None if unavailable)."""
    if arm == "morgan":
        return batched_encode(morgan_features, smiles, device)

    if arm == "untrained_chemprop":
        torch.manual_seed(SEED)  # reproducible random init
        encoder = MoleculeEncoder(hidden_size=HIDDEN_SIZE).eval().to(device)
        return batched_encode(encoder, smiles, device)

    if arm == "trained_chemprop":
        path = Path(TRAINED_ENCODER_PATH)
        path = path if path.is_absolute() else data.REPO_ROOT / path
        if not path.exists():
            print(f"  [skip] trained_chemprop: no encoder at {path}")
            return None
        return batched_encode(load_encoder(path, device), smiles, device)

    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------- #
# One active-learning run: seed -> acquire -> repeat, counting hits           #
# --------------------------------------------------------------------------- #
def run_one_al(
    head: TabPFNHead,
    X: torch.Tensor,
    y: torch.Tensor,
    hit_set: set[int],
    strategy: str,
    seed_idx: int,
) -> list[int]:
    """Run the AL loop once; return hits-found after the seed and each round.

    The list has length N_ROUNDS + 1 (index 0 = after the seed set), so budget
    at position k is SEED_SIZE + k * BATCH_SIZE.
    """
    rng = random.Random(SEED + seed_idx)  # same seed set across arms/strategies -> fair
    n = X.shape[0]
    labeled = set(rng.sample(range(n), SEED_SIZE))
    unlabeled = [i for i in range(n) if i not in labeled]

    hits = [len(labeled & hit_set)]
    for _ in range(N_ROUNDS):
        if not unlabeled:
            hits.append(len(labeled & hit_set))
            continue

        if strategy == "random":
            picks = rng.sample(unlabeled, min(BATCH_SIZE, len(unlabeled)))
        else:
            # Score a fresh random candidate subset of the unlabeled pool (or all
            # of it if CANDIDATE_POOL_SIZE is 0/larger than the pool).
            if CANDIDATE_POOL_SIZE and len(unlabeled) > CANDIDATE_POOL_SIZE:
                candidates = rng.sample(unlabeled, CANDIDATE_POOL_SIZE)
            else:
                candidates = unlabeled
            ctx = sorted(labeled)
            mu, sigma = head.predict_dist(X[ctx], y[ctx], X[candidates])
            best = float(y[ctx].max())
            scores = acquisition_scores(strategy, mu.cpu(), sigma.cpu(), best, xi=XI, kappa=KAPPA)
            top = torch.topk(scores, min(BATCH_SIZE, len(candidates))).indices.tolist()
            picks = [candidates[i] for i in top]

        labeled.update(picks)
        picks_set = set(picks)
        unlabeled = [i for i in unlabeled if i not in picks_set]
        hits.append(len(labeled & hit_set))
    return hits


def _save_curves(results: dict, budgets: list[int], total_hits: int) -> Path | None:
    """Write the averaged hits-vs-budget curves to CSV. Called after EACH arm so a
    later crash can't cost you the arms already finished."""
    if not SAVE_CURVES_TO:
        return None
    out = Path(SAVE_CURVES_TO)
    out = out if out.is_absolute() else data.REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "strategy", "budget", "mean_hits", "recall", "total_hits"])
        for (arm, strategy), mean_hits in results.items():
            for budget, h in zip(budgets, mean_hits):
                writer.writerow([arm, strategy, budget, f"{h:.3f}", f"{h / total_hits:.4f}", total_hits])
    return out


# --------------------------------------------------------------------------- #
# The experiment                                                              #
# --------------------------------------------------------------------------- #
def main() -> None:
    _load_hf_token()
    t0 = time.time()

    # 1. Data + the definition of a "hit".
    print("loading S. aureus target ...")
    smiles, y = data.load_target()
    n = len(smiles)
    total_hits = max(1, round(n * ACTIVE_QUANTILE))
    hit_idx = torch.topk(y, total_hits).indices.tolist()  # the true actives
    hit_set = set(hit_idx)
    budgets = [SEED_SIZE + k * BATCH_SIZE for k in range(N_ROUNDS + 1)]
    print(f"  {n} molecules | {total_hits} hits (top {ACTIVE_QUANTILE:.0%} pMIC) | "
          f"budget {budgets[0]} -> {budgets[-1]} labels")

    # 2. Device. We build a FRESH TabPFN head PER ARM (below), not one shared
    #    head: TabPFN caches the fitted preprocessing schema, and arms have
    #    different feature widths (Morgan 2048 vs Chemprop 300), so a reused head
    #    hits a schema mismatch on TabPFN's GPU preprocessing path. One head per
    #    arm keeps the feature spaces isolated.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device} | n_estimators: {N_ESTIMATORS} | "
          f"seeds: {N_SEEDS} | strategies: {STRATEGIES}\n")

    # 3. For each representation, precompute features, then run every strategy.
    results = {}  # (arm, strategy) -> mean hits curve (list of floats)
    for arm in ARMS:
        print(f"[{arm}] featurizing pool ...  ({time.time() - t0:.0f}s)")
        X = build_arm_features(arm, smiles, device)
        if X is None:
            continue
        head = TabPFNHead(n_estimators=N_ESTIMATORS)  # fresh per arm — no schema mixing
        for strategy in STRATEGIES:
            per_seed = [run_one_al(head, X, y, hit_set, strategy, s) for s in range(N_SEEDS)]
            mean_hits = [sum(run[k] for run in per_seed) / N_SEEDS for k in range(N_ROUNDS + 1)]
            results[(arm, strategy)] = mean_hits
            auc = oracle_efficiency(mean_hits, budgets, total_hits)
            print(f"    {strategy:>7s}: hits@{budgets[-1]}={mean_hits[-1]:5.1f}/{total_hits} "
                  f"| efficiency {auc:.3f}  ({time.time() - t0:.0f}s)")
        # Persist after every arm — the arms already done survive a later crash.
        saved = _save_curves(results, budgets, total_hits)
        if saved:
            print(f"    (curves so far -> {saved})")
        # Free this arm's TabPFN + features before loading the next arm's head.
        del head, X
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Summary table: final hits found and oracle-efficiency (how fast).
    #    efficiency ~ hit-rate: 1.0 = oracle, ~ACTIVE_QUANTILE = random floor.
    print("\n" + "=" * 68)
    print(f"ACTIVE LEARNING on S. aureus  (found {total_hits} hits; higher = better)")
    print("=" * 68)
    print(f"  {'representation':20s} {'strategy':>8s} {'hits@'+str(budgets[-1]):>9s} {'efficiency':>11s}")
    for (arm, strategy), mean_hits in results.items():
        auc = oracle_efficiency(mean_hits, budgets, total_hits)
        print(f"  {arm:20s} {strategy:>8s} {mean_hits[-1]:>9.1f} {auc:>11.3f}")

    # 5. Headline verdict: does the trained encoder beat Morgan and untrained,
    #    using EI, in how fast it finds hits (oracle-efficiency)? Only printed if
    #    all three arms ran (e.g. skipped when running trained-only).
    if all((a, "ei") in results for a in ("morgan", "untrained_chemprop", "trained_chemprop")):
        auc = {a: oracle_efficiency(results[(a, "ei")], budgets, total_hits) for a in
               ("morgan", "untrained_chemprop", "trained_chemprop")}
        print("\nVERDICT (EI acquisition, oracle-efficiency; random floor ~"
              f"{ACTIVE_QUANTILE:.2f}):")
        print(f"  trained {auc['trained_chemprop']:.3f} vs untrained {auc['untrained_chemprop']:.3f} "
              f"({auc['trained_chemprop'] - auc['untrained_chemprop']:+.3f}) "
              f"-> {'HELPS' if auc['trained_chemprop'] - auc['untrained_chemprop'] > 0.01 else 'no real gain'}")
        print(f"  trained {auc['trained_chemprop']:.3f} vs Morgan    {auc['morgan']:.3f} "
              f"({auc['trained_chemprop'] - auc['morgan']:+.3f}) "
              f"-> {'beats baseline' if auc['trained_chemprop'] - auc['morgan'] > 0 else 'below baseline'}")

    # 6. Final curve save (already written incrementally per arm above).
    saved = _save_curves(results, budgets, total_hits)
    if saved:
        print(f"\nsaved curves -> {saved}")

    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
