"""scripts/run_active_learning_finetune.py — THE thesis experiment. Press Run.

THE QUESTION THIS ANSWERS
    Static Experiment A showed the molecular representation barely matters for
    TabPFN when the encoder is FROZEN. But the thesis mechanism is a *trainable*
    encoder that ADAPTS to the target. Active learning is the only setting where
    that adaptation can happen, so this asks the real question:

        In a round-by-round campaign, does FINE-TUNING the Chemprop encoder on the
        target labels we accumulate each round find active molecules FASTER than
        the SAME encoder left frozen?

    This is the controlled contrast that decides RQ4:
        * finetune — each round, fine-tune the encoder on all revealed labels,
                     re-encode, then acquire. (The thesis mechanism.)
        * frozen   — the IDENTICAL random-init encoder, never fine-tuned.
                     (The control: isolates the effect of fine-tuning alone.)
    Both arms share the same random initialization per seed, so any difference is
    the fine-tuning, not the starting weights.

RETROSPECTIVE AL BENCHMARK (how we score it honestly)
    * We know every label but HIDE them. "Hit" = top ACTIVE_QUANTILE of the target.
    * Round 0: reveal a small random SEED_SIZE of labels.
    * Each round: (finetune arm only) fine-tune the encoder on revealed labels;
      then TabPFN scores a random CANDIDATE_POOL_SIZE subset of the hidden pool
      via Expected Improvement; reveal the BATCH_SIZE highest-scoring molecules.
    * Track hits found vs labels spent; average over N_SEEDS random starts.
    * Metric = oracle-efficiency (1.0 = perfect, ~ACTIVE_QUANTILE = random floor).

COMPUTE
    The per-round fine-tune does a differentiable BACKWARD through TabPFN — the
    expensive op. Defaults below are a CPU-friendly SMOKE TEST (1 seed, few
    rounds/steps, n_estimators=1). For real thesis numbers scale up
    (N_SEEDS>=3, N_ROUNDS>=15, FINETUNE_STEPS>=30, N_ESTIMATORS=8, HIDDEN_SIZE=300)
    and run on a CUDA cluster — NOT on a Mac (MPS is ~7x slower than CPU here).

All settings are constants below. No flags / env vars — open and press Run.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent          # .../src
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402
from models.acquisition import acquisition_scores  # noqa: E402
from models.encoder import MoleculeEncoder  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run. Defaults = CPU smoke test.            #
# --------------------------------------------------------------------------- #
# Target dataset: (display name, csv rel. path, smiles col, target col).
TARGET = ("S. aureus", "data/curated/s_aureus_curated_with_scores_filtered.csv", "Smiles", "pMIC")
# Others: ("D2R", "data/curated/D2R_final.csv", "SMILES", "affinity")
#         ("USP7", "data/curated/USP7_final.csv", "SMILES", "affinity")

ARMS = ["finetune", "frozen"]  # the thesis arm and its controlled baseline
STRATEGY = "ei"                # acquisition (ei = the Bayesian-optimisation choice)

ACTIVE_QUANTILE = 0.05         # top 5% of the target counts as a "hit"
SEED_SIZE = 20                 # labels revealed for free at round 0
BATCH_SIZE = 10                # labels acquired per round
N_ROUNDS = 5                   # acquisition rounds  (SMOKE: 5; real: >=15)
N_SEEDS = 1                    # random restarts, averaged  (SMOKE: 1; real: >=3)

# Per-round fine-tuning of the encoder on the accumulated labeled set.
FINETUNE_STEPS = 15            # gradient steps per round  (SMOKE: 15; real: >=30)
FINETUNE_LR = 1e-3            # AdamW step size for the encoder
MIN_LABELS_TO_FINETUNE = 8    # skip fine-tuning until this many labels exist
MAX_FINETUNE_CONTEXT = 64     # cap the fine-tune episode's context (memory)
MAX_FINETUNE_QUERY = 64       # cap the fine-tune episode's query

N_ESTIMATORS = 1              # TabPFN passes to average  (SMOKE: 1; real: 8)
CANDIDATE_POOL_SIZE = 1000    # random unlabeled subset scored each round (0 = all)
MAX_SCORE_CONTEXT = 256       # cap the labeled context fed to TabPFN when scoring
HIDDEN_SIZE = 128             # encoder width  (SMOKE: 128; real/thesis: 300)
XI = 0.0                      # EI exploration margin

SEED = 0
SAVE_CURVES_TO = "experiments/al_finetune_curves.csv"  # "" = don't save


# --------------------------------------------------------------------------- #
# Encoding helpers                                                             #
# --------------------------------------------------------------------------- #
def encode(encoder: MoleculeEncoder, smiles: list[str], device: torch.device) -> torch.Tensor:
    """Encode SMILES -> [N, d] with the current encoder weights (no grad, scrubbed)."""
    with torch.no_grad():
        X = encoder(smiles).to(device)
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def finetune_encoder(
    encoder: MoleculeEncoder,
    optimizer: torch.optim.Optimizer,
    head: TabPFNHead,
    smiles: list[str],
    y: torch.Tensor,
    labeled: list[int],
    rng: random.Random,
) -> None:
    """Fine-tune the encoder for FINETUNE_STEPS on the revealed labels.

    Each step splits the labeled set into a context + query episode and minimizes
    TabPFN's NLL — the same differentiable objective as pre-training, but on the
    TARGET's own accumulated labels. Weights persist across rounds (cumulative
    adaptation); the optimizer is created once per run.
    """
    encoder.train()
    for _ in range(FINETUNE_STEPS):
        idx = list(labeled)
        rng.shuffle(idx)
        n_ctx = max(5, min(len(idx) // 2, MAX_FINETUNE_CONTEXT))  # >=5 context (notes)
        ctx_idx = idx[:n_ctx]
        qry_idx = idx[n_ctx:n_ctx + MAX_FINETUNE_QUERY]
        if len(qry_idx) < 1:
            break
        y_ctx = y[ctx_idx]
        if float(y_ctx.std(unbiased=False)) < 1e-6:  # unusable context (all equal)
            continue
        optimizer.zero_grad()
        emb_ctx = encoder([smiles[i] for i in ctx_idx])
        emb_qry = encoder([smiles[i] for i in qry_idx])
        loss = head.loss(emb_ctx, y_ctx, emb_qry, y[qry_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
    encoder.eval()


# --------------------------------------------------------------------------- #
# One active-learning run for one arm                                         #
# --------------------------------------------------------------------------- #
def run_one_al(
    arm: str,
    head: TabPFNHead,
    smiles: list[str],
    y: torch.Tensor,
    hit_set: set[int],
    seed_idx: int,
    device: torch.device,
) -> list[int]:
    """Run the AL loop once for `arm`; return hits found after seed + each round."""
    # Same random init for both arms (fair) -> the only difference is fine-tuning.
    torch.manual_seed(SEED + seed_idx)
    encoder = MoleculeEncoder(hidden_size=HIDDEN_SIZE).to(device).eval()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=FINETUNE_LR)

    rng = random.Random(SEED + seed_idx)  # same seed set across arms -> fair
    n = len(smiles)
    labeled = list(rng.sample(range(n), SEED_SIZE))
    labeled_set = set(labeled)
    unlabeled = [i for i in range(n) if i not in labeled_set]

    hits = [len(labeled_set & hit_set)]
    for _ in range(N_ROUNDS):
        if not unlabeled:
            hits.append(len(labeled_set & hit_set))
            continue

        # 1. (finetune arm) adapt the encoder to the labels revealed so far.
        if arm == "finetune" and len(labeled) >= MIN_LABELS_TO_FINETUNE:
            finetune_encoder(encoder, optimizer, head, smiles, y, labeled, rng)

        # 2. Score a random candidate subset of the unlabeled pool with TabPFN.
        if CANDIDATE_POOL_SIZE and len(unlabeled) > CANDIDATE_POOL_SIZE:
            candidates = rng.sample(unlabeled, CANDIDATE_POOL_SIZE)
        else:
            candidates = list(unlabeled)

        ctx = labeled if len(labeled) <= MAX_SCORE_CONTEXT else rng.sample(labeled, MAX_SCORE_CONTEXT)
        X_ctx = encode(encoder, [smiles[i] for i in ctx], device)
        X_cand = encode(encoder, [smiles[i] for i in candidates], device)
        mu, sigma = head.predict_dist(X_ctx, y[ctx], X_cand)
        best = float(y[ctx].max())
        scores = acquisition_scores(STRATEGY, mu.cpu(), sigma.cpu(), best, xi=XI)
        top = torch.topk(scores, min(BATCH_SIZE, len(candidates))).indices.tolist()
        picks = [candidates[i] for i in top]

        # 3. Reveal the picked labels; update bookkeeping.
        labeled.extend(picks)
        picks_set = set(picks)
        labeled_set |= picks_set
        unlabeled = [i for i in unlabeled if i not in picks_set]
        hits.append(len(labeled_set & hit_set))
    return hits


def efficiency(mean_hits: list[float], budgets: list[int], total_hits: int) -> float:
    """Oracle-efficiency in [0,1]: 1.0 = perfect, ~ACTIVE_QUANTILE = random floor."""
    if total_hits == 0:
        return float("nan")
    fracs = [h / min(b, total_hits) for h, b in zip(mean_hits, budgets)]
    return sum(fracs) / len(fracs)


def save_curves(results: dict, budgets: list[int], total_hits: int) -> None:
    if not SAVE_CURVES_TO:
        return
    import csv
    out = Path(SAVE_CURVES_TO)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "budget", "mean_hits", "recall", "total_hits"])
        for arm, mean_hits in results.items():
            for b, h in zip(budgets, mean_hits):
                w.writerow([arm, b, f"{h:.3f}", f"{h / total_hits:.4f}", total_hits])
    print(f"\nsaved curves -> {out}")


# --------------------------------------------------------------------------- #
# The experiment                                                              #
# --------------------------------------------------------------------------- #
def main() -> None:
    _load_hf_token()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_name, rel_path, smiles_col, target_col = TARGET
    print(f"loading {ds_name} target ...")
    smiles, y = data.load_target(REPO_ROOT / rel_path, smiles_col=smiles_col, target_col=target_col)
    n = len(smiles)
    total_hits = max(1, round(n * ACTIVE_QUANTILE))
    hit_set = set(torch.topk(y, total_hits).indices.tolist())
    budgets = [SEED_SIZE + k * BATCH_SIZE for k in range(N_ROUNDS + 1)]
    print(f"  {n} molecules | {total_hits} hits (top {ACTIVE_QUANTILE:.0%}) | "
          f"budget {budgets[0]} -> {budgets[-1]} labels")
    print(f"  device: {device} | arms: {ARMS} | strategy: {STRATEGY} | "
          f"seeds: {N_SEEDS} | rounds: {N_ROUNDS} | finetune_steps: {FINETUNE_STEPS}\n")

    head = TabPFNHead(n_estimators=N_ESTIMATORS)  # one head: all arms share the encoder feature width
    results = {}  # arm -> mean hits curve
    for arm in ARMS:
        print(f"[{arm}] running {N_SEEDS} seed(s) ...  ({time.time() - t0:.0f}s)")
        per_seed = [run_one_al(arm, head, smiles, y, hit_set, s, device) for s in range(N_SEEDS)]
        mean_hits = [sum(run[k] for run in per_seed) / N_SEEDS for k in range(N_ROUNDS + 1)]
        results[arm] = mean_hits
        eff = efficiency(mean_hits, budgets, total_hits)
        print(f"    hits@{budgets[-1]} = {mean_hits[-1]:.1f}/{total_hits} | "
              f"efficiency {eff:.3f}  ({time.time() - t0:.0f}s)")

    # ---- results -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"ACTIVE LEARNING (fine-tune vs frozen) on {ds_name}")
    print("=" * 60)
    print(f"  {'arm':12s} {'hits@'+str(budgets[-1]):>10s} {'efficiency':>12s}")
    for arm, mean_hits in results.items():
        print(f"  {arm:12s} {mean_hits[-1]:>10.1f} {efficiency(mean_hits, budgets, total_hits):>12.3f}")

    if "finetune" in results and "frozen" in results:
        d = efficiency(results["finetune"], budgets, total_hits) - \
            efficiency(results["frozen"], budgets, total_hits)
        print(f"\nVERDICT: fine-tune - frozen = {d:+.3f}  "
              f"-> {'FINE-TUNING HELPS' if d > 0.01 else 'no real gain from fine-tuning'}")
        print("  (SMOKE-TEST defaults — scale up + run on GPU before trusting this.)")

    save_curves(results, budgets, total_hits)
    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
