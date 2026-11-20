"""scripts/sample_efficiency.py — does TabPFN react better to a few new labels? Press Run.

The 2026-08-20 supervision asked exactly this, after the benchmark showed TabPFN's margin
over Random Forest grows as the acquisition batch shrinks (report §5e Result 1b):

    "why is small batch size better? is this a TabPFN thing?
     Can TabPFN react better to adding few new samples?"

§5e can only answer it halfway. There, smaller batches help EVERY surrogate (8/8 for each
of TabPFN, RF and GP), so finer acquisition is first of all a property of active learning
— but TabPFN converts it into hits better than either baseline, beating RF's gain in 8 of
8 comparisons. What that cannot separate is *why*: inside an AL loop the model also
chooses what it is trained on next, so a better model gets a better labelled set, and
model quality and data quality are hopelessly entangled.

THIS SCRIPT REMOVES THE LOOP. No acquisition function, no rounds, no feedback: label n
molecules drawn UNIFORMLY AT RANDOM, fit each surrogate on exactly those, and score how
well it ranks a held-out query set. Sweep n. What remains is pure model
sample-efficiency.

    if TabPFN's curve rises faster at small n  -> yes, it reacts better to few samples,
                                                  and that explains the §5e differential
    if the curves are parallel                 -> the AL advantage comes from acquisition
                                                  dynamics, not from the model update

PAIRING IS THE POINT. At every (target, representation, n, seed) all three surrogates see
the SAME context molecules and the SAME query set. Differences are therefore the model
alone, never the draw — the same discipline the paired comparisons in §5e use.

TWO METRICS, because they answer different questions:
  spearman   ranking quality over the whole query set — the clean statistical signal.
  hits@k     fraction of the true actives captured in the top k predictions — what a
             campaign actually cares about, and the metric CEF is built from.

The marginal-gain table at the end is the direct answer: how much does each surrogate
gain when the labelled set doubles? That is "reacting to new samples", quantified.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SRC_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "scripts"))

from benchmark_al import TARGETS, featurize_pool, load_pool, make_model  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Targets to sweep. TYK2 is the one target whose signal separates representations
# (§5e), so it is the most informative; S. aureus is the largest pool.
TARGET_NAMES = ["tyk2", "saureus"]

# Representations. Chemeleon is the strong reference; Chemprop random-init is the weak
# one where §5e Result 1c says TabPFN's advantage is largest. Keeping both tests whether
# sample-efficiency depends on the representation.
REPRESENTATIONS = {
    "chemeleon": ("chemeleon", None),
    "chemprop_untrained": ("untrained", None),
}

# Labelled-set sizes. Geometric so "doubling" is one step — the marginal-gain table
# below reads straight off consecutive entries. The small end is the claim under test.
CONTEXT_SIZES = [5, 10, 20, 40, 80, 160, 320, 640]

N_QUERY = 1500        # held-out molecules scored each time (fixed per seed)
N_SEEDS = 5           # repeats; the same draws are shared by all three surrogates
TOP_K_FRAC = 0.05     # hits@k with k = 5 % of the query set
N_ESTIMATORS = 4      # TabPFN passes to average — matches benchmark_al.py
MODELS = ["tabpfn", "rf", "gp"]
SAVE_TO = "experiments/sample_efficiency.csv"


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation, computed without scipy (already a project convention)."""
    ar = torch.argsort(torch.argsort(a)).double()
    br = torch.argsort(torch.argsort(b)).double()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = ar.norm() * br.norm()
    return float((ar @ br) / denom) if denom > 0 else float("nan")


def hits_at_k(pred: torch.Tensor, is_hit: torch.Tensor, k: int) -> float:
    """Fraction of the query set's true actives that land in the top-k predictions."""
    total = int(is_hit.sum())
    if total == 0:
        return float("nan")
    top = torch.topk(pred, k).indices
    return float(is_hit[top].sum()) / total


def evaluate(target: str, rep_name: str, kind: str, ckpt: str | None,
             device: torch.device, writer) -> None:
    """One (target, representation): sweep context size for all three surrogates."""
    cfg = TARGETS[target]
    smiles, y, hit_set = load_pool(cfg)
    X = featurize_pool(kind, ckpt, smiles, device)
    y = y.flatten().float()
    is_hit_all = torch.zeros(len(smiles), dtype=torch.bool)
    is_hit_all[list(hit_set)] = True
    k = max(1, round(N_QUERY * TOP_K_FRAC))
    print(f"\n=== {target} / {rep_name} — pool {len(smiles):,}, {int(is_hit_all.sum())} hits, "
          f"features {tuple(X.shape)} ===")
    print(f"{'n':>5} " + " ".join(f"{m:>21}" for m in MODELS))

    for n_ctx in CONTEXT_SIZES:
        if n_ctx + N_QUERY > len(smiles):
            print(f"{n_ctx:5d}  skipped — pool too small")
            continue
        rows = {m: {"sp": [], "hk": []} for m in MODELS}
        for seed in range(N_SEEDS):
            g = torch.Generator().manual_seed(1000 * seed + n_ctx)
            perm = torch.randperm(len(smiles), generator=g)
            # The SAME context and query for every surrogate — see the docstring.
            c_idx, q_idx = perm[:n_ctx], perm[n_ctx:n_ctx + N_QUERY]
            xc, yc, xq = X[c_idx], y[c_idx], X[q_idx]
            yq, hq = y[q_idx], is_hit_all[q_idx]
            if float(yc.std()) < 1e-8:      # a degenerate draw teaches nothing
                continue
            for m in MODELS:
                model = make_model(m, device)
                try:
                    mu, _ = model.predict_dist(xc, yc, xq)
                except Exception as exc:    # a surrogate may refuse a tiny context
                    print(f"    [warn] {m} failed at n={n_ctx} seed={seed}: "
                          f"{type(exc).__name__}")
                    continue
                rows[m]["sp"].append(spearman(mu, yq))
                rows[m]["hk"].append(hits_at_k(mu, hq, k))
                del model

        cells = []
        for m in MODELS:
            sp, hk = rows[m]["sp"], rows[m]["hk"]
            if not sp:
                cells.append(f"{'--':>21}")
                continue
            sp_m, hk_m = float(np.mean(sp)), float(np.mean(hk))
            cells.append(f"rho {sp_m:+.3f}  h@k {hk_m:.3f}")
            writer.writerow([target, rep_name, m, n_ctx, N_QUERY, k, len(sp),
                             f"{sp_m:.4f}", f"{float(np.std(sp)):.4f}",
                             f"{hk_m:.4f}", f"{float(np.std(hk)):.4f}"])
        print(f"{n_ctx:5d} " + " ".join(cells))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"sweep: {TARGET_NAMES} x {list(REPRESENTATIONS)} x {MODELS} "
          f"x n={CONTEXT_SIZES} x {N_SEEDS} seeds")

    out = REPO_ROOT / SAVE_TO
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["target", "representation", "model", "n_context", "n_query",
                         "top_k", "n_seeds", "mean_spearman", "std_spearman",
                         "mean_hits_at_k", "std_hits_at_k"])
        for target in TARGET_NAMES:
            for rep_name, (kind, ckpt) in REPRESENTATIONS.items():
                evaluate(target, rep_name, kind, ckpt, device, writer)

    print(f"\nwrote -> {out}")
    print("Read the MARGINAL gain (change per doubling of n) — that is the supervision's")
    print("'reacts better to adding few new samples'. A steeper TabPFN curve at small n")
    print("confirms it; parallel curves point at acquisition dynamics instead.")


if __name__ == "__main__":
    main()
