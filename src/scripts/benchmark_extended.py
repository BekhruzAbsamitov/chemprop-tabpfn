"""scripts/benchmark_extended.py — the follow-up AL experiments of the 2026-09-29 supervision.

The experiments, selected by name (cluster jobs pass it as the first argument):

  acquisition   E2-E5. Same protocol as the primary benchmark (benchmark_al.py: seeds,
                budgets, 20 rounds, 1,000-candidate subsample), but on three representations
                (CheMeleon, Morgan, random-init Chemprop) and with the acquisition variants the supervision asked
                about:
                  - EI and PI relative to the incumbent y+ AND to the hit threshold y_thresh
                  - for TabPFN, EI and PI computed both from the Gaussian approximation
                    (mu, sigma) and from TabPFN's full predictive (bar) distribution
                  - a Gaussian process with a Tanimoto kernel, on Morgan fingerprints
                  - a third metric: the mean activity of the acquired compounds
  subsampling   E6. Does scoring only 1,000 candidates per round change the comparison?
                D2R and USP7 (the two smallest pools), CheMeleon, all three surrogates,
                greedy and EI, with the 1,000-candidate subsample AND the full pool.

  final         The last complete run (2026-11-04), replacing the primary benchmark, its
                three RQ4 transfer runs and the acquisition experiment above with ONE run:
                every representation of those runs (12 shared, plus each target's own
                domain-matched encoder at two training lengths) and the seed-1 replicate of
                the epoch-level encoder (epochs 21 and 26), all four surrogates, and
                every criterion of `acquisition` (greedy; EI and PI at y+ and y_thresh; for
                TabPFN from the Gaussian form and from the bar distribution). The Tanimoto
                GP runs on every representation here, not on Morgan only.
  final_subsampling  The full-pool arms of E6 for the final run, with every criterion; the
                1,000-candidate arms they are compared with are those of `final`.
  seed1_control Added after the final run (2026-11-09): the untrained starting point
                of the seed-1 encoders, benchmarked as one more representation with the same
                four surrogates and criteria as `final`. The seed-1 encoders (epochs 21 and
                26) start from another initialisation than "chemprop_untrained" (seed 0), so
                their gain over random initialisation is measured against THIS control. It is
                a later job of the same protocol, paired with the arms of `final` by pool and
                seed as every job of that run is (each was a separate cluster job too).
                A separate control, not part of the benchmark grid: its file name does not
                match "final_*.csv", so the analyses of the final run do not pick it up.
  repro_check   Submitted with seed1_control: the D2R / Morgan job of `final` run again
                under another name. Morgan is computed on the CPU and reproduced bit for bit
                between earlier runs, so this file must equal final_d2r_morgan.csv in every
                column except `experiment` and `seconds`. If it does, the code and the
                environment of the later jobs are those of the final run, and seed1_control
                can be paired with it.

Every campaign is recorded PER SEED and PER ROUND (the primary benchmark stored seed
means only), so later analyses can pair at the level of individual seeds.

PAIRING: all arms of one run share the pool, the seed sets and the session, so they can be
compared with each other. They must NOT be paired with rows of the primary benchmark
(report §5e: cross-run GPU nondeterminism).

Run locally (one target, one representation):   press Run with the settings below.
Cluster:  python src/scripts/benchmark_extended.py <experiment> <target> <representation>
Output:   experiments/benchmark_extended/<experiment>_<target>_<representation>.csv
          (experiments/benchmark_final/ for the final run), written row by row. An existing
          file is resumed (RESUME): finished campaigns are kept and skipped, a campaign cut
          off mid-way is dropped and re-run. Never run two jobs on the same file at once.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SRC_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_al import (  # noqa: E402  (same pools, featurisation and protocol)
    BATCH_SIZES, INIT_SIZES, N_FOLDS, N_ROUNDS, SEED, TARGETS, featurize_pool, load_pool)
from benchmark_al import REPRESENTATIONS as AL_REPRESENTATIONS  # noqa: E402
from metrics import cef  # noqa: E402
from models.acquisition import expected_improvement, probability_of_improvement  # noqa: E402
from models.baselines import (  # noqa: E402
    GaussianProcessHead, GaussianProcessTanimotoHead, InferenceTabPFN, RandomForestHead)
from models.tabpfn_regressor import _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run (the cluster overrides the first three). #
# --------------------------------------------------------------------------- #
EXPERIMENT = "acquisition"         # "acquisition" | "subsampling"
TARGET = "d2r"
REPRESENTATION = "chemeleon"
N_ESTIMATORS = 4                   # TabPFN ensemble size, as in the primary benchmark
OUT_DIR = "experiments/benchmark_extended"   # default; an experiment may name its own out_dir
RESUME = True                      # continue an interrupted job instead of refusing to start

# Every representation of the earlier AL runs, with the definitions of benchmark_al.py:
# the primary benchmark's arms and those of its three RQ4 transfer runs, plus the seed-1
# replicate of the epoch-level encoder, never benchmarked before. "chemeleon",
# "morgan" and "chemprop_untrained" (random-init Chemprop D-MPNN, seed 0) are the three of
# the acquisition experiment, unchanged.
FINAL_SHARED = [
    "morgan", "physchem", "mordred3d", "chemeleon", "chemprop_untrained",
    "chemprop_trained_full",                                            # all ChEMBL
    "chemprop_trained_full_sp", "chemprop_trained_full_sp_best",       # single-protein ChEMBL
    "chemprop_trained_full_sp_10ep", "chemprop_trained_full_sp_10ep_best",
    "chemprop_trained_full_sp_50ep_best", "chemprop_trained_full_sp_50ep_final",
    "chemprop_trained_full_sp_50ep_seed1_best", "chemprop_trained_full_sp_50ep_seed1_final",
]
REPRESENTATIONS = {r: AL_REPRESENTATIONS[r] for r in FINAL_SHARED}
# Each target's own domain-matched encoder at two training lengths (benchmark_al.py,
# representations_for); valid on that target only. TYK2 has none.
OWN_ENCODER_TARGET: dict[str, str] = {}
for _target, _cfg in TARGETS.items():
    if _cfg.get("own_encoder"):
        for _suffix in ("", "_long"):
            _name = f"chemprop_trained_own_{_cfg['own_encoder']}{_suffix}"
            REPRESENTATIONS[_name] = (
                "encoder", f"experiments/trained_encoder_{_cfg['own_encoder']}{_suffix}.pt")
            OWN_ENCODER_TARGET[_name] = _target

# The control of the seed-1 encoders, added after the final run. training.py saves
# `epoch_000.pt` as "epoch 0: the untrained baseline", before the first update, so it is the
# exact starting point of the seed-1 encoders. Kept out of REPRESENTATIONS so that the
# `final` experiment stays what was run.
SEED1_CONTROL_REPRESENTATIONS = {
    "chemprop_untrained_seed1": (
        "encoder", "experiments/trained_encoder_full_sp_50ep_seed1_epochs/epoch_000.pt"),
}
ALL_REPRESENTATIONS = {**REPRESENTATIONS, **SEED1_CONTROL_REPRESENTATIONS}

# A strategy is (acquisition, reference value tau, predictive distribution).
GREEDY = ("greedy", "-", "-")
GAUSS_VARIANTS = [(a, t, "gauss") for a in ("ei", "pi") for t in ("yplus", "ythresh")]
FULL_VARIANTS = [(a, t, "full") for a in ("ei", "pi") for t in ("yplus", "ythresh")]

EXPERIMENTS = {
    "acquisition": dict(
        targets=["saureus", "tyk2", "usp7", "d2r"],
        representations=["chemeleon", "morgan", "chemprop_untrained"],
        models=["tabpfn", "rf", "gp", "gp_tanimoto"],
        strategies=[GREEDY] + GAUSS_VARIANTS + FULL_VARIANTS,
        candidate_sizes=[1000]),
    "subsampling": dict(
        targets=["d2r", "usp7"], representations=["chemeleon"],
        models=["tabpfn", "rf", "gp"],
        strategies=[GREEDY, ("ei", "yplus", "gauss")],
        candidate_sizes=[1000, 0]),          # 0 = score every untested molecule
    "final": dict(
        targets=["saureus", "tyk2", "usp7", "d2r"],
        representations=list(REPRESENTATIONS),
        models=["tabpfn", "rf", "gp", "gp_tanimoto"],
        strategies=[GREEDY] + GAUSS_VARIANTS + FULL_VARIANTS,
        candidate_sizes=[1000],
        tanimoto_on_all=True,                # the Tanimoto GP on every representation
        out_dir="experiments/benchmark_final"),
    "final_subsampling": dict(
        targets=["d2r", "usp7"], representations=["chemeleon"],
        models=["tabpfn", "rf", "gp"],
        strategies=[GREEDY] + GAUSS_VARIANTS + FULL_VARIANTS,
        candidate_sizes=[0],                 # the 1,000-candidate arms are those of "final"
        out_dir="experiments/benchmark_final"),
    # Not named "final_*": the analyses of the final run read final_*.csv, and these files
    # must not enter its grid (see the docstring).
    "seed1_control": dict(
        targets=["saureus", "tyk2", "usp7", "d2r"],
        representations=list(SEED1_CONTROL_REPRESENTATIONS),
        models=["tabpfn", "rf", "gp", "gp_tanimoto"],
        strategies=[GREEDY] + GAUSS_VARIANTS + FULL_VARIANTS,
        candidate_sizes=[1000],
        tanimoto_on_all=True,
        out_dir="experiments/benchmark_final"),
    "repro_check": dict(                     # the final run's D2R / Morgan job, rerun
        targets=["d2r"],
        representations=["morgan"],
        models=["tabpfn", "rf", "gp", "gp_tanimoto"],
        strategies=[GREEDY] + GAUSS_VARIANTS + FULL_VARIANTS,
        candidate_sizes=[1000],
        tanimoto_on_all=True,
        out_dir="experiments/benchmark_final"),
}

COLUMNS = ["experiment", "target", "representation", "model", "acquisition", "tau", "dist",
           "candidates", "init_size", "batch_size", "seed", "round", "budget", "new_hits",
           "cum_hits", "cef", "mean_y_acquired", "best_y", "seconds"]
# The columns that identify one campaign (with experiment, target and representation, which
# are fixed within one output file).
KEY_COLUMNS = ["model", "acquisition", "tau", "dist", "candidates", "init_size", "batch_size",
               "seed"]


def finished_campaigns(path: Path) -> set[tuple[str, ...]]:
    """The campaigns an earlier, interrupted start of this job completed (all N_ROUNDS rows).

    Rows of a campaign cut off mid-way are removed from the file, so that campaign is re-run
    whole; finished campaigns are never touched.
    """
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    if header is not None and list(header) != COLUMNS:
        raise SystemExit(f"{path} has other columns than this script writes; move it away first")
    counts = Counter(tuple(r[c] for c in KEY_COLUMNS) for r in rows)
    done = {k for k, n in counts.items() if n == N_ROUNDS}
    keep = [r for r in rows if tuple(r[c] for c in KEY_COLUMNS) in done]
    if header is None or len(keep) < len(rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(keep)
    return done


def arm_is_valid(model: str, representation: str, strategy: tuple,
                 tanimoto_on_all: bool = False) -> bool:
    """Tanimoto on Morgan only unless the experiment says otherwise (the kernel is the
    convention for binary fingerprints but valid on any non-zero real vectors); the full
    distribution only exists for TabPFN."""
    if model == "gp_tanimoto" and representation != "morgan" and not tanimoto_on_all:
        return False
    if strategy[2] == "full" and model != "tabpfn":
        return False
    return True


def make_model(name: str, device: torch.device):
    if name == "tabpfn":
        return InferenceTabPFN(n_estimators=N_ESTIMATORS, device=str(device))
    if name == "rf":
        return RandomForestHead()
    if name == "gp":
        return GaussianProcessHead()
    if name == "gp_tanimoto":
        return GaussianProcessTanimotoHead()
    raise ValueError(f"unknown model {name!r}")


def hit_threshold(cfg: dict, y: torch.Tensor, hit_set: set[int]) -> float:
    """The activity value that defines a hit on this pool (the y_thresh of EI/PI)."""
    if "hit_threshold" in cfg:
        return float(cfg["hit_threshold"])
    return float(min(float(y[i]) for i in hit_set))   # the weakest of the top-k molecules


def score(model, strategy, x_ctx, y_ctx, x_cand, y_thresh) -> torch.Tensor:
    """Acquisition scores for the candidates (higher = acquire first)."""
    acq, tau_kind, dist = strategy
    if dist == "full":
        mu, sigma, criterion, logits = model.predict_full(x_ctx, y_ctx, x_cand)
    else:
        mu, sigma = model.predict_dist(x_ctx, y_ctx, x_cand)
    mu, sigma = mu.cpu(), sigma.cpu()
    if acq == "greedy":
        return mu
    tau = float(y_ctx.max()) if tau_kind == "yplus" else y_thresh
    if dist == "full":
        fn = criterion.ei if acq == "ei" else criterion.pi
        return fn(logits, tau).detach().float().cpu()
    fn = expected_improvement if acq == "ei" else probability_of_improvement
    return fn(mu, sigma, tau)


def run_campaign(model, strategy, X, y, hit_set, y_thresh, init, batch, seed, n_cand,
                 base_rate):
    """One AL campaign; returns one dict per round. Seed sets match benchmark_al.py."""
    rng = random.Random(SEED + seed)          # identical seed set to the primary benchmark
    n = X.shape[0]
    labeled = set(rng.sample(range(n), init))
    unlabeled = [i for i in range(n) if i not in labeled]
    cumulative = len(labeled & hit_set)
    acquired: list[int] = []
    rows = []
    for r in range(N_ROUNDS):
        t0 = time.time()
        new = 0
        if unlabeled:
            if n_cand and len(unlabeled) > n_cand:
                cands = rng.sample(unlabeled, n_cand)
            else:
                cands = unlabeled
            ctx = sorted(labeled)
            s = score(model, strategy, X[ctx], y[ctx], X[cands], y_thresh)
            top = torch.topk(s, min(batch, len(cands))).indices.tolist()
            picks = [cands[i] for i in top]
            new = len(set(picks) & hit_set)
            cumulative += new
            labeled.update(picks)
            acquired.extend(picks)
            ps = set(picks)
            unlabeled = [i for i in unlabeled if i not in ps]
        budget = init + (r + 1) * batch
        rows.append(dict(round=r + 1, budget=budget, new_hits=new, cum_hits=cumulative,
                         cef=f"{cef(cumulative, budget, base_rate):.4f}",
                         mean_y_acquired=f"{float(y[acquired].mean()):.4f}" if acquired else "",
                         best_y=f"{float(y[sorted(labeled)].max()):.4f}",
                         seconds=f"{time.time() - t0:.2f}"))
    return rows


def main() -> None:
    experiment, target, representation = EXPERIMENT, TARGET, REPRESENTATION
    if len(sys.argv) >= 4:
        experiment, target, representation = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = EXPERIMENTS[experiment]
    if (target not in spec["targets"] or representation not in spec["representations"]
            or OWN_ENCODER_TARGET.get(representation, target) != target):
        raise SystemExit(f"{experiment} does not include {target} / {representation}")

    out = REPO_ROOT / spec.get("out_dir", OUT_DIR) / f"{experiment}_{target}_{representation}.csv"
    done: set[tuple[str, ...]] = set()
    if out.exists():
        if not RESUME:
            raise SystemExit(f"refusing to overwrite {out}; move it away first")
        done = finished_campaigns(out)
        print(f"resuming {out}: {len(done)} campaigns already complete")
    out.parent.mkdir(parents=True, exist_ok=True)
    _load_hf_token()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TARGETS[target]
    smiles, y, hit_set = load_pool(cfg)
    y = y.flatten().float()
    y_thresh = hit_threshold(cfg, y, hit_set)
    base_rate = len(hit_set) / len(smiles)
    kind, ckpt = ALL_REPRESENTATIONS[representation]
    X = featurize_pool(kind, ckpt, smiles, device).float().cpu()
    batch_sizes = cfg.get("batch_sizes", BATCH_SIZES)
    print(f"{experiment} | {target} ({len(smiles):,} molecules, {len(hit_set)} hits, "
          f"y_thresh {y_thresh:.3f}) | {representation} {tuple(X.shape)} | device {device}")

    if not out.exists():
        with out.open("w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

    t_start = time.time()
    for model_name in spec["models"]:
        strategies = [s for s in spec["strategies"]
                      if arm_is_valid(model_name, representation, s, spec.get("tanimoto_on_all", False))]
        if not strategies:
            continue
        model = make_model(model_name, device)
        for strategy in strategies:
            for n_cand in spec["candidate_sizes"]:
                for init in INIT_SIZES:
                    for batch in batch_sizes:
                        last = None
                        for seed in range(N_FOLDS):
                            key = (model_name, *strategy, str(n_cand or "all"), str(init),
                                   str(batch), str(seed))
                            if key in done:
                                continue
                            rows = run_campaign(model, strategy, X, y, hit_set, y_thresh,
                                                init, batch, seed, n_cand, base_rate)
                            with out.open("a", newline="") as f:
                                w = csv.writer(f)
                                for row in rows:
                                    w.writerow([experiment, target, representation, model_name,
                                                *strategy, n_cand or "all", init, batch, seed,
                                                *[row[c] for c in COLUMNS[11:]]])
                            last = (seed, rows[-1])
                        if last is None:
                            continue
                        seed, row = last
                        print(f"  {model_name:11s} {'/'.join(strategy):20s} cand {n_cand or 'all':>4} "
                              f"init {init:>3} batch {batch:>2} | seed {seed}: CEF {row['cef']} "
                              f"hits {row['cum_hits']} | {(time.time() - t_start) / 60:.0f} min")
    print(f"\nwrote {out}  ({(time.time() - t_start) / 3600:.1f} h)")


if __name__ == "__main__":
    main()
