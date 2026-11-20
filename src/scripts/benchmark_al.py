"""scripts/benchmark_al.py — AL benchmark: TabPFN vs baseline models × representations.

For every (target × representation × model × strategy × initial-size × batch-size) it runs a
retrospective AL loop, averages over seeds, and reports the CEF curve (cumulative enrichment
factor) + new hits found.

The benchmark of the thesis is run by benchmark_extended.py, which imports the pools, the
featurisation and the protocol from this file.

LEAKAGE is tracked per (representation, target) in the LEAKAGE table below and written
to the output CSV; arms above 1 % molecule overlap are flagged [LEAKY] in the log. That is
MOLECULE overlap; the overlap that can leak labels (same molecule, same target) is much
smaller except for the full-ChEMBL encoder on S. aureus, see leakage_by_target.py.

RUNNING. The targets are independent, so pass one as the first argument to run it alone:

    python benchmark_al.py tyk2       # one target -> experiments/al_benchmark_tyk2.csv
    python benchmark_al.py            # all four   -> experiments/al_benchmark.csv

Merge the per-target CSVs afterwards with `merge_benchmark_csvs()`.
"""

from __future__ import annotations

import csv
import gc
import random
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from active_learning import batched_encode  # noqa: E402
from data_utils import data  # noqa: E402
from metrics import cef, mean_std  # noqa: E402
from models.acquisition import acquisition_scores  # noqa: E402
from models.baselines import GaussianProcessHead, InferenceTabPFN, RandomForestHead  # noqa: E402
from models.checkpoints import load_encoder  # noqa: E402
from models.chemeleon import CheMeleonFingerprint  # noqa: E402
from models.encoder import MoleculeEncoder  # noqa: E402
from models.mordred3d import mordred3d_features  # noqa: E402
from models.morgan import morgan_features  # noqa: E402
from models.physchem import physchem_features  # noqa: E402
from models.rigr_encoder import RIGREncoder  # noqa: E402
from models.tabpfn_regressor import _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Benchmark pools. Two ways to define a hit:
#   hit_threshold  — an absolute cut on the value (the chempfn S. aureus convention)
#   hit_top_frac   — the top fraction by value; 0.05 on TYK2 reproduces its `top_5p`
#                    column exactly (500/9,997), and unlike a boolean column it
#                    survives load_target's de-duplication by SMILES.
# `batch_sizes` is overridden per target so the largest campaign stays ~10-15 % of the
# pool: past ~25 % you are exhausting the pool rather than searching it, and CEF
# collapses toward 1.0 by construction. `own_encoder` is the adaptive encoder trained
# on THAT target's domain (with its 100k counterpart) — the fairest test of whether
# domain-matched training pays off on its own target. TYK2 has none (no kinase
# adaptive set was built), so it runs the shared arms only.
TARGETS = {
    # The original S_aureus.csv stores log10(MIC + 1) in ug/mL, which rises with MIC. The corrected
    # pool keeps exact measurements only and uses molar pMIC = -log10(MIC_ug / MW) + 3
    # (src/scripts/build_saureus_corrected.py); hits are the top 5 %, as for the other pools.
    # Results on the original pool are NOT comparable.
    "saureus": dict(csv="data/curated/S_aureus_corrected.csv", smiles_col="Smiles",
                    value_col="pMIC", hit_top_frac=0.05,
                    own_encoder="saureus"),                       # 31,085 mols → 3.5 %
    "tyk2": dict(csv="data/curated/TYK2.csv", smiles_col="SMILES",
                 value_col="affinity", hit_top_frac=0.05),        #  9,997 mols → 11 %
    "usp7": dict(csv="data/curated/USP7.csv", smiles_col="SMILES",
                 value_col="affinity", hit_top_frac=0.05,
                 batch_sizes=[10, 25], own_encoder="usp7"),       #  4,535 mols → 13 %
    "d2r": dict(csv="data/curated/D2R.csv", smiles_col="SMILES",
                value_col="affinity", hit_top_frac=0.05,
                batch_sizes=[5, 10], own_encoder="d2r"),          #  2,502 mols → 12 %
}

# Arms run on EVERY target -> (kind, checkpoint-or-None). Each target additionally gets its
# own domain-matched encoder at two training lengths (see `representations_for`).
REPRESENTATIONS = {
    # Classical descriptors, no learning. Widths differ by 10x (physchem 200 ... Morgan 2048).
    "morgan": ("morgan", None),             # 2048, sparse binary
    "physchem": ("physchem", None),         #  200, dense, interpretable
    "mordred3d": ("mordred3d", None),       # 1826, dense, needs 3D conformers
    # Learned representations.
    "chemeleon": ("chemeleon", None),       # 2048, pretrained, frozen
    "chemprop_untrained": ("untrained", None),   # 300, random init — the control
    "chemprop_trained_full": ("encoder", "experiments/trained_encoder_full.pt"),
    # Trained on the single-protein ChEMBL set for 100k steps: `_best` is the checkpoint with the
    # best validation (step 64k), the plain arm is the final one.
    "chemprop_trained_full_sp": ("encoder", "experiments/trained_encoder_full_sp.pt"),
    "chemprop_trained_full_sp_best": ("encoder", "experiments/trained_encoder_full_sp_best.pt"),
    # The same training set carried to 10 full epochs.
    "chemprop_trained_full_sp_10ep": ("encoder", "experiments/trained_encoder_full_sp_10ep.pt"),
    "chemprop_trained_full_sp_10ep_best": ("encoder", "experiments/trained_encoder_full_sp_10ep_best.pt"),
    # The epoch-level run at learning rate 3e-5: best validation loss at epoch 23, last epoch 26.
    # Both checkpoints are taken from the per-epoch directory.
    "chemprop_trained_full_sp_50ep_best": (
        "encoder", "experiments/trained_encoder_full_sp_50ep_epochs/epoch_023.pt"),
    "chemprop_trained_full_sp_50ep_final": (
        "encoder", "experiments/trained_encoder_full_sp_50ep_epochs/epoch_026.pt"),
    # Its seed-1 replicate (preset full_sp_50ep_seed1): best validation loss at epoch 21, early
    # stopping at epoch 26.
    "chemprop_trained_full_sp_50ep_seed1_best": (
        "encoder", "experiments/trained_encoder_full_sp_50ep_seed1_epochs/epoch_021.pt"),
    "chemprop_trained_full_sp_50ep_seed1_final": (
        "encoder", "experiments/trained_encoder_full_sp_50ep_seed1_epochs/epoch_026.pt"),
    # The group's pretrained D-MPNN: 512-d, RIGR featurisation (models/rigr_encoder.py), frozen.
    # Its random-init twin is the matched control.
    "rigr_pretrained": ("rigr", "data/weights/chemicl_mpnn.pt"),
    "rigr_untrained": ("rigr", None),
}

# MOLECULE overlap between an encoder's TRAINING split and a benchmark pool, as a fraction
# of the pool: the pool molecule occurs in some training assay, after RDKit canonicalisation
# of both sides (src/scripts/leakage_by_target.py). Everything not listed is 0 %: the adaptive
# encoders had every benchmark compound removed at build time, and that holds for their
# _long variants too (same training parquet).
#
# Molecule overlap is not label leakage: a molecule measured on an unrelated protein says
# nothing about its activity on the benchmark target. The same-target overlap is in
# experiments/leakage_by_target.csv, and it is small except full ChEMBL on S. aureus:
#   full ChEMBL      S. aureus 58.8 %  TYK2 0.15 %  USP7 3.7 %  D2R 3.8 %
#   single-protein   S. aureus  0.0 %  TYK2 0.15 %  USP7 2.6 %  D2R 4.4 %
# S. aureus values are for the corrected pool (S_aureus_corrected.csv).
LEAKAGE = {
    ("chemprop_trained_full", "saureus"): 0.8436,  # corrected pool; 0.772 on the original
    ("chemprop_trained_full", "d2r"): 0.911,   # by molecule; only 3.8 % on the D2 target
    ("chemprop_trained_full", "usp7"): 0.044,
    ("chemprop_trained_full", "tyk2"): 0.0015,  # essentially clean
    # UNKNOWN, not zero: the training molecules of `rigr_pretrained` are not known, so its overlap
    # with these pools cannot be measured. NaN records that.
    ("rigr_pretrained", "saureus"): float("nan"),
    ("rigr_pretrained", "d2r"): float("nan"),
    ("rigr_pretrained", "usp7"): float("nan"),
    ("rigr_pretrained", "tyk2"): float("nan"),
    # Measured against the TRAIN SPLIT of the parquet the encoder actually saw.
    ("chemprop_trained_full_sp", "saureus"): 0.0421,
    ("chemprop_trained_full_sp", "d2r"): 0.9137,
    ("chemprop_trained_full_sp", "usp7"): 0.0289,
    ("chemprop_trained_full_sp", "tyk2"): 0.0015,
    ("chemprop_trained_full_sp_best", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_best", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_best", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_best", "tyk2"): 0.0015,
    ("chemprop_trained_full_sp_10ep", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_10ep", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_10ep", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_10ep", "tyk2"): 0.0015,
    ("chemprop_trained_full_sp_10ep_best", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_10ep_best", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_10ep_best", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_10ep_best", "tyk2"): 0.0015,
    # same parquet and train split as the other single-protein encoders
    ("chemprop_trained_full_sp_50ep_best", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_50ep_best", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_50ep_best", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_50ep_best", "tyk2"): 0.0015,
    ("chemprop_trained_full_sp_50ep_final", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_50ep_final", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_50ep_final", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_50ep_final", "tyk2"): 0.0015,
    # the seed-1 replicate: the seed does not change the train/test assay split
    ("chemprop_trained_full_sp_50ep_seed1_best", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_50ep_seed1_best", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_50ep_seed1_best", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_50ep_seed1_best", "tyk2"): 0.0015,
    ("chemprop_trained_full_sp_50ep_seed1_final", "saureus"): 0.0421,
    ("chemprop_trained_full_sp_50ep_seed1_final", "d2r"): 0.9137,
    ("chemprop_trained_full_sp_50ep_seed1_final", "usp7"): 0.0289,
    ("chemprop_trained_full_sp_50ep_seed1_final", "tyk2"): 0.0015,
}

# Every benchmark run as a named preset: (RUN_TAG, ONLY_REPRESENTATIONS); an empty list runs
# every representation. Comparisons are paired within one run (same pool, same order, same
# seeds), so a focused run carries its own reference arms. "{own}" stands for the target's
# own domain-matched encoder and is skipped on a target without one (TYK2).
RUNS: dict[str, tuple[str, list[str]]] = {
    "primary": ("", ["morgan", "physchem", "mordred3d", "chemeleon", "chemprop_untrained",
                     "chemprop_trained_full", "chemprop_trained_own_{own}",
                     "chemprop_trained_own_{own}_long"]),
    "fullsp": ("fullsp", ["chemprop_trained_full_sp", "chemprop_trained_full_sp_best",
                          "chemprop_untrained", "chemeleon"]),
    "10ep": ("10ep", ["chemprop_trained_full_sp_10ep", "chemprop_trained_full_sp_10ep_best",
                      "chemprop_trained_full_sp", "chemprop_untrained", "chemeleon"]),
    "50ep": ("50ep", [
        "chemprop_trained_full_sp_50ep_best",   # THE question: best epoch (23) of the lr-3e-5 run
        "chemprop_trained_full_sp_50ep_final",  # its final epoch (26)
        "chemprop_trained_full_sp_10ep",        # the 1e-4 run, for a within-run lr comparison
        "chemprop_untrained",                   # the control all of them must beat
        "chemeleon",                            # the strong pretrained reference
    ]),
}
# The run to execute. Press Run in the IDE to use this one; a cluster job may name another
# as the SECOND argument (python benchmark_al.py <target> <run>).
RUN = "50ep"
# Only when this file is the script being run: benchmark_extended.py, calibration_study.py and
# runtime_study.py import it, and their own command-line arguments mean something else.
if __name__ == "__main__" and len(sys.argv) > 2:
    RUN = sys.argv[2]
    if RUN not in RUNS:
        raise SystemExit(f"unknown run {RUN!r}; choose from {list(RUNS)}")
ONLY_REPRESENTATIONS: list[str] = RUNS[RUN][1]

MODELS = ["tabpfn", "rf", "gp"]
STRATEGIES = ["ei", "greedy"]

INIT_SIZES = [10, 100]
BATCH_SIZES = [10, 50]
N_ROUNDS = 20
N_FOLDS = 5                 # random-seed restarts, averaged (like the chempfn folds)

CANDIDATE_POOL_SIZE = 1000  # score a fresh random subset of the pool each round; 0 = all
N_ESTIMATORS = 4           # TabPFN passes to average
UNTRAINED_HIDDEN = 300
SEED = 0
# Names this run's outputs. MUST be changed for any run that is not a full re-do of the
# whole grid, otherwise a per-target job overwrites the previous run's part file of the
# same name and its data is gone. "" = the canonical full-grid outputs.
RUN_TAG = RUNS[RUN][0]

SAVE_TO = f"experiments/al_benchmark{'_' + RUN_TAG if RUN_TAG else ''}.csv"
# Per-target jobs write here, one file each; merge_benchmark_csvs() combines them into SAVE_TO.
PARTS_DIR = f"experiments/al_benchmark_parts{'_' + RUN_TAG if RUN_TAG else ''}"


def leakage(rep: str, target: str) -> float:
    return LEAKAGE.get((rep, target), 0.0)


def representations_for(cfg: dict) -> dict[str, tuple[str, str | None]]:
    """The shared arms, plus this target's OWN domain-matched encoder if it has one.

    All adaptive encoders are leakage-clean against every benchmark (their training
    parquets had the benchmark compounds removed at build time), so no extra leakage
    bookkeeping is needed for the `own` arms.
    """
    reps = dict(REPRESENTATIONS)
    own = cfg.get("own_encoder")
    if own:
        reps[f"chemprop_trained_own_{own}"] = (
            "encoder", f"experiments/trained_encoder_{own}.pt")
        reps[f"chemprop_trained_own_{own}_long"] = (
            "encoder", f"experiments/trained_encoder_{own}_long.pt")
    if ONLY_REPRESENTATIONS:
        wanted = [r.format(own=own) for r in ONLY_REPRESENTATIONS if own or "{own}" not in r]
        missing = [r for r in wanted if r not in reps]
        if missing:  # a typo here would silently shrink the sweep — fail loudly instead
            raise SystemExit(f"ONLY_REPRESENTATIONS names unknown arms: {missing}")
        reps = {r: reps[r] for r in wanted}
    return reps


def load_pool(cfg: dict) -> tuple[list[str], torch.Tensor, set[int]]:
    """(smiles, values, hit indices) for one benchmark, by whichever hit rule is set."""
    smiles, y = data.load_target(REPO_ROOT / cfg["csv"], smiles_col=cfg["smiles_col"],
                                 target_col=cfg["value_col"])
    if "hit_threshold" in cfg:
        hit_set = {i for i in range(len(smiles)) if float(y[i]) >= cfg["hit_threshold"]}
    else:  # top fraction by value (higher = better on both pools)
        k = max(1, round(len(smiles) * cfg["hit_top_frac"]))
        hit_set = set(torch.topk(y.flatten(), k).indices.tolist())
    return smiles, y, hit_set


def featurize_pool(kind: str, ckpt: str | None, smiles: list[str], device: torch.device) -> torch.Tensor:
    """Turn the whole pool into a [N, d] feature matrix for one representation (once)."""
    if kind == "chemeleon":
        return batched_encode(CheMeleonFingerprint(device=str(device)), smiles, device)
    if kind == "encoder":
        return batched_encode(load_encoder(REPO_ROOT / ckpt, device), smiles, device)
    if kind == "untrained":
        torch.manual_seed(SEED)
        enc = MoleculeEncoder(hidden_size=UNTRAINED_HIDDEN).eval().to(device)
        return batched_encode(enc, smiles, device)
    if kind == "rigr":
        # ckpt None -> the random-init control; the encoder reports which one it loaded.
        torch.manual_seed(SEED)
        enc = RIGREncoder(weights_path=ckpt).to(device)
        return batched_encode(enc, smiles, device)
    # Static descriptors: computed on CPU for the whole pool at once, not batched
    # through a network. Mordred caches to disk (see models/mordred3d.py) because
    # generating a conformer per molecule is the one genuinely slow step here.
    if kind == "morgan":
        return morgan_features(smiles)
    if kind == "physchem":
        return physchem_features(smiles)
    if kind == "mordred3d":
        return mordred3d_features(smiles)
    raise ValueError(f"unknown representation kind {kind!r}")


def make_model(name: str, device: torch.device):
    if name == "tabpfn":
        return InferenceTabPFN(n_estimators=N_ESTIMATORS, device=str(device))
    if name == "rf":
        return RandomForestHead()
    if name == "gp":
        return GaussianProcessHead()
    raise ValueError(f"unknown model {name!r}")


def run_al(model, X, y, hit_set, strategy, init, batch, seed, global_hit_rate):
    """One AL trajectory. Returns (new_hits, cumulative_hits, cef) curves of N_ROUNDS.

    The two hit counts differ by the hits that happened to fall in the INITIAL random
    draw: `new_hits` counts only what the model acquired, while `cumulative_hits` (and
    therefore CEF, which is derived from it) describes the whole labelled set. At
    init=100 with a ~5 % base rate that is ~5 free hits, so the two are not
    interchangeable — report which one you mean.
    """
    rng = random.Random(SEED + seed)
    n = X.shape[0]
    labeled = set(rng.sample(range(n), init))
    unlabeled = [i for i in range(n) if i not in labeled]
    cumulative = len(labeled & hit_set)  # seeded with the initial draw's hits
    new_hits = 0
    hits_curve, cum_curve, cef_curve = [], [], []

    for r in range(N_ROUNDS):
        if unlabeled:
            if CANDIDATE_POOL_SIZE and len(unlabeled) > CANDIDATE_POOL_SIZE:
                cands = rng.sample(unlabeled, CANDIDATE_POOL_SIZE)
            else:
                cands = unlabeled
            ctx = sorted(labeled)
            mu, sigma = model.predict_dist(X[ctx], y[ctx], X[cands])
            best = float(y[ctx].max())
            scores = acquisition_scores(strategy, mu.cpu(), sigma.cpu(), best)
            top = torch.topk(scores, min(batch, len(cands))).indices.tolist()
            picks = [cands[i] for i in top]
            got = len(set(picks) & hit_set)
            new_hits += got
            cumulative += got
            labeled.update(picks)
            ps = set(picks)
            unlabeled = [i for i in unlabeled if i not in ps]
        total_sampled = init + (r + 1) * batch
        hits_curve.append(new_hits)
        cum_curve.append(cumulative)
        cef_curve.append(cef(cumulative, total_sampled, global_hit_rate))
    return hits_curve, cum_curve, cef_curve


def selected_targets() -> tuple[dict, str]:
    """Which targets to run, and where to write them.

    The four targets are completely independent, so passing one as the first CLI
    argument lets the cluster run them as parallel jobs — one submit file with
    `queue TARGET in (...)` finishes in roughly a quarter of the wall-clock. Each job
    writes its own CSV so they never collide; `merge_benchmark_csvs()` stitches them
    back together. No argument = all four sequentially (the IDE Run-button path).
    """
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in TARGETS:
            raise SystemExit(f"unknown target {name!r}; choose from {list(TARGETS)}")
        # A subdirectory, NOT experiments/al_benchmark_<target>.csv, which would collide with the
        # file of an earlier single-target run.
        return {name: TARGETS[name]}, f"{PARTS_DIR}/{name}.csv"
    return TARGETS, SAVE_TO


def merge_benchmark_csvs(out_path: str = SAVE_TO) -> int:
    """Concatenate the per-target part files into one, for analysis code."""
    parts = sorted((REPO_ROOT / PARTS_DIR).glob("*.csv"))
    rows, header = [], None
    for p in parts:
        with p.open() as f:
            r = list(csv.reader(f))
        header, rows = r[0], rows + r[1:]
    if header is None:
        raise SystemExit(f"no part files found in {PARTS_DIR}/")
    out = REPO_ROOT / out_path
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"merged {len(parts)} part files -> {out_path}  ({len(rows)} rows)")
    return len(rows)


def main() -> None:
    _load_hf_token()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets, save_to = selected_targets()

    print(f"device {device} | targets {list(targets)} -> {save_to}")
    # Print the arms that will ACTUALLY run, i.e. after ONLY_REPRESENTATIONS is applied.
    _arms = list(representations_for(next(iter(targets.values()))))
    if ONLY_REPRESENTATIONS:
        print(f"  [subset] ONLY_REPRESENTATIONS is set — running {len(_arms)} of "
              f"{len(REPRESENTATIONS)} shared arms")
    print(f"sweep per target: {_arms} x {MODELS} x {STRATEGIES} x "
          f"init{INIT_SIZES} x batch{BATCH_SIZES} x {N_FOLDS} folds\n")

    out = REPO_ROOT / save_to
    # Never overwrite a previous run's results: a job requeued by HTCondor restarts from scratch
    # and would otherwise rewrite this file from its first rows.
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Both hit counts are stored: new_hits excludes the initial random draw's hits,
    # cum_hits includes them (and is what CEF is computed from). Standard deviations
    # across the N_FOLDS seeds are kept so differences can be judged against the spread
    # rather than eyeballed.
    header = ["target", "representation", "leakage_frac", "model", "strategy", "init_size",
              "batch_size", "round", "budget",
              "mean_new_hits", "std_new_hits", "mean_cum_hits", "std_cum_hits",
              "mean_cef", "std_cef", "base_rate"]

    def persist() -> None:
        """Rewrite the CSV after every arm, so a crash keeps everything finished so far."""
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    for target, cfg in targets.items():
        smiles, y, hit_set = load_pool(cfg)
        n = len(smiles)
        global_hit_rate = len(hit_set) / n
        rule = (f"{cfg['value_col']}>={cfg['hit_threshold']}" if "hit_threshold" in cfg
                else f"top {cfg['hit_top_frac']:.0%} by {cfg['value_col']}")
        print(f"##### {target}: {n} molecules | {len(hit_set)} hits ({rule}) | "
              f"base rate {global_hit_rate:.4f}")
        batch_sizes = cfg.get("batch_sizes", BATCH_SIZES)
        budget = max(INIT_SIZES) + N_ROUNDS * max(batch_sizes)
        note = f"max budget {budget} labels = {budget/n:.0%} of the pool"
        if budget > 0.25 * n:  # past ~25% the pool is being exhausted, not searched
            print(f"      WARNING: {note} — CEF collapses toward 1.0 when the pool "
                  f"runs out. Lower `batch_sizes` for this target.")
        else:
            print(f"      batches {batch_sizes} | {note}")
        print()

        for rep, (kind, ckpt) in representations_for(cfg).items():
            leak = leakage(rep, target)
            tag = "  [LEAKY]" if leak > 0.01 else ""
            print(f"=== {target} / {rep} (leakage {leak:.1%}){tag} — featurizing ...  "
                  f"({time.time()-t0:.0f}s)")
            X = featurize_pool(kind, ckpt, smiles, device)
            for model_name in MODELS:
                model = make_model(model_name, device)  # one per (rep, model); reused below
                for strategy in STRATEGIES:
                    for init in INIT_SIZES:
                        for batch in batch_sizes:
                            curves = [run_al(model, X, y, hit_set, strategy, init, batch, s,
                                             global_hit_rate) for s in range(N_FOLDS)]
                            for r in range(N_ROUNDS):
                                mh, sh = mean_std([c[0][r] for c in curves])   # new hits
                                mu, su = mean_std([c[1][r] for c in curves])   # cumulative
                                mc, sc = mean_std([c[2][r] for c in curves])   # CEF
                                rows.append([target, rep, f"{leak:.3f}", model_name, strategy,
                                             init, batch, r + 1, init + (r + 1) * batch,
                                             f"{mh:.3f}", f"{sh:.3f}", f"{mu:.3f}", f"{su:.3f}",
                                             f"{mc:.4f}", f"{sc:.4f}", f"{global_hit_rate:.4f}"])
                            f = rows[-1]
                            print(f"    {model_name:>7s} {strategy:>6s} init{init:>3} "
                                  f"batch{batch:>2} | final CEF {f[13]}+-{f[14]} "
                                  f"hits {f[9]} (cum {f[11]})  ({time.time()-t0:.0f}s)")
                del model  # free the model (and its GPU memory) before the next one
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            persist()
            print(f"    (saved -> {out.relative_to(REPO_ROOT)})\n")
            del X
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"done. results -> {SAVE_TO}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
