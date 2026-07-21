from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch
from scipy.stats import spearmanr

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent  # .../src
REPO_ROOT = SRC_DIR.parent  # repo root
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402  (our src/data_utils/data.py)
from models.encoder import MoleculeEncoder  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, _load_hf_token  # noqa: E402

DATASETS = [
    ("S. aureus", "data/curated/s_aureus_curated_with_scores_filtered.csv", "Smiles", "pMIC"),
    ("D2R", "data/curated/D2R_final.csv", "SMILES", "affinity"),
    ("USP7", "data/curated/USP7_final.csv", "SMILES", "affinity"),
]

N_CONTEXT = 200  # labeled molecules TabPFN learns from, per split
N_QUERY = 200  # molecules it ranks, per split
N_SAMPLES = 3  # random context/query draws per dataset (the notes: "3 samples")

RANDOM_HIDDEN_SIZE = 128  # width of the random-init Chemprop arm
INCLUDE_TRAINED = True  # add the ChEMBL-trained Chemprop arm?
TRAINED_ENCODER_PATH = "experiments/trained_encoder_full.pt"  # where run_experiment_gpu.py saves
INCLUDE_CHEMELEON = True  # add the Chemeleon arm?

N_ESTIMATORS = 8  # TabPFN passes to average (8 = fair across arms; 1 = fast local)
LOSS_FN = "nll"  # head needs a loss even though eval only calls predict()
SEED = 0


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def score(predictions: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    p = predictions.detach().cpu().numpy()
    t = truth.detach().cpu().numpy()
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - float(((t - p) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else float("nan")
    rho = float(spearmanr(p, t).statistic)
    return rho, r2


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v == v]  # drop NaNs
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, var ** 0.5


class CheMeleonFingerprint:

    _ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

    def __init__(self, device: str | None = None):
        from urllib.request import urlretrieve

        from chemprop import featurizers, nn
        from chemprop.models import MPNN
        from chemprop.nn import RegressionFFN

        self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        if not mp_path.exists():
            print("  downloading Chemeleon weights from Zenodo (once) ...")
            urlretrieve(self._ZENODO_URL, mp_path)

        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        self.model = MPNN(
            message_passing=mp,
            agg=nn.MeanAggregation(),
            predictor=RegressionFFN(input_dim=mp.output_dim),  # required, unused
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str]) -> torch.Tensor:
        from chemprop.data import BatchMolGraph
        from rdkit.Chem import MolFromSmiles

        bmg = BatchMolGraph([self._featurizer(MolFromSmiles(s)) for s in molecules])
        bmg.to(device=self.model.device)
        with torch.no_grad():
            return self.model.fingerprint(bmg)


def to_tensor(x) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


def build_arms(device: str) -> list[tuple[str, object]]:
    """Return [(arm_name, featurize_fn), ...]. featurize_fn: list[str] -> tensor."""
    arms: list[tuple[str, object]] = []

    # Chemprop-random: a fresh, untrained encoder.
    random_encoder = MoleculeEncoder(hidden_size=RANDOM_HIDDEN_SIZE).to(device).eval()
    arms.append(("Chemprop-random", lambda s: random_encoder(s)))

    # Chemprop-trained: load the ChEMBL-meta-trained encoder (matches its width).
    if INCLUDE_TRAINED:
        ckpt_path = Path(TRAINED_ENCODER_PATH)
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            trained_encoder = MoleculeEncoder(hidden_size=ckpt["hidden_size"]).to(device)
            trained_encoder.load_state_dict(ckpt["state_dict"])
            trained_encoder.eval()
            arms.append(("Chemprop-trained", lambda s: trained_encoder(s)))
        else:
            print(f"  WARNING: trained encoder not found at {ckpt_path} — skipping that arm.")

    # Chemeleon: pretrained Chemprop fingerprinter (weights auto-download/cached).
    if INCLUDE_CHEMELEON:
        fingerprinter = CheMeleonFingerprint(device=device)
        arms.append(("Chemeleon", lambda s: fingerprinter(s)))

    return arms


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def make_splits(n: int, rng: random.Random) -> list[tuple[list[int], list[int]]]:
    """N_SAMPLES fixed (context, query) index splits, shared across all arms."""
    idx = list(range(n))
    splits = []
    for _ in range(N_SAMPLES):
        rng.shuffle(idx)
        splits.append((idx[:N_CONTEXT], idx[N_CONTEXT: N_CONTEXT + N_QUERY]))
    return [(list(c), list(q)) for c, q in splits]


def evaluate(featurize, head, smiles, y, splits) -> tuple[list[float], list[float]]:
    """Spearman & R2 for each split (static one-shot ranking)."""
    rhos, r2s = [], []
    for ctx, qry in splits:
        with torch.no_grad():
            x_ctx = to_tensor(featurize([smiles[i] for i in ctx])).to(head.device)
            x_qry = to_tensor(featurize([smiles[i] for i in qry])).to(head.device)
            preds = head.predict(x_ctx, y[ctx], x_qry)
        rho, r2 = score(preds, y[qry])
        rhos.append(rho)
        r2s.append(r2)
    return rhos, r2s


# --------------------------------------------------------------------------- #
# The experiment                                                              #
# --------------------------------------------------------------------------- #
def main() -> None:
    _load_hf_token()
    torch.manual_seed(SEED)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arms = build_arms(device)
    # One frozen TabPFN head per arm (arms live in different feature spaces).
    heads = {name: TabPFNHead(n_estimators=N_ESTIMATORS, loss_fn=LOSS_FN) for name, _ in arms}
    print(f"device: {device} | arms: {[n for n, _ in arms]} | "
          f"{N_SAMPLES} samples x {N_CONTEXT} ctx / {N_QUERY} qry\n")

    # rho[arm][dataset] = (mean, std) over N_SAMPLES; same for r2.
    rho_table: dict[str, dict[str, tuple[float, float]]] = {n: {} for n, _ in arms}
    r2_table: dict[str, dict[str, tuple[float, float]]] = {n: {} for n, _ in arms}

    for ds_name, rel_path, smiles_col, target_col in DATASETS:
        print(f"=== {ds_name} ===")
        smiles, y = data.load_target(
            REPO_ROOT / rel_path, smiles_col=smiles_col, target_col=target_col
        )
        print(f"  {len(smiles)} unique molecules | target '{target_col}' "
              f"mean {float(y.mean()):.2f}")
        if len(smiles) < N_CONTEXT + N_QUERY:
            print(f"  WARNING: only {len(smiles)} molecules (< {N_CONTEXT + N_QUERY}); "
                  "skipping this dataset.")
            continue

        # Same splits for every arm on this dataset -> a fair comparison.
        splits = make_splits(len(smiles), random.Random(SEED))

        for arm_name, featurize in arms:
            rhos, r2s = evaluate(featurize, heads[arm_name], smiles, y, splits)
            rho_table[arm_name][ds_name] = mean_std(rhos)
            r2_table[arm_name][ds_name] = mean_std(r2s)
            m, s = rho_table[arm_name][ds_name]
            print(f"  {arm_name:18s} Spearman {m:6.3f} +/- {s:.3f}  "
                  f"({time.time() - t0:.0f}s)")
        print()

    # ---- results tables ------------------------------------------------------
    ds_names = [d[0] for d in DATASETS]

    def print_table(title: str, table: dict[str, dict[str, tuple[float, float]]]) -> None:
        print("=" * (22 + 16 * len(ds_names) + 12))
        print(title)
        print("=" * (22 + 16 * len(ds_names) + 12))
        header = f"  {'arm':18s}" + "".join(f"{d:>15s} " for d in ds_names) + f"{'overall':>10s}"
        print(header)
        for arm_name, _ in arms:
            row = f"  {arm_name:18s}"
            per_ds_means = []
            for d in ds_names:
                if d in table[arm_name]:
                    m, s = table[arm_name][d]
                    row += f"{m:7.3f} +/-{s:5.3f} "
                    per_ds_means.append(m)
                else:
                    row += f"{'-':>15s} "
            overall, _ = mean_std(per_ds_means)
            row += f"{overall:>10.3f}"
            print(row)
        print()

    print()
    print_table("Spearman (mean +/- std over samples) — higher is better", rho_table)
    print_table("R2 (mean +/- std over samples) — higher is better", r2_table)
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
