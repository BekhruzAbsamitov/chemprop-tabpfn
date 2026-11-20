"""scripts/cka_study.py — did training actually CHANGE the encoder? Press Run.

Does the representation itself change during training, and by how much?

For each held-out assay we embed the SAME molecules with two encoders and score them
with CKA (centered kernel alignment): 1.0 = the same representation up to rotation
and scaling, ~0 = unrelated.

    E_initial = chemprop_initial(X)     # the untrained encoder training started from
    E_trained = chemprop_trained(X)     # the encoder after training
    CKA(E_initial, E_trained)           # how far did training move it?

A CKA score means nothing on its own, so every score
is reported next to three REFERENCES, all measured against the same E_initial:

    other-init   a second random encoder (different seed)  -> how far does merely
                 reseeding move the representation?
    Chemeleon    a genuinely different, pretrained encoder -> the scale of a real
    Morgan       a different representation family            representational gap

Read it as: if CKA(initial, trained) is close to 1.0 and far above the other-init
reference, training barely moved the encoder — it was effectively a no-op. If it is
down near the other-init reference, the encoder moved as far as a different random
seed would, i.e. it changed a lot.

E_initial: taken from `<encoder>_steps/step_000000.pt` when that exists (the exact
init the run started from). Otherwise it is REBUILT from the seed, which is only
valid if the same torch version produced it — the script says which one it used.

TWO PARTS, both run when you press Run:
  1. STATIC     — CKA(E_initial, E_trained) for each finished encoder + the references.
  2. TRAJECTORY — CKA(E_initial, E_step_k) across training, from the step-tagged
                  checkpoints the 100k runs write every `checkpoint_every` steps.
                  It works on a
                  job still in progress: pull whatever checkpoints exist so far
                  (`cluster/sync.sh pull`) and it uses those; missing directories are
                  skipped with a note. Reference anchors are constant across steps, so
                  on a plot they are horizontal lines giving the curve its scale.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from active_learning import batched_encode  # noqa: E402
from data_utils import data  # noqa: E402
from metrics import mean_std  # noqa: E402
from models.checkpoints import load_encoder  # noqa: E402
from models.chemeleon import CheMeleonFingerprint  # noqa: E402
from models.encoder import MoleculeEncoder  # noqa: E402
from models.morgan import morgan_features  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
# Which trained encoders to study -> (checkpoint, held-out assays to embed).
# The parquet is the encoder's OWN training file; we read its `test` split, so the
# molecules are held out from that encoder's training.
ENCODERS = {
    "d2r": ("experiments/trained_encoder_d2r.pt",
            "data/curated/train_adaptive_D2R.parquet"),
    "usp7": ("experiments/trained_encoder_usp7.pt",
             "data/curated/train_adaptive_USP7.parquet"),
    "saureus": ("experiments/trained_encoder_saureus.pt",
                "data/curated/train_adaptive_S_aureus.parquet"),
    # The full-ChEMBL encoder reads the big cleaned parquet — slower to load.
    "full": ("experiments/trained_encoder_full.pt", ""),
}

# TRAJECTORY: step-tagged encoder
# directories written by run_experiment.py when `checkpoint_every` is set. We compute
# CKA(E_initial, E_step_k) for every checkpoint, where E_initial is step_000000.pt —
# the exact untrained encoder that run started from. A directory that does not exist
# yet is skipped with a note, so this is safe to run on a job still in progress
# (pull partial results with `cluster/sync.sh pull`).
TRAJECTORIES = {
    "saureus_long": ("experiments/trained_encoder_saureus_long_steps",
                     "data/curated/train_adaptive_S_aureus.parquet"),
    "d2r_long": ("experiments/trained_encoder_d2r_long_steps",
                 "data/curated/train_adaptive_D2R.parquet"),
    "usp7_long": ("experiments/trained_encoder_usp7_long_steps",
                  "data/curated/train_adaptive_USP7.parquet"),
}

N_ASSAYS = 20        # held-out assays to score (CKA is computed per assay)
MIN_ASSAY_MOLS = 25  # skip tiny assays — CKA on a handful of rows is unstable
INIT_SEED = 0        # the seed the training runs used (rebuild fallback only)
OTHER_INIT_SEED = 1  # the "how far does reseeding move it?" reference
SAVE_TO = "experiments/cka_study.csv"
SAVE_TRAJECTORY_TO = "experiments/cka_trajectory.csv"


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """CKA between two embeddings of the SAME molecules (Kornblith et al. 2019).

    x, y are [N, d1] / [N, d2] — the feature dimensions need not match. Returns 1.0
    when the two representations are identical up to rotation and isotropic scaling,
    and ~0 when they are unrelated. Computed in float64: the Frobenius norms below
    square the features, which loses precision badly in float32.
    """
    x = x.double()
    y = y.double()
    x = x - x.mean(dim=0, keepdim=True)  # CKA is defined on centered features
    y = y - y.mean(dim=0, keepdim=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    cross = (y.T @ x).norm() ** 2
    scale = (x.T @ x).norm() * (y.T @ y).norm()
    if scale == 0:  # a constant representation carries no similarity information
        return float("nan")
    return float(cross / scale)


def held_out_assays(parquet: str) -> list[data.Assay]:
    """The `test` split of an encoder's own training file — molecules it never saw."""
    path = (REPO_ROOT / parquet) if parquet else None
    assays = data.load_chembl_assays("test", use_full=not parquet, path=path)
    big = [a for a in assays if len(a.smiles) >= MIN_ASSAY_MOLS]
    return big[:N_ASSAYS]


def initial_encoder(ckpt_path: Path, hidden_size: int, device: torch.device):
    """E_initial: the untrained encoder the run started from.

    Prefer the step-0 checkpoint if the run saved one (exact). Otherwise rebuild it
    from INIT_SEED — only valid if the same torch version produced the original, so
    the caller reports which source was used.
    """
    step0 = ckpt_path.with_name(f"{ckpt_path.stem}_steps") / "step_000000.pt"
    if step0.exists():
        return load_encoder(step0, str(device)), "step_000000.pt (exact)"
    torch.manual_seed(INIT_SEED)  # matches run_experiment.main()
    enc = MoleculeEncoder(hidden_size=hidden_size).eval().to(device)
    return enc, f"rebuilt from seed {INIT_SEED} (UNVERIFIED)"


def random_encoder(seed: int, hidden_size: int, device: torch.device) -> MoleculeEncoder:
    torch.manual_seed(seed)
    return MoleculeEncoder(hidden_size=hidden_size).eval().to(device)


def reference_embeddings(smiles, initial, other, chemeleon, device) -> dict:
    """The three anchors, embedded for one assay. Constant across training steps, so
    on a trajectory plot they are horizontal lines that give the y-axis its meaning."""
    return {
        "other-init": batched_encode(other, smiles, device, warn=False),
        "chemeleon": batched_encode(chemeleon, smiles, device, warn=False),
        "morgan": batched_encode(lambda s: morgan_features(s), smiles, device, warn=False),
    }


def run_trajectory(name, steps_dir: Path, parquet: str, chemeleon, device) -> list[list]:
    """CKA(E_initial, E_step_k) for every step-tagged checkpoint of one run.

    E_initial is step_000000.pt — the exact encoder that run started from, so unlike
    the static study there is nothing to rebuild and no seed assumption. Safe to run
    while the job is still training: it simply uses whatever checkpoints exist.
    """
    ckpts = sorted(steps_dir.glob("step_*.pt"))
    init_path = steps_dir / "step_000000.pt"
    if not ckpts:
        print(f"=== {name}: SKIPPED — no checkpoints in {steps_dir.name} yet\n")
        return []
    if not init_path.exists():
        print(f"=== {name}: SKIPPED — {steps_dir.name}/step_000000.pt missing "
              f"(E_initial is required)\n")
        return []

    initial = load_encoder(init_path, str(device))
    hidden = torch.load(init_path, map_location="cpu", weights_only=False)["hidden_size"]
    other = random_encoder(OTHER_INIT_SEED, hidden, device)
    assays = held_out_assays(parquet)
    steps = [(int(p.stem.split("_")[1]), p) for p in ckpts]
    print(f"=== {name} trajectory (d_h={hidden}) | {len(steps)} checkpoints "
          f"(step {steps[0][0]}–{steps[-1][0]}) | {len(assays)} held-out assays")

    # Load every checkpoint once (~1.3 MB each), then reuse across all assays.
    encoders = [(step, load_encoder(p, str(device))) for step, p in steps]
    per_step: dict[int, list[float]] = {step: [] for step, _ in encoders}
    per_ref: dict[str, list[float]] = {k: [] for k in ("other-init", "chemeleon", "morgan")}

    for assay in assays:
        smiles = assay.smiles
        e_init = batched_encode(initial, smiles, device, warn=False)
        for key, emb in reference_embeddings(smiles, initial, other, chemeleon, device).items():
            per_ref[key].append(linear_cka(e_init, emb))
        for step, enc in encoders:
            per_step[step].append(
                linear_cka(e_init, batched_encode(enc, smiles, device, warn=False)))

    rows = []
    for step, vals in sorted(per_step.items()):
        m, s = mean_std(vals)
        rows.append([name, "trained", step, f"{m:.4f}", f"{s:.4f}", len(vals)])
        print(f"    step {step:>6} | CKA(E_initial, trained) = {m:.3f} +- {s:.3f}")
    for key, vals in per_ref.items():  # constant reference lines
        m, s = mean_std(vals)
        rows.append([name, key, "", f"{m:.4f}", f"{s:.4f}", len(vals)])
        print(f"    reference  {key:<11s} = {m:.3f} +- {s:.3f}")
    print()
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CKA study | device {device} | {N_ASSAYS} held-out assays per encoder "
          f"(>= {MIN_ASSAY_MOLS} molecules each)\n")

    chemeleon = CheMeleonFingerprint(device=str(device))
    rows = []

    for name, (ckpt, parquet) in ENCODERS.items():
        ckpt_path = REPO_ROOT / ckpt
        if not ckpt_path.exists():
            print(f"=== {name}: SKIPPED — no checkpoint at {ckpt}\n")
            continue

        trained = load_encoder(ckpt_path, str(device))
        hidden = trained.hidden_size if hasattr(trained, "hidden_size") else None
        if hidden is None:  # width lives in the checkpoint payload
            hidden = torch.load(ckpt_path, map_location="cpu",
                                weights_only=False)["hidden_size"]
        initial, init_source = initial_encoder(ckpt_path, hidden, device)
        other = random_encoder(OTHER_INIT_SEED, hidden, device)

        assays = held_out_assays(parquet)
        print(f"=== {name}  (d_h={hidden}, {len(assays)} assays) | E_initial: {init_source}")

        # CKA(E_initial, ?) for each comparison, one score per held-out assay.
        scores: dict[str, list[float]] = {k: [] for k in
                                          ("trained", "other-init", "chemeleon", "morgan")}
        for assay in assays:
            smiles = assay.smiles
            e_init = batched_encode(initial, smiles, device, warn=False)
            against = {
                "trained": batched_encode(trained, smiles, device, warn=False),
                "other-init": batched_encode(other, smiles, device, warn=False),
                "chemeleon": batched_encode(chemeleon, smiles, device, warn=False),
                "morgan": batched_encode(lambda s: morgan_features(s), smiles, device,
                                         warn=False),
            }
            for key, emb in against.items():
                scores[key].append(linear_cka(e_init, emb))

        for key, vals in scores.items():
            m, s = mean_std(vals)
            print(f"    CKA(E_initial, {key:<11s}) = {m:.3f} +- {s:.3f}")
            rows.append([name, key, f"{m:.4f}", f"{s:.4f}", len(vals), init_source])
        print()

    out = REPO_ROOT / SAVE_TO
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "compared_to", "cka_mean", "cka_std", "n_assays",
                    "e_initial_source"])
        w.writerows(rows)
    print(f"saved -> {SAVE_TO}")
    print("\nRead: CKA near 1.0 = the representation barely moved. Compare the "
          "'trained' row\nagainst 'other-init' (reseeding) and against "
          "'chemeleon'/'morgan' (a real gap).\n")

    # --- Trajectory: CKA across training ---------------------------------------- #
    if TRAJECTORIES:
        print("=" * 70)
        print("TRAJECTORY — CKA(E_initial, E_step_k) across training\n")
        traj_rows = []
        for name, (steps_dir, parquet) in TRAJECTORIES.items():
            traj_rows += run_trajectory(name, REPO_ROOT / steps_dir, parquet,
                                        chemeleon, device)
        if traj_rows:
            tout = REPO_ROOT / SAVE_TRAJECTORY_TO
            with tout.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["run", "compared_to", "step", "cka_mean", "cka_std", "n_assays"])
                w.writerows(traj_rows)
            print(f"saved -> {SAVE_TRAJECTORY_TO}")
            print("\nRead: plot cka_mean vs step for compared_to='trained'. The three "
                  "reference\nrows (blank step) are horizontal lines — they set the "
                  "scale the curve is read against.")
        else:
            print("No trajectories available yet — run `cluster/sync.sh pull` once the "
                  "100k jobs\nhave written some checkpoints, then re-run this script.")


if __name__ == "__main__":
    main()
