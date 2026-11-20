"""scripts/cka_epoch_level.py — how far does the epoch-level encoder move from its random initialisation? Press Run.

Chapter 5 follows one encoder, the epoch-level run on the single-protein corpus (learning rate 3e-5, 26 epochs,
two seeds). cka_study.py and cka_debiased.py measure CKA only for the all-ChEMBL and domain-matched encoders;
this script does the same for the epoch-level run:

  CKA(E_epoch_0, E_epoch_k)   for every saved epoch k = 1..26 of each seed, with E_epoch_0 the saved epoch_000.pt
  anchors                     CKA(E_epoch_0, .) for the OTHER seed's epoch_000.pt (a change of seed), CheMeleon
                              and Morgan fingerprints

with both estimators of cka_debiased.py (linear, Kornblith et al. 2019; debiased, Song et al. 2012), per held-out
assay of the single-protein corpus with at least MIN_ASSAY_MOLS molecules (the first N_ASSAYS of them, as in
cka_debiased.py), mean and standard deviation over assays.

Output: SAVE_TO (refuses to overwrite), with the columns of cka_debiased.csv. Read by plot_training.py
(fig:epoch_change) and appendix_d_tables.py (tab:appd_cka_epoch). Runs on the CPU in a few minutes.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cka_debiased import (  # noqa: E402
    CheMeleonFingerprint, anchors, batched_encode, debiased_cka, held_out_assays, linear_cka, load_encoder, summarise)

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
SINGLE_PROTEIN = "data/curated/chembl36_single_protein_clean.parquet"   # its `test` split gives the held-out assays
RUNS = {  # run name in the output -> directory of its epoch_XXX.pt checkpoints (epoch_000.pt = random initialisation)
    "full_sp_50ep": "experiments/trained_encoder_full_sp_50ep_epochs",
    "full_sp_50ep_seed1": "experiments/trained_encoder_full_sp_50ep_seed1_epochs",
}
DEVICE = "cpu"
SAVE_TO = "experiments/cka_epoch_level.csv"


def epoch_checkpoints(directory: Path) -> dict[int, Path]:
    """Epoch number -> checkpoint; the number is read after `epoch_` (some files carry a suffix)."""
    return {int(re.match(r"epoch_(\d+)", p.stem).group(1)): p for p in sorted(directory.glob("epoch_*.pt"))}


def main() -> None:
    out = REPO_ROOT / SAVE_TO
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    device = torch.device(DEVICE)
    chemeleon = CheMeleonFingerprint(device=str(device))
    assays = held_out_assays(SINGLE_PROTEIN)
    print(f"{len(assays)} held-out assays, {min(len(a.smiles) for a in assays)} to "
          f"{max(len(a.smiles) for a in assays)} molecules each")
    ckpts = {name: epoch_checkpoints(REPO_ROOT / d) for name, d in RUNS.items()}
    initial = {name: load_encoder(c[0], str(device)) for name, c in ckpts.items()}
    rows = []
    for name, c in ckpts.items():
        other = initial[next(n for n in RUNS if n != name)]          # the other seed's random initialisation
        encoders = {e: load_encoder(p, str(device)) for e, p in c.items() if e > 0}
        per_epoch = {e: ([], []) for e in encoders}
        per_ref = {k: ([], []) for k in ("other-init", "chemeleon", "morgan")}
        for assay in assays:
            e0 = batched_encode(initial[name], assay.smiles, device, warn=False)
            for k, e in anchors(assay.smiles, other, chemeleon, device).items():
                per_ref[k][0].append(linear_cka(e0, e))
                per_ref[k][1].append(debiased_cka(e0, e))
            for epoch, enc in encoders.items():
                e = batched_encode(enc, assay.smiles, device, warn=False)
                per_epoch[epoch][0].append(linear_cka(e0, e))
                per_epoch[epoch][1].append(debiased_cka(e0, e))
        for epoch, (lin, deb) in sorted(per_epoch.items()):
            rows.append(summarise("epoch-level", name, "trained", epoch, lin, deb))
        for k, (lin, deb) in per_ref.items():
            rows.append(summarise("epoch-level", name, k, "", lin, deb))
        last = [r for r in rows if r[1] == name and r[2] == "trained"][-1]
        ref = {r[2]: r for r in rows if r[1] == name and r[2] != "trained"}
        for i, label in ((4, "linear"), (6, "debiased")):
            moved = 1 - float(last[i])
            print(f"  {name:19s} {label:8s} epoch {last[3]}: CKA {last[i]} (1 - CKA {moved:.3f}); CheMeleon "
                  f"{ref['chemeleon'][i]} ({moved / (1 - float(ref['chemeleon'][i])):.0%} of its distance); change of "
                  f"seed {ref['other-init'][i]} ({moved / (1 - float(ref['other-init'][i])):.1f} times its distance)")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "run", "compared_to", "step", "cka_linear_mean", "cka_linear_std",
                    "cka_debiased_mean", "cka_debiased_std", "n_assays"])
        w.writerows(rows)
    print(f"\nsaved -> {SAVE_TO}")


if __name__ == "__main__":
    main()
