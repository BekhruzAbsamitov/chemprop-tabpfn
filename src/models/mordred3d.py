"""src/models/mordred3d.py — Mordred descriptors including 3D geometry.

1,826 calculated descriptors per molecule. Unlike Morgan (topological hashing) and
physchem (whole-molecule properties), this set includes **3D** terms — geometrical
moments, surface-area distributions, spatial autocorrelations — which require an actual
conformer, not just the graph. Each molecule therefore gets hydrogens added and a single
ETKDGv3 conformer embedded before the descriptors are computed.

COST. Descriptor calculation dominates at ~150 ms/molecule, so a 40k pool is ~2 h on one
core (~20 min at `n_proc=8`). That is why this module CACHES: features are keyed by the
exact SMILES list they were computed for, written once, and reloaded thereafter. Any
benchmark re-run is then free.

NOTE ON WIDTH. At 1,826 features this is by far the widest representation in the
comparison (Chemeleon 2048 is comparable, Chemprop 300 and physchem 200 are not), which
is exactly the dimensionality confound the ablation in the report addresses.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")  # mordred/rdkit are noisy on unusual structures
from mordred import Calculator, descriptors  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

CONFORMER_SEED = 42   # fixed so the descriptors are reproducible run to run
N_PROC = 4            # mordred worker processes; matches the cluster's request_CPUs
CACHE_DIR = "data/features"


def _cache_file(smiles: list[str], cache_dir: Path) -> Path:
    """One cache file per exact SMILES list (order included, so slicing can't collide)."""
    digest = hashlib.sha1("\n".join(smiles).encode()).hexdigest()[:16]
    return cache_dir / f"mordred3d_{len(smiles)}_{digest}.npy"


def mordred3d_features(
        smiles: list[str],
        *,
        n_proc: int = N_PROC,
        cache_dir: str | Path | None = CACHE_DIR,
        verbose: bool = True,
) -> torch.Tensor:
    """SMILES -> (n_molecules, 1826) float tensor of Mordred 3D descriptors.

    Molecules that fail to parse, or whose conformer embedding fails, become all-zero
    rows — the same convention as the other featurizers, so the output always has one
    row per input. Non-finite descriptor values are zeroed for the same reason.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = _cache_file(smiles, Path(cache_dir))
        if cache_path.exists():
            if verbose:
                print(f"  [mordred3d] cache hit -> {cache_path.name}")
            return torch.from_numpy(np.load(cache_path))

    calc = Calculator(descriptors, ignore_3D=False)
    out = np.zeros((len(smiles), len(calc.descriptors)), dtype=np.float32)

    # 1. Parse + embed one conformer per molecule, remembering which rows succeeded.
    mols, rows = [], []
    for row, smi in enumerate(smiles):
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = CONFORMER_SEED
        if AllChem.EmbedMolecule(mol, params) == -1:  # embedding failed
            continue
        mols.append(mol)
        rows.append(row)

    if verbose:
        print(f"  [mordred3d] {len(mols)}/{len(smiles)} conformers embedded; "
              f"computing {len(calc.descriptors)} descriptors (n_proc={n_proc}) ...")

    # 2. Descriptors for the molecules that made it, scattered back to their rows.
    if mols:
        df = calc.pandas(mols, nproc=n_proc, quiet=True)
        values = (df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
                    .fillna(0.0).to_numpy(dtype=np.float32))
        out[np.asarray(rows)] = values

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
        if verbose:
            print(f"  [mordred3d] cached -> {cache_path.name}")
    return torch.from_numpy(out)


if __name__ == "__main__":
    x = mordred3d_features(
        ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "not-a-molecule"], cache_dir=None)
    print(f"mordred3d shape: {tuple(x.shape)}")
    print(f"  unparseable row is all-zero: {bool((x[3] == 0).all())}")
    print(f"  finite everywhere: {bool(torch.isfinite(x).all())}")
