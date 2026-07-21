"""src/models/morgan.py — Morgan (ECFP) fingerprints as a static baseline.

The simplest molecular representation: a fixed 2048-bit circular fingerprint per
molecule, computed by RDKit with no learning at all. It is the static baseline
every trainable representation must beat. Same interface as the encoders —
`morgan_features(smiles) -> tensor` — so it plugs into the same TabPFN head.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

RADIUS = 2      # ECFP4-style neighbourhood radius
N_BITS = 2048   # default fingerprint length


def morgan_features(smiles: list[str], radius: int = RADIUS, n_bits: int = N_BITS) -> torch.Tensor:
    """SMILES -> (n_molecules, n_bits) float tensor of Morgan fingerprints.

    An unparseable SMILES becomes an all-zero row (same as the encoders' handling
    of degenerate molecules), so the output always has one row per input.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
        else:
            rows.append(generator.GetFingerprintAsNumPy(mol).astype(np.float32))
    return torch.from_numpy(np.stack(rows))


if __name__ == "__main__":
    fp = morgan_features(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    print(f"Morgan fingerprints shape: {tuple(fp.shape)} | nonzero bits in row 0: "
          f"{int(fp[0].sum())}")
