"""src/models/physchem.py — RDKit physicochemical descriptors as a static baseline.

200 interpretable whole-molecule properties (molecular weight, logP, TPSA, hydrogen-bond
donor/acceptor counts, ring counts, fragment counts, ...), computed by descriptastorus's
`RDKit2DNormalized`. The *Normalized* variant maps each descriptor through its empirical
CDF so every feature lands in [0, 1] — without it the raw descriptors span wildly
different scales (molecular weight in the hundreds next to counts in single digits),
which distorts distance-based models.

This is the smallest representation in the comparison (200-d, dense) and the only one
whose individual features are human-readable. Same interface as the encoders —
`physchem_features(smiles) -> tensor` — so it plugs into the same TabPFN head.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")  # descriptastorus is noisy on odd molecules
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors  # noqa: E402

NORMALIZED = True  # CDF-normalized descriptors; False = raw, wildly different scales


def physchem_features(smiles: list[str], *, normalized: bool = NORMALIZED) -> torch.Tensor:
    """SMILES -> (n_molecules, 200) float tensor of physicochemical descriptors.

    A molecule descriptastorus cannot process becomes an all-zero row, so the output
    always has one row per input — the same convention as `morgan_features`. Non-finite
    values (a few descriptors overflow on unusual structures) are zeroed too, because
    TabPFN's preprocessing indexes on feature values and a NaN crashes it.
    """
    generator = (rdNormalizedDescriptors.RDKit2DNormalized() if normalized
                 else rdDescriptors.RDKit2D())
    n_features = len(generator.columns)
    out = np.zeros((len(smiles), n_features), dtype=np.float32)

    for row, smi in enumerate(smiles):
        if not isinstance(smi, str) or not smi.strip():
            continue
        result = generator.process(smi)
        # result[0] is a success flag; the descriptors follow it.
        if result is None or not result[0]:
            continue
        out[row] = np.asarray(result[1:], dtype=np.float32)

    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(out)


if __name__ == "__main__":
    x = physchem_features(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "not-a-molecule"])
    print(f"physchem shape: {tuple(x.shape)}")
    print(f"  row 0 range: [{x[0].min():.3f}, {x[0].max():.3f}]  (normalized -> [0,1])")
    print(f"  unparseable row is all-zero: {bool((x[3] == 0).all())}")
