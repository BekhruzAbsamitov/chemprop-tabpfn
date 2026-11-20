"""src/models/rigr_encoder.py — the supervisor's pretrained Chemprop encoder. Press Run.

Supplied at the 2026-08-20 supervision (`notes/notes_2026_08_20.md`) as a new
representation arm for the AL benchmark: a Chemprop D-MPNN the group pretrained
themselves, to be compared against Chemeleon, Morgan, physchem, Mordred 3D and our own
trained/untrained encoders.

TWO THINGS DIFFER from `models/encoder.py`, and both matter:

1. **RIGR featurisation.** Resonance-Invariant Graph Representation — atoms and bonds are
   described so that two resonance forms of the same molecule (e.g. the two Kekulé
   structures of benzene, or a carboxylate drawn either way) featurise IDENTICALLY. The
   standard featuriser does not guarantee that, so the same molecule can get different
   features depending on how its SMILES was written. Dimensions are much smaller as a
   result: 52 atom features and **2** bond features, against 72 and 14 for
   `SimpleMoleculeMolGraphFeaturizer`.

2. **The embedding is the FULL forward pass, not `fingerprint()`.** Our encoder stops
   after message passing + aggregation. This one sets `n_tasks = n_hidden` on the
   RegressionFFN and uses its *output* as the embedding — the predictor head is part of
   the encoder here, not a discarded appendage. Calling `.fingerprint()` on it would
   return the 256-d intermediate instead of the intended 512-d embedding.

Architecture is fixed by the checkpoint and must not be edited: d_v=52, d_e=2,
d_h=256 (= n_hidden // 2), GELU activation, no batch norm, embedding dim 512.

WEIGHTS. Set `WEIGHTS_PATH` to the .pt the supervisor provides. Without it the module
still builds and runs (random init), which is enough to verify wiring but is NOT the
pretrained representation — every entry point says loudly which one you got.
"""

from __future__ import annotations

from pathlib import Path

import chemprop
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.featurizers.atom import RIGRAtomFeaturizer
from chemprop.featurizers.bond import RIGRBondFeaturizer
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from torch import nn
from torch.nn import GELU

# --------------------------------------------------------------------------- #
# SETTINGS                                                                     #
# --------------------------------------------------------------------------- #
WEIGHTS_PATH = "data/weights/chemicl_mpnn.pt"  # the supervisor's pretrained checkpoint
N_HIDDEN = 512      # embedding dim; also n_tasks on the predictor
D_V = 52            # RIGR atom feature dim  — fixed by the checkpoint
D_E = 2             # RIGR bond feature dim  — fixed by the checkpoint


def build_rigr_mpnn(n_hidden: int = N_HIDDEN) -> chemprop.models.MPNN:
    """The exact architecture the checkpoint was saved from (supervisor's snippet)."""
    act = GELU
    mp = chemprop.nn.BondMessagePassing(d_v=D_V, d_e=D_E, d_h=n_hidden // 2,
                                        activation=act())
    agg = chemprop.nn.MeanAggregation(d_h=n_hidden // 2, activation=act())
    # n_tasks = n_hidden: the predictor's OUTPUT is the embedding, not a prediction.
    predictor = chemprop.nn.RegressionFFN(input_dim=n_hidden // 2, n_tasks=n_hidden,
                                          activation=act())
    return chemprop.models.MPNN(mp, agg, predictor, batch_norm=False)


class RIGREncoder(nn.Module):
    """SMILES -> (n_molecules, 512) embeddings from the group's pretrained D-MPNN.

    Frozen and in eval mode: this is a fixed representation arm for the benchmark, the
    same status as Chemeleon or Morgan. Nothing here is trained.

    UNPARSEABLE SMILES RAISE, they do not become zero rows. That differs from
    morgan/physchem/mordred3d but matches the other *neural* arms — CheMeleonFingerprint
    and MoleculeEncoder both raise too, because Chemprop cannot build a graph from a
    molecule RDKit refuses to parse. The benchmark pools are pre-cleaned, so this has
    never fired; keep it that way rather than silently embedding a zero molecule.
    """

    def __init__(self, weights_path: str | Path | None = WEIGHTS_PATH,
                 n_hidden: int = N_HIDDEN, verbose: bool = True):
        super().__init__()
        self.featurizer = SimpleMoleculeMolGraphFeaturizer(
            atom_featurizer=RIGRAtomFeaturizer(),
            bond_featurizer=RIGRBondFeaturizer(),
        )
        self.mpnn = build_rigr_mpnn(n_hidden)
        self.output_dim = n_hidden

        self.pretrained = False
        path = Path(weights_path) if weights_path else None
        if path is not None and not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if path is not None and path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            # Accept a bare state_dict or one nested under a common wrapper key.
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            self.mpnn.load_state_dict(state)
            self.pretrained = True
            if verbose:
                print(f"  [rigr] loaded pretrained weights <- {path.name}")
        elif weights_path is None:
            # Deliberate: the matched random-init control arm. Not a problem.
            if verbose:
                print("  [rigr] random init (no weights requested) — control arm")
        elif verbose:
            # A path was asked for and is missing. Almost certainly a mistake, so say so
            # loudly rather than silently benchmarking a random encoder as 'pretrained'.
            print(f"  [rigr] !! WEIGHTS NOT FOUND at {path} — falling back to RANDOM "
                  f"INIT. This is NOT the pretrained representation.")

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, smiles: list[str]) -> torch.Tensor:
        """Embed a batch of SMILES. Full forward pass — see the module docstring."""
        if not smiles:
            raise ValueError("RIGREncoder.forward got an empty list of SMILES")
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles]
        dataset = MoleculeDataset(datapoints, featurizer=self.featurizer)
        batch = collate_batch([dataset[i] for i in range(len(dataset))])
        graph = batch.bmg
        graph.to(self._device())
        with torch.no_grad():
            return self.mpnn(graph)


def rigr_features(smiles: list[str], **kwargs) -> torch.Tensor:
    """Convenience one-shot featuriser, matching morgan_features / physchem_features."""
    return RIGREncoder(**kwargs)(smiles)


if __name__ == "__main__":
    enc = RIGREncoder()
    smi = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    x = enc(smi)
    print(f"pretrained weights loaded: {enc.pretrained}")
    print(f"embedding shape: {tuple(x.shape)}  (expected ({len(smi)}, {N_HIDDEN}))")
    print(f"finite everywhere: {bool(torch.isfinite(x).all())}")
    print(f"parameters: {sum(p.numel() for p in enc.parameters()):,}")

    # RIGR's defining property: resonance forms must embed identically. Benzene written
    # aromatic vs. as one Kekule structure is the standard check.
    pair = enc(["c1ccccc1", "C1=CC=CC=C1"])
    delta = float((pair[0] - pair[1]).abs().max())
    print(f"\nresonance invariance (benzene aromatic vs Kekule): max |diff| = {delta:.2e}")
