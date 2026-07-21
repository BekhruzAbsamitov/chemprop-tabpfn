"""src/encoder.py — the Chemprop part: turn a molecule into numbers.

A molecule arrives as a SMILES string (e.g. "CCO" = ethanol). A computer can't
do math on a string, so we need to convert each molecule into a fixed-length
list of numbers (a "fingerprint" / "embedding"). That is this file's only job.

We use Chemprop, a graph neural network. It reads the molecule AS A GRAPH (atoms
= nodes, bonds = edges), passes messages between neighbouring atoms a few times,
then averages everything into one vector per molecule. Crucially this is a
*trainable* network: the numbers it produces can be improved by training, which
is the whole point of the thesis.

Pipeline for one batch of SMILES:

    SMILES strings
        │  (featurize: describe each atom & bond as numbers)
        ▼
    graph batch  ("BatchMolGraph")
        │  (Chemprop message passing + averaging)
        ▼
    embeddings   (one row of numbers per molecule)   <-- what we return

Press Run on this file to encode a few example molecules and see the output.
"""

from __future__ import annotations

import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from torch import nn

# --------------------------------------------------------------------------- #
# SETTINGS — the size/shape of the encoder. Edit and re-run to experiment.     #
# --------------------------------------------------------------------------- #
HIDDEN_SIZE = 128  # length of each molecule's number-vector (bigger = more capacity)
DEPTH = 3  # how many rounds atoms exchange messages with neighbours
NORMALIZE = True  # rescale outputs to a tidy range so TabPFN sees consistent numbers


class MoleculeEncoder(nn.Module):
    def __init__(
            self,
            hidden_size: int = HIDDEN_SIZE,
            depth: int = DEPTH,
            normalize: bool = NORMALIZE,
    ):
        super().__init__()

        # Featurizer: describes each atom and bond as raw numbers (the graph's
        # starting features), before any learning happens.
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

        # The learnable network, in three parts:
        message_passing = BondMessagePassing(d_h=hidden_size, depth=depth)  # atoms talk to neighbours
        aggregation = MeanAggregation()  # average atoms -> one vector
        # MPNN insists on a prediction head, but we NEVER use its predictions —
        # we only take the embedding before it. It's here just to build the model.
        unused_head = RegressionFFN(input_dim=hidden_size)
        self.mpnn = MPNN(message_passing, aggregation, unused_head)

        # Run one molecule through to discover the embedding length automatically.
        with torch.no_grad():
            self.output_dim: int = self._embed(["CCO"]).shape[1]

        # Optional final rescaling. LayerNorm puts every embedding on a similar
        # scale (mean ~0, spread ~1), which makes TabPFN's job more stable.
        self.norm: nn.Module = nn.LayerNorm(self.output_dim) if normalize else nn.Identity()

    def _device(self) -> torch.device:
        """The device (cpu/gpu) the model's weights currently live on."""
        return next(self.parameters()).device

    def _embed(self, smiles: list[str]) -> torch.Tensor:
        """SMILES strings -> (n_molecules, hidden_size) raw embeddings (no norm)."""
        # 1. Wrap each SMILES as a Chemprop datapoint, collect into a dataset.
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles]
        dataset = MoleculeDataset(datapoints, featurizer=self.featurizer)

        # 2. Collate the molecules into ONE batched graph (BatchMolGraph).
        batch = collate_batch([dataset[i] for i in range(len(dataset))])
        graph = batch.bmg
        graph.to(self._device())  # move graph onto the model's device (in place)

        # 3. Run message passing + averaging -> one embedding per molecule.
        return self.mpnn.fingerprint(graph)

    def forward(self, smiles: list[str]) -> torch.Tensor:
        """Encode a list of SMILES into embeddings. Gradients flow through this."""
        if not smiles:
            raise ValueError("MoleculeEncoder.forward got an empty list of SMILES")
        return self.norm(self._embed(smiles))


# --------------------------------------------------------------------------- #
# Press Run to see the encoder turn a few molecules into numbers.             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    encoder = MoleculeEncoder()
    example_smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]  # ethanol, benzene, aspirin

    embeddings = encoder(example_smiles)

    print(f"encoder embedding length (output_dim): {encoder.output_dim}")
    print(f"input: {len(example_smiles)} molecules  ->  output shape: {tuple(embeddings.shape)}")
    print(f"trainable numbers (weights) in the encoder: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"\nfirst 8 numbers of the aspirin embedding:\n{embeddings[2, :8]}")
