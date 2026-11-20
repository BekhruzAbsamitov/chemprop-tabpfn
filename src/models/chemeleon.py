"""src/models/chemeleon.py — the Chemeleon pretrained fingerprinter.

Chemeleon is a Chemprop message-passing network pretrained on PubChem to predict
Mordred descriptors. We use it as a FROZEN feature extractor: SMILES -> a fixed
embedding, via the same `.fingerprint()` interface as our own MoleculeEncoder.

The checkpoint auto-downloads once from Zenodo and is cached in ~/.chemprop, so
no Hugging Face token or manual download is needed. Adapted from the official
Chemeleon usage snippet.
"""

from __future__ import annotations

from pathlib import Path

import torch


class CheMeleonFingerprint:
    """Turn SMILES into pretrained-Chemeleon embeddings via `.fingerprint()`."""

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


if __name__ == "__main__":
    fp = CheMeleonFingerprint(device="cpu")
    emb = fp(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    print(f"Chemeleon embedding shape: {tuple(emb.shape)}")
