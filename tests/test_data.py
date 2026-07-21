"""Tests for src/data_utils/data.py — episode splitting and dataset loading."""

from __future__ import annotations

import random

import polars as pl
import torch

from data_utils import data
from data_utils.data import Assay, load_chembl_assays, load_target, make_episode


def _numbered_assay(n: int) -> Assay:
    """Assay where molecule 'C{i}' has activity value i (so we can check alignment)."""
    return Assay("A", [f"C{i}" for i in range(n)], torch.arange(float(n)))


def test_make_episode_split_is_disjoint_and_covers_all():
    ep = make_episode(_numbered_assay(10), context_frac=0.5, min_context=2, rng=random.Random(0))
    assert ep is not None
    assert len(ep.context_smiles) == 5 and len(ep.query_smiles) == 5
    assert set(ep.context_smiles).isdisjoint(ep.query_smiles)
    assert set(ep.context_smiles) | set(ep.query_smiles) == {f"C{i}" for i in range(10)}


def test_make_episode_keeps_labels_aligned_with_molecules():
    ep = make_episode(_numbered_assay(10), rng=random.Random(1))
    # molecule "C{k}" must carry activity value k in both context and query
    for smi, val in zip(ep.context_smiles, ep.context_y.tolist()):
        assert int(smi[1:]) == val
    for smi, val in zip(ep.query_smiles, ep.query_y.tolist()):
        assert int(smi[1:]) == val


def test_make_episode_is_deterministic_for_same_seed():
    a = make_episode(_numbered_assay(10), rng=random.Random(42))
    b = make_episode(_numbered_assay(10), rng=random.Random(42))
    assert a.context_smiles == b.context_smiles and a.query_smiles == b.query_smiles


def test_make_episode_too_small_returns_none():
    # 4 molecules < min_context(4) + min_query(1)
    assay = Assay("A", ["C", "CC", "CCC", "CCCC"], torch.arange(4.0))
    assert make_episode(assay, min_context=4, min_query=1, rng=random.Random(0)) is None


def test_make_episode_constant_labels_returns_none():
    # every label identical -> context std is 0 -> unusable
    assay = Assay("A", [f"C{i}" for i in range(10)], torch.full((10,), 3.0))
    assert make_episode(assay, rng=random.Random(0)) is None


def test_make_episode_respects_min_context():
    # context_frac would give 2, but min_context=5 must win
    ep = make_episode(_numbered_assay(20), context_frac=0.1, min_context=5, rng=random.Random(0))
    assert len(ep.context_smiles) == 5


def test_make_episode_caps_size_and_keeps_labels_aligned():
    # A 400-molecule assay would give 200 context + 200 query; the caps must
    # bound both (this is the GPU-OOM guard), while keeping each label with its
    # molecule.
    ep = make_episode(_numbered_assay(400), max_context=30, max_query=10, rng=random.Random(0))
    assert len(ep.context_smiles) == 30 and len(ep.query_smiles) == 10
    for smi, val in zip(ep.context_smiles, ep.context_y.tolist()):
        assert int(smi[1:]) == val
    assert set(ep.context_smiles).isdisjoint(ep.query_smiles)


def test_load_chembl_assays_groups_filters_and_drops_nulls(tmp_path, monkeypatch):
    df = pl.DataFrame(
        {
            "assay_x_type_x_doc": ["A1", "A1", "A1", "A2", "A2", "A3", "A1"],
            "canonical_smiles": ["C", "CC", "CCC", "CCCC", "CCCCC", "c1ccccc1", None],
            "target_transformed": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0],
            "split": ["train", "train", "train", "train", "train", "val", "train"],
            "junk_column": [0, 0, 0, 0, 0, 0, 0],  # the other 38 columns must be ignored
        }
    )
    path = tmp_path / "dev.parquet"
    df.write_parquet(path)
    monkeypatch.setattr(data, "CHEMBL_DEV_FILE", path)

    assays = {a.assay_id: a for a in load_chembl_assays("train")}

    assert set(assays) == {"A1", "A2"}  # A3 is 'val', excluded
    assert len(assays["A1"].smiles) == 3  # the null-SMILES row was dropped
    assert assays["A1"].y.tolist() == [1.0, 2.0, 3.0]
    assert assays["A2"].smiles == ["CCCC", "CCCCC"]


def test_load_target_dedups_by_median(tmp_path):
    path = tmp_path / "target.csv"
    pl.DataFrame({"Smiles": ["CCO", "CCO", "CCO", "CCN"], "pMIC": [1.0, 3.0, 5.0, 2.0]}).write_csv(path)

    smiles, y = load_target(path)
    values = dict(zip(smiles, y.tolist()))

    assert len(smiles) == 2
    assert values["CCO"] == 3.0  # median of 1, 3, 5
    assert values["CCN"] == 2.0
    assert y.dtype == torch.float32
