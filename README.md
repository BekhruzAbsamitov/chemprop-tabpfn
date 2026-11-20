# Exploring Tabular Foundation Models for Active Learning in Drug Discovery

Code for the Master's thesis of Bekhruz Absamitov (Saarland University).

In early drug discovery, active learning chooses which compounds to measure next, so that a limited
budget finds as many actives as possible. The thesis uses **TabPFN**, a tabular foundation model, as
the surrogate model that guides these choices, and asks four questions:

1. Is TabPFN a better surrogate for hit discovery than a random forest and Gaussian processes?
2. Does its predictive uncertainty improve the acquisition of compounds?
3. How much does the molecular representation matter?
4. Can a **Chemprop** message-passing encoder be trained through the frozen TabPFN, so that it
   learns a representation tailored to it?

The first three are answered by simulated active learning campaigns on four pools (D2R, USP7, TYK2
and a whole-cell *S. aureus* screen). For the fourth, the encoder is meta-trained on bioactivity
assays from ChEMBL, with gradients flowing from TabPFN's prediction error back into Chemprop.

## Main results

- TabPFN is the strongest surrogate, but modestly: it ends ahead of the random forest by
  0.22 ± 0.05 in cumulative enrichment factor, although not on every target, and it costs more per
  candidate.
- Expected Improvement over the best value so far loses to greedy selection on every target. The
  probability of exceeding the hit threshold beats greedy selection on three of four targets.
- CheMeleon, a pretrained encoder, is the best representation for TabPFN.
- The encoder trained through TabPFN receives gradients and improves on held-out assays, but does
  not transfer usefully to new targets.

The thesis has the details: the benchmark in Chapter 4, the encoder training in Chapter 5.

---

## Setup

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync                  # builds .venv from pyproject.toml
cp .env.example .env     # then put your Hugging Face token into .env
```

The TabPFN checkpoint is downloaded from a gated Hugging Face repository, so the account behind
`HF_TOKEN` must have accepted access to it (`.env.example` explains this). `tabpfn` is pinned to
8.1.0, the version every result was produced with; later versions change the numerics.

## Running

Each script in `src/scripts/` is one self-contained file. Its settings are constants at the top of
the file: open it in the IDE, edit them and press Run. No environment variables or `PYTHONPATH` are
needed. From a terminal:

```bash
uv run python src/scripts/run_experiment.py     # the default preset is a fast local run
uv run pytest                                   # unit tests
```

The benchmark and the full-scale training runs need a GPU and were run on an HTCondor cluster;
`cluster/README.md` describes the set-up, and `cluster/sync.sh` copies code and results between
the laptop and the cluster.

---

## The data

No data and no model weights are in the repository. The scripts expect their input files under
`data/`; each script names the files it reads and writes as constants at its top, and results go
to `experiments/`.

The datasets themselves (the four benchmark pools and the training corpora, with their sizes,
sources and preprocessing) are described in Chapter 3 of the thesis.

---

## Project layout

```
data/                 datasets and weights (not in the repository)
src/                  pipeline modules: encoder, TabPFN regressor, baselines, acquisition, training
src/scripts/          runnable scripts — open in your IDE and press Run
cluster/              HTCondor job files for the GPU runs
notebooks/            data curation
tests/                unit tests
archive/              earlier scripts and job files that no thesis result depends on
experiments/          results written by the scripts (not in the repository)
```

## Which script produces what in the thesis

| Thesis | Script (`src/scripts/`) |
|---|---|
| Benchmark runs behind Chapter 4 and the transfer results of Chapter 5 | `benchmark_extended.py` (pools, representations and protocol from `benchmark_al.py`), submitted with `cluster/run_gpu_benchmark_final.sub` and `cluster/run_gpu_benchmark_final_seed1_control.sub` |
| Encoder training (Chapter 5) | `run_experiment.py`, `reevaluate_epochs.py`; corpora from `build_single_protein_dataset.py`, `build_adaptive_dataset.py`, `clean_chembl_targets.py`; *S. aureus* labels from `build_saureus_corrected.py` |
| Figures 4.1 to 4.6, Table 4.1 | `plot_benchmark.py` |
| Appendix B (runtime, candidate subsampling) | `runtime_study.py`, `runtime_capacity.py`, `appendix_b_tables.py` |
| Appendix C (detailed benchmark results) | `calibration_study.py`, `gp_mordred_geometry.py`, `appendix_c_tables.py` |
| Chapter 5 figures and Appendix D | `plot_training.py`, `cka_study.py`, `cka_debiased.py`, `cka_epoch_level.py`, `feature_stats.py`, `feature_stats_standardised.py`, `appendix_d_tables.py` |
| Overlap between training corpora and pools (Chapter 3) | `leakage_by_target.py` |

## License

MIT, see `LICENSE`.
