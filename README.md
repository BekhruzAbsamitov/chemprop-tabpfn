# Chemprop → TabPFN for Active Learning in Drug Discovery

Master's thesis code (Bekhruz Absamitov). The idea in one sentence:

> Train a **Chemprop** graph network to turn a molecule into numbers that a frozen
> **TabPFN** model likes best, so TabPFN predicts molecular activity more accurately.

**Chemprop** reads a molecule (from its SMILES string) and produces a numeric
"fingerprint". **TabPFN** is a frozen tabular foundation model that takes those
numbers and predicts activity. We let gradients flow from TabPFN's prediction
error *back into* Chemprop, so Chemprop learns a representation tailored to
TabPFN. The research question: **does training Chemprop this way actually
make TabPFN better than off-the-shelf molecular features?**

---

## The data

Everything lives in `data/`. There are three files.

### 1. `data/curated/chembl36_curated.parquet` — the full dataset
The complete, cleaned ChEMBL activity dataset (produced by the supervisor's
curation notebook). This is the "everything" file used for large-scale training.

| | |
|---|---|
| Rows (activity measurements) | **4,472,273** |
| Columns | 42 |
| Unique assays (`assay_x_type_x_doc`) | **248,060** |
| Unique compounds (`compound_chembl_id`) | 1,509,947 |
| Split (train / val / test) | 3,134,043 / 673,001 / 665,229 |

### 2. `data/curated/chembl36_dev.parquet` — the small dev subset
A **small, reproducible sample** drawn from the curated file (~1% of the assays),
with the **same 42 columns and the same split labels**. Use this for fast local
experiments so you don't wait on the 4.5-million-row file. Same shape, same code
path — just smaller.

| | |
|---|---|
| Rows | **46,257** |
| Columns | 42 (identical schema to curated) |
| Unique assays | **2,481** |
| Unique compounds | 41,476 |
| Split (train / val / test) | 32,592 / 7,140 / 6,525 |

**curated vs dev:** dev is a subset of curated — same schema, same meaning,
roughly 1% the size. Develop and debug on `dev`; run the real thing on
`curated` (on the GPU cluster).

### 3. `~/University/thesis/data/s_aureus_curated_with_scores_filtered.csv` — the target
The real evaluation target: *S. aureus* antibacterial screening data. This is the
dataset the thesis compares models on in the active-learning loop.

| | |
|---|---|
| Molecules (unique SMILES) | ~40,270 |
| Key columns | `Smiles` (molecule), `pMIC` (potency; higher = more potent) |
| pMIC range | 0 – 8.1 (mean ≈ 1.3) |

### The columns that matter
Both parquet files share these key columns:

| Column | Meaning |
|---|---|
| `assay_x_type_x_doc` | **Assay id** — one assay = one prediction task (TabPFN learns "in context" per assay) |
| `canonical_smiles` | The molecule, as a SMILES string |
| `target_transformed` | The value to predict (normalized activity) |
| `compound_chembl_id` | Compound id |
| `split` | `train` / `val` / `test`, assigned **per assay** (an assay's molecules are never split across sets — no leakage) |
| `group` | Grouping key used to build the split |

---

## Project layout

```
data/                 datasets (not in the repository)
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

