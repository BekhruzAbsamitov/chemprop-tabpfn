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
data/                 datasets
src/                  reusable pipeline modules (encoder, TabPFN head, training)
scripts/              runnable experiment scripts — open in your IDE and press Run
notebooks/            data curation + proposal
tests/                unit tests
```

