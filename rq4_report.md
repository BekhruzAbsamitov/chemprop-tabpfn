# RQ4 — Training a Chemprop encoder toward a frozen TabPFN for active learning

**Question:** Does training a molecular encoder (Chemprop) to feed a frozen TabPFN
improve hit-finding on *S. aureus* antibacterial activity — better than an untrained
encoder, and better than Morgan fingerprints?

- **Author:** Bekhruz Absamitov · **Advisor:** Joschka Groß · **Supervisor:** Prof. Verena Wolf
- **Institution:** Saarland University · **Date:** 15 July 2026
- **Reproduce:** `run_experiment_gpu.py` (static), `run_active_learning_gpu.py` (AL); curves in `experiments/al_curves.csv`

---

## Talking points (for the meeting)

> 1. **Coverage — be precise, don't say "a quarter."** Training was 4,000 gradient steps,
>    one random assay per step sampled *with replacement* from 174,023 assays — so it saw a
>    broad but **small sample, ~2–3% of the assays**, roughly one pass each. Not a quarter of
>    the dataset.
> 2. **Result.** The training loss was **flat from step 250** (never dropped, see §3), and on
>    *S. aureus* the **trained encoder is slightly *worse* than an untrained one** — static
>    ranking 0.440 → 0.408, active-learning (EI) 0.135 → 0.113. So training the encoder toward
>    TabPFN doesn't help; a random-init GNN is already the best representation.
> 3. **Framing.** This negative sits inside a **positive method** — GNN + TabPFN + uncertainty
>    acquisition beats Morgan and finds hits ~3× faster than random. Open question we can't yet
>    rule out: TabPFN being *representation-agnostic* (its internal normalization absorbs encoder
>    changes) vs. simply *undertrained*. The flat loss leans toward the former. **Next:** fine-tune
>    the encoder on *S. aureus* labels *inside* the loop — the one arm that could flip RQ4 positive.

---

## TL;DR

- We built a **differentiable Chemprop → frozen TabPFN** pipeline: the loss gradient
  flows *through* the frozen TabPFN back into Chemprop, so the encoder can be trained
  to produce representations TabPFN predicts well from.
- **Headline (negative):** training the encoder **did not help — it slightly hurt.**
  An **untrained (random-init) Chemprop encoder is the best representation**, in both
  a static ranking test and the active-learning loop. Consistent across two experiments.
- **Headline (positive):** the method still works well. GNN + TabPFN + uncertainty-guided
  acquisition beats Morgan fingerprints and finds hits **~3× faster than random screening.**
- The lever that works is the **representation + acquisition strategy**, not training
  the encoder toward the foundation model.

---

## 1. Data and splits

### Evaluation target — *S. aureus*
- **40,270 unique molecules**, each a SMILES + measured **pMIC** (higher = more active).
- Replicate measurements collapsed to the **median** pMIC.
- Never trained on — used only to measure "how fast does the method find the actives?"

### Training corpus — ChEMBL
- **174,023 assays** (full); a ~1,740-assay "dev" subset for fast local iteration.
- Molecules grouped by `assay × type × document` (one assay = measurements made the same way).

### Split strategy — by assay, not molecule
- **Assay-level 3-way split** (train / val / test).
- Splitting by *assay* prevents leakage: one assay's measurements never straddle two
  splits. Train drives encoder updates; val monitors; test is held out.

### Episodic framing — one assay = one task
- TabPFN is an **in-context** learner, so training is **episodic**.
- Each step: sample one random assay → split into a **context** set (labeled examples
  TabPFN learns from) + a **query** set (to predict), 50/50, capped at **128 + 128**
  molecules to bound GPU memory on very large assays.

### Features (model inputs) and target (label)

**Target — what we predict (`y`):** a single scalar activity per molecule.
- *S. aureus evaluation:* the **pMIC** value (`pMIC` column). Higher = more active.
- *ChEMBL training:* the assay's transformed activity (`target_transformed` column).

**Input features — what we predict from (`X`):** each molecule turned into a numeric
vector. Three representations are compared (one per "arm"), all fed to the same TabPFN:
- **Chemprop embedding (trained / untrained):** a **300-dim** vector produced by the GNN
  from the molecular graph. The graph's *raw* atom/bond features (before any learning,
  from Chemprop's `SimpleMoleculeMolGraphFeaturizer`) are standard descriptors — atoms:
  element, degree, formal charge, hybridization, aromaticity, H-count; bonds: bond type,
  conjugation, ring membership, stereo. Message passing turns these into the 300-dim
  embedding.
- **Morgan fingerprint:** a **2048-bit** ECFP4 vector (presence/absence of circular
  substructures out to radius 2). Fixed, not learned.

TabPFN then applies its own internal normalization/preprocessing to whichever feature
vector it receives before predicting.

---

## 2. The pipeline

```
SMILES ──▶ Chemprop MPNN ──▶ embedding ──▶ frozen TabPFN ──▶ activity + uncertainty
           (d_h=300, depth 3)  (300-dim)     (in-context)
              ▲                                      │  gradients (NLL loss)
              └──────── flow back THROUGH frozen TabPFN ─────┘
```

- **Encoder (trainable):** Chemprop 2.2.4 MPNN — bond message passing (width 300,
  depth 3), mean aggregation, LayerNorm → one 300-dim embedding per molecule. Only
  part with trainable weights.
- **Head (frozen, differentiable):** TabPFN v3 (8.1.0). Given a labeled context + a
  query set in one forward pass, predicts query activity **plus calibrated uncertainty**
  (needed for acquisition). Weights never updated, but differentiable — so gradients
  pass through it into Chemprop.
- **Loss:** TabPFN's native **NLL** (distributional — trains the full predictive
  distribution incl. uncertainty, not just a point).
- **Baselines:** **Morgan** fingerprints (ECFP4, 2048 bits, fixed) and an **untrained**
  (random-init) Chemprop encoder. The trained encoder must beat both.

---

## 3. How the encoder was trained

- Full-scale run on Saarland HTCondor cluster (NVIDIA A100).
- Settings: **full 174,023-assay ChEMBL**, encoder width 300, 8-estimator TabPFN
  ensemble, **4,000 episodic steps**, LR 1e-4, NLL loss. One random assay per step.

### What these settings mean

- **4,000 episodic steps.** One *step* = one *episode* = one gradient update. Each step
  samples a single random assay (with replacement) from the full 174,023, splits it into
  context + query, and updates the encoder once. So 4,000 is the number of **gradient
  updates**, not epochs — it is *not* "4,000 assays" in a coverage sense (with replacement
  it touches ~4,000 draws ≈ 2.3% of the pool). The encoder learns from a broad *sample* of
  tasks, not a full pass over the data; more steps = train longer, independent of dataset size.
- **LR = 1e-4 (learning rate).** The step size of the AdamW optimizer that updates the
  Chemprop weights — how far each gradient update moves them. Small = stable but slow;
  gradients are clipped to norm 1.0 to keep updates well-behaved. (TabPFN's weights are
  frozen and never updated.)
- **How the training loss is calculated (NLL).** For each episode: (1) encode the context
  molecules and *fit* TabPFN on them (context embeddings + their labels); (2) it predicts a
  **probability distribution** over activity for each query molecule; (3) the loss is the
  **negative log-likelihood** — how *unlikely* the true query pMIC values are under those
  predicted distributions — averaged over the query molecules. Computed in TabPFN's own
  z-normalized label space, with the target clipped to ±5 for stability. **Lower loss =
  TabPFN assigns higher probability to the true values**, i.e. the encoder's embeddings let
  TabPFN model the assay better. Because NLL scores the whole distribution (not just a point
  prediction), it also trains the **uncertainty** the active-learning acquisition relies on.

- **Observation: the training loss stayed flat** (~1.9–2.2, no downward trend). The
  encoder's updates did not reduce TabPFN's loss — first sign training buys little.

  Recorded running-average NLL (logged every 250 steps):

  | Step | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 |
  |------|----:|----:|----:|-----:|-----:|-----:|-----:|
  | Avg loss | 1.891 | 2.116 | 2.168 | 2.085 | 1.983 | 2.271 | 2.104 |

  There is **no downward trend** — the values just wobble in a ~1.9–2.3 band. If the
  encoder were learning, this curve would fall; a flat curve means the gradient isn't
  usefully reshaping it (a different failure mode than "needs more training").
  - *Caveat:* a flat loss can also mean a **weak gradient signal** (TabPFN's heavy
    internal normalization may absorb representation changes), so "fundamentally
    useless" vs. "this setup didn't learn" are not fully separated.

---

## 4. Experiment A — static ranking

One-shot check: fit TabPFN on 500 labeled *S. aureus* molecules, predict 500 held-out,
over 10 random splits. Score ranking (Spearman) and calibration (R²).

| Representation          | Spearman ρ | R²    |
|-------------------------|-----------:|------:|
| **Untrained Chemprop** (best) | **0.440** | 0.200 |
| Trained Chemprop        | 0.408      | 0.172 |
| Morgan fingerprints     | 0.407      | 0.183 |

*Averaged over 10 splits; higher is better.*

- Training **slightly hurt** ranking (0.440 → 0.408, **−0.032**).
- Untrained GNN is strongest and already edges out Morgan.

---

## 5. Experiment B — active learning (core test)

Retrospective benchmark: hide all labels; simulate a screening campaign.
- **Hit** = top **5%** of true pMIC (2,014 of 40,270).
- Start with 10 random labels; each round TabPFN scores the hidden pool and we reveal
  the top 10; 20 rounds → up to **210 labels**. Averaged over **5 seeds**.
- Strategies: **EI**, **PI** (uncertainty-guided), **greedy** (mean only), **random** (floor).
- Metric: **oracle-efficiency** = hits ÷ best-possible at each budget, averaged over the
  campaign. `1.0 = oracle`, `~0.05 = random`.

### Results — oracle-efficiency (hits@210 in the last column)

| Representation          |   EI  |   PI  | Greedy | Random | Hits@210 (EI) |
|-------------------------|------:|------:|-------:|-------:|--------------:|
| **Untrained Chemprop** (best) | **0.135** | 0.133 | 0.074 | 0.045 | 33.2 |
| Trained Chemprop        | 0.113 | 0.100 | 0.095  | 0.045  | 26.6 |
| Morgan fingerprints     | 0.084 | 0.079 | 0.096  | 0.045  | 24.8 |

### Hits-found curve (EI acquisition), by budget

| Labels | Untrained | Trained | Morgan | Random |
|-------:|----------:|--------:|-------:|-------:|
|  10 |  0.6 |  0.6 |  0.6 |  0.6 |
|  50 |  5.4 |  5.0 |  2.6 |  2.2 |
| 100 | 15.6 | 11.0 |  7.4 |  4.4 |
| 150 | 22.8 | 18.4 | 15.0 |  6.4 |
| 210 | **33.2** | 26.6 | 24.8 |  9.6 |

*(Full per-round curve for all arms × strategies in `experiments/al_curves.csv`.)*

### What the numbers say
- **Training hurt here too:** untrained EI **0.135** > trained EI **0.113** — a
  consistent ~6-hit gap across *both* EI and PI, so not seed noise. Same direction as
  the static test.
- **GNN beats Morgan in the loop:** untrained EI 0.135 vs Morgan's best 0.096.
- **Uncertainty matters — but only on GNN features:** on both Chemprop arms EI/PI ≫
  greedy (0.135 vs 0.074); on Morgan, greedy ≈ EI. The GNN gives TabPFN
  better-calibrated uncertainty that Expected-Improvement can exploit.

---

## 6. Interpretation

- Two independent experiments agree: **training the Chemprop encoder toward a frozen
  TabPFN does not improve, and slightly degrades, *S. aureus* performance.** A random-init
  GNN is best in both. RQ4 (as posed: ChEMBL-pretraining the encoder) is a well-supported
  **negative**.
- That negative sits inside a **positive** method: TabPFN + a GNN representation +
  uncertainty acquisition is a strong active learner (~3× enrichment, beats fingerprints).
  The working lever is representation + acquisition, not gradient-training the encoder.

---

## 7. Limitations & next steps

- **Only Morgan as a fixed baseline so far.** Physicochemical descriptors, Mordred 3D,
  and CheMeleon embeddings (RQ1–RQ3) not yet run.
- **Flat-loss cause not fully diagnosed** — weak signal vs. fundamental ceiling; worth
  an ablation (larger LR, unfreeze normalization).
- **Five seeds** — more would tighten confidence intervals.
- **Untested arm that could flip RQ4 positive:** fine-tune the encoder on *S. aureus*
  labels *as they arrive inside the loop* (target adaptation, not ChEMBL transfer). The
  harness is built; recommended next experiment.
