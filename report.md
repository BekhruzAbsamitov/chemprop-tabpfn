# Progress Report — Chemprop → TabPFN Trainability

**Author:** Bekhruz Absamitov
**Date:** 2026-07-20
**For:** Supervision meeting (follow-up to 2026-07-16)

---

## 1. Summary

The headline question from last meeting was: *training loss on the full run is
not decreasing — is this a bug or a real learning problem?*

I ran the **overfit sanity check** you suggested (memorize a small batch of
5 tiny assays). **It passes decisively.** The pipeline is wired correctly:
gradients reach the Chemprop encoder and the NLL loss can be driven to ~0 when
the data is memorizable. Therefore the flat loss on the full run is **not a
wiring bug** — it is a data / optimization property of the training task.

---

## 2. Overfit sanity check

**Setup.** Freeze 5 tiny ChEMBL-dev assays (10–18 molecules each in context and
query), reuse the *exact same* episodes every step (so overfitting is possible),
start from a **randomly initialized** encoder (a warm start would make a low loss
ambiguous). Everything else — encoder, frozen TabPFN, NLL loss, AdamW, gradient
clipping — is identical to the real training loop.

Config: CPU, `hidden_size=128`, `n_estimators=1`, `loss=nll`, `lr=1e-3`,
300 steps. Script: `src/scripts/sanity_overfit.py` (one self-contained file).

**Result.**

| Quantity | Value |
|---|---|
| Mean NLL, step 1 | **1.85** |
| Mean NLL, step 300 | **0.06** |
| Relative drop | **−96.6 %** |
| Memorization Spearman (fixed queries, mean) | **0.79** (0.55–0.95 per episode) |
| Gradient flow into encoder | **6 / 10 params non-zero** (see note) |

The loss falls smoothly and monotonically toward zero:

```
step   1/300 | loss 1.8525
step  40/300 | loss 1.0729
step 100/300 | loss 0.9364
step 160/300 | loss 0.3906
step 200/300 | loss 0.1659
step 260/300 | loss 0.1528
step 300/300 | loss 0.0637
```

**Verdict: PASS.** A correctly-wired model with this capacity *should* be able
to memorize a handful of tiny assays, and it does.

---

## 3. What this tells us

- **Not a bug.** Gradients flow to Chemprop, loss/target shapes are correct, the
  differentiable path through frozen TabPFN works. The stuck full-run loss is
  *not* a mechanical failure (no detached targets, dead gradients, or broken loss).

- **Resolves an earlier ambiguity.** After the full-scale GPU run I could not
  distinguish "weak/absent gradient signal (bug)" from "the approach has little
  headroom (fundamental)." The sanity check distinguishes them: the pipeline
  *can* drive loss to zero when signal exists, so the flat full-run loss is the
  **fundamental / weak-signal** side — predicting a held-out assay from a *fresh
  random assay every step*, under TabPFN's per-episode normalization, leaves
  little for the encoder to learn.

---

## 4. Two observations from the run (neither is a bug)

1. **6 of 10 encoder parameters receive gradients.** The 4 that don't are an
   unused `RegressionFFN` head that only exists to satisfy Chemprop's `MPNN`
   constructor; it is never on the embedding (`.fingerprint()`) path. Harmless,
   though the optimizer needlessly carries 4 dead parameters — an easy cleanup.

2. **Mixed label scales confirmed** (the concern you raised). Among the 5 assays,
   one is on a raw scale (`[11, 85]`) while the others are log-scale (`[-2, 8]`).
   TabPFN normalizes **per episode**, so within a single assay this is handled
   (`target_z` std ≈ 1.1, 0 / 69 query labels hit the ±5 clip) and ranking is
   still learned (Spearman 0.93 on the raw-scale assay). This motivates the open
   question below.

---

## 4b. Is TabPFN's built-in label normalization enabled in training?

Your explicit question. Traced through the TabPFN v3 source (package v8.1.0).
There are **two** label-normalization layers; our differentiable pipeline uses
only the first.

| Layer | In our training? | Notes |
|---|---|---|
| **Global z-standardization** `y → (y−mean)/std` | **ENABLED** | Recomputed per episode inside `fit_with_differentiable_input` (`regressor.py:1015`). Our `tabpfn_regressor.py` already accounts for it (`y_train_mean_`/`y_train_std_`, `target_z`). |
| **Target reshaping (`safepower`)** | **DISABLED** | TabPFN's default is `REGRESSION_Y_PREPROCESS_TRANSFORMS = (None, "safepower")`, but the differentiable path hardcodes `target_transforms=[None]` (`regressor.py:771`). |

**Why `safepower` is off:** it is a non-differentiable (numpy/sklearn) power
transform; applying it would break the gradient graph back to Chemprop. So
differentiability is bought by dropping it. This is by design, not a bug.

**Implications:**
- No train/eval mismatch *inside* our pipeline — `predict()`/`predict_dist()`
  also route through the differentiable path, so both skip `safepower`.
- Our pipeline runs a "lighter" TabPFN than a stock `fit()` inference. Negligible
  at `n_estimators=1`; more relevant at `n_estimators=8` (the GPU run), where
  stock inference would blend in `safepower` ensemble members and ours does not.
- `safepower` de-skews the target distribution — exactly what would help the
  raw-scale assays (e.g. the raw-MIC `[11,85]` assay in the sanity check). Our
  per-episode z-norm fixes *scale* but not *skew/shape*.

**Answer in one line:** basic label standardization **is** enabled; TabPFN's
richer built-in target transform is **not**, and cannot be, in the differentiable
setting.

---

## 5. Experiment A — expanded static ranking

Repeat of the static one-shot ranking, broadened per the notes: **3 targets**
(S. aureus, D2R, USP7) × **3 representations**, **3 random context/query samples**
each (200 context / 200 query), all arms scored on the *same* splits.
Script: `src/scripts/run_experiment_a.py`. Config: CPU, NLL head, static (no
fine-tuning at eval). Run at **`n_estimators=8` for every arm** — the fair setting
(see the caveat at the end for why this matters).

**Spearman, mean ± std over 3 samples (higher is better):**

| Representation | S. aureus | D2R | USP7 | **Overall** |
|---|---|---|---|---|
| Chemprop-random | 0.388 ± 0.066 | 0.454 ± 0.053 | 0.818 ± 0.029 | **0.553** |
| Chemprop-trained | 0.376 ± 0.061 | 0.441 ± 0.068 | 0.832 ± 0.024 | **0.550** |
| Chemeleon | 0.394 ± 0.072 | 0.466 ± 0.040 | 0.843 ± 0.018 | **0.568** |

(R² overall: 0.361 / 0.359 / 0.350 for random / trained / Chemeleon.)

**Findings:**
1. **All three representations are statistically tied.** Overall Spearman spans
   just 0.550–0.568, far inside the ±0.04–0.09 per-cell std bands. For TabPFN at
   this scale, whether the encoder is a random GNN, a ChEMBL-trained GNN, or a
   PubChem-pretrained GNN barely changes the ranking.
2. **Training neither clearly helps nor hurts** once ensembling is fair
   (trained 0.550 ≈ random 0.553; trained even *beats* random on USP7). This
   corrects the earlier `n_estimators=1` result — see below.
3. **Chemeleon is marginally highest** (+0.015 overall) but within noise, and its
   R² is actually the lowest.

**⚠️ Why `n_estimators=8`, and a corrected result.** An earlier run left
`n_estimators=1`, but TabPFN auto-scales it to 5 for Chemeleon (2048-dim features;
one estimator covers only 500) while the low-dim Chemprop arms stayed at 1. That
run showed trained < random on all 3 targets (−0.04 overall) and a bigger Chemeleon
lead (+0.03). **Both effects largely vanish at a fair `n_estimators=8`**: the
apparent "training hurts" was mostly low-estimator noise, not a real effect. The
fair run above is the one to trust.

| Comparison (overall) | n_est=1/5 | n_est=8 (fair) |
|---|---|---|
| trained − random | −0.042 ("hurts") | −0.003 (tied) |
| Chemeleon − random | +0.028 | +0.015 |

---

## 6. Status of the 2026-07-16 action items

| Item from notes | Status |
|---|---|
| Overfit a small batch (5 assays, 20–50 cpds) | **Done — PASS** |
| Does Chemprop receive gradients? | **Yes** (6/10 params; rest are dead unused head) |
| Check loss/target shapes & warnings | **Checked — clean** |
| Push code to repo | Pending |
| Is TabPFN v3 built-in label normalization enabled in training? | **Answered — see §4b** |
| Expand Experiment A: + Chemeleon, + d2r/usp7, 3 samples, mean ± std | **Done — see §5** |
| Merge assays into one batch / batch dimension | Deferred (efficiency, not correctness) |

---

## 7. Proposed next steps

1. The static metric is now well-characterized: **representation barely matters**
   (all arms tied at a fair `n_estimators=8`), and the sanity check shows the
   pipeline can learn. Together these say the static one-shot task simply has
   little headroom for representation — not a bug, but not where the thesis wins.
2. **Move to the active-learning loop** — the actual untested hypothesis: the
   encoder is fine-tuned on accumulated target labels each round, and the metric
   is actives-found vs. #labels (Hit/CEF). The D2R/USP7 datasets carry `top_2p`/
   `top_5p` hit flags, ready for this.
3. If revisiting training signal: the fresh-resample-every-step design, context
   sizing, and safepower-style label de-skewing are the levers — but given #1,
   this is lower priority than the AL loop.

---

*Reproduce:* run `src/scripts/sanity_overfit.py` (settings are constants at the
top of the file).
