"""scripts/sanity_overfit.py — the "can we overfit a tiny batch?" sanity check. Press Run.

WHY THIS FILE EXISTS
    Training loss on the full run is not decreasing. Before hunting for subtle
    causes, we answer one blunt question:

        Given only a HANDFUL of tiny assays, and allowed to look at them over
        and over, can the encoder drive the training loss clearly DOWN?

    A correctly-wired model with this much capacity should be able to *memorize*
    5 small assays almost perfectly. Generalization is NOT the point here — we
    WANT it to cheat. This is a diagnostic, not an experiment.

HOW TO READ THE RESULT
    * loss falls a lot and plateaus low  -> the pipeline is wired correctly;
      the full-run problem is data / optimization, not a bug.
    * loss barely moves / is flat        -> strong evidence of a BUG (gradients
      not reaching Chemprop, wrong shapes, detached targets, dead LR, ...).
    * loss is NaN / explodes             -> a numerical bug (normalization / LR).

    We also print three bug-catching diagnostics the supervision notes asked for:
      1. gradient flow      — do the encoder's weights actually receive gradients?
      2. shapes & targets   — context/query shapes, label ranges, target_z stats.
      3. memorization score  — after training, how well does it fit the (fixed)
                              query points it was allowed to see repeatedly?

WHAT IS DELIBERATELY DIFFERENT FROM run_experiment.py
    The real loop draws a FRESH random assay + FRESH split every step. Here we
    freeze a few tiny episodes ONCE and reuse the exact same data every step —
    that is what makes overfitting possible and the diagnosis unambiguous.

    We start from a RANDOMLY INITIALIZED encoder (NOT the saved
    trained_encoder_full.pt). A warm start would make a low loss ambiguous —
    we could not tell learning from a lucky initialization.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch
from scipy.stats import spearmanr

# Make `src/` importable so this runs from the IDE Run button with no PYTHONPATH.
# This file is at src/scripts/, so `src/` is its parent directory.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_utils import data  # noqa: E402  (our src/data_utils/data.py)
from models.encoder import MoleculeEncoder  # noqa: E402
from models.tabpfn_regressor import TabPFNHead, TARGET_Z_CLIP, _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
N_ASSAYS = 5           # how many tiny assays to try to overfit
MIN_ASSAY_SIZE = 20    # only use assays with at least this many molecules ...
MAX_ASSAY_SIZE = 50    # ... and at most this many (the notes: "like 20~50 compounds")

N_STEPS = 300          # training steps over the SAME frozen episodes
LEARNING_RATE = 1e-3   # bigger than the real run's 1e-4: we WANT to overfit fast
HIDDEN_SIZE = 128      # encoder width (match the real run for a fair diagnosis)
N_ESTIMATORS = 1       # TabPFN passes to average (1 = fast; fine for a diagnostic)
LOSS_FN = "nll"        # match the real run: "nll" | "mse" | "huber"

CONTEXT_FRAC = 0.5     # fraction of each assay used as context (rest is query)
CLIP_GRAD_NORM = 1.0   # same gradient clipping as the real training loop
SEED = 0
PRINT_EVERY = 20       # print the loss curve every N steps

# Pass verdict thresholds (only a hint — read the curve yourself too).
PASS_REL_DROP = 0.30   # final mean loss must be >= this fraction below the start


# --------------------------------------------------------------------------- #
# Pick the tiny fixed episodes                                                 #
# --------------------------------------------------------------------------- #
def pick_fixed_episodes(rng: random.Random) -> list[data.Episode]:
    """Choose N_ASSAYS small assays and freeze ONE episode from each.

    The episodes are built once and reused every training step — identical data
    every time, which is what lets the model memorize (overfit) them.
    """
    assays = data.load_chembl_assays("train", use_full=False)
    small = [a for a in assays if MIN_ASSAY_SIZE <= len(a.smiles) <= MAX_ASSAY_SIZE]
    small.sort(key=lambda a: a.assay_id)  # deterministic order

    episodes: list[data.Episode] = []
    for assay in small:
        # A fixed per-assay rng => the same context/query split every run.
        ep = data.make_episode(
            assay,
            context_frac=CONTEXT_FRAC,
            rng=random.Random(hash(assay.assay_id) & 0xFFFF),
        )
        if ep is None:  # too small, or context labels nearly identical
            continue
        episodes.append(ep)
        if len(episodes) == N_ASSAYS:
            break

    if not episodes:
        raise RuntimeError(
            f"No assays with {MIN_ASSAY_SIZE}-{MAX_ASSAY_SIZE} usable molecules "
            "found in the ChEMBL dev 'train' split — widen the size window."
        )
    if len(episodes) < N_ASSAYS:
        print(f"  WARNING: only found {len(episodes)} usable assays (wanted {N_ASSAYS}).")
    return episodes


# --------------------------------------------------------------------------- #
# Diagnostics the supervision notes asked for                                  #
# --------------------------------------------------------------------------- #
def report_shapes_and_targets(head: TabPFNHead, episodes: list[data.Episode]) -> None:
    """Print context/query shapes and label ranges — catches shape/scale bugs."""
    print("\nshapes & targets (per frozen episode):")
    print(f"  {'#':>2} {'context':>8} {'query':>7} {'y_ctx range':>18} {'y_qry range':>18}")
    for i, ep in enumerate(episodes):
        cy, qy = ep.context_y, ep.query_y
        print(
            f"  {i:>2} {len(ep.context_smiles):>8} {len(ep.query_smiles):>7} "
            f"[{float(cy.min()):>7.2f},{float(cy.max()):>7.2f}] "
            f"[{float(qy.min()):>7.2f},{float(qy.max()):>7.2f}]"
        )
    # How the query labels look in TabPFN's rescaled ("z") space, incl. clipping.
    all_z, clipped = [], 0
    for ep in episodes:
        m, s = float(ep.context_y.mean()), float(ep.context_y.std(unbiased=False))
        s = s if s > 1e-6 else 1.0
        z = (ep.query_y.flatten() - m) / s
        clipped += int((z.abs() > TARGET_Z_CLIP).sum())
        all_z.append(z.clamp(-TARGET_Z_CLIP, TARGET_Z_CLIP))
    z = torch.cat(all_z)
    print(
        f"  target_z (clipped to +/-{TARGET_Z_CLIP}): "
        f"mean {float(z.mean()):+.2f}  std {float(z.std()):.2f}  "
        f"min {float(z.min()):+.2f}  max {float(z.max()):+.2f}  "
        f"({clipped} of {z.numel()} query labels hit the clip)"
    )


def report_gradient_flow(encoder: MoleculeEncoder) -> None:
    """After one backward pass, confirm the encoder's weights got gradients."""
    total = sum(1 for _ in encoder.parameters())
    got = sum(1 for p in encoder.parameters() if p.grad is not None)
    nonzero = sum(
        1 for p in encoder.parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0.0
    )
    grad_norm = sum(
        float(p.grad.norm()) ** 2 for p in encoder.parameters() if p.grad is not None
    ) ** 0.5
    print("\ngradient flow into the Chemprop encoder:")
    print(f"  params with a gradient : {got}/{total}")
    print(f"  params with a NON-zero gradient: {nonzero}/{total}")
    print(f"  total encoder grad norm: {grad_norm:.3e}")
    if nonzero == 0:
        print("  !! NO gradients reached the encoder — this is the bug. !!")


@torch.no_grad()
def memorization_score(head: TabPFNHead, encoder: MoleculeEncoder,
                       episodes: list[data.Episode]) -> None:
    """After training, how well does it fit the query points it saw every step?"""
    encoder.eval()
    print("\nmemorization of the (fixed) query points after training:")
    print(f"  {'#':>2} {'spearman':>9} {'mse':>10}")
    rhos = []
    for i, ep in enumerate(episodes):
        x_ctx = encoder(ep.context_smiles).to(head.device)
        x_qry = encoder(ep.query_smiles).to(head.device)
        preds = head.predict(x_ctx, ep.context_y, x_qry)
        p = preds.detach().cpu()
        t = ep.query_y.flatten().cpu()
        rho = float(spearmanr(p.numpy(), t.numpy()).statistic) if p.numel() > 1 else float("nan")
        mse = float(((p - t) ** 2).mean())
        rhos.append(rho)
        print(f"  {i:>2} {rho:>9.3f} {mse:>10.3f}")
    good = [r for r in rhos if r == r]  # drop NaNs
    if good:
        print(f"  mean spearman on memorized queries: {sum(good) / len(good):.3f}  "
              "(near 1.0 => it successfully memorized the ranking)")


# --------------------------------------------------------------------------- #
# The sanity check                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    _load_hf_token()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()

    print("picking tiny fixed episodes to overfit ...")
    episodes = pick_fixed_episodes(rng)
    sizes = ", ".join(f"{len(e.context_smiles)}+{len(e.query_smiles)}" for e in episodes)
    print(f"  {len(episodes)} episodes (context+query sizes): {sizes}")

    encoder = MoleculeEncoder(hidden_size=HIDDEN_SIZE)
    head = TabPFNHead(n_estimators=N_ESTIMATORS, loss_fn=LOSS_FN)
    encoder.to(head.device)
    print(f"  device: {head.device} | loss: {LOSS_FN} | hidden_size: {HIDDEN_SIZE} | "
          f"lr: {LEARNING_RATE} | steps: {N_STEPS}")

    report_shapes_and_targets(head, episodes)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)

    def mean_loss_over_episodes() -> torch.Tensor:
        """Mean NLL across all frozen episodes = one clean number per step."""
        losses = []
        for ep in episodes:
            emb_ctx = encoder(ep.context_smiles)
            emb_qry = encoder(ep.query_smiles)
            losses.append(head.loss(emb_ctx, ep.context_y, emb_qry, ep.query_y))
        return torch.stack(losses).mean()

    print(f"\noverfitting the same {len(episodes)} episodes for {N_STEPS} steps ...")
    start_loss = None
    final_loss = float("nan")
    for step in range(1, N_STEPS + 1):
        encoder.train()
        optimizer.zero_grad()
        loss = mean_loss_over_episodes()
        loss.backward()

        if step == 1:  # capture the untrained baseline + prove gradients flow
            start_loss = float(loss.item())
            report_gradient_flow(encoder)
            print("\n  loss curve:")

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP_GRAD_NORM)
        optimizer.step()

        final_loss = float(loss.item())
        if step == 1 or step % PRINT_EVERY == 0 or step == N_STEPS:
            print(f"    step {step:>4}/{N_STEPS} | mean loss {final_loss:8.4f}  "
                  f"({time.time() - t0:.0f}s)")

    memorization_score(head, encoder, episodes)

    # ---- verdict -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("OVERFIT SANITY CHECK")
    print("=" * 60)
    print(f"  start loss (step 1): {start_loss:8.4f}")
    print(f"  final loss (step {N_STEPS}): {final_loss:8.4f}")
    drop = (start_loss - final_loss) / abs(start_loss) if start_loss else float("nan")
    print(f"  relative drop:       {drop:+7.1%}")
    if final_loss != final_loss:  # NaN
        print("  VERDICT: FAIL — loss went NaN. Numerical bug (normalization / LR).")
    elif drop >= PASS_REL_DROP:
        print("  VERDICT: PASS — loss fell clearly. The pipeline can learn; the "
              "full-run problem is data / optimization, not a wiring bug.")
    else:
        print("  VERDICT: FAIL — loss did not drop enough on a tiny memorizable "
              "set. Strong evidence of a BUG (see the gradient-flow report above).")
    print(f"\ntotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
