"""scripts/runtime_capacity.py — how many candidates fit in ONE TabPFN predict call on a 40 GB A100? Press Run.



The benchmark scores 1,000 candidates per round because the protocol subsamples them, not because of the GPU, and
runtime_study.py already scored the whole S. aureus pool (about 31,000 candidates) in one call. It recorded no GPU
memory, however. For each representation and
labelled-set size it fits TabPFN once and then makes ONE predict call for a growing number of candidates, and
records for every size

  status                 ok, or the error that stopped it (e.g. CUDA out of memory)
  seconds_predict        wall-clock time of the one predict call (median of the repeats)
  gpu_peak_allocated_gb  peak GPU memory held by PyTorch tensors during that call (weights included)
  gpu_peak_reserved_gb   peak GPU memory reserved by PyTorch's allocator
  gpu_peak_share         gpu_peak_allocated_gb as a fraction of the GPU's memory
  host_peak_rss_gb       peak host RAM of the process so far (this cannot be reset, so it never falls)

Candidate sets larger than the pool are made by REPEATING the pool's untested molecules. The cost of a predict
call depends on the number and width of the rows, not on their values, so repeated rows are a fair stand-in for a
larger library. The call is `reg.predict(X, output_type="mean")`; one acquisition round of the benchmark makes two
such calls (mean, then quantiles), so a round costs about twice the time and no more memory.

The sizes run in ascending order across all settings, and a row with status `started` is written BEFORE every
attempt. If the job is killed from outside (the host-RAM limit of the job, which Python cannot catch), the file
therefore ends with a `started` row that names the setting it died in, and every smaller size is already saved.
After a failure, the larger sizes of that representation and labelled-set size are skipped. A setting whose input
matrix alone exceeds MAX_INPUT_GB of host RAM is skipped and marked, so that host RAM, which the job requests
conservatively (cluster/run_gpu_capacity_a100.sub), does not decide a question about the GPU.

Output: OUTPUT_CSV (refuses to overwrite) and a short summary at the end of the log.
Cluster: condor_submit cluster/run_gpu_capacity_a100.sub   (about an hour)
"""

from __future__ import annotations

import csv
import gc
import platform
import random
import resource
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SRC_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_al import TARGETS, featurize_pool, load_pool  # noqa: E402
from models.baselines import InferenceTabPFN  # noqa: E402
from models.tabpfn_regressor import _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
DATASET = "saureus"                       # the largest pool (31,085 molecules), as in runtime_study.py
REPRESENTATIONS = {"chemeleon": ("chemeleon", None),            # 2,048-d: TabPFN's best and widest representation
                   "chemprop_untrained": ("untrained", None)}   # 300-d: the width of every trained encoder
CONTEXT_SIZES = [100, 1100]               # labelled molecules: a small set and the largest budget of the benchmark
CANDIDATE_SIZES = [1_000, 10_000, 30_000, 100_000, 200_000, 400_000, 700_000, 1_000_000]   # 1,000 = the benchmark
REPEATS = 2                               # timed calls per setting ...
SINGLE_REPEAT_FROM = 200_000              # ... but one call from this many candidates on (they take minutes)
N_ESTIMATORS = 4                          # as in the benchmark
MAX_INPUT_GB = 3.0                        # skip a setting whose float32 input matrix alone is larger (host RAM)
OUTPUT_CSV = "experiments/runtime_capacity_a100.csv"
EXPECT_GPU = "A100-PCIE-40GB"             # refuse to start on any other GPU, before anything is written ("" = accept any)
COLUMNS = ["dataset", "representation", "dim", "context_size", "candidates", "pool_multiple", "status",
           "seconds_fit", "seconds_predict", "ms_per_candidate", "repeats",
           "gpu_peak_allocated_gb", "gpu_peak_reserved_gb", "gpu_total_gb", "gpu_peak_share", "host_peak_rss_gb",
           "n_estimators", "host", "gpu", "tabpfn_version", "torch_version"]
GB = 1024 ** 3


def host_peak_rss_gb() -> float:
    """Peak resident memory of this process so far. Linux reports kilobytes, macOS bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2 if platform.system() == "Linux" else GB)


def append(out: Path, row: dict) -> None:
    with out.open("a", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in COLUMNS])


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    import tabpfn

    out = REPO_ROOT / OUTPUT_CSV
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else "none"
    if EXPECT_GPU and EXPECT_GPU not in gpu_name:
        raise SystemExit(f"this job runs on {gpu_name!r}, not on {EXPECT_GPU!r}; nothing was written")
    gpu_total = torch.cuda.get_device_properties(0).total_memory / GB if cuda else float("nan")
    device = torch.device("cuda" if cuda else "cpu")
    fixed = dict(dataset=DATASET, n_estimators=N_ESTIMATORS, host=socket.gethostname(), gpu=gpu_name,
                 gpu_total_gb=f"{gpu_total:.1f}", tabpfn_version=tabpfn.__version__, torch_version=torch.__version__)
    print("setup:", fixed)

    out.parent.mkdir(parents=True, exist_ok=True)
    _load_hf_token()
    smiles, y, _ = load_pool(TARGETS[DATASET])
    y = y.flatten().float().numpy()
    with out.open("w", newline="") as f:
        csv.writer(f).writerow(COLUMNS)

    features = {}
    for rep, (kind, ckpt) in REPRESENTATIONS.items():
        features[rep] = featurize_pool(kind, ckpt, smiles, device).float().cpu().numpy()
        print(f"{rep}: {features[rep].shape[0]:,} molecules x {features[rep].shape[1]:,} features")
    free_gpu()

    model = InferenceTabPFN(n_estimators=N_ESTIMATORS, device=str(device))
    failed: set[tuple[str, int]] = set()          # (representation, context size) that failed at a smaller size
    results = []
    for n_cand in CANDIDATE_SIZES:                # ascending: every smaller size is saved before a larger one runs
        for rep, X in features.items():
            for n_ctx in CONTEXT_SIZES:
                perm = list(range(len(smiles)))
                random.Random(n_ctx).shuffle(perm)
                ctx, rest = perm[:n_ctx], np.asarray(perm[n_ctx:])
                row = dict(fixed, representation=rep, dim=X.shape[1], context_size=n_ctx, candidates=n_cand,
                           pool_multiple=f"{n_cand / len(rest):.2f}")
                label = f"{rep:19s} context {n_ctx:>5,} candidates {n_cand:>9,}"
                if (rep, n_ctx) in failed:
                    append(out, dict(row, status="skipped: a smaller size failed"))
                    print(f"{label}: skipped, a smaller size failed")
                    continue
                input_gb = n_cand * X.shape[1] * 4 / GB
                if input_gb > MAX_INPUT_GB:
                    append(out, dict(row, status=f"skipped: input matrix {input_gb:.1f} GB exceeds MAX_INPUT_GB"))
                    print(f"{label}: skipped, the input matrix alone is {input_gb:.1f} GB of host RAM")
                    continue
                append(out, dict(row, status="started"))             # survives a kill from outside
                print(f"{label}: started", flush=True)
                repeats = 1 if n_cand >= SINGLE_REPEAT_FROM else REPEATS
                times, status, fit_seconds, peak_alloc, peak_reserved = [], "ok", float("nan"), float("nan"), float("nan")
                try:
                    xq = X[np.resize(rest, n_cand)]                   # the untested molecules, repeated as needed
                    t0 = time.perf_counter()
                    model.reg.fit(X[ctx], y[ctx])
                    fit_seconds = time.perf_counter() - t0
                    if n_cand == CANDIDATE_SIZES[0]:
                        model.reg.predict(xq, output_type="mean")    # warm-up: weights, CUDA initialisation
                    if cuda:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    for _ in range(repeats):
                        t0 = time.perf_counter()
                        model.reg.predict(xq, output_type="mean")    # the ONE predict call
                        if cuda:
                            torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                    if cuda:
                        peak_alloc = torch.cuda.max_memory_allocated() / GB
                        peak_reserved = torch.cuda.max_memory_reserved() / GB
                except Exception as exc:  # noqa: BLE001 — TabPFN raises its own TabPFNCUDAOutOfMemoryError, not a RuntimeError
                    message = str(exc).splitlines()[0][:120] if str(exc) else ""
                    status = f"failed: {type(exc).__name__}: {message}"
                    failed.add((rep, n_ctx))
                    if cuda:
                        peak_alloc = torch.cuda.max_memory_allocated() / GB
                        peak_reserved = torch.cuda.max_memory_reserved() / GB
                xq = None
                free_gpu()
                med = statistics.median(times) if times else float("nan")
                done = dict(row, status=status, seconds_fit=f"{fit_seconds:.3f}",
                            seconds_predict=f"{med:.3f}" if times else "",
                            ms_per_candidate=f"{1000 * med / n_cand:.4f}" if times else "", repeats=len(times),
                            gpu_peak_allocated_gb=f"{peak_alloc:.2f}", gpu_peak_reserved_gb=f"{peak_reserved:.2f}",
                            gpu_peak_share=f"{peak_alloc / gpu_total:.3f}", host_peak_rss_gb=f"{host_peak_rss_gb():.2f}")
                append(out, done)
                results.append(done)
                print(f"{label}: {status}; predict {med:8.2f} s ({1000 * med / n_cand:.3f} ms per candidate); GPU peak "
                      f"{peak_alloc:5.2f} GB allocated, {peak_reserved:5.2f} GB reserved of {gpu_total:.1f} GB; host peak "
                      f"{host_peak_rss_gb():.2f} GB", flush=True)

    print("\nSUMMARY — one predict call per setting")
    for rep in features:
        for n_ctx in CONTEXT_SIZES:
            mine = [r for r in results if r["representation"] == rep and r["context_size"] == n_ctx]
            ok = [r for r in mine if r["status"] == "ok"]
            bad = [r for r in mine if r["status"].startswith("failed")]
            if not ok:
                print(f"  {rep}, context {n_ctx:,}: nothing completed")
                continue
            first, last = ok[0], ok[-1]
            line = (f"  {rep}, context {n_ctx:,}: {first['candidates']:,} candidates use {first['gpu_peak_allocated_gb']} GB "
                    f"({100 * float(first['gpu_peak_share']):.0f} % of the GPU); the largest call that completed has "
                    f"{last['candidates']:,} candidates ({last['pool_multiple']} x the pool), {last['seconds_predict']} s, "
                    f"{last['gpu_peak_allocated_gb']} GB ({100 * float(last['gpu_peak_share']):.0f} %)")
            if bad:
                line += f"; {bad[0]['candidates']:,} candidates {bad[0]['status']}"
            print(line)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
