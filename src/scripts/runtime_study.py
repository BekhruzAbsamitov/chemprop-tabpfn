"""scripts/runtime_study.py — wall-clock cost of one acquisition round per surrogate.

One active learning round = fit (or condition) on the labelled set +
predict mean and uncertainty for the candidates; this script times exactly that call,
predict_dist, which is what the benchmark loop runs every round.

Settings swept: labelled-set size (the context) and the number of candidates scored per
round (the benchmark's 1,000-candidate subsample, and the whole S. aureus pool).
TabPFN runs on the GPU; the random forest and the Gaussian processes on the CPU, as in
the benchmark. Each setting is repeated REPEATS times after one warm-up call, and the
median is reported, so one-off costs (loading weights, CUDA initialisation) are excluded.

Output: OUTPUT_CSV (refuses to overwrite). The thesis uses experiments/runtime_study_a100.csv, the sweep on an
A100-PCIe-40GB (EXPECT_GPU), the GPU of most benchmark campaigns, with the host, GPU and CPU of each row in
the last four columns. The script refuses to start on any other GPU.
Cluster: condor_submit cluster/run_gpu_runtime_a100.sub   (about half an hour to an hour)
"""

from __future__ import annotations

import csv
import os
import platform
import random
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
for p in (SRC_DIR, SRC_DIR / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_al import TARGETS, featurize_pool, load_pool  # noqa: E402
from models.baselines import (  # noqa: E402
    GaussianProcessHead, GaussianProcessTanimotoHead, InferenceTabPFN, RandomForestHead)
from models.tabpfn_regressor import _load_hf_token  # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS — edit these, then press Run.                                       #
# --------------------------------------------------------------------------- #
DATASET = "saureus"  # the largest pool (31,085 molecules)
REPRESENTATIONS = {"chemeleon": ("chemeleon", None), "morgan": ("morgan", None),
                   "chemprop_untrained": ("untrained", None)}  # random-init D-MPNN, 300-d
MODELS = ["tabpfn", "rf", "gp", "gp_tanimoto"]
CONTEXT_SIZES = [10, 100, 300, 1100]  # 1,100 = the largest budget in the benchmark
CANDIDATE_SIZES = [1000, 0]  # 0 = every molecule not in the context
REPEATS = 3
N_ESTIMATORS = 4
OUTPUT_CSV = "experiments/runtime_study_a100.csv"  # the first run, on an H200, is experiments/runtime_study.csv
EXPECT_GPU = "A100-PCIE-40GB"  # refuse to start on any other GPU, before anything is written ("" = accept any)
COLUMNS = ["dataset", "representation", "dim", "model", "device", "context_size",
           "candidates", "median_seconds", "min_seconds", "max_seconds", "status",
           "host", "gpu", "cpu", "cpus_allowed"]


def hardware() -> dict[str, str]:
    """Where this sweep runs: the host, its GPU and CPU model, and how many CPU cores the job may use."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass  # not Linux: keep the platform name
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS has no affinity call
        allowed = os.cpu_count() or 0
    return dict(host=socket.gethostname(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                cpu=cpu, cpus_allowed=str(allowed))


def make_model(name: str, device: torch.device):
    return {"tabpfn": lambda: InferenceTabPFN(n_estimators=N_ESTIMATORS, device=str(device)),
            "rf": RandomForestHead, "gp": GaussianProcessHead,
            "gp_tanimoto": GaussianProcessTanimotoHead}[name]()


def main() -> None:
    out = REPO_ROOT / OUTPUT_CSV
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it away first")
    hw = hardware()
    print("hardware:", hw)
    if EXPECT_GPU and EXPECT_GPU not in hw["gpu"]:
        raise SystemExit(f"this job runs on {hw['gpu']!r}, not on {EXPECT_GPU!r}; nothing was written")
    out.parent.mkdir(parents=True, exist_ok=True)
    _load_hf_token()
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles, y, _ = load_pool(TARGETS[DATASET])
    y = y.flatten().float()
    with out.open("w", newline="") as f:
        csv.writer(f).writerow(COLUMNS)

    for rep, (kind, ckpt) in REPRESENTATIONS.items():
        X = featurize_pool(kind, ckpt, smiles, gpu).float().cpu()
        for m in MODELS:
            if m == "gp_tanimoto" and rep != "morgan":
                continue
            device = gpu if m == "tabpfn" else torch.device("cpu")
            model = make_model(m, device)
            for n_ctx in CONTEXT_SIZES:
                for n_cand in CANDIDATE_SIZES:
                    rng = random.Random(n_ctx)
                    perm = list(range(len(smiles)))
                    rng.shuffle(perm)
                    ctx = perm[:n_ctx]
                    cand = perm[n_ctx:n_ctx + n_cand] if n_cand else perm[n_ctx:]
                    times, status = [], "ok"
                    try:
                        model.predict_dist(X[ctx], y[ctx], X[cand])  # warm-up
                        for _ in range(REPEATS):
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            model.predict_dist(X[ctx], y[ctx], X[cand])
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            times.append(time.perf_counter() - t0)
                    except (RuntimeError, MemoryError) as exc:  # e.g. CUDA out of memory
                        status = f"failed: {type(exc).__name__}"
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    med = statistics.median(times) if times else float("nan")
                    with out.open("a", newline="") as f:
                        csv.writer(f).writerow([
                            DATASET, rep, X.shape[1], m, device.type, n_ctx, len(cand),
                            f"{med:.3f}", f"{min(times):.3f}" if times else "",
                            f"{max(times):.3f}" if times else "", status,
                            hw["host"], hw["gpu"], hw["cpu"], hw["cpus_allowed"]])
                    print(f"{rep:9s} {m:11s} ({device.type}) context {n_ctx:>4} candidates "
                          f"{len(cand):>6}: {med:8.2f} s  {status}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
