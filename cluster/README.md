# Running the GPU jobs on the Saarland HTCondor cluster

Every GPU run of the thesis (encoder training, the active learning benchmark, the calibration and
runtime studies) was an **HTCondor** job in the **docker universe**. Each job is a pair of files:

- a submit file `run_gpu_<name>.sub`, which requests the resources and names the executable;
- a shell script `run_gpu_<name>.sh`, which runs inside the container and starts one Python script
  from `src/scripts/`.

The jobs run inside a PyTorch CUDA docker image. Your home directory is mounted into the container
(`+WantGPUHomeMounted = true`), so the repository, the data, the Python environment and the TabPFN
checkpoint all live in `$HOME` and nothing is transferred per job. The environment is built once
(step 2) and every job uses it.

Placeholders below: `<USER>`, `<CLUSTER>` (the submit host), and the repository path in `$HOME`,
assumed to be `~/chemprop-tabpfn`.

---

## 1. Code and data onto the cluster

Git does not carry `data/`, `.env` or `uv.lock`, so the repository is copied with `rsync` instead
of cloned. Write your login into the one-line file `cluster/remote.local` (not in the repository):

```
<USER>@<CLUSTER>:~/chemprop-tabpfn
```

Then, from the repository root on the laptop:

```bash
cluster/sync.sh push --dry    # show what would be transferred
cluster/sync.sh push          # code, data/ and uv.lock -> cluster
```

`sync.sh` leaves out `.env`, `.venv/`, `experiments/`, `runlogs/` and the local notes. Only changed
bytes move, so later pushes are fast. Add the Hugging Face token on the cluster by hand:

```bash
ssh <USER>@<CLUSTER>
printf 'HF_TOKEN=hf_xxx\n' > ~/chemprop-tabpfn/.env
```

## 2. One-time environment build

On the submit node, which has internet access:

```bash
cd ~/chemprop-tabpfn
bash cluster/setup_env.sh
```

This installs `uv`, builds `.venv` on Python 3.13 with the CUDA 12.4 build of torch, and downloads
the TabPFN checkpoint into `.hf_cache/` inside the repository. The batch jobs run offline
(`HF_HUB_OFFLINE=1`) and find both there.

## 3. Submit a job

```bash
cd ~/chemprop-tabpfn
mkdir -p runlogs
condor_submit -dry-run /dev/null cluster/run_gpu_benchmark_final.sub   # validate only
condor_submit cluster/run_gpu_benchmark_final.sub
```

Read the comment at the top of a submit file before submitting it: several list what must be on
the cluster first (a trained encoder in `experiments/`, an output file that must not exist yet).

## 4. Monitor and collect

```bash
condor_q                          # job status
tail -f runlogs/<job>.*.out       # live output
```

Results are written to `experiments/` on the cluster. Back on the laptop:

```bash
cluster/sync.sh pull              # experiments/ and runlogs/ -> laptop
```

---

## Files in this folder

| File | What it is |
|---|---|
| `setup_env.sh` | builds the uv environment and caches the TabPFN checkpoint (step 2) |
| `sync.sh` | `push`, `pull` and `watch` between laptop and cluster |
| `run_gpu_<name>.sub` / `.sh` | one job each, see below |

### Encoder training (`src/scripts/run_experiment.py <preset>`)

The preset is the first argument of the Python script; each `.sh` passes its own. The presets and
their settings are defined in `run_experiment.py`.

| Job | Preset | Training data |
|---|---|---|
| `run_gpu` | `full` | all of ChEMBL |
| `run_gpu_full_sp` | `full_sp` | single-protein ChEMBL |
| `run_gpu_full_sp_10ep`, `run_gpu_full_sp_10ep_lowlr` | `full_sp_10ep`, `full_sp_10ep_lowlr` | single-protein ChEMBL, 10 epochs, two learning rates |
| `run_gpu_full_sp_50ep`, `run_gpu_full_sp_50ep_seed1` | `full_sp_50ep`, `full_sp_50ep_seed1` | single-protein ChEMBL, up to 50 epochs with early stopping, two seeds |
| `run_gpu_d2r`, `run_gpu_usp7`, `run_gpu_saureus` | `d2r`, `usp7`, `saureus` | the target's own domain-matched set |
| `run_gpu_d2r_long`, `run_gpu_usp7_long`, `run_gpu_saureus_long` | `d2r_long`, `usp7_long`, `saureus_long` | the same, 100k steps |
| `run_gpu_reevaluate_epochs` | (runs `reevaluate_epochs.py`) | recomputes the validation of saved epochs of the 50-epoch run |

### Benchmark and studies

| Job | Python script | What it produces |
|---|---|---|
| `run_gpu_benchmark_final.sub` | `benchmark_extended.py` | the final benchmark: 64 jobs, one per (target, representation), into `experiments/benchmark_final/` |
| `run_gpu_benchmark_final_seed1_control.sub` | `benchmark_extended.py` | the untrained seed-1 control and a reproducibility check, 5 jobs |
| `run_gpu_calibration.sub` | `calibration_study.py` | `experiments/calibration_study.csv` |
| `run_gpu_runtime_a100.sub` | `runtime_study.py` | `experiments/runtime_study_a100.csv` |
| `run_gpu_capacity_a100.sub` | `runtime_capacity.py` | `experiments/runtime_capacity_a100.csv` |

Both benchmark submit files use `run_gpu_benchmark_extended.sh`; the runtime and capacity jobs use
`run_gpu_runtime.sh` and `run_gpu_capacity.sh`. A benchmark job that is requeued resumes its output
file, so never submit a benchmark file twice while its jobs are still queued or running.

Job files of earlier runs are in `archive/cluster/`.

## Things you may need to edit

- **Repository path**: every `.sh` assumes `~/chemprop-tabpfn`. Change `REPO=` if yours differs.
- **Domain**: the submit files require `UidDomain == "cs.uni-saarland.de"`.
- **Docker image**: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`. Our own Python and torch are
  installed with `uv` on top, so the image mainly provides CUDA.

## Troubleshooting

- **`CUDA available: False`** in the log: the torch wheel does not match the GPU driver of the
  node. `pyproject.toml` takes torch from the `cu124` index; change it there if the cluster's
  driver differs, then run `setup_env.sh` again.
- **`HF_TOKEN not set`**: create `.env` (step 1) before running `setup_env.sh`.
- **Job stays IDLE in `condor_q`**: check `condor_q -better-analyze <JobId>`. A common cause is
  a `request_memory` that is too large: the A100 hosts are partitionable, and a 16 GB request
  once left jobs idle beside free GPUs. Most jobs here ask for 7 GB.
- **Job held, "gone over memory limit"**: the opposite case, raise `request_memory`. That setting
  is **host RAM only**.
- **`torch.OutOfMemoryError` / `CUBLAS_STATUS_ALLOC_FAILED`**: this is *GPU* memory, which
  `request_memory` does not affect. The differentiable TabPFN backward needs about 40 GB at
  `hidden=300, n_estimators=8`, so the submit files constrain the GPU:

  ```
  require_gpus = (GlobalMemoryMb >= 40000)
  ```

  or, in the older form, `requirements = ... && (GPUs_GlobalMemoryMb >= 40000)`. To see what the
  pool has:

  ```
  condor_status -constraint 'GPUs > 0' -af Machine GPUs_GlobalMemoryMb | sort -u -k2 -n
  ```

- **Short jobs and the shared GPU machines**: some machines only accept jobs that set
  `+MaxJobRetirementTime`. The calibration, runtime and capacity jobs set it; the day-long
  benchmark and training jobs do not.
