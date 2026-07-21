# Running the full-scale RQ4 test on the Saarland HTCondor cluster

This runs `src/scripts/run_experiment_gpu.py` (full ChEMBL, wider encoder,
stronger TabPFN ensemble) on one GPU, using **HTCondor** with the **docker
universe** — matching the `example/*.sub` files.

How it works: the job runs inside a PyTorch CUDA docker image. Your home
directory is mounted into the container (`+WantGPUHomeMounted = true`), so we
build a `uv` environment **once** in your home (inside the same image), and the
batch job just uses it. No big-file transfers — the repo and data live in `$HOME`.

Placeholders: `<USER>`, `<CLUSTER>` (submit host), and the repo path in `$HOME`
(default assumed: `~/chemprop-tabpfn`).

---

## 1. Code + data onto the cluster

Git does **not** carry `data/`, `.env`, or `uv.lock` (all gitignored). The full
run needs the big ChEMBL file (566 MB).

```bash
# cluster (submit node):
ssh <USER>@<CLUSTER>
git clone <YOUR_REPO_URL> ~/chemprop-tabpfn
cd ~/chemprop-tabpfn

# laptop: copy the gitignored files into place
ssh <USER>@<CLUSTER> "mkdir -p ~/chemprop-tabpfn/data/curated"
scp data/curated/chembl36_curated.parquet \
    data/curated/s_aureus_curated_with_scores_filtered.csv \
    <USER>@<CLUSTER>:~/chemprop-tabpfn/data/curated/
scp uv.lock <USER>@<CLUSTER>:~/chemprop-tabpfn/
```

Then add your Hugging Face token (on the cluster):
```bash
printf 'HF_TOKEN=hf_xxx\n' > ~/chemprop-tabpfn/.env
```

## 2. One-time environment build (interactive container)

Build the env *inside* the same docker image the job uses, so the libraries match:
```bash
cd ~/chemprop-tabpfn
condor_submit -i cluster/interactive.sub     # drops you into a shell in the container
# --- now inside the container: ---
bash cluster/setup_env.sh                     # installs uv, builds .venv, caches TabPFN
exit                                          # leave the interactive session
```

## 3. Submit the batch job
```bash
cd ~/chemprop-tabpfn
mkdir -p runlogs
condor_submit cluster/run_gpu.sub
```

## 4. Monitor and collect
```bash
condor_q                                      # job status
tail -f runlogs/run_gpu.*.out                 # live output (the results table prints here)
```
The run also saves the trained encoder to `experiments/trained_encoder_full.pt`
(in your mounted home). Copy results back to your laptop:
```bash
scp <USER>@<CLUSTER>:~/chemprop-tabpfn/experiments/trained_encoder_full.pt ./experiments/
```

---

## Files in this folder
| File | What it is |
|---|---|
| `interactive.sub` | interactive container for the one-time setup (`condor_submit -i`) |
| `setup_env.sh`    | builds the uv env + caches the checkpoint (run inside the container) |
| `run_gpu.sub`     | the batch job (`condor_submit`) |
| `run_gpu.sh`      | what the job actually executes inside the container |

## Things you may need to edit
- **Repo path**: `run_gpu.sh` assumes `~/chemprop-tabpfn`. Change `REPO=` if different.
- **Resources**: `run_gpu.sub` requests 1 GPU / 8 CPUs / 64 GB. Adjust if needed.
- **Docker image**: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` (from your examples).
  We install our own Python 3.12 + torch via `uv` on top, so the image mainly
  provides CUDA.

## Full-scale settings (in `src/scripts/run_experiment_gpu.py`)
```
USE_FULL_DATA = True     # all 248k ChEMBL assays
HIDDEN_SIZE   = 300      # thesis-scale encoder
N_ESTIMATORS  = 8        # stronger TabPFN ensemble
N_TRAIN_STEPS = 4000
N_CONTEXT/N_QUERY = 500
N_SPLITS      = 10
```

## Troubleshooting
- **`CUDA available: False`** in the log → the torch wheel `uv` installed doesn't
  match the cluster's GPU driver. In `pyproject.toml` pin a CUDA build compatible
  with the image (e.g. a `cu121` torch), then re-run `setup_env.sh`.
- **Setup can't download (uv/pip/HF)** → the interactive node had no internet;
  ask cluster admins how outbound access works, or pre-cache on a node that has it.
- **`HF_TOKEN not set`** → create `.env` (step 1) before running `setup_env.sh`.
- **Job idle in `condor_q`** → no matching GPU slot yet; wait, or check
  `condor_q -better-analyze <JobId>`.
- **Out of memory** → raise `request_memory` in `run_gpu.sub`.
