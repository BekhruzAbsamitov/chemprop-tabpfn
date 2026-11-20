#!/bin/bash
# The program HTCondor runs INSIDE the docker container for the SEED-1 REPLICATE of the 50-epoch
# single-protein run (preset full_sp_50ep_seed1).
# It uses the uv environment + Hugging Face cache that cluster/setup_env.sh
# prepared once in your (mounted) home directory.
set -euo pipefail

# TODO: set this to where your repo lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

# Keep a separate, timestamped copy of this start's output. If HTCondor requeues the job,
# the restart writes to the same .out file and would erase the first start's log (that is
# how the original 50-epoch run's per-epoch log was lost on 2026-10-01).
mkdir -p runlogs
exec > >(tee -a "runlogs/full_sp_50ep_seed1.$(date -u +%Y%m%dT%H%M%SZ).tee.log") 2>&1

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$REPO/.hf_cache" # HF cache inside the (mounted) repo — no ambiguity
export HF_HUB_OFFLINE=1          # the checkpoint was cached here during setup
# Let the CUDA allocator grow segments instead of fragmenting — the long
# training loop repeatedly allocates/frees TabPFN activations of varying size.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && { set -a; source .env; set +a; }

echo "host: $(hostname)"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || true
echo "start: $(date)"

# Fail early and loudly if the dataset was not synced — otherwise the job burns a GPU
# slot only to die at the first batch.
test -f data/curated/chembl36_single_protein_clean.parquet || {
  echo "MISSING data/curated/chembl36_single_protein_clean.parquet"
  echo "Build it locally (src/scripts/build_single_protein_dataset.py) and sync.sh push."
  exit 1
}

# src/ on the path so 'from models.../from data_utils...' imports resolve.
# PYTHONUNBUFFERED=1 so prints appear in the log live instead of being buffered.
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/run_experiment.py full_sp_50ep_seed1

echo "end: $(date)"
