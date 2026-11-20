#!/bin/bash
# Runs inside the docker container: reevaluate_epochs.py.
set -euo pipefail

# Where the repository lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

# Keep a separate, timestamped copy of this start's output: if HTCondor requeues the job,
# the restart would otherwise overwrite the .out file of the first start.
mkdir -p runlogs
exec > >(tee -a "runlogs/reevaluate_epochs.$(date -u +%Y%m%dT%H%M%SZ).tee.log") 2>&1

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

test -d experiments/trained_encoder_full_sp_50ep_epochs || {
  echo "MISSING experiments/trained_encoder_full_sp_50ep_epochs"; exit 1; }

PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/reevaluate_epochs.py

echo "end: $(date)"
