#!/bin/bash
# Runs inside the docker container: runtime_study.py.
set -euo pipefail

# Where the repository lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

# Keep a separate, timestamped copy of this start's output: if HTCondor requeues the job,
# the restart would otherwise overwrite the .out file of the first start.
mkdir -p runlogs
exec > >(tee -a "runlogs/runtime_study.$(date -u +%Y%m%dT%H%M%SZ).tee.log") 2>&1

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$REPO/.hf_cache" # HF cache inside the (mounted) repo — no ambiguity
export HF_HUB_OFFLINE=1          # the checkpoint was cached here during setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && { set -a; source .env; set +a; }

echo "host: $(hostname)"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || true
echo "start: $(date)"

PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/runtime_study.py

echo "end: $(date)"
