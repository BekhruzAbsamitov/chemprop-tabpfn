#!/bin/bash
# The program HTCondor runs INSIDE the docker container for the AL benchmark.
# It uses the uv environment + Hugging Face cache that cluster/setup_env.sh
# prepared once in your (mounted) home directory.
set -euo pipefail

# TODO: set this to where your repo lives in your home directory on the cluster.
REPO="$HOME/chemprop-tabpfn"
cd "$REPO"

# Keep a separate, timestamped copy of this start's output (a requeued restart would
# otherwise overwrite the .out file of the first start).
mkdir -p runlogs
exec > >(tee -a "runlogs/benchmark_rerun_${1}_${2}.$(date -u +%Y%m%dT%H%M%SZ).tee.log") 2>&1

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$REPO/.hf_cache" # HF cache inside the (mounted) repo — no ambiguity
export HF_HUB_OFFLINE=1          # the checkpoint was cached here during setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ -f .env ]] && { set -a; source .env; set +a; }

echo "host: $(hostname)"
.venv/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || true
echo "start: $(date)"

# $1 = target, $2 = run preset (RUNS in benchmark_al.py): re-runs one target inside an
# existing benchmark run, with that run's arms. Refuses to overwrite an existing part file.
test -f data/curated/S_aureus_corrected.csv || { echo "MISSING data/curated/S_aureus_corrected.csv"; exit 1; }
PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python src/scripts/benchmark_al.py "$1" "$2"

echo "end: $(date)"
