#!/bin/bash
# One-command sync between this repo and the cluster (rsync over SSH).
#
#   cluster/sync.sh push          # local -> cluster (code + data), the default
#   cluster/sync.sh push --dry    # show what WOULD transfer, change nothing
#   cluster/sync.sh pull          # cluster -> local (results: experiments/ + runlogs/)
#   cluster/sync.sh watch         # auto-push on every local file change (needs fswatch)
#
# Only changed bytes move, so repeated pushes are fast. Set REMOTE once below.
set -euo pipefail

# --------------------------------------------------------------------------- #
# SET THIS ONCE: your cluster login + the repo path in your (mounted) home.    #
# Put it in the one-line file cluster/remote.local (not in the repository),    #
# for example:   <USER>@<CLUSTER>:~/chemprop-tabpfn                            #
# --------------------------------------------------------------------------- #
REMOTE_FILE="$(dirname "$0")/remote.local"
if [[ ! -f "$REMOTE_FILE" ]]; then
  echo "cluster/remote.local is missing: write <USER>@<CLUSTER>:~/chemprop-tabpfn into it." >&2
  exit 1
fi
REMOTE="$(head -n 1 "$REMOTE_FILE")"

# Never push these (cluster has its own venv / caches / results we don't clobber).
EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.hf_cache/'
  --exclude '.cache/'         # any local cache dir
  --exclude 'notes/'          # meeting notes — not needed on the cluster
  --exclude '.idea/'          # JetBrains IDE metadata
  --exclude '.claude/'        # Claude Code local state
  --exclude '.env'            # secrets stay put; set them on the cluster once
  --exclude 'experiments/'    # trained encoders + curves come FROM the cluster
  --exclude 'runlogs/'        # condor logs are written ON the cluster
  --exclude '.DS_Store'
  --exclude '.pytest_cache/'
  --exclude 'data/curated/chembl36_curated_single_protein.csv'
)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mode="${1:-push}"
dry=""
[[ "${2:-}" == "--dry" ]] && dry="--dry-run"

case "$mode" in
  push)
    echo "PUSH  $REPO_ROOT/  ->  $REMOTE   ${dry:+(dry run)}"
    rsync -avz $dry "${EXCLUDES[@]}" ./ "$REMOTE/"
    ;;
  pull)
    # Bring results back: trained encoders, val curves, condor logs.
    echo "PULL  $REMOTE/{experiments,runlogs}  ->  $REPO_ROOT/   ${dry:+(dry run)}"
    rsync -avz $dry "$REMOTE/experiments/" experiments/
    rsync -avz $dry "$REMOTE/runlogs/"     runlogs/ 2>/dev/null || true
    ;;
  watch)
    command -v fswatch >/dev/null || { echo "need fswatch: brew install fswatch"; exit 1; }
    echo "WATCH $REPO_ROOT — auto-push on change (Ctrl-C to stop)"
    "$0" push
    fswatch -o -r --exclude '\.git' --exclude '\.venv' --exclude '__pycache__' . | \
      while read -r _; do "$0" push; done
    ;;
  *)
    echo "usage: cluster/sync.sh [push|pull|watch] [--dry]"; exit 1;;
esac
