#!/bin/bash
# Slurm job for one gateway batch (DESIGN.md 6). The gateway submits it from the
# environment's directory, ~/sd_service/<env>/:
#   sbatch --job-name=<batch_id> --dependency=singleton --time=... sd_job.sh <batch_id>
# Reads batches/<batch_id>/batch.json, writes out/<batch_id>/.

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.out

set -euo pipefail
BATCH=$1
ENV_DIR=$SLURM_SUBMIT_DIR            # ~/sd_service/<env>
ROOT=$(dirname "$ENV_DIR")           # ~/sd_service: shared container and model
SIF=$ROOT/pytorch.sif
TMP=/tmp/$USER-$SLURM_JOB_ID
trap 'rm -rf "$TMP"' EXIT

# A failed sbatch can still create the job, so one batch may have several jobs with
# its name. --dependency=singleton keeps them from running at once; only the lowest
# job ID does the work.
FIRST=$(sacct -n -X -u "$USER" --name="$SLURM_JOB_NAME" -S now-7days -o JobID 2>/dev/null | awk '{print $1}' | sort -n | head -1)
if [ -n "$FIRST" ] && [ "$FIRST" != "$SLURM_JOB_ID" ]; then
    echo "duplicate of job $FIRST, exiting"
    exit 0
fi

# Model weights (a few big files) live in ~/sd_service/hf_cache. diffusers is about
# 1,000 small files, which shared storage writes very slowly, so it is installed on
# the node's local disk in every job. --no-deps keeps the container's own torch,
# transformers and huggingface_hub; requests, and peft for LoRA styles, are what the container lacks.
export APPTAINERENV_HF_HOME=$ROOT/hf_cache
export APPTAINERENV_PYTHONPATH=$TMP/pylib
export APPTAINERENV_PIP_CACHE_DIR=$TMP/pip
export APPTAINERENV_PYTHONUNBUFFERED=1
run() { apptainer exec --nv -B /tmp:/tmp "$SIF" "$@"; }
stamp() { echo "[$(date +%T)] $*"; }

stamp "batch $BATCH, node $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
stamp "installing packages"

run pip install -q --disable-pip-version-check --no-deps --target "$TMP/pylib" \
    "diffusers==0.42.1" "peft==0.21.2" requests urllib3 idna charset-normalizer certifi
run python -c "import scipy" 2>/dev/null ||
    run pip install -q --disable-pip-version-check --no-deps --target "$TMP/pylib" "scipy==1.18.1"

stamp "packages installed, starting generate.py"
run python "$ENV_DIR/generate.py" --batch "$ENV_DIR/batches/$BATCH/batch.json" --out "$ENV_DIR/out/$BATCH" \
    --loras "$ROOT/loras"
stamp "finished $(date)"
