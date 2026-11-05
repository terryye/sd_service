#!/bin/bash
# Slurm job: generate illustrations for one CSV of words on one GPU.
# Usage (on the login node): sbatch ~/sd_service/sd_job.sh <words.csv> <out_dir> [generate.py options]

#SBATCH --job-name=sd_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ye.te/sd_service/logs/%x.%j.out
#SBATCH --error=/home/ye.te/sd_service/logs/%x.%j.out

set -euo pipefail
SVC=$HOME/sd_service
SIF=/shared/container_repository/explorer/pytorch/pytorch_training.sif
TMP=/tmp/$USER-$SLURM_JOB_ID
trap 'rm -rf "$TMP"' EXIT

# Model weights (~7 GB, a few big files) live in home. diffusers is ~1,000 small
# files, which shared storage writes very slowly, so it goes on the node's local
# disk each job. --no-deps keeps the container's own torch/transformers/hub;
# requests is the one diffusers dependency the container lacks.
export APPTAINERENV_HF_HOME=$SVC/hf_cache
export APPTAINERENV_PYTHONPATH=$TMP/pylib
export APPTAINERENV_PIP_CACHE_DIR=$TMP/pip
export APPTAINERENV_PYTHONUNBUFFERED=1
run() { apptainer exec --nv -B /tmp:/tmp "$SIF" "$@"; }

# submit.sh may leave duplicate submissions under the same job name (see there).
# Only the first one does the work.
FIRST=$(sacct -n -X -u "$USER" --name="$SLURM_JOB_NAME" -S now-7days -o JobID 2>/dev/null | awk '{print $1}' | sort -n | head -1)
if [ -n "$FIRST" ] && [ "$FIRST" != "$SLURM_JOB_ID" ]; then
    echo "duplicate of job $FIRST, exiting"
    exit 0
fi

echo "node $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run pip install -q --disable-pip-version-check --no-deps --target "$TMP/pylib" \
    "diffusers==0.42.1" requests urllib3 idna charset-normalizer certifi
run python -c "import scipy" 2>/dev/null ||
    run pip install -q --disable-pip-version-check --no-deps --target "$TMP/pylib" "scipy==1.18.1"
echo "packages installed $(date)"

run python "$SVC/generate.py" --input "$1" --out "$2" "${@:3}"
echo "finished $(date)"
