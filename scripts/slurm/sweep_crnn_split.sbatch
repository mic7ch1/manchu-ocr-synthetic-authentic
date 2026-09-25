#!/bin/bash
# Sweep all checkpoints of one CRNN variant on one eval split.
# Usage:
#   sbatch --job-name=sweep-real-realval --time=02:00:00 \
#       scripts/slurm/sweep_crnn_split.sbatch crnn-step1-real real_val
#   sbatch --job-name=sweep-syn-test --time=02:00:00 \
#       scripts/slurm/sweep_crnn_split.sbatch crnn-step1-syn test

#SBATCH --account=YOUR_ACCOUNT  # set to your Slurm account
#SBATCH --partition=YOUR_PARTITION  # set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f .env ]; then
  set -a; source .env; set +a
fi

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

MODEL_NAME=${1:?model name required}
SPLIT=${2:?split required (validation|test|real_val)}

echo "=== run info ==="
echo "node:        $(hostname)"
echo "python:      $(which python) ($(python --version))"
echo "model:       $MODEL_NAME"
echo "split:       $SPLIT"
echo "================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

shift 2 2>/dev/null || true
srun python scripts/sweep_crnn_split.py --target-model "$MODEL_NAME" --split "$SPLIT" "$@"
