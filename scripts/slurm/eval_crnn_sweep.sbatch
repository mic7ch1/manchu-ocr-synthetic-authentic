#!/bin/bash
# Run a full checkpoint sweep evaluation for a CRNN model.
# Usage:
#   sbatch --job-name=eval-crnn-real --time=02:00:00 scripts/slurm/eval_crnn_sweep.sbatch crnn-step1-real

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

echo "=== run info ==="
echo "node:        $(hostname)"
echo "python:      $(which python) ($(python --version))"
echo "model:       $MODEL_NAME"
echo "================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

srun python scripts/evaluate_checkpoints.py --target-model "$MODEL_NAME"
