#!/bin/bash
# Run a single-checkpoint evaluation on 1 GPU.
# Usage:
#   sbatch --job-name=eval-real --time=00:45:00 scripts/slurm/eval_single.sbatch \
#     llama-step1-real models/VLM/llama-step1-real/checkpoints/checkpoint-3500

#SBATCH --account=YOUR_ACCOUNT  # set to your Slurm account
#SBATCH --partition=YOUR_PARTITION  # set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f .env ]; then
  set -a; source .env; set +a
fi

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

MODEL_NAME=${1:?model name required}
CHECKPOINT=${2:?checkpoint path required}
shift 2
EXTRA_ARGS=("$@")

echo "=== run info ==="
echo "node:       $(hostname)"
echo "python:     $(which python) ($(python --version))"
echo "torch cuda: $(python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())')"
echo "model:      $MODEL_NAME"
echo "checkpoint: $CHECKPOINT"
echo "================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

srun python scripts/evaluate_single_checkpoint.py \
  --model-name "$MODEL_NAME" \
  --checkpoint "$CHECKPOINT" \
  "${EXTRA_ARGS[@]}"
