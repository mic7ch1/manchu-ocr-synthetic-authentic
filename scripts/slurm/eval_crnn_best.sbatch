#!/bin/bash
# Run evaluate_best.py on a CRNN model (val + test on best_model).
# Prereqs: best_model/ must exist (created by evaluate_checkpoints.py sweep).
# Usage:
#   sbatch --job-name=eval-best-crnn-real scripts/slurm/eval_crnn_best.sbatch crnn-step1-real

#SBATCH --account=YOUR_ACCOUNT  # set to your Slurm account
#SBATCH --partition=YOUR_PARTITION  # set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
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
echo "model:       $MODEL_NAME"
echo "================"

srun python scripts/evaluate_best.py --target-model "$MODEL_NAME"
