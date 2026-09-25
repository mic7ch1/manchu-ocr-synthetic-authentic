#!/bin/bash
# Slurm launcher for CRNN baseline training (single-GPU).
# Hyperparameters are sourced from configs/training.yaml (crnn-base section),
# which is copied verbatim from github.com/mic7ch1/ManchuAI-OCR.
#
# Usage:
#   sbatch --gres=gpu:1 scripts/slurm/train_crnn.sbatch --target_model crnn-base

#SBATCH --job-name=crnn
#SBATCH --account=YOUR_ACCOUNT  # set to your Slurm account
#SBATCH --partition=YOUR_PARTITION  # set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

echo "=== run info ==="
echo "node:        $(hostname)"
echo "python:      $(which python) ($(python --version))"
echo "torch cuda:  $(python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())')"
echo "gpus asked:  ${SLURM_GPUS_ON_NODE:-1}"
echo "cuda_vis:    ${CUDA_VISIBLE_DEVICES:-}"
echo "args:        $*"
echo "================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

echo "Launching CRNN training (single-GPU)"
srun python scripts/train.py "$@"
