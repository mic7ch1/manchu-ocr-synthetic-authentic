#!/bin/bash
# Slurm launcher for Manchu OCR training.
# Pass train.py args after the script path:
#   sbatch --gres=gpu:8 scripts/slurm/train.sbatch --target_model llama-final --bf16
# For profiling with fewer steps:
#   sbatch --gres=gpu:1 --time=00:30:00 scripts/slurm/train.sbatch --target_model llama-step1-syn --max_steps 20 --bf16 --output_suffix smoke
#
# Single-GPU → plain python. Multi-GPU → torchrun DDP (Unsloth auto-detects).

#SBATCH --job-name=manchu
#SBATCH --account=YOUR_ACCOUNT  # set to your Slurm account
#SBATCH --partition=YOUR_PARTITION  # set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load HF_TOKEN and any other vars from project .env
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Activate the uv-managed venv
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

# BNB resume workaround: delete optimizer.pt from latest checkpoint of the target model
# (prevents bnb 4-bit optimizer-state crash when resuming; HF Trainer reinits state cleanly)
ARGS="$@"
TARGET_MODEL=$(echo "$ARGS" | grep -oE -- '--target_model [a-zA-Z0-9_-]+' | awk '{print $2}' | head -1 || true)
OUTPUT_SUFFIX=$(echo "$ARGS" | grep -oE -- '--output_suffix [a-zA-Z0-9_-]+' | awk '{print $2}' | head -1 || true)
if [ -n "$TARGET_MODEL" ]; then
  DIR_NAME="$TARGET_MODEL"
  [ -n "$OUTPUT_SUFFIX" ] && DIR_NAME="${TARGET_MODEL}-${OUTPUT_SUFFIX}"
  CKPT_DIR="models/VLM/$DIR_NAME/checkpoints"
  if [ -d "$CKPT_DIR" ]; then
    LATEST=$(ls -1dt "$CKPT_DIR"/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/optimizer.pt" ]; then
      rm -f "$LATEST/optimizer.pt"
      echo "[bnb-workaround] removed $LATEST/optimizer.pt to avoid 4-bit optimizer-state crash on resume"
    fi
  fi
fi

# Print sanity info
echo "=== run info ==="
echo "node:        $(hostname)"
echo "python:      $(which python) ($(python --version))"
echo "torch cuda:  $(python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())')"
echo "gpus asked:  ${SLURM_GPUS_ON_NODE:-1}"
echo "cuda_vis:    ${CUDA_VISIBLE_DEVICES:-}"
echo "args:        $*"
echo "================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

n_gpus=${SLURM_GPUS_ON_NODE:-1}

if [ "$n_gpus" -gt 1 ]; then
  # Pick a unique NCCL rendezvous port per job so two co-located DDP jobs
  # on the same node don't collide on port 29500 (which would cause the
  # second job to silently hang in rendezvous — see PAPER_NOTES §7.8).
  RDZV_PORT=$((20000 + SLURM_JOB_ID % 20000))
  echo "Launching DDP on $n_gpus GPUs via torchrun (rdzv port: $RDZV_PORT)"
  srun torchrun \
    --nproc_per_node="$n_gpus" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${RDZV_PORT}" \
    scripts/train.py "$@"
else
  echo "Launching single-GPU run"
  srun python scripts/train.py "$@"
fi
