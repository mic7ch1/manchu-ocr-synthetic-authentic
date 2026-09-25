#!/bin/bash
# Launches the three Pixtral Wave-1 trainings on a Slurm cluster.
# Each gets 1 node × 8 H100.
#
# Per-variant dataset routing:
#   step1-syn  → mic7ch/manchu-2025-0033        (60k synthetic, mic7ch validation)
#   step1-real → HenryKingCN/scidb_manchu_v2    (20k real; scidb's `test` used as val signal)
#   mix        → mic7ch + scidb concatenated    (80,306 train; mic7ch validation as val)
#
# Usage (on the cluster):
#   ./scripts/slurm/launch_pixtral_wave1.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

TIME_LIMIT="${TIME_LIMIT:-05:00:00}"

submit() {
  local target="$1"; shift
  local jobid
  jobid=$(sbatch --parsable \
    --gres=gpu:4 \
    --time="$TIME_LIMIT" \
    --job-name="train-${target}" \
    scripts/slurm/train.sbatch \
      --target_model "$target" \
      --bf16 \
      --grad_accum 1 \
      "$@")
  echo "submitted $target → jobid $jobid"
}

submit pixtral-step1-syn
submit pixtral-step1-real \
  --dataset_name HenryKingCN/scidb_manchu_v2 \
  --train_split train \
  --val_split test
submit pixtral-mix \
  --dataset_name "mic7ch/manchu-2025-0033,HenryKingCN/scidb_manchu_v2" \
  --train_split train \
  --val_split validation

echo
echo "--- queue snapshot ---"
squeue -u "$USER" -o "%.8i %.18j %.8T %.10M %.6D %R"
