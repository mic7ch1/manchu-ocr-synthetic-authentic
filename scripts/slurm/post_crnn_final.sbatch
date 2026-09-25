#!/bin/bash
# Post-training pipeline for crnn-final:
# 1. Flatten doubly-nested checkpoints dir (artifact of warm-started CRNN trainer)
# 2. Submit 3 sweep jobs (validation, test, real_val) with proper warmup ordering
# 3. Print job IDs so they can be tracked
#
# Run after crnn-final training finishes:
#   sbatch --dependency=afterok:<train_jobid> scripts/slurm/post_crnn_final.sbatch

#SBATCH --account=YOUR_ACCOUNT  # set to your Slurm account
#SBATCH --partition=YOUR_PARTITION  # set to your Slurm partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --job-name=post-crnn-final

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CRNN_DIR="models/CRNN/crnn-final"
NESTED="$CRNN_DIR/checkpoints/checkpoints"
FLAT="$CRNN_DIR/checkpoints"

echo "=== flatten ==="
if [ -d "$NESTED" ]; then
  N_FILES=$(ls "$NESTED"/checkpoint-*.pth 2>/dev/null | wc -l)
  echo "moving $N_FILES checkpoints from $NESTED to $FLAT"
  # Move all .pth files up one level
  for f in "$NESTED"/checkpoint-*.pth; do
    mv "$f" "$FLAT/"
  done
  # Also move history.json if present in nested (shouldn't be, but safe)
  if [ -f "$NESTED/history.json" ]; then
    mv "$NESTED/history.json" "$FLAT/"
  fi
  rmdir "$NESTED"
  echo "after flatten: $(ls "$FLAT"/checkpoint-*.pth 2>/dev/null | wc -l) checkpoints in $FLAT"
else
  echo "no nested dir; assuming already flat with $(ls "$FLAT"/checkpoint-*.pth 2>/dev/null | wc -l) checkpoints"
fi

echo
echo "=== submit sweeps ==="
# Validation first to warm up (memory: rv-only cold-start hangs)
JID_VAL=$(sbatch --parsable --job-name=sweep-final-validation --time=02:00:00 \
  scripts/slurm/sweep_crnn_split.sbatch crnn-final validation)
echo "validation sweep:    $JID_VAL"

JID_TEST=$(sbatch --parsable --job-name=sweep-final-test --time=02:00:00 \
  --dependency=afterok:$JID_VAL \
  scripts/slurm/sweep_crnn_split.sbatch crnn-final test)
echo "test sweep:          $JID_TEST"

JID_RV=$(sbatch --parsable --job-name=sweep-final-real_val --time=02:00:00 \
  --dependency=afterok:$JID_VAL \
  scripts/slurm/sweep_crnn_split.sbatch crnn-final real_val)
echo "real_val sweep:      $JID_RV"

echo
echo "All sweep jobs queued. Check progress with: squeue -u \$USER"
