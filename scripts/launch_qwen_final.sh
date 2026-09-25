#!/bin/bash
# Stage-2 launch helper: pick real_val-best qwen-step1-syn checkpoint and launch qwen-final
#
# Prereqs: qwen-step1-syn real_val sweep must have enough data to identify a peak.
# Run on cluster login node.
set -euo pipefail

# Resolve project root from this script's location.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Canonical step1 dir for qwen is `qwen-step1-syn-v2` (a post-v1 retrain;
# see configs/base.yaml comment). Allow override via env var for forks.
SYN_DIR="${QWEN_SYN_DIR:-qwen-step1-syn-v2}"
if [[ ! -d "$ROOT/results/metrics/$SYN_DIR" ]]; then
    echo "ERROR: results/metrics/$SYN_DIR not found. Set QWEN_SYN_DIR=<dir> or run the step1 sweep first."
    exit 1
fi
echo "Reading sweep results from: $SYN_DIR"

# 1. Find best qwen-syn checkpoint on real_val
BEST_CKPT=$(ROOT="$ROOT" SYN_DIR="$SYN_DIR" python3 <<'PYEOF'
import json, glob, os
root = os.environ["ROOT"]
syn = os.environ["SYN_DIR"]
best = None
best_wa = -1
for f in sorted(glob.glob(f"{root}/results/metrics/{syn}/*/real_val.json")):
    ckpt = f.split("/")[-2]
    d = json.load(open(f))
    wa = d.get("manchu_word_accuracy", 0)
    if wa > best_wa:
        best_wa = wa
        best = ckpt
if not best:
    print("NO_RESULTS")
else:
    print(f"{best}|{best_wa*100:.2f}")
PYEOF
)

if [[ "$BEST_CKPT" == "NO_RESULTS" ]]; then
    echo "ERROR: No $SYN_DIR real_val results on disk. Wait for sweep to complete first."
    exit 1
fi

CKPT_NAME=$(echo $BEST_CKPT | cut -d'|' -f1)
WA=$(echo $BEST_CKPT | cut -d'|' -f2)
echo "Best $SYN_DIR on real_val: $CKPT_NAME @ ${WA}%"

# 2. Determine source path (checkpoints/, best_model/, or _preserved/)
MODEL_DIR_NAME="${SYN_DIR}"
if [[ -d "$ROOT/models/VLM/$MODEL_DIR_NAME/checkpoints/$CKPT_NAME" ]]; then
    BASE_PATH="local:models/VLM/$MODEL_DIR_NAME/checkpoints/$CKPT_NAME"
elif [[ -d "$ROOT/models/VLM/$MODEL_DIR_NAME/best_model" && "$CKPT_NAME" == "best_model" ]]; then
    BASE_PATH="local:models/VLM/$MODEL_DIR_NAME/best_model"
elif [[ -d "$ROOT/models/VLM/$MODEL_DIR_NAME/_preserved/$CKPT_NAME" ]]; then
    BASE_PATH="local:models/VLM/$MODEL_DIR_NAME/_preserved/$CKPT_NAME"
else
    echo "ERROR: $CKPT_NAME not found on disk under models/VLM/$MODEL_DIR_NAME/"
    exit 1
fi

echo "Base model path: $BASE_PATH"

# 3. Update configs/base.yaml
cp "$ROOT/configs/base.yaml" "$ROOT/configs/base.yaml.bak.$(date +%s)"
ROOT="$ROOT" BASE_PATH="$BASE_PATH" python3 <<'PYEOF'
import os, re
p = os.path.join(os.environ["ROOT"], "configs", "base.yaml")
src = open(p).read()
new_base = os.environ["BASE_PATH"]
src = re.sub(
    r'(- name: qwen-final\s*\n\s+base_model: )local:[^\n]+',
    r'\\1' + new_base,
    src,
    count=1,
)
open(p, "w").write(src)
print("Updated base.yaml: qwen-final → " + new_base)
PYEOF

echo "---verify---"
grep -A2 "name: qwen-final" "$ROOT/configs/base.yaml" | head -3

if [[ -d "$ROOT/models/VLM/qwen-final" ]]; then
    echo "Note: existing models/VLM/qwen-final/ will be used by Trainer. If any stale checkpoints, rename first."
    ls "$ROOT/models/VLM/qwen-final/" 2>/dev/null || true
fi

# 4. Submit with 24h wall time
echo "---submitting---"
sbatch \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    --mem=400G \
    --time=24:00:00 \
    --job-name=train-qwen-final \
    scripts/slurm/train.sbatch \
        --target_model qwen-final --bf16 --grad_accum 1 \
        --dataset_name HenryKingCN/scidb_manchu_v2 \
        --train_split train --val_split test

echo "---done---"
