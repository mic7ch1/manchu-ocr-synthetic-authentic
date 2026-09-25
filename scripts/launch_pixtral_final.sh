#!/bin/bash
# Stage-2 launch helper: pick real_val-best pixtral-step1-syn checkpoint and launch pixtral-final
#
# Prereqs: pixtral-step1-syn real_val sweep must have enough data to identify a peak.
# Run on cluster login node.
set -euo pipefail

# Resolve project root from this script's location.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Find best pixtral-syn checkpoint on real_val
BEST_CKPT=$(ROOT="$ROOT" python3 <<'PYEOF'
import json, glob, os
root = os.environ["ROOT"]
best = None
best_wa = -1
for f in sorted(glob.glob(f"{root}/results/metrics/pixtral-step1-syn/*/real_val.json")):
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
    echo "ERROR: No pixtral-step1-syn real_val results on disk. Wait for sweep to complete first."
    exit 1
fi

CKPT_NAME=$(echo $BEST_CKPT | cut -d'|' -f1)
WA=$(echo $BEST_CKPT | cut -d'|' -f2)
echo "Best pixtral-step1-syn on real_val: $CKPT_NAME @ ${WA}%"

# 2. Determine source path (checkpoints/, best_model/, or _preserved/)
if [[ -d "$ROOT/models/VLM/pixtral-step1-syn/checkpoints/$CKPT_NAME" ]]; then
    BASE_PATH="local:models/VLM/pixtral-step1-syn/checkpoints/$CKPT_NAME"
elif [[ -d "$ROOT/models/VLM/pixtral-step1-syn/_preserved/$CKPT_NAME" ]]; then
    BASE_PATH="local:models/VLM/pixtral-step1-syn/_preserved/$CKPT_NAME"
else
    echo "ERROR: $CKPT_NAME not found on disk in checkpoints/ or _preserved/"
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
    r'(- name: pixtral-final\s*\n\s+base_model: )local:[^\n]+',
    r'\\1' + new_base,
    src,
    count=1,
)
open(p, "w").write(src)
print("Updated base.yaml: pixtral-final → " + new_base)
PYEOF

echo "---verify---"
grep -A2 "name: pixtral-final" "$ROOT/configs/base.yaml" | head -3

if [[ -d "$ROOT/models/VLM/pixtral-final" ]]; then
    echo "Note: existing models/VLM/pixtral-final/ will be used by Trainer. If any stale checkpoints, rename first."
    ls "$ROOT/models/VLM/pixtral-final/" 2>/dev/null || true
fi

# 4. Submit with 24h wall time
echo "---submitting---"
sbatch \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    --mem=400G \
    --time=24:00:00 \
    --job-name=train-pixtral-final \
    scripts/slurm/train.sbatch \
        --target_model pixtral-final --bf16 --grad_accum 1 \
        --dataset_name HenryKingCN/scidb_manchu_v2 \
        --train_split train --val_split test

echo "---done---"
