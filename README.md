# manchu-ocr-synthetic-authentic

Companion repository to the manuscript *Building OCR for Low-Resource
Historical Collections: Synthetic and Authentic Training Data in Manchu*
(under review). It holds the training and evaluation code, the exact
data splits, the per-checkpoint selection records, and the prediction
files of all sixteen configurations reported in the paper: three
vision-language model families (LLaMA 3.2 11B Vision, Pixtral 12B,
Qwen3-VL 8B) and a CRNN, each trained under four data regimes (SYN,
REAL, JOINT, SEQ). Fine-tuned checkpoints are published on
[Hugging Face](https://huggingface.co/mic7ch).

## Data

The word images are not included in this repository. They are available
from the public sources cited in the paper:

- Synthetic corpus (SYN-train 60,000 / SYN-val 15,000): Hugging Face
  dataset [`mic7ch/manchu-2025-0033`](https://huggingface.co/datasets/mic7ch/manchu-2025-0033).
- Authentic corpus (SCI-train 20,306 / SCI-val 3,359): Science Data Bank,
  <https://doi.org/10.57760/sciencedb.25676> (*A dataset of Manchu ancient
  book words for OCR*, V2). The training scripts read the Hugging Face
  mirror `HenryKingCN/scidb_manchu_v2` of the same release.
- ARCH-test (753 images): released with Chung and Choi (2026),
  *Computational Humanities Research*; it is the `test` split of
  `mic7ch/manchu-2025-0033`.

`data/train/labels.json`, `data/validation/labels.json`, and
`data/test/labels.json` list the exact image filenames and labels
(Manchu script and romanisation) of the splits used in the paper.

## Experimental Environment

- **Scheduler**: Slurm-managed GPU cluster
- **GPU**: 4× NVIDIA H100 80GB per VLM training job (the two retrained LLaMA cells, `llama-step1-real-v2` and `llama-final-v2`, and all reported evaluation ran on single NVIDIA H200 141GB nodes after a cluster upgrade)
- **CPU**: 32 cores per node
- **RAM**: 400 GB per node
- **Driver / CUDA**: CUDA 12.4

A single H100 with ≥80 GB VRAM is sufficient for evaluation. Training requires 4× H100 (LLaMA 11B, Pixtral 12B) or 1× H100 (CRNN, ~12 M params).

## Installation

```bash
uv sync
```

All datasets and checkpoints used here are public; no Hugging Face
token is required. The base models (`unsloth/Llama-3.2-11B-Vision-Instruct`
and the Pixtral and Qwen equivalents named in `configs/base.yaml`) are
downloaded on first use.

## Quick Start

The `scripts/` folder contains the main entry points:

### 1. Train Models

Dataset routing is NOT automatic — `train.py` trains on the default
synthetic dataset unless `--dataset_name` / `--train_split` /
`--val_split` are supplied (the `data_routing` table in
`configs/base.yaml` documents the correct values per cell but is not
consumed by `train.py`). Always launch through the routing-aware
helpers, or pass the flags yourself. Canonical commands per regime
(one family shown; substitute `llama` / `pixtral` / `qwen`):

```bash
# SYN (step1-syn) — default synthetic routing, no flags needed
python scripts/train.py --target_model llama-step1-syn

# REAL (step1-real)
python scripts/train.py --target_model llama-step1-real   --dataset_name HenryKingCN/scidb_manchu_v2 --train_split train --val_split test

# JOINT (mix)
python scripts/train.py --target_model llama-mix   --dataset_name mic7ch/manchu-2025-0033,HenryKingCN/scidb_manchu_v2   --train_split train --val_split validation

# SEQ (final) — warm-starts from the step1-syn checkpoint in configs/base.yaml
python scripts/train.py --target_model llama-final   --dataset_name HenryKingCN/scidb_manchu_v2 --train_split train --val_split test

# CRNN cells route via labels.json and need no dataset flags
python scripts/train.py --target_model crnn-final
```

`train.py` dispatches to `src/training/{vlm,crnn}_trainer.py` based on each model's `model_class` field in `configs/base.yaml`. Output: `models/{VLM,CRNN}/{model_name}/checkpoints/`.

### 2. Evaluate Models

Three evaluation entry points, each with a different scope:

```bash
# Sweep all checkpoints of one model on one split — picks the real_val-peak step
python scripts/sweep_crnn_split.py --target-model crnn-final --split real_val

# Evaluate the real_val-peak checkpoint of every model on val + test
python scripts/evaluate_best.py
python scripts/evaluate_best.py --target-model llama-final

# Sweep all checkpoints of every VLM model (used to identify peaks before evaluate_best)
python scripts/evaluate_checkpoints.py
python scripts/evaluate_checkpoints.py --target-model qwen-mix
```

Note: `train.py` uses `--target_model` (underscore); the `evaluate_*.py` scripts use `--target-model` (hyphen).

Results land under `results/predictions/{model}/checkpoint-{N}/{split}.json` and `results/metrics/{model}/checkpoint-{N}/{split}.json`. The `real_val` split is the complete 3,359-image held-out SCI-DB validation set; checkpoint selection uses the full split, never the test set.

### 3. Run on a Slurm cluster

`scripts/slurm/` contains sbatch wrappers for the same entry points. Before submitting, set `--account=YOUR_ACCOUNT` and `--partition=YOUR_PARTITION` in each `.sbatch` file (or override at `sbatch` invocation):

```bash
# 4× GPU LoRA training (24 h time limit)
sbatch --gres=gpu:4 --time=24:00:00 --cpus-per-task=32 --mem=400G \
  --job-name=train-llama-final scripts/slurm/train.sbatch \
  --target_model llama-final --bf16 --grad_accum 1

# CRNN (single GPU)
sbatch scripts/slurm/train_crnn.sbatch crnn-final

# Per-checkpoint sweep on one split (CRNN)
sbatch --job-name=sweep-final-real_val --time=02:00:00 \
  scripts/slurm/sweep_crnn_split.sbatch crnn-final real_val

# Single VLM evaluation
sbatch scripts/slurm/eval_single.sbatch llama-final 6000 validation
```

### 4. Standalone CRNN OCR demo

`manchu_ocr_crnn.py` is self-contained — it downloads the pre-trained CRNN from Hugging Face and runs OCR on a single image with no project imports:

```bash
python manchu_ocr_crnn.py --image path/to/manchu_word.png
# or pin a specific HF repo:
python manchu_ocr_crnn.py --image word.png --repo mic7ch/manchu-ocr-crnn-final
```

The script applies the required preprocessing (Resize to 64×480, ImageNet normalize, RGB convert) — this is **not** the same as the VLMs, which take arbitrary-sized PIL images via the model's processor.

## Configuration

Configs are YAML, deep-merged at load time. The `default:` section of each file is the baseline; per-model sections override individual fields.

```python
from src.utils.config import ConfigLoader

loader = ConfigLoader()
training_config = loader.get_config("training", "llama-final")  # default + llama-final overrides
```

### 1. Training (`configs/training.yaml`)

```yaml
default:
  training:
    per_device_train_batch_size: 4   # × 4 ranks × 1 grad_accum = effective batch 16
    gradient_accumulation_steps: 1
    warmup_steps: 100
    num_train_epochs: 6
    learning_rate: 2.0e-4
    bf16: true                       # mixed-precision; fp16 disabled
    eval_strategy: "no"              # disabled — Unsloth VLM + DDP crash on DynamicCache during eval
    optim: "paged_adamw_8bit"
    lr_scheduler_type: "cosine_with_restarts"
    save_strategy: "steps"
    save_steps: 500
    save_total_limit: 50
    metric_for_best_model: "manchu_cer"
    greater_is_better: false
  loading:
    load_in_4bit: true
    use_gradient_checkpointing: unsloth
    attn_implementation: eager
  peft:
    r: 32
    lora_alpha: 64
    lora_dropout: 0.05
    target_modules: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# Per-VLM overrides (all 12 VLM variants share these — extends defaults)
llama-final:
  training:
    warmup_steps: 1000          # 10× the default for the longer step-2 schedule
    learning_rate: 1.0e-4       # halved vs. default for fine-tuning stability
    num_train_epochs: 5
  loading:  { use_gradient_checkpointing: unsloth }
  peft:     { use_gradient_checkpointing: unsloth }

# CRNN family (separate optimizer stack — 100 epochs, batch=16, lr=1e-3, AdamW + cosine_warm_restarts)
crnn-step1-syn: &crnn-base
  training:
    num_train_epochs: 100
    batch_size: 16
    learning_rate: 1.0e-3
    optimizer: { type: AdamW, lr: 1.0e-3, weight_decay: 0.05 }
    scheduler: { type: CosineAnnealingWarmRestarts, T_0: 10, T_mult: 2, eta_min: 1.0e-6 }
    gradient_clipping: { max_norm: 1.0 }
crnn-step1-real: *crnn-base
crnn-mix:        *crnn-base
crnn-final:      *crnn-base
```

### 2. Evaluation (`configs/evaluation.yaml`)

```yaml
default:
  validation:                # 1000 seeded samples from mic7ch validation
    num_samples: 1000
  real_val:                  # 1000 seeded samples from a held-out real-image set built from scidb
    num_samples: 1000
  test:                      # 753 = full mic7ch/manchu-2025-0033 test split (paper's reported test)
    num_samples: 753
    step_num: best
```

`step_num: best` selects the real_val-peak checkpoint per model; pass an explicit step (e.g. `--step 6000`) to override at the command line. The CRNN family uses the same three-split block via a YAML anchor.

### 3. Data and Models (`configs/base.yaml`)

A single file holds the dataset config, the per-variant dataset routing, and the master model list:

```yaml
data:                                  # Default — used by step1-syn (mic7ch synthetic)
  dataset_name: mic7ch/manchu-2025-0033
  train_split: train
  val_split: validation
  test_split: test
  image_key: im
  text_key: [manchu, roman]
  cache: true
  instruction: |
    You are an expert OCR system for Manchu script. ...

# Per-variant dataset routing — see launch helpers in scripts/launch_*.sh for the
# exact CLI flags. Note: scidb_manchu_v2 has only train (20,306) / test (3,359);
# its `test` is used as a TRAINING-TIME validation signal for step1-real / *-final.
# It is NOT the paper's reported test set.
data_routing:
  llama-step1-syn:    {dataset_name: "mic7ch/manchu-2025-0033",                          train_split: train, val_split: validation}
  llama-step1-real:   {dataset_name: "HenryKingCN/scidb_manchu_v2",                       train_split: train, val_split: test}
  llama-mix:          {dataset_name: "mic7ch/manchu-2025-0033,HenryKingCN/scidb_manchu_v2", train_split: train, val_split: validation}
  llama-final:        {dataset_name: "HenryKingCN/scidb_manchu_v2",                       train_split: train, val_split: test}
  # ...same routing pattern for pixtral-*, qwen-*, crnn-*

models:
  - name: llama-final
    base_model: local:models/VLM/llama-step1-syn-v2/checkpoints/checkpoint-17000
    model_class: VLM
  - name: llama-step1-syn
    base_model: unsloth/Llama-3.2-11B-Vision-Instruct
    model_class: VLM
  - name: crnn-final
    base_model: local:models/CRNN/crnn-step1-syn/checkpoints/checkpoint-251250.pth
    model_class: CRNN
  - name: crnn-step1-syn
    base_model: crnn
    model_class: CRNN
  ...
```

`base_model` accepts three forms:

- A Hugging Face repo (e.g., `unsloth/Llama-3.2-11B-Vision-Instruct`) — used for stage-1 ("step1") models trained from scratch on the dataset.
- A `local:` prefix pointing to a checkpoint directory (VLM) or `.pth` file (CRNN), resolved relative to the repo root — used for stage-2 ("final") models that warm-start from a stage-1 output.
- The literal string `crnn` — only meaningful for `model_class: CRNN` step-1 models, signalling "build fresh".

`data_routing` keys are consumed by the launch helpers (`scripts/launch_*_*.sh`) which translate them to `--dataset_name`, `--train_split`, and `--val_split` CLI flags for `train.py`. Running `python scripts/train.py --target_model llama-step1-real` directly without those flags would (incorrectly) train on the default mic7ch synthetic data — use the canonical commands in the Quick Start above, go through the launch helpers, or pass the routing flags yourself.

### 4. CRNN warm-start

CRNN warm-starting is implemented by `_resolve_local_warmstart()` in `src/training/crnn_trainer.py`. After the model is built fresh, if the config's `base_model` is a `local:` reference, the helper resolves it to a `.pth` file (or scans for `best_model.pth` / `model.pth` inside a directory) and loads its `state_dict` with `strict=True`. The `char2idx` mapping must match exactly, otherwise it raises rather than silently mismatching the output head.

## Models

Three VLM families and one CRNN family. Each family has four variants:

- **`step1-syn`** — synthetic Manchu words only (paper name: **SYN**).
- **`step1-real`** — real photographed words only (paper name: **REAL**).
- **`mix`** — synthetic + real mixed in a single training run (paper name: **JOINT**).
- **`final`** — warm-started from `step1-syn`, fine-tuned on real data (paper name: **SEQ**).

### Paper terminology

The on-disk cell names above are the identifiers used by every script,
config, and results directory. The paper refers to the same four
training regimes as SYN / REAL / JOINT / SEQ:

| Repo (legacy) | Paper | Regime |
|---|---|---|
| `step1-syn` | SYN | Synthetic-only training |
| `step1-real` | REAL | Real-only training |
| `mix` | JOINT | Joint synthetic--real training |
| `final` | SEQ | Sequential synthetic-to-real fine-tuning |

Paper display names combine family and regime: `llama-final` →
**LLaMA-SEQ**, `crnn-mix` → **CRNN-JOINT**, and so on for all 16
cells. The `-v2` / `-v5` / `-rv3359`
suffixes on disk and HF names are lineage/campaign versioning and have
no paper-level counterpart.

### Vision Language Models (12 models, 3 families × 4 variants)

| Family | Variant | HF repo |
|---|---|---|
| **LLaMA 3.2 11B Vision** | `llama-step1-syn` | [`mic7ch/manchu-ocr-llama-step1-syn-v2`](https://huggingface.co/mic7ch/manchu-ocr-llama-step1-syn-v2) |
| | `llama-step1-real` | [`mic7ch/manchu-ocr-llama-step1-real-v2`](https://huggingface.co/mic7ch/manchu-ocr-llama-step1-real-v2) |
| | `llama-mix` | [`mic7ch/manchu-ocr-llama-mix-v2`](https://huggingface.co/mic7ch/manchu-ocr-llama-mix-v2) |
| | `llama-final` | [`mic7ch/manchu-ocr-llama-final-v2`](https://huggingface.co/mic7ch/manchu-ocr-llama-final-v2) |
| **Pixtral 12B** | `pixtral-step1-syn` | [`mic7ch/manchu-ocr-pixtral-step1-syn-v2`](https://huggingface.co/mic7ch/manchu-ocr-pixtral-step1-syn-v2) |
| | `pixtral-step1-real` | [`mic7ch/manchu-ocr-pixtral-step1-real`](https://huggingface.co/mic7ch/manchu-ocr-pixtral-step1-real) |
| | `pixtral-mix` | [`mic7ch/manchu-ocr-pixtral-mix-v5`](https://huggingface.co/mic7ch/manchu-ocr-pixtral-mix-v5) |
| | `pixtral-final` | [`mic7ch/manchu-ocr-pixtral-final`](https://huggingface.co/mic7ch/manchu-ocr-pixtral-final) |
| **Qwen3-VL 8B** | `qwen-step1-syn` | [`mic7ch/manchu-ocr-qwen-step1-syn-v2`](https://huggingface.co/mic7ch/manchu-ocr-qwen-step1-syn-v2) |
| | `qwen-step1-real` | [`mic7ch/manchu-ocr-qwen-step1-real`](https://huggingface.co/mic7ch/manchu-ocr-qwen-step1-real) |
| | `qwen-mix` | [`mic7ch/manchu-ocr-qwen-mix-v2`](https://huggingface.co/mic7ch/manchu-ocr-qwen-mix-v2) |
| | `qwen-final` | [`mic7ch/manchu-ocr-qwen-final`](https://huggingface.co/mic7ch/manchu-ocr-qwen-final) |

The `-v2` / `-v5` suffixes are internal lineage versioning carried over from the training campaign (e.g. `pixtral-mix-v5` denotes the seed-3407 reproduction that supersedes earlier seeds). Each variant has a single canonical HF repo — the one linked above. The full `variant → HF repo` mapping is in `scripts/upload_to_hf.py::MODEL_TO_REPO`.

All VLMs are fine-tuned via [Unsloth](https://github.com/unslothai/unsloth) `FastVisionModel` + LoRA + TRL `SFTTrainer`. Each model emits `Manchu: {text}\nRoman: {text}`; the evaluator splits on those line prefixes.

### CRNN (4 models, 1 family × 4 variants)

| Variant | HF repo |
|---|---|
| `crnn-step1-syn` | [`mic7ch/manchu-ocr-crnn-step1-syn`](https://huggingface.co/mic7ch/manchu-ocr-crnn-step1-syn) |
| `crnn-step1-real` | [`mic7ch/manchu-ocr-crnn-step1-real`](https://huggingface.co/mic7ch/manchu-ocr-crnn-step1-real) |
| `crnn-mix` | [`mic7ch/manchu-ocr-crnn-mix`](https://huggingface.co/mic7ch/manchu-ocr-crnn-mix) |
| `crnn-final` | [`mic7ch/manchu-ocr-crnn-final`](https://huggingface.co/mic7ch/manchu-ocr-crnn-final) |

CRNN architecture: 9-layer CNN feature extractor → 4-layer BiLSTM (hidden=256 per direction, 30% inter-layer dropout) → CTC head. ~12 M params (12,061,664). Outputs Manchu only; the romanization column is empty for CRNN models. Inputs are resized to 64×480, ImageNet-normalized.

## Results

The leaderboard below is each canonical variant's `real_val`-peak
checkpoint under the full-split selection protocol: every saved
checkpoint of every cell (733 in all) is scored on the complete
3,359-image `real_val` split, ties in word accuracy broken by lower
stored `real_val` CER, then the earlier step. `test` is the 753-image
archival split, never used for selection. All numbers are Manchu word
accuracy (%).

| Rank | Model | Family | Step | real_val | test |
|---:|---|---|---:|---:|---:|
| 1 | `llama-final` | VLM | 6000 | 98.81 | **96.28** |
| 2 | `crnn-step1-real` | CRNN | 92710 | 99.32 | 95.88 |
| 3 | `crnn-final` | CRNN | 19050 | 99.14 | 95.75 |
| 4 | `pixtral-final` | VLM | 5500 | 99.08 | 95.75 |
| 5 | `crnn-mix` | CRNN | 346380 | 99.35 | 95.48 |
| 6 | `llama-mix` | VLM | 19000 | 98.18 | 95.09 |
| 7 | `pixtral-mix` | VLM | 23000 | 98.90 | 95.09 |
| 8 | `llama-step1-real` | VLM | 5500 | 98.12 | 91.23 |
| 9 | `pixtral-step1-real` | VLM | 6350 | 98.45 | 89.11 |
| 10 | `llama-step1-syn` | VLM | 17000 | 87.85 | 87.92 |
| 11 | `qwen-final` | VLM | 6000 | 95.62 | 86.98 |
| 12 | `qwen-mix` | VLM | 21500 | 95.33 | 85.39 |
| 13 | `pixtral-step1-syn` | VLM | 12000 | 84.82 | 81.94 |
| 14 | `qwen-step1-real` | VLM | 6000 | 92.20 | 70.92 |
| 15 | `crnn-step1-syn` | CRNN | 367500 | 73.74 | 64.28 |
| 16 | `qwen-step1-syn` | VLM | 18500 | 69.78 | 62.42 |

The canonical table (including per-cell disk names and stored CER) is
[`results/leaderboard_rv3359.csv`](results/leaderboard_rv3359.csv);
`results/leaderboard_real_val_peak.csv` is a compatibility view of the
same selections. Per-item predictions and per-checkpoint metrics for
every selected cell live under `results/predictions/` and
`results/metrics/` in the `<cell>-rv3359` namespaces.

The reported step is each variant's `real_val`-peak checkpoint, not the
final step of its training schedule; models routinely peak well before
the configured epoch bound, which is why checkpoints are saved densely
and swept.

## Repository layout

```
configs/
  base.yaml             # dataset + model registry, per-cell data routing
  training.yaml         # per-model training hyperparameters
  evaluation.yaml       # per-model evaluation pin (step_num)
data/
  train/labels.json     # SYN-train filenames + labels (60,000)
  validation/labels.json# SYN-val (15,000)
  test/labels.json      # ARCH-test (753)
fonts/
  NotoSansMongolian-Regular.ttf
scripts/
  train.py              # main training entry (dispatches by model_class)
  evaluate_best.py      # eval real_val-peak checkpoint on val + test
  evaluate_checkpoints.py  # sweep all VLM checkpoints
  evaluate_single_checkpoint.py
  sweep_crnn_split.py   # CRNN per-split per-checkpoint sweep
  build_best_checkpoint_links.py
  submit_all_evals.py   # batch-submit eval jobs (slurm)
  launch_*_final.sh     # routing-aware launch helpers
  slurm/                # sbatch templates for cluster runs
src/
  CRNN/                 # standalone CRNN architecture + trainer
  training/             # vlm_trainer.py, crnn_trainer.py
  evaluation/           # vlm_evaluator.py, crnn_evaluator.py, metrics.py
  utils/                # config.py (ConfigLoader), dataset.py, files.py
manchu_ocr_crnn.py      # standalone CRNN demo (downloads from HF, runs OCR)
results/
  leaderboard_rv3359.csv        # the 16 selected checkpoints (paper Table 2)
  metrics/<cell>-rv3359/        # per-checkpoint metrics on real_val / test
  predictions/<cell>-rv3359/    # per-item predictions of the selected checkpoint
  trainer_states_terminal/      # Hugging Face trainer_state.json of the 12 VLM runs
pyproject.toml
requirements.txt
uv.lock
```

## Citation

If you use this code or these files, please cite:

```bibtex
@misc{manchu-ocr-synthetic-authentic-2026,
  author       = {mic7ch},
  title        = {Building OCR for Low-Resource Historical Collections:
                  Synthetic and Authentic Training Data in Manchu
                  (code, splits, and prediction files)},
  year         = {2026},
  howpublished = {\url{https://github.com/mic7ch1/manchu-ocr-synthetic-authentic}},
  note         = {Companion repository to a manuscript under review}
}
```

Please also cite Chung and Choi (2026), *Computational Humanities
Research*, for the synthetic corpus and the 753-image test set, and
the Science Data Bank release (doi:10.57760/sciencedb.25676) for the
authentic corpus.

## Licence

Code is released under the MIT License (see `LICENSE`). The label
files and prediction files under `data/` and `results/` are released
under CC BY 4.0.
