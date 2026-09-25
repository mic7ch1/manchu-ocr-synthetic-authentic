import glob
import os
import numpy as np
import torch
import re
import random
from pathlib import Path
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from src.evaluation.metrics import calculate_cer

from src.utils.files import create_dir

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_base_model(name):
    """Resolve base_model strings. `local:relative/path` → absolute path to a prior
    training output (typically `models/VLM/<upstream>/best_model`). Other strings
    are passed through unchanged to HuggingFace."""
    if isinstance(name, str) and name.startswith("local:"):
        rel = name[len("local:"):].lstrip("/")
        resolved = (_PROJECT_ROOT / rel).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"base_model 'local:{rel}' not found at {resolved}. "
                f"Train the upstream model first, or fix base_model in base.yaml."
            )
        return str(resolved)
    return name


def train_vlm_model(
    model_config,
    training_config,
    training_output,
    converted_train_data,
    converted_val_data,
):
    # Pin each DDP rank to its own GPU *before* any CUDA allocation.
    # Without this, Unsloth's FastVisionModel.from_pretrained loads every rank
    # onto cuda:0 and OOMs. LOCAL_RANK is injected by torchrun.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    base_model = model_config.get("base_model")
    model_name = model_config.get("name")

    loading_config = training_config.get("loading", {})
    peft_config = training_config.get("peft", {})
    sft_config = training_config.get("training", {}).copy()

    model, tokenizer = load_model_and_tokenizer(base_model, loading_config, peft_config)

    training_output_subdir = training_output / "checkpoints"
    create_dir(training_output_subdir)

    print(f"Training {model_name}...")
    print(f"Training output: {training_output_subdir}")

    FastVisionModel.for_training(model)

    # TODO: make this configurable.
    num_eval_samples = 400
    random.seed(42)
    random.shuffle(converted_val_data)
    eval_dataset = converted_val_data[:num_eval_samples]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer, resize="max"),
        train_dataset=converted_train_data,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_pred: compute_manchu_cer(eval_pred, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits,
        args=SFTConfig(
            **sft_config,
            output_dir=training_output_subdir,
            run_name=f"{model_name}_training",
        ),
    )

    checkpoints = glob.glob(os.path.join(training_output_subdir, "checkpoint-*"))
    recent_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None

    if recent_checkpoint:
        print(f"Resuming from checkpoint {recent_checkpoint}")
    else:
        print("Starting training from scratch")

    try:
        trainer.train(resume_from_checkpoint=recent_checkpoint)
        print(f"VLM training finished for {model_name}.")
        best_model_path = training_output / "best_model"
        trainer.save_model(str(best_model_path))
        print(f"Best model saved to: {best_model_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        # Attempt to save whatever state we have before bailing — helps
        # recovery when training crashes mid-run (e.g., eval failures).
        try:
            crash_save = training_output / "crash_save"
            trainer.save_model(str(crash_save))
            print(f"Saved crash state to: {crash_save}")
        except Exception as save_err:
            print(f"Additionally failed to save crash state: {save_err}")
        # Re-raise so Slurm sees a non-zero exit and --requeue can fire.
        raise


def load_model_and_tokenizer(base_model_name, loading_config=None, peft_config=None):
    if loading_config is None:
        loading_config = {}
    if peft_config is None:
        peft_config = {}

    # DDP: force each rank to load onto its own GPU. Without an explicit
    # device_map, HF's caching_allocator_warmup allocates on cuda:0 for every
    # rank, causing cascading OOM on GPU 0.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and "device_map" not in loading_config:
        loading_config["device_map"] = {"": local_rank}
    print(f"[rank {local_rank}] loading with device_map={loading_config.get('device_map')}")

    base_model_name = resolve_base_model(base_model_name)

    # Step-2 continue-training: if base_model_name is a LoRA adapter
    # directory (contains adapter_config.json), FastVisionModel.from_pretrained
    # returns a model with adapters already attached. Skip get_peft_model —
    # otherwise Unsloth raises 'You already added LoRA adapters'. This is the
    # intended path for <family>-final, which resumes training on real data
    # from <family>-step1-syn/best_model.
    is_adapter_dir = (
        os.path.isdir(base_model_name)
        and os.path.exists(os.path.join(base_model_name, "adapter_config.json"))
    )

    model, tokenizer = FastVisionModel.from_pretrained(
        base_model_name, **loading_config
    )

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_adapter_dir:
        print(f"[rank {local_rank}] base is LoRA adapter dir — skip get_peft_model (step-2 continue-training)")
        FastVisionModel.for_training(model)
    else:
        model = FastVisionModel.get_peft_model(model, **peft_config)

    return model, tokenizer


def compute_manchu_cer(eval_pred, tokenizer):
    MANCHU_RE = re.compile(r"manchu\s*:\s*(.*)", flags=re.I)
    extract_manchu = lambda text: next(
        (
            m.group(1).strip()
            for line in text.splitlines()
            if (m := MANCHU_RE.match(line.strip()))
        ),
        "",
    )

    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
    decoded_labels = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

    print("\n[Eval Debug] Sample model output:")
    if decoded_labels and decoded_preds:
        print(f"  GT : {decoded_labels[0]}")
        print(f"  PR : {decoded_preds[0]}\n")

    cer_scores = [
        calculate_cer(extract_manchu(l), extract_manchu(p))
        for p, l in zip(decoded_preds, decoded_labels)
    ]
    mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    return {"manchu_cer": mean_cer}


def preprocess_logits(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = torch.argmax(logits, dim=-1)
    return logits
