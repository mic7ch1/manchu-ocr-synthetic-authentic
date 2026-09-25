"""Evaluate a single checkpoint on one or more splits in a single model load.

Checkpoint-selection sweeps should target held-out splits (validation, real_val,
or both) to avoid test-set leakage. Reserve test only for the single final
checkpoint per model, or run it together with the val splits to save model-load
overhead.

Example (single split):
  python scripts/evaluate_single_checkpoint.py \
    --model-name llama-step1-real \
    --checkpoint models/VLM/llama-step1-real/checkpoints/checkpoint-3500 \
    --splits real_val --num-samples 1000

Example (combined — one model load, three splits written as separate JSONs):
  python scripts/evaluate_single_checkpoint.py \
    --model-name llama-step1-real \
    --checkpoint models/VLM/llama-step1-real/checkpoints/checkpoint-3500 \
    --splits validation,real_val,test
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.dataset import prepare_evaluation_datasets, load_split
from src.utils.config import ConfigLoader
from src.evaluation.vlm_evaluator import inference_vlm
from src.evaluation.metrics import calculate_metrics
from src.evaluation.utils import load_vlm_model
from src.utils.files import save_json


# Per-split sample cap. Test is small and fixed; val splits use user-provided
# num_samples or all available if smaller.
DEFAULT_SAMPLES_PER_SPLIT = {
    "validation": 1000,
    "real_val":   1000,
    "test":       753,
}


def _load_split_dataset(split, dataset_config):
    """Return the eval Dataset for a given split name."""
    if split == "real_val":
        data_root = Path(__file__).resolve().parent.parent / "data"
        return load_split(data_root, "real_val")
    val_ds, test_ds = prepare_evaluation_datasets(
        dataset_config["val_split"], dataset_config["test_split"]
    )
    return val_ds if split == "validation" else test_ds


def main(model_name, checkpoint_path, num_samples, max_new_tokens, splits):
    config_loader = ConfigLoader()
    dataset_config = config_loader.get_config("data")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    # Extract step number; for "best_model" dirs, fall back to the latest
    # numbered sibling checkpoint, or -1 if none.
    try:
        step_num = int(checkpoint_path.name.split("-")[-1])
    except ValueError:
        sibling_ckpts = (checkpoint_path.parent / "checkpoints").glob("checkpoint-*")
        step_num = max(
            (int(p.name.split("-")[-1]) for p in sibling_ckpts),
            default=-1,
        )

    print(f"Model : {model_name}")
    print(f"Ckpt  : {checkpoint_path}")
    print(f"Step  : {step_num}")
    print(f"Splits: {splits}")
    print(f"Sample cap per split (user): {num_samples}")

    # Load model once, reuse across splits.
    model, tokenizer = load_vlm_model(str(checkpoint_path))

    metrics_dir = project_root / "results" / "metrics" / model_name / f"checkpoint-{step_num}"
    preds_dir = project_root / "results" / "predictions" / model_name / f"checkpoint-{step_num}"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        ds = _load_split_dataset(split, dataset_config)
        # Use user cap if provided, else per-split default
        if num_samples > 0:
            n = min(num_samples, len(ds))
        else:
            n = min(DEFAULT_SAMPLES_PER_SPLIT.get(split, 1000), len(ds))

        print(f"\n--- Evaluating split={split}  N={n} (dataset has {len(ds)}) ---")
        results = inference_vlm(
            model, tokenizer, ds, dataset_config, n, max_new_tokens
        )
        metrics = calculate_metrics(results, model_name, step_num)
        metrics["best_step"] = step_num
        metrics["checkpoint_path"] = str(checkpoint_path)
        metrics["split"] = split
        metrics["num_samples"] = n

        out_name = f"{split}.json"
        save_json(metrics_dir / out_name, metrics)
        save_json(preds_dir / out_name, results)
        print(f"Saved: {metrics_dir / out_name}")

        print("=== Key metrics for", split, "===")
        for k in ("manchu_word_accuracy", "manchu_cer", "roman_word_accuracy", "roman_cer"):
            v = metrics.get(k)
            if v is not None:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a single checkpoint on one or more splits (single model load)"
    )
    parser.add_argument("--model-name", required=True, help="e.g. llama-step1-real")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint dir (relative to project root or absolute)"
    )
    parser.add_argument(
        "--splits", default="real_val",
        help="Comma-separated splits: validation, real_val, test. Default: real_val"
    )
    # Back-compat: --split (singular) still accepted; if both given, --splits wins.
    parser.add_argument(
        "--split", default=None,
        help="(legacy, singular) one of: validation, real_val, test"
    )
    parser.add_argument(
        "--num-samples", type=int, default=0,
        help="Max samples per split. 0 = use per-split defaults (val=1000, real_val=1000, test=753)."
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    args = parser.parse_args()

    raw_splits = args.splits if args.splits else args.split
    if not raw_splits:
        raise SystemExit("Must provide --splits or --split")
    splits = [s.strip() for s in raw_splits.split(",") if s.strip()]
    valid = {"validation", "real_val", "test"}
    bad = [s for s in splits if s not in valid]
    if bad:
        raise SystemExit(f"Invalid split(s): {bad}. Choose from {sorted(valid)}")

    main(
        args.model_name,
        args.checkpoint,
        args.num_samples,
        args.max_new_tokens,
        splits,
    )
