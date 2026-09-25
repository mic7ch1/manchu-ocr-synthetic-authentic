"""
Sweep CRNN checkpoints on a single eval split.

Mirrors the structure of evaluate_checkpoints.py but evaluates on ONE split
(validation, test, or real_val), writing `checkpoint-{step}_{split}.json`
files alongside the existing validation outputs.

Used to retroactively populate test/real_val curves for CRNN variants whose
sweep evaluation was originally run only on validation.

Usage:
    python scripts/sweep_crnn_split.py --target-model crnn-step1-real --split real_val
    python scripts/sweep_crnn_split.py --target-model crnn-mix --split test --num-samples 753
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import ConfigLoader
from src.utils.dataset import load_split
from src.evaluation.crnn_evaluator import inference_crnn
from src.evaluation.metrics import calculate_metrics
from src.evaluation.utils import load_crnn_model, cleanup_gpu, print_header
from src.utils.files import load_json, save_json


SPLIT_DEFAULTS = {
    "validation": 1000,
    "real_val":   1000,
    "test":        753,
}


def main():
    parser = argparse.ArgumentParser(
        description="Sweep all checkpoints of a CRNN variant on one eval split"
    )
    parser.add_argument("--target-model", required=True,
                        help="CRNN variant name (e.g. crnn-step1-real)")
    parser.add_argument("--split", required=True,
                        choices=["validation", "test", "real_val"],
                        help="Which eval split to score every checkpoint on")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override sample count (default per split)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached outputs exist")
    parser.add_argument("--out-name", default=None,
                        help="Output dir name under results/ (default: target-model). "
                             "Use to keep a re-evaluation from overwriting historical metrics.")
    args = parser.parse_args()

    config_loader = ConfigLoader()
    dataset_config = config_loader.get_config("data")
    models_config_list = config_loader.get_config("models")
    config_map = {m["name"]: m for m in models_config_list}

    if args.target_model not in config_map:
        print(f"Model '{args.target_model}' not in configs/base.yaml")
        sys.exit(1)

    model_config = config_map[args.target_model]
    if model_config.get("model_class") != "CRNN":
        print(f"'{args.target_model}' is not a CRNN model")
        sys.exit(1)

    num_samples = args.num_samples or SPLIT_DEFAULTS[args.split]

    data_root = project_root / "data"
    eval_dataset = load_split(data_root, args.split)
    print(f"Loaded split '{args.split}' with {len(eval_dataset)} samples; using {num_samples}")

    out_name = args.out_name or args.target_model
    metrics_dir = project_root / "results" / "metrics" / out_name
    predictions_dir = project_root / "results" / "predictions" / out_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = project_root / "models" / "CRNN" / args.target_model / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"No checkpoints dir at {checkpoints_dir}")
        sys.exit(1)

    checkpoint_paths = sorted(
        checkpoints_dir.glob("checkpoint-*.pth"),
        key=lambda p: int(p.stem.split("-")[-1]),
    )
    if not checkpoint_paths:
        print(f"No checkpoints in {checkpoints_dir}")
        sys.exit(1)

    print_header(f"Sweep {args.target_model} on '{args.split}' ({len(checkpoint_paths)} ckpts, {num_samples} samples)")

    for ckpt_path in checkpoint_paths:
        step_num = int(ckpt_path.stem.split("-")[-1])
        metrics_file = metrics_dir / f"checkpoint-{step_num}_{args.split}.json"
        predictions_file = predictions_dir / f"checkpoint-{step_num}_{args.split}.json"

        if not args.force and metrics_file.exists() and predictions_file.exists():
            print(f"[cached] checkpoint-{step_num} on {args.split}")
            continue

        print(f"Evaluating checkpoint-{step_num} on {args.split} ...")
        model = load_crnn_model(str(ckpt_path))
        results = inference_crnn(model, eval_dataset, dataset_config, num_samples)
        metrics = calculate_metrics(results, args.target_model, step_num)

        save_json(metrics_file, metrics)
        save_json(predictions_file, results)

        wa = metrics.get("manchu_word_accuracy", 0)
        cer = metrics.get("manchu_cer", 0)
        print(f"  ckpt-{step_num} on {args.split}: WA={wa:.4f} CER={cer:.4f}")

        del model
        cleanup_gpu()

    print_header("Sweep complete")


if __name__ == "__main__":
    main()
