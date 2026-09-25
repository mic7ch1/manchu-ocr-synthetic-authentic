import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import ConfigLoader
from src.utils.dataset import prepare_training_datasets, convert_to_conversation
from src.training.vlm_trainer import train_vlm_model
from src.training.crnn_trainer import train_crnn_model


def main(
    target_models=None,
    max_steps=None,
    bf16=False,
    per_device_batch_size=None,
    grad_accum=None,
    output_suffix="",
    dataset_name=None,
    train_split=None,
    val_split=None,
):
    config_loader = ConfigLoader()
    models_config_list = config_loader.get_config("models")
    dataset_config = config_loader.get_config("data")
    if target_models:
        models_config_list = [
            m for m in models_config_list if m["name"] in target_models
        ]
    print(f"target_models: {target_models}")

    if dataset_name:
        # Override: load directly from HF dataset(s), bypassing labels.json disk cache.
        # Comma-separated names concatenate train splits (used for the 'mix' variant).
        # Validation always comes from the first dataset in the list.
        from datasets import load_dataset, concatenate_datasets
        names = [n.strip() for n in dataset_name.split(",") if n.strip()]
        ts = train_split or dataset_config.get("train_split", "train")
        vs = val_split or dataset_config.get("val_split", "validation")
        if len(names) == 1:
            print(f"Loading HF dataset: {names[0]} (train={ts}, val={vs})")
            train_dataset = load_dataset(names[0], split=ts)
            val_dataset = load_dataset(names[0], split=vs)
        else:
            print(f"Mixing HF datasets: {names} — concatenating '{ts}' splits")
            trains = [load_dataset(n, split=ts) for n in names]
            train_dataset = concatenate_datasets(trains)
            print(f"Validation from {names[0]} split={vs}")
            val_dataset = load_dataset(names[0], split=vs)
    else:
        train_dataset, val_dataset = prepare_training_datasets(
            dataset_config.get("train_split"), dataset_config.get("val_split")
        )

    overrides = {}
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    if bf16:
        overrides["bf16"] = True
        overrides["fp16"] = False
    if per_device_batch_size is not None:
        overrides["per_device_train_batch_size"] = per_device_batch_size
    if grad_accum is not None:
        overrides["gradient_accumulation_steps"] = grad_accum
    if overrides:
        print(f"CLI overrides to training config: {overrides}")

    for model_config in models_config_list:
        model_name = model_config["name"]

        if not model_config:
            print(f"Warning: Model {model_name} not found in models.yaml. Skipping.")
            continue

        model_class = model_config.get("model_class")
        training_config = config_loader.get_config("training", model_config["name"])
        if not training_config:
            training_config = training_config.get("default")

        if overrides:
            training_config = {
                **training_config,
                "training": {**training_config.get("training", {}), **overrides},
            }

        output_name = model_name + (f"-{output_suffix}" if output_suffix else "")
        training_output = project_root / "models" / model_class / output_name
        training_output.mkdir(parents=True, exist_ok=True)

        if model_class == "VLM":
            converted_train_data_vlm = [
                convert_to_conversation(s, dataset_config) for s in train_dataset
            ]
            converted_val_data_vlm = [
                convert_to_conversation(s, dataset_config) for s in val_dataset
            ]

            train_vlm_model(
                model_config,
                training_config,
                training_output,
                converted_train_data_vlm,
                converted_val_data_vlm,
            )

        elif model_class == "CRNN":
            train_crnn_model(
                model_config,
                training_config,
                dataset_config,
                train_dataset,
                val_dataset,
                training_output,
            )
            print(f"CRNN training pipeline finished for {model_config['name']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Manchu OCR models")
    parser.add_argument(
        "--target_model",
        type=str,
        nargs='*',
        default=None,
        help="Model(s) to train (space-separated). Available models: qwen-25-3b, qwen-25-7b, llama-32-11b, crnn-base"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Cap training at N steps (overrides num_train_epochs). For smoke tests and profiling sweeps."
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 training (recommended for H100). Overrides YAML's bf16/fp16."
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=None,
        help="Override training.per_device_train_batch_size."
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Override training.gradient_accumulation_steps."
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Appended to output dir (e.g. 'profile-bf16') so profiling runs don't stomp production checkpoints."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset to use instead of the base project dataset (e.g. 'HenryKingCN/scidb_manchu_v2'). Loads directly from HF."
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Train split name when --dataset_name is set (default: configured train_split)."
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="Validation split name when --dataset_name is set (default: configured val_split)."
    )

    args = parser.parse_args()

    target_models = args.target_model if args.target_model else None

    main(
        target_models=target_models,
        max_steps=args.max_steps,
        bf16=args.bf16,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        output_suffix=args.output_suffix,
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        val_split=args.val_split,
    )
