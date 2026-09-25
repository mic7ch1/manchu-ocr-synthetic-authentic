import shutil
from pathlib import Path

import torch

from src.CRNN.model import CRNN
from src.CRNN.trainer import CRNNTrainer, load_model_and_tokenizer
from src.utils.files import create_dir


def _resolve_local_warmstart(base_model: str) -> Path | None:
    """If `base_model` is `local:...`, return absolute path to the .pth to load.
    Returns None if not a `local:` reference. Raises FileNotFoundError if it
    cannot resolve to an existing .pth."""
    if not isinstance(base_model, str) or not base_model.startswith("local:"):
        return None
    rel = base_model[len("local:"):].lstrip("/")
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / rel
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in ("best_model.pth", "model.pth", "final_model.pth"):
            p = candidate / name
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"warm-start path '{base_model}' did not resolve to a .pth file "
        f"(checked {candidate} and common subnames)"
    )


def train_crnn_model(
    model_config,
    training_config,
    dataset_config,
    train_dataset,
    val_dataset,
    training_output,
):
    model_name = model_config["name"]
    train_config = training_config["training"]

    training_output_subdir = training_output / "checkpoints"
    create_dir(training_output_subdir)

    print(f"Training {model_name}...")
    print(f"Training output: {training_output_subdir}")
    model, tokenizer = load_model_and_tokenizer(
        model_config["base_model"],
        train_config=train_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    warmstart_pth = _resolve_local_warmstart(model_config["base_model"])
    if warmstart_pth is not None:
        print(f"Warm-starting {model_name} from {warmstart_pth}")
        ckpt = torch.load(warmstart_pth, map_location="cpu", weights_only=False)
        ckpt_char2idx = ckpt.get("char2idx")
        if ckpt_char2idx != tokenizer["char2idx"]:
            new_chars = set(tokenizer["char2idx"]) - set(ckpt_char2idx or {})
            if new_chars:
                # Encoding maps chars missing from char2idx to <unk> (idx 1),
                # so these degrade only the affected samples, not the run.
                print(
                    f"  WARNING: {len(new_chars)} dataset chars absent from ckpt "
                    f"vocab {sorted(new_chars)} — they will encode as <unk>"
                )
            # Keep the ckpt's char2idx so label ids stay aligned with the
            # pretrained CTC head.
            print(
                f"  adopting ckpt char2idx ({len(ckpt_char2idx)} classes; "
                f"dataset build had {len(tokenizer['char2idx'])})"
            )
            tokenizer = {
                "char2idx": ckpt_char2idx,
                "idx2char": {i: c for c, i in ckpt_char2idx.items()},
            }
            model = CRNN(
                len(ckpt_char2idx),
                train_config["hidden_size"],
                train_config["dropout"],
            )
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
        print(f"  loaded weights — missing={len(missing) if missing else 0}, "
              f"unexpected={len(unexpected) if unexpected else 0}")

    trainer = CRNNTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=train_config,
        output_dir=training_output_subdir,
    )

    checkpoints = list(training_output_subdir.glob("checkpoint-epoch-*.pth"))
    recent_checkpoint = (
        max(checkpoints, key=lambda x: x.stat().st_ctime) if checkpoints else None
    )

    if recent_checkpoint:
        print(f"Found checkpoint: {recent_checkpoint}")
        print("Note: Resume functionality not yet implemented for CRNN")
    elif warmstart_pth is None:
        print("Starting training from scratch")

    try:
        trainer.train(resume_from_checkpoint=recent_checkpoint)
        print(f"CRNN training finished for {model_name}.")

        best_model_path = training_output / "best_model"
        best_model_path.mkdir(exist_ok=True)

        best_checkpoint = training_output_subdir / "best_model.pth"
        if best_checkpoint.exists():
            shutil.copy2(best_checkpoint, best_model_path / "best_model.pth")
            print(f"Best model saved to: {best_model_path}")
        else:
            print(f"Warning: Best checkpoint not found at {best_checkpoint}")

    except Exception as e:
        print(f"Error during training: {e}")
        return
