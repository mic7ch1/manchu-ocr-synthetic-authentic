"""Materialize `best_checkpoint/` shortcuts for every model in the leaderboard.

The publishable repo ships per-checkpoint metrics (`checkpoint-N/{split}.json` for
VLMs, `checkpoint-N_{split}.json` for CRNNs), but several downstream consumers
(error/perf table generators, eval-best loaders) look for a stable
`results/metrics/{disk}/best_checkpoint/{split}.json` path.

This script reads `results/leaderboard_real_val_peak.csv`, picks the rv_peak_step
for each model, and creates `best_checkpoint/` either as a symlink (VLM nested)
or as a small dir of copied JSONs (CRNN flat). Run after editing the leaderboard
to keep best_checkpoint in sync with the canonical step choice.
"""

import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = PROJECT_ROOT / "results" / "leaderboard_real_val_peak.csv"
METRICS_ROOT = PROJECT_ROOT / "results" / "metrics"
SPLITS = ("validation", "real_val", "test")


def materialize_vlm(model_dir: Path, step: int) -> tuple[bool, str]:
    src = model_dir / f"checkpoint-{step}"
    dst = model_dir / "best_checkpoint"
    if not src.exists():
        return False, f"missing {src.name}"
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src.name, target_is_directory=True)
    found = [s for s in SPLITS if (src / f"{s}.json").exists()]
    return True, f"-> {src.name} ({', '.join(found) or 'no splits'})"


def materialize_crnn(model_dir: Path, step: int) -> tuple[bool, str]:
    dst = model_dir / "best_checkpoint"
    if dst.exists() and not dst.is_symlink():
        shutil.rmtree(dst)
    elif dst.is_symlink():
        dst.unlink()
    dst.mkdir()
    copied = []
    for split in SPLITS:
        src = model_dir / f"checkpoint-{step}_{split}.json"
        if src.exists():
            shutil.copy2(src, dst / f"{split}.json")
            copied.append(split)
    if not copied:
        dst.rmdir()
        return False, f"no checkpoint-{step}_*.json found"
    return True, f"step {step} ({', '.join(copied)})"


def main():
    if not LEADERBOARD.exists():
        sys.exit(f"leaderboard not found: {LEADERBOARD}")

    rows = list(csv.DictReader(LEADERBOARD.open()))
    n_ok = n_fail = 0
    for row in rows:
        disk = row["model"]
        family = row["family"]
        step = int(row["rv_peak_step"])
        model_dir = METRICS_ROOT / disk
        if not model_dir.exists():
            print(f"  SKIP {disk}: no metrics dir")
            n_fail += 1
            continue
        if family == "vlm":
            ok, msg = materialize_vlm(model_dir, step)
        elif family == "crnn":
            ok, msg = materialize_crnn(model_dir, step)
        else:
            print(f"  SKIP {disk}: unknown family {family!r}")
            n_fail += 1
            continue
        marker = "OK" if ok else "FAIL"
        print(f"  {marker} {disk:<25} {msg}")
        n_ok += int(ok)
        n_fail += int(not ok)
    print(f"\n{n_ok} ok, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
