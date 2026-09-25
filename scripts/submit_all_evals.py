#!/usr/bin/env python3
"""Plan and submit combined-splits evaluations for every (model, checkpoint)
missing at least one of {validation, real_val, test}.

Idempotent in three ways:
  1. Skips (model, ckpt) with all 3 splits already on disk.
  2. Skips (model, ckpt) currently PENDING or RUNNING in Slurm with an
     eval-<model>-<ckpt>* job name — avoids duplicate submissions.
  3. Uses --splits with ONLY the missing splits, so partial completion
     (e.g. real_val done, val+test missing) only requests the missing 2.

USAGE (on cluster login node):
  python3 submit_all_evals.py --dry-run            # preview everything
  python3 submit_all_evals.py --mode featured      # paper-main variants (~60)
  python3 submit_all_evals.py --mode minimum       # only missing real_val + test on best
  python3 submit_all_evals.py --mode all           # every missing split
  python3 submit_all_evals.py --model pixtral-mix  # one specific model

MODES:
  featured = step1-real × 3, mix × 3, pixtral-final (and other finals once trained)
  minimum  = only enough to pick val-best and report test-on-best per model
  all      = every missing split for every (model, ckpt) on disk
"""
import os
import glob
import re
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
SPLITS = ["validation", "real_val", "test"]

# Featured: the 9 paper-main variants (7 existing + 2 not-yet-trained)
FEATURED = {
    # step1-real × 3
    "llama-step1-real", "pixtral-step1-real", "qwen-step1-real",
    # mix v1 × 3 (kept for historical comparison) + v2 retrains (canonical)
    "llama-mix", "pixtral-mix", "qwen-mix",
    "llama-mix-v2", "pixtral-mix-v2", "qwen-mix-v2",
    # final × 3
    "llama-final", "pixtral-final", "qwen-final",
    # step1-syn × 3 (v2 is the canonical qwen-syn; also pixtral-step1-syn-v2 retrain in progress)
    "llama-step1-syn", "pixtral-step1-syn", "qwen-step1-syn-v2",
    "pixtral-step1-syn-v2", "llama-step1-syn-v2", "pixtral-mix-v4",
}

# Also worth evaluating in "all" mode
SECONDARY = set()  # now empty; all -syn variants promoted to FEATURED

# Known-corrupted checkpoints to never evaluate (kept for diagnostic comparison,
# but should not enter the main eval pipeline)
CORRUPTED_CKPTS = {
    # llama-mix: original run 3734's resume used mic7ch only (not concat),
    # so ckpts 12000-18750 are syn-only continuations. Valid range: <=11500.
    "llama-mix": lambda ckpt_name: (
        ckpt_name == "best_model" or
        (ckpt_name.startswith("checkpoint-") and
         int(ckpt_name.replace("checkpoint-", "")) > 11500)
    ),
}

# Legacy / not useful
SKIP = {"qwen-step1-syn",          # v1 overfit, replaced by v2
        "qwen-final-v1-abandoned"} # abandoned


def list_checkpoints(model):
    """Yield (ckpt_name, full_path) across checkpoints/, _preserved/, best_model/."""
    is_corrupted = CORRUPTED_CKPTS.get(model, lambda _: False)
    out = []
    ckpt_dir = f"{ROOT}/models/VLM/{model}/checkpoints"
    preserved_dir = f"{ROOT}/models/VLM/{model}/_preserved"
    best_model = f"{ROOT}/models/VLM/{model}/best_model"
    seen = set()
    if os.path.isdir(ckpt_dir):
        for d in sorted(os.listdir(ckpt_dir)):
            if d.startswith("checkpoint-") and os.path.isdir(f"{ckpt_dir}/{d}"):
                out.append((d, f"{ckpt_dir}/{d}"))
                seen.add(d)
    if os.path.isdir(preserved_dir):
        for d in sorted(os.listdir(preserved_dir)):
            if d.startswith("checkpoint-") and os.path.isdir(f"{preserved_dir}/{d}") and d not in seen:
                out.append((d, f"{preserved_dir}/{d}"))
    if os.path.isdir(best_model):
        out.append(("best_model", best_model))
    # Filter out known-corrupted ckpts for this model
    out = [(n, p) for (n, p) in out if not is_corrupted(n)]
    return out


def ckpt_step(ckpt_name):
    if ckpt_name == "best_model":
        return 10_000_000
    try:
        return int(ckpt_name.replace("checkpoint-", ""))
    except ValueError:
        return -1


def existing_splits(model, ckpt_name):
    """Return set of splits already written. best_model maps to checkpoint-<max>."""
    if ckpt_name == "best_model":
        sib_dir = f"{ROOT}/models/VLM/{model}/checkpoints"
        steps = []
        if os.path.isdir(sib_dir):
            for d in os.listdir(sib_dir):
                if d.startswith("checkpoint-"):
                    try: steps.append(int(d.replace("checkpoint-", "")))
                    except ValueError: pass
        step = max(steps) if steps else -1
        result_dir = f"{ROOT}/results/metrics/{model}/checkpoint-{step}"
    else:
        result_dir = f"{ROOT}/results/metrics/{model}/{ckpt_name}"
    return {s for s in SPLITS if os.path.exists(f"{result_dir}/{s}.json")}


def jobs_in_queue():
    """Return set of job-name prefixes that are PENDING or RUNNING.
    Used to avoid resubmitting already-queued work."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ["USER"],
             "--Format=Name:80", "-h"],
            text=True,
        )
    except Exception:
        return set()
    names = set()
    for ln in out.splitlines():
        names.add(ln.strip())
    return names


def already_queued(model, ckpt_name, queued_names):
    """True if any queued job matches by (model, ckpt) identity,
    regardless of splits suffix or naming-scheme variant."""
    ckpt_tag = ckpt_name.replace('checkpoint-', '')
    # Accept multiple naming schemes:
    #   eval-<model>-<ckpt>-...         (script-generated)
    #   eval-<model-without-step1->-<ckpt>-...   (manual, e.g. llamasyn)
    #   eval-<model-no-hyphens>-<ckpt>-...       (manual, e.g. llamastep1syn)
    targets = [
        f'eval-{model}-{ckpt_tag}',
        f"eval-{model.replace('step1-','').replace('-','')}-{ckpt_tag}",
        f"eval-{model.replace('-','')}-{ckpt_tag}",
    ]
    for n in queued_names:
        for t in targets:
            if n.startswith(t):
                return True
    return False


def minimum_target_ckpts(model):
    """For 'minimum' mode, select just the 'likely val-best' checkpoint per model.
    Prefers best_model if present, else the highest-numbered checkpoint."""
    ckpts = list_checkpoints(model)
    if any(c[0] == "best_model" for c in ckpts):
        return [c for c in ckpts if c[0] == "best_model"]
    if ckpts:
        latest = max(ckpts, key=lambda c: ckpt_step(c[0]))
        return [latest]
    return []


def plan(mode, model_filter=None):
    jobs = []
    queued = jobs_in_queue()
    models_dir = f"{ROOT}/models/VLM"
    if not os.path.isdir(models_dir):
        return jobs

    if mode == "minimum":
        allowed = FEATURED | SECONDARY
    elif mode == "featured":
        allowed = FEATURED
    elif mode == "all":
        allowed = FEATURED | SECONDARY
    else:
        raise ValueError(f"unknown mode: {mode}")

    for model in sorted(os.listdir(models_dir)):
        if model in SKIP:
            continue
        if model not in allowed:
            continue
        if model_filter and model != model_filter:
            continue

        if mode == "minimum":
            ckpt_list = minimum_target_ckpts(model)
        else:
            ckpt_list = list_checkpoints(model)

        for ckpt_name, ckpt_path in ckpt_list:
            done = existing_splits(model, ckpt_name)
            missing = [s for s in SPLITS if s not in done]
            if not missing:
                continue

            # Avoid resubmitting queue duplicates
            ckpt_tag = ckpt_name.replace("checkpoint-", "")
            job_name = f"eval-{model}-{ckpt_tag}-{'-'.join(missing)[:15]}"
            if already_queued(model, ckpt_name, queued):
                continue

            jobs.append({
                "model": model, "ckpt": ckpt_name, "path": ckpt_path,
                "missing": missing, "job_name": job_name,
            })

    jobs.sort(key=lambda j: (j["model"], ckpt_step(j["ckpt"])))
    return jobs


def submit(jobs, dry_run=False):
    for j in jobs:
        rel = os.path.relpath(j["path"], ROOT)
        splits_arg = ",".join(j["missing"])
        cmd = [
            "sbatch",
            "--time=02:30:00",
            f"--job-name={j['job_name']}",
            "scripts/slurm/eval_single.sbatch",
            j["model"], rel,
            "--splits", splits_arg,
        ]
        missing_fmt = ",".join(j["missing"])
        print(f"  {j['model']:<25} {j['ckpt']:<18} missing={missing_fmt}")
        if not dry_run:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            print(f"    -> {r.stdout.strip() or r.stderr.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="featured",
                    choices=["minimum", "featured", "all"],
                    help="featured = step1-real + mix + final variants (default); "
                         "minimum = best_model only, missing splits; "
                         "all = featured + secondary (syn variants)")
    ap.add_argument("--model", default=None, help="Restrict to one model name")
    ap.add_argument("--top-up", type=int, default=None,
                    help="Cap: submit only enough to bring this user's eval-* job count to this. Good citizenship.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = plan(args.mode, model_filter=args.model)
    # top-up throttle: only submit what we need to reach --top-up
    if args.top_up is not None:
        queued = jobs_in_queue()
        eval_queued = sum(1 for n in queued if n.startswith("eval-"))
        deficit = args.top_up - eval_queued
        if deficit <= 0:
            print(f"Queue has {eval_queued} eval jobs already (target: {args.top_up}). Nothing to add.")
            return
        print(f"Queue has {eval_queued} eval jobs; target {args.top_up}; will submit up to {deficit}.")
        jobs = jobs[:deficit]
    if not jobs:
        print("Nothing to submit — all covered or all queued.")
        return

    missing_hist = {}
    model_hist = {}
    for j in jobs:
        k = len(j["missing"])
        missing_hist[k] = missing_hist.get(k, 0) + 1
        model_hist[j["model"]] = model_hist.get(j["model"], 0) + 1

    print(f"=== PLAN: {len(jobs)} combined-splits jobs ({args.mode} mode) ===")
    for n in sorted(missing_hist):
        print(f"  {n} split(s) missing: {missing_hist[n]}")
    for m in sorted(model_hist):
        print(f"  {m}: {model_hist[m]}")
    print()

    submit(jobs, dry_run=args.dry_run)
    if args.dry_run:
        print("\n(dry-run)")


if __name__ == "__main__":
    main()
