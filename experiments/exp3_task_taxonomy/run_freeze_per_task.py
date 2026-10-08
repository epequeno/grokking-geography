"""
Experiment 3 (interventional) — Freeze-one sweep across task types.

Reframing relative to descriptive taxonomy:
  2602.18523 (Geometry of Multi-Task Grokking, Feb 2026) already covers
  descriptive multi-task taxonomy. This experiment is differentiated by
  asking a causal question per task:

  "For each task type, which component updates are causally necessary
   for the grokking transition?"

  Specifically: is the MLP always the bottleneck, or does the necessary
  component shift for non-abelian tasks (S5, dihedral) or differently-
  structured tasks (XOR, parity)?

  Exp1 showed for mod_add:
    - MLP is the only hard-necessary component
    - Freezing attn has no effect (attention converges by memorisation step)
    - Freezing embedding causes 10x delay (model finds alternative circuit)

  Predictions for other tasks:
    - mod_mul:     Same as mod_add — similar abelian Fourier structure
    - S5:          MLP may still be necessary but embedding bottleneck
                   may be *harder* (no simple Fourier basis for non-abelian)
    - dihedral:    Similar to S5
    - xor_6bit:    MLP still necessary; Fourier basis exists (Walsh-Hadamard)
                   so embedding may be less of a bottleneck than mod_add
    - parity:      May not grok at all — if it does, compare bottleneck

Run (full sweep — ~3-6 hours on CPU):
    uv run python experiments/exp3_task_taxonomy/run_freeze_per_task.py

Run (quick smoke test):
    uv run python experiments/exp3_task_taxonomy/run_freeze_per_task.py --quick

Run (single task):
    uv run python experiments/exp3_task_taxonomy/run_freeze_per_task.py --tasks mod_mul s5_compose
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import time
import torch
from copy import deepcopy

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition, ModularMultiplication, ModularExponentiation, ModularDivision
from src.tasks.groups import S5Composition, DihedralComposition
from src.tasks.boolean import XORTask, ParityTask
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager

# ── task definitions ──────────────────────────────────────────────────────────

def build_tasks(quick=False):
    p = 97 if quick else 113
    return {
        "mod_add":      ModularAddition(p=p,  train_frac=0.3),
        "mod_mul":      ModularMultiplication(p=p, train_frac=0.3),
        "s5_compose":   S5Composition(train_frac=0.3),
        "dihedral_12":  DihedralComposition(n=12, train_frac=0.3),
        "xor_6bit":     XORTask(n_bits=6, train_frac=0.3),
        "parity_8bit":  ParityTask(n_bits=8, train_frac=0.3),
    }


COMPONENTS = [
    "mlp_all", "embedding", "unembedding",
    "attn_all", "attn_Q", "attn_K", "attn_V", "attn_O",
    "mlp_in", "mlp_out",
]

# ── single freeze-one experiment ──────────────────────────────────────────────

def run_one(task_name, task, target_comp, freeze_at_step,
            model_cfg, train_cfg, seed):
    torch.manual_seed(seed)
    model = GrokTransformer(model_cfg)
    fm    = FreezeManager(model)

    cfg = deepcopy(train_cfg)
    cfg.run_name = f"{task_name}_{target_comp}_s{seed}"
    cfg.freeze_schedule = [(freeze_at_step, "freeze", target_comp)]

    t0 = time.time()
    metrics = GrokTrainer(model, task, cfg, freeze_manager=fm).train()

    return {
        "task":          task_name,
        "component":     target_comp,
        "freeze_at":     freeze_at_step,
        "seed":          seed,
        "grokked":       metrics.grok_step is not None,
        "grok_step":     metrics.grok_step,
        "mem_step":      metrics.mem_step,
        "grok_delay":    metrics.grok_delay,
        "final_test":    metrics.test_acc[-1]  if metrics.test_acc  else 0.0,
        "final_train":   metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "elapsed":       time.time() - t0,
    }


# ── baseline (no freeze) per task ─────────────────────────────────────────────

def run_baseline(task_name, task, model_cfg, train_cfg, seed):
    torch.manual_seed(seed)
    model = GrokTransformer(model_cfg)
    cfg   = deepcopy(train_cfg)
    cfg.run_name = f"{task_name}_baseline_s{seed}"

    t0 = time.time()
    metrics = GrokTrainer(model, task, cfg).train()

    return {
        "task":       task_name,
        "component":  "baseline",
        "freeze_at":  None,
        "seed":       seed,
        "grokked":    metrics.grok_step is not None,
        "grok_step":  metrics.grok_step,
        "mem_step":   metrics.mem_step,
        "grok_delay": metrics.grok_delay,
        "final_test": metrics.test_acc[-1]  if metrics.test_acc  else 0.0,
        "final_train":metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "elapsed":    time.time() - t0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    nargs="+",  default=None,
                        help="Subset of tasks to run (default: all)")
    parser.add_argument("--quick",    action="store_true",
                        help="Reduced steps and smaller primes for smoke testing")
    parser.add_argument("--steps",    type=int,   default=80_000)
    parser.add_argument("--d_model",  type=int,   default=128)
    parser.add_argument("--n_heads",  type=int,   default=4)
    parser.add_argument("--d_mlp",    type=int,   default=512)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--wd",       type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out_dir",  type=str,   default="results/exp3_freeze_per_task")
    args = parser.parse_args()

    if args.quick:
        args.steps = 20_000

    os.makedirs(args.out_dir, exist_ok=True)

    all_tasks = build_tasks(quick=args.quick)
    tasks_to_run = {k: v for k, v in all_tasks.items()
                    if args.tasks is None or k in args.tasks}

    out_path = os.path.join(args.out_dir, f"freeze_per_task_s{args.seed}.json")

    # Load prior results for resuming
    all_results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        done = {(r["task"], r["component"], r["seed"]) for r in all_results}
        print(f"Resuming — {len(all_results)} results already saved.")
    else:
        done = set()

    def save():
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    for task_name, task in tasks_to_run.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name}  (vocab={task.transformer_vocab_size}, "
              f"op={task.cfg.operation}, abelian={task.cfg.is_commutative})")

        model_cfg = TransformerConfig(
            vocab_size=task.transformer_vocab_size,
            n_ctx=3, d_model=args.d_model,
            n_heads=args.n_heads, d_head=args.d_model // args.n_heads,
            d_mlp=args.d_mlp, n_layers=1,
        )
        train_cfg = TrainConfig(
            n_steps=args.steps, lr=args.lr, weight_decay=args.wd,
            log_every=500, seed=args.seed,
        )

        # Step 1: baseline (no freeze) to get mem_step
        key = (task_name, "baseline", args.seed)
        if key in done:
            baseline = next(r for r in all_results
                            if r["task"] == task_name
                            and r["component"] == "baseline"
                            and r["seed"] == args.seed)
            print(f"  baseline (cached): grokked={baseline['grokked']} "
                  f"mem={baseline['mem_step']} grok={baseline['grok_step']}")
        else:
            print(f"  Running baseline…", flush=True)
            baseline = run_baseline(task_name, task, model_cfg, train_cfg, args.seed)
            all_results.append(baseline)
            done.add(key)
            save()
            print(f"  baseline: grokked={baseline['grokked']} "
                  f"mem={baseline['mem_step']} grok_step={baseline['grok_step']} "
                  f"test={baseline['final_test']:.3f}")

        # Skip freeze sweep if the task never groks
        if not baseline["grokked"]:
            print(f"  Task did not grok in {args.steps} steps — skipping freeze sweep.")
            continue

        freeze_at = baseline["mem_step"] or 1000

        # Step 2: freeze-one sweep
        print(f"\n  Freeze-one sweep (freeze at step {freeze_at}):")
        print(f"  {'Component':<15} {'Grokked':<9} {'Delay ratio':<14} {'Test acc'}")
        print(f"  {'-'*55}")

        for comp in COMPONENTS:
            key = (task_name, comp, args.seed)
            if key in done:
                r = next(x for x in all_results if x["task"] == task_name
                         and x["component"] == comp and x["seed"] == args.seed)
                ratio = (f"{r['grok_delay']/baseline['grok_delay']:.1f}×"
                         if r["grok_delay"] and baseline["grok_delay"] else "NEVER")
                print(f"  {comp:<15} (cached) {str(r['grokked']):<9} {ratio:<14} {r['final_test']:.3f}")
                continue

            r = run_one(task_name, task, comp, freeze_at,
                        model_cfg, train_cfg, args.seed)
            all_results.append(r)
            done.add(key)
            save()

            if r["grokked"] and baseline["grok_delay"]:
                ratio = f"{r['grok_delay']/baseline['grok_delay']:.1f}×"
            else:
                ratio = "NEVER"
            print(f"  {comp:<15} {str(r['grokked']):<9} {ratio:<14} {r['final_test']:.3f}")

    # Print final summary table
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<14} {'Component':<15} {'Grokked':<9} {'Delay ratio':<14} {'Test acc'}")
    print(f"{'-'*60}")

    baselines = {r["task"]: r for r in all_results if r["component"] == "baseline"}
    for task_name in tasks_to_run:
        b = baselines.get(task_name)
        if not b:
            continue
        for comp in ["baseline"] + COMPONENTS:
            runs = [r for r in all_results
                    if r["task"] == task_name and r["component"] == comp
                    and r["seed"] == args.seed]
            if not runs:
                continue
            r = runs[0]
            if comp == "baseline":
                ratio = "—"
            elif r["grokked"] and b["grok_delay"]:
                ratio = f"{r['grok_delay']/b['grok_delay']:.1f}×"
            else:
                ratio = "NEVER"
            print(f"{task_name:<14} {comp:<15} {str(r['grokked']):<9} {ratio:<14} {r['final_test']:.3f}")
        print()

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
