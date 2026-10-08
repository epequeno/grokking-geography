"""
Experiment 3b — Depth / Capacity Ablation for Hard Tasks.

Key question from review: Is S5's difficulty "fundamental non-grokkability"
or just a 1-layer capacity limit?

This experiment runs S5, dihedral, and mod_add (as control) with varying
model depths (1-layer vs 2-layer) to separate architectural capacity from
task-intrinsic difficulty.

If S5 groks significantly faster or more reliably with 2 layers, the
"non-abelian tasks can't grok" claim needs to be narrowed to
"non-abelian tasks can't grok in a 1-layer transformer."

Run:
    uv run python experiments/exp3_task_taxonomy/run_depth_ablation.py
    uv run python experiments/exp3_task_taxonomy/run_depth_ablation.py --quick
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import time
import numpy as np
import torch
from copy import deepcopy

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition
from src.tasks.groups import S5Composition, DihedralComposition
from src.training.trainer import GrokTrainer, TrainConfig
from src.results import save_result_batch


def build_tasks():
    return {
        "mod_add":     ModularAddition(p=113, train_frac=0.3),
        "s5_compose":  S5Composition(train_frac=0.3),
        "dihedral_12": DihedralComposition(n=12, train_frac=0.3),
    }


def run_one(task_name, task, n_layers, n_steps, seed, d_model=128, n_heads=4, d_mlp=512):
    """Run a single (task, depth, seed) experiment."""
    torch.manual_seed(seed)

    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3,
        d_model=d_model,
        n_heads=n_heads,
        d_head=d_model // n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
    )
    model = GrokTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())

    train_cfg = TrainConfig(
        n_steps=n_steps,
        lr=1e-3,
        weight_decay=1.0,
        log_every=500,
        seed=seed,
        run_name=f"depth_{task_name}_L{n_layers}_s{seed}",
    )

    t0 = time.time()
    metrics = GrokTrainer(model, task, train_cfg).train()
    elapsed = time.time() - t0

    result = {
        "task": task_name,
        "n_layers": n_layers,
        "seed": seed,
        "n_params": n_params,
        "grokked": metrics.grok_step is not None,
        "grok_step": metrics.grok_step,
        "mem_step": metrics.mem_step,
        "grok_delay": metrics.grok_delay,
        "final_test_acc": metrics.test_acc[-1] if metrics.test_acc else 0.0,
        "final_train_acc": metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "elapsed_seconds": elapsed,
    }

    status = "✓ GROKKED" if result["grokked"] else "✗ no grok"
    print(f"  {task_name:12s} L={n_layers} seed={seed} | {status} | "
          f"grok_step={result['grok_step']} delay={result['grok_delay']} "
          f"test={result['final_test_acc']:.3f} ({elapsed:.0f}s)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Depth ablation: does adding layers help hard tasks grok?")
    parser.add_argument("--steps",    type=int,   default=120_000,
                        help="Training steps (default: 120K for slow-grokking tasks)")
    parser.add_argument("--n_seeds",  type=int,   default=5)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--layers",   type=int,   nargs="+", default=[1, 2],
                        help="Layer counts to test (default: 1 2)")
    parser.add_argument("--quick",    action="store_true",
                        help="Reduced steps and seeds for smoke testing")
    parser.add_argument("--out_dir",  type=str,   default="results/exp3_depth_ablation")
    args = parser.parse_args()

    if args.quick:
        args.steps = 30_000
        args.n_seeds = 2

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = build_tasks()
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    print(f"Depth ablation: {list(tasks.keys())} × layers={args.layers} × {args.n_seeds} seeds")
    print(f"Step limit: {args.steps}")
    print()

    all_results = []

    for task_name, task in tasks.items():
        for n_layers in args.layers:
            for seed in seeds:
                result = run_one(task_name, task, n_layers, args.steps, seed)
                all_results.append(result)

    # Save immutable artifact
    config = {
        "experiment": "exp3_depth_ablation",
        "tasks": list(tasks.keys()),
        "layers": args.layers,
        "seeds": seeds,
        "n_seeds": args.n_seeds,
        "steps": args.steps,
    }
    save_result_batch(
        experiment="exp3_depth_ablation",
        results=all_results,
        config=config,
        run_name="depth_ablation",
    )

    # Also save legacy format
    out_path = os.path.join(args.out_dir, "depth_ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in all_results:
        by_key[(r["task"], r["n_layers"])].append(r)

    print(f"\n{'='*85}")
    print(f"DEPTH ABLATION SUMMARY ({args.n_seeds} seeds)")
    print(f"{'='*85}")
    print(f"{'Task':<14} {'Layers':<8} {'Params':<10} {'Grok Rate':<11} "
          f"{'Delay (mean±sd)':<20} {'Test Acc (mean±sd)'}")
    print("-" * 85)

    for (task_name, n_layers), runs in sorted(by_key.items()):
        n_grokked = sum(1 for r in runs if r["grokked"])
        grok_rate = f"{n_grokked}/{len(runs)}"
        n_params = runs[0]["n_params"]

        delays = [r["grok_delay"] for r in runs if r["grok_delay"] is not None]
        delay_str = f"{np.mean(delays):.0f} ± {np.std(delays):.0f}" if delays else "—"

        accs = [r["final_test_acc"] for r in runs]
        acc_str = f"{np.mean(accs):.3f} ± {np.std(accs):.3f}"

        print(f"{task_name:<14} {n_layers:<8} {n_params:<10,} {grok_rate:<11} "
              f"{delay_str:<20} {acc_str}")

    # Highlight the key comparison
    print(f"\n--- Key comparison: S5 depth effect ---")
    for n_layers in args.layers:
        runs = by_key.get(("s5_compose", n_layers), [])
        if runs:
            n_grokked = sum(1 for r in runs if r["grokked"])
            accs = [r["final_test_acc"] for r in runs]
            print(f"  S5 L={n_layers}: grokked {n_grokked}/{len(runs)}, "
                  f"test acc {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
