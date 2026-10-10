"""
Experiment 3 — Task Taxonomy (Grokking Geography Map).

Train models on a battery of tasks and measure grokking behavior for each.
The goal: find structural task properties that predict grokking timeline.

Tasks:
  - mod_add (Z_p addition) — abelian, prime order, circular structure
  - mod_mul (Z_p multiplication) — abelian, prime order
  - mod_exp (Z_p exponentiation) — non-commutative
  - mod_div (Z_p division) — non-commutative
  - s5_compose (S_5 permutation) — non-abelian, order 120
  - dihedral (D_n dihedral group) — non-abelian
  - xor (n-bit XOR) — abelian, power-of-2 order
  - parity — binary output, power-of-2 input

For each task, measure:
  - Whether grokking occurs
  - Grokking delay Δt
  - Final test accuracy
  - Weight norm trajectory shape

Hypothesis: grokking is easier for tasks with:
  - Abelian group structure
  - Prime group order
  - Circular/Fourier-alignable representations

Run:
    python experiments/exp3_task_taxonomy/run_taxonomy.py
    python experiments/exp3_task_taxonomy/run_taxonomy.py --quick  # smaller runs
"""

import os

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models.transformer import TransformerConfig, build_model
from src.tasks.modular import ModularAddition, ModularMultiplication, ModularExponentiation, ModularDivision
from src.tasks.groups import S5Composition, DihedralComposition
from src.tasks.boolean import XORTask, ParityTask
from src.training.trainer import GrokTrainer, TrainConfig


def build_task_list(seed: int, quick: bool = False):
    """Build the tasks for one seed (the seed also fixes the train/test split)."""
    p = 97 if quick else 113
    return [
        ("mod_add",    ModularAddition(p=p, seed=seed)),
        ("mod_mul",    ModularMultiplication(p=p, seed=seed)),
        ("mod_exp",    ModularExponentiation(p=97, seed=seed)),
        ("mod_div",    ModularDivision(p=p, seed=seed)),
        ("s5_compose", S5Composition(seed=seed)),
        ("dihedral_12", DihedralComposition(n=12, seed=seed)),
        ("xor_6bit",   XORTask(n_bits=6, seed=seed)),
        ("xor_5bit",   XORTask(n_bits=5, seed=seed)),
        ("parity_8bit", ParityTask(n_bits=8, seed=seed)),
    ]


def run_task(name: str, task, n_steps: int, seed: int, use_wandb: bool) -> dict:
    """Run a single task and return grokking metrics."""
    print(f"\n  Task: {name}")

    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3,
        d_model=128,
        n_heads=4,
        d_head=32,
        d_mlp=512,
        n_layers=1,
    )
    model = build_model(model_cfg, seed)

    train_cfg = TrainConfig(
        n_steps=n_steps,
        lr=1e-3,
        weight_decay=1.0,
        log_every=200,
        seed=seed,
        use_wandb=use_wandb,
        wandb_project="grokking-geography",
        run_name=f"taxonomy_{name}_s{seed}",
    )

    trainer = GrokTrainer(model, task, train_cfg)
    metrics = trainer.train()

    result = {
        "task_name": name,
        "seed": seed,
        "grokked": metrics.grok_step is not None,
        "grok_step": metrics.grok_step,
        "mem_step": metrics.mem_step,
        "grok_delay": metrics.grok_delay,
        "final_test_acc": metrics.test_acc[-1] if metrics.test_acc else 0.0,
        "final_train_acc": metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "peak_test_acc": metrics.peak_test_acc,
        "grok_lost_step": metrics.grok_lost_step,
        "n_train": len(task.dataset.train_labels),
        "n_test": len(task.dataset.test_labels),
        "elapsed_seconds": metrics.elapsed_seconds,
        # Task structural properties (from TaskConfig)
        "vocab_size": task.cfg.vocab_size,
        "is_commutative": task.cfg.is_commutative,
        "is_group": task.cfg.is_group,
        "group_order": task.cfg.group_order,
        "operation": task.cfg.operation,
        # Learning curves (for visualization)
        "steps": metrics.steps,
        "train_acc": metrics.train_acc,
        "test_acc": metrics.test_acc,
    }

    print(f"    grokked={result['grokked']} | grok_step={result['grok_step']} | "
          f"delay={result['grok_delay']} | test_acc={result['final_test_acc']:.3f}")

    return result


def plot_taxonomy(all_results: list, out_dir: str):
    """Generate the grokking geography map visualization."""
    # Average over seeds per task
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in all_results:
        by_task[r["task_name"]].append(r)

    tasks = list(by_task.keys())
    grokked = [any(r["grokked"] for r in by_task[t]) for t in tasks]
    grok_delays = [
        np.mean([r["grok_delay"] for r in by_task[t] if r["grok_delay"] is not None])
        if any(r["grok_delay"] is not None for r in by_task[t]) else None
        for t in tasks
    ]
    test_accs = [np.mean([r["final_test_acc"] for r in by_task[t]]) for t in tasks]
    commutative = [by_task[t][0]["is_commutative"] for t in tasks]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Grokking Geography: Task Taxonomy", fontsize=14)

    colors = ["steelblue" if c else "coral" for c in commutative]
    legend_patches = [
        mpatches.Patch(color="steelblue", label="Commutative"),
        mpatches.Patch(color="coral",     label="Non-commutative"),
    ]

    # 1. Grokking delay by task
    ax = axes[0]
    y_pos = range(len(tasks))
    delays = [d if d else 0 for d in grok_delays]
    ax.barh(list(y_pos), delays, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Grokking Delay (steps)")
    ax.set_title("Grokking Delay by Task")
    ax.legend(handles=legend_patches, loc="lower right")
    # Mark tasks that didn't grok
    for i, (g, t) in enumerate(zip(grokked, tasks)):
        if not g:
            ax.text(ax.get_xlim()[1] * 0.5, i, "NO GROK", va="center", color="red", fontsize=9)

    # 2. Final test accuracy by task
    ax = axes[1]
    ax.barh(list(y_pos), test_accs, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Final Test Accuracy")
    ax.set_title("Final Test Accuracy by Task")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.99, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_patches)

    # 3. Learning curves
    ax = axes[2]
    cmap = plt.cm.tab10
    for i, (t, results) in enumerate(by_task.items()):
        # Average test_acc over seeds
        avg_steps = results[0]["steps"]
        avg_test_acc = np.mean([[r["test_acc"][j] if j < len(r["test_acc"]) else r["test_acc"][-1]
                                  for j in range(len(avg_steps))]
                                 for r in results], axis=0)
        ax.plot(avg_steps, avg_test_acc, label=t, color=cmap(i / len(by_task)), alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Learning Curves by Task")
    ax.legend(loc="lower right", fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "task_taxonomy.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",     type=int,   default=120_000,
                        help="Step limit (default: 120K to avoid false negatives on slow-grokking tasks)")
    parser.add_argument("--n_seeds",   type=int,   default=5,
                        help="Number of seeds per task (default: 5 for statistical robustness)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--quick",     action="store_true",
                        help="Run with fewer steps and smaller primes for debugging")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--out_dir",   type=str,   default="results/exp3_taxonomy")
    args = parser.parse_args()

    if args.quick:
        args.steps = 20_000
        args.n_seeds = 1

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Running taxonomy × {args.n_seeds} seeds")

    all_results = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        for name, task in build_task_list(seed, quick=args.quick):
            all_results.append(run_task(name, task, args.steps, seed, args.use_wandb))

    # Save (exclude large arrays)
    out_path = os.path.join(args.out_dir, "taxonomy_results.json")
    with open(out_path, "w") as f:
        save = [{k: v for k, v in r.items() if k not in ("steps", "train_acc", "test_acc")}
                for r in all_results]
        json.dump(save, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    plot_taxonomy(all_results, args.out_dir)

    # Print summary
    print(f"\n=== TAXONOMY SUMMARY ({args.n_seeds} seeds per task) ===")
    print(f"{'Task':<16} {'Grok Rate':<11} {'Delay (mean±sd)':<20} {'Test Acc (mean±sd)':<22} {'Commutative'}")
    print("-" * 85)
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in all_results:
        by_task[r["task_name"]].append(r)
    for task_name, runs in by_task.items():
        n_grokked = sum(1 for r in runs if r["grokked"])
        grok_rate = f"{n_grokked}/{len(runs)}"
        delays = [r["grok_delay"] for r in runs if r["grok_delay"] is not None]
        if delays:
            delay_str = f"{np.mean(delays):.0f} ± {np.std(delays):.0f}"
        else:
            delay_str = "—"
        accs = [r["final_test_acc"] for r in runs]
        acc_str = f"{np.mean(accs):.3f} ± {np.std(accs):.3f}"
        comm = runs[0]["is_commutative"]
        print(f"{task_name:<16} {grok_rate:<11} {delay_str:<20} {acc_str:<22} {comm}")


if __name__ == "__main__":
    main()
