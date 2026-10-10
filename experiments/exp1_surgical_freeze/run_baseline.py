"""
Experiment 1 — Baseline grokking run.

Reproduce standard modular addition grokking before any freezing experiments.
This confirms our setup matches Nanda 2023 and gives us the baseline grokking step.

Run:
    python experiments/exp1_surgical_freeze/run_baseline.py
    python experiments/exp1_surgical_freeze/run_baseline.py --prime 97 --steps 100000
"""

import argparse
import json
import os
import matplotlib.pyplot as plt

from src.models.transformer import TransformerConfig, build_model
from src.tasks import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.results import save_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime",       type=int,   default=113)
    parser.add_argument("--train_frac",  type=float, default=0.3)
    parser.add_argument("--steps",       type=int,   default=50_000)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--d_mlp",       type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--wd",          type=float, default=1.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--log_every",   type=int,   default=100)
    parser.add_argument("--use_wandb",   action="store_true")
    parser.add_argument("--out_dir",     type=str,   default="results/exp1_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Task
    task = ModularAddition(p=args.prime, train_frac=args.train_frac, seed=args.seed)
    print(f"Task: mod_add p={args.prime}, train pairs: {len(task.dataset.train_pairs)}, "
          f"test pairs: {len(task.dataset.test_pairs)}")

    # Model
    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=1,
    )
    model = build_model(model_cfg, args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # Train
    train_cfg = TrainConfig(
        n_steps=args.steps,
        lr=args.lr,
        weight_decay=args.wd,
        log_every=args.log_every,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project="grokking-geography",
        run_name=f"baseline_p{args.prime}_wd{args.wd}",
    )

    trainer = GrokTrainer(model, task, train_cfg)
    metrics = trainer.train()

    # Save results (immutable artifact + legacy format)
    result_dict = metrics.to_dict()
    baseline_config = {
        "experiment": "exp1_baseline",
        "prime": args.prime,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "steps": args.steps,
        "lr": args.lr,
        "wd": args.wd,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_mlp": args.d_mlp,
    }
    save_result(
        experiment="exp1_baseline",
        result=result_dict,
        config=baseline_config,
        run_name=f"baseline_p{args.prime}_s{args.seed}",
    )

    # Also save legacy format for backward compatibility
    out_path = os.path.join(args.out_dir, f"baseline_p{args.prime}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Memorization step: {metrics.mem_step}")
    print(f"Grokking step:     {metrics.grok_step}")
    print(f"Grokking delay:    {metrics.grok_delay}")
    print(f"Final train acc:   {metrics.train_acc[-1]:.4f}")
    print(f"Final test acc:    {metrics.test_acc[-1]:.4f}")
    print(f"Elapsed:           {metrics.elapsed_seconds:.1f}s")
    print(f"Results saved to:  {out_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    steps = metrics.steps

    # Panel 1: accuracy curves
    ax1 = axes[0]
    ax1.plot(steps, metrics.train_acc, label="Train acc", color="blue")
    ax1.plot(steps, metrics.test_acc,  label="Test acc",  color="orange")
    ax1.axhline(0.99, color="gray", linestyle="--", alpha=0.5, label="Grok threshold")
    if metrics.grok_step:
        ax1.axvline(metrics.grok_step, color="green", linestyle="--", alpha=0.7, label=f"Grok @ {metrics.grok_step}")
    if metrics.mem_step:
        ax1.axvline(metrics.mem_step, color="red", linestyle="--", alpha=0.7, label=f"Mem @ {metrics.mem_step}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Grokking: mod_add p={args.prime}")
    ax1.legend()

    # Panel 2: Fourier alignment — the continuous grokking progress measure.
    # Should rise from ~0 to ~1 during circuit formation (phase 2), completing
    # before or at the accuracy jump. This is the key three-phase diagnostic.
    ax2 = axes[1]
    ax2.plot(steps, metrics.fourier_alignment, label="Fourier alignment", color="purple")
    if metrics.grok_step:
        ax2.axvline(metrics.grok_step, color="green", linestyle="--", alpha=0.7, label=f"Grok @ {metrics.grok_step}")
    if metrics.mem_step:
        ax2.axvline(metrics.mem_step, color="red", linestyle="--", alpha=0.7, label=f"Mem @ {metrics.mem_step}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Fourier Alignment Score")
    ax2.set_title("Grokking Progress Measure\n(Fourier concentration in embedding)")
    ax2.set_ylim(0, 1.05)
    ax2.legend()

    # Panel 3: weight norms
    ax3 = axes[2]
    for comp in ["embedding", "attn_all", "mlp_all", "unembedding"]:
        if comp in metrics.weight_norms:
            ax3.plot(steps, metrics.weight_norms[comp], label=comp)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("L2 Norm")
    ax3.set_title("Component Weight Norms")
    ax3.legend()

    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, f"baseline_p{args.prime}_s{args.seed}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")


if __name__ == "__main__":
    main()
