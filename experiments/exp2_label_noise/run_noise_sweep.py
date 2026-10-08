"""
Experiment 2 — Grokking Under Label Noise.

Train modular addition with varying levels of label noise.
For each noise level, measure:
  1. Does grokking occur?
  2. If yes: how much is the delay changed?
  3. Fourier alignment score — does the model still learn the true circuit?
  4. Final test accuracy on the CLEAN test set.

Novel contribution: first rigorous mechanistic study of grokking under label noise.
We go beyond just measuring accuracy curves to examine the circuit structure.

Run:
    python experiments/exp2_label_noise/run_noise_sweep.py
    python experiments/exp2_label_noise/run_noise_sweep.py --noise_levels 0.0 0.05 0.1 0.2 0.3 0.4 0.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.analysis.fourier import fourier_alignment_score, fourier_power_spectrum


def run_noise_experiment(
    noise_frac: float,
    p: int,
    train_frac: float,
    model_cfg: TransformerConfig,
    train_cfg: TrainConfig,
    seed: int,
) -> dict:
    """Run a single noise level experiment."""
    import dataclasses
    from src.analysis.fourier import dominant_frequencies

    task = ModularAddition(p=p, train_frac=train_frac, noise_frac=noise_frac, seed=seed)

    torch.manual_seed(seed)
    model = GrokTransformer(model_cfg)
    cfg   = dataclasses.replace(train_cfg, run_name=f"noise_{noise_frac:.2f}_s{seed}")

    trainer      = GrokTrainer(model, task, cfg)
    train_metrics = trainer.train()   # Fourier alignment now logged by trainer internally

    # Final circuit analysis
    with torch.no_grad():
        W_E  = model.embed.W_E.detach().cpu()
        final_fourier_score = train_metrics.fourier_alignment[-1] if train_metrics.fourier_alignment else 0.0
        spectrum            = fourier_power_spectrum(W_E, p).numpy()
        dom_freqs           = dominant_frequencies(W_E, p, top_k=3)

    result = {
        "noise_frac":          noise_frac,
        "seed":                seed,
        "grokked":             train_metrics.grok_step is not None,
        "grok_step":           train_metrics.grok_step,
        "mem_step":            train_metrics.mem_step,
        "grok_delay":          train_metrics.grok_delay,
        "final_test_acc":      train_metrics.test_acc[-1]  if train_metrics.test_acc  else 0.0,
        "final_train_acc":     train_metrics.train_acc[-1] if train_metrics.train_acc else 0.0,
        "final_fourier_score": final_fourier_score,
        "dominant_frequencies": dom_freqs,
        "fourier_spectrum":    spectrum.tolist(),
        "n_corrupted_labels":  int(len(task.dataset.train_labels) * noise_frac),
        "n_train":             len(task.dataset.train_labels),
        # Full time-series (for visualisation)
        "steps":               train_metrics.steps,
        "test_acc":            train_metrics.test_acc,
        "train_acc":           train_metrics.train_acc,
        "fourier_alignment":   train_metrics.fourier_alignment,
        "elapsed_seconds":     train_metrics.elapsed_seconds,
    }

    print(f"  noise={noise_frac:.2f} seed={seed}: grokked={result['grokked']} "
          f"grok_step={result['grok_step']} fourier={final_fourier_score:.3f} "
          f"test={result['final_test_acc']:.3f} dom_freqs={dom_freqs}")

    return result


def plot_results(all_results: list, out_dir: str, p: int):
    """Generate summary plots for the noise sweep."""
    noise_levels = [r["noise_frac"] for r in all_results]
    grokked = [r["grokked"] for r in all_results]
    grok_delays = [r["grok_delay"] if r["grok_delay"] is not None else None for r in all_results]
    fourier_scores = [r["final_fourier_score"] for r in all_results]
    test_accs = [r["final_test_acc"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Grokking Under Label Noise (mod_add p={p})", fontsize=14)

    # 1. Test accuracy vs noise
    ax = axes[0, 0]
    ax.plot(noise_levels, test_accs, "o-", color="steelblue")
    ax.set_xlabel("Label Noise Fraction")
    ax.set_ylabel("Final Test Accuracy (clean test set)")
    ax.set_title("Test Accuracy vs Noise")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.99, color="gray", linestyle="--", alpha=0.5)

    # 2. Grokking delay vs noise
    ax = axes[0, 1]
    valid = [(n, d) for n, d in zip(noise_levels, grok_delays) if d is not None]
    if valid:
        ns, ds = zip(*valid)
        ax.plot(ns, ds, "o-", color="orange")
    ax.set_xlabel("Label Noise Fraction")
    ax.set_ylabel("Grokking Delay (steps)")
    ax.set_title("Grokking Delay vs Noise")

    # 3. Fourier alignment vs noise
    ax = axes[1, 0]
    ax.plot(noise_levels, fourier_scores, "o-", color="green")
    ax.set_xlabel("Label Noise Fraction")
    ax.set_ylabel("Fourier Alignment Score")
    ax.set_title("Embedding Fourier Structure vs Noise\n(1.0 = perfect Fourier features)")
    ax.set_ylim(0, 1.05)

    # 4. Learning curves (test acc) for each noise level
    ax = axes[1, 1]
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(all_results)))
    for r, color in zip(all_results, colors):
        if r["test_acc"]:
            ax.plot(r["steps"], r["test_acc"],
                    label=f"noise={r['noise_frac']:.2f}", color=color, alpha=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Learning Curves by Noise Level")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"noise_sweep_p{p}.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime",        type=int,   default=113)
    parser.add_argument("--train_frac",   type=float, default=0.3)
    parser.add_argument("--steps",        type=int,   default=80_000)
    parser.add_argument("--d_model",      type=int,   default=128)
    parser.add_argument("--n_heads",      type=int,   default=4)
    parser.add_argument("--d_mlp",        type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--wd",           type=float, default=1.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_seeds",      type=int,   default=3,
                        help="Number of seeds to run per noise level")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--use_wandb",    action="store_true")
    parser.add_argument("--out_dir",      type=str,   default="results/exp2_label_noise")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_cfg = TransformerConfig(
        vocab_size=args.prime + 1,
        n_ctx=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=1,
    )

    train_cfg = TrainConfig(
        n_steps=args.steps,
        lr=args.lr,
        weight_decay=args.wd,
        log_every=200,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project="grokking-geography",
    )

    LARGE_KEYS = {"fourier_spectrum", "steps", "test_acc", "train_acc", "fourier_alignment"}
    curves_path  = os.path.join(args.out_dir, f"noise_sweep_curves_p{args.prime}.json")
    summary_path = os.path.join(args.out_dir, f"noise_sweep_p{args.prime}.json")

    # Load any prior results so we can resume after a timeout
    all_results = []
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            all_results = json.load(f)
        done = {(r["noise_frac"], r["seed"]) for r in all_results}
        print(f"Resuming — {len(all_results)} results already saved.")
    else:
        done = set()

    for noise in args.noise_levels:
        print(f"\nNoise level: {noise:.2f}")
        for seed in range(args.seed, args.seed + args.n_seeds):
            if (noise, seed) in done:
                print(f"  noise={noise:.2f} seed={seed}: already done, skipping")
                continue
            result = run_noise_experiment(
                noise_frac=noise,
                p=args.prime,
                train_frac=args.train_frac,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                seed=seed,
            )
            all_results.append(result)
            done.add((noise, seed))

            # Save incrementally after every run
            with open(curves_path, "w") as f:
                json.dump(all_results, f, indent=2)
            with open(summary_path, "w") as f:
                json.dump([{k: v for k, v in r.items() if k not in LARGE_KEYS}
                           for r in all_results], f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Curves saved to:  {curves_path}")

    # Average over seeds for plotting
    from collections import defaultdict
    by_noise = defaultdict(list)
    for r in all_results:
        by_noise[r["noise_frac"]].append(r)

    avg_results = []
    for noise, runs in sorted(by_noise.items()):
        # Use the shortest run length so steps and curves stay aligned.
        # Different seeds can stop at different points (early stopping on grokking),
        # so we truncate all series to the shortest one to avoid shape mismatches.
        min_len = min(len(r["test_acc"]) for r in runs)
        ref_run = min(runs, key=lambda r: len(r["test_acc"]))  # the run that determines length
        avg = {
            "noise_frac":          noise,
            "grokked":             any(r["grokked"] for r in runs),
            "final_test_acc":      np.mean([r["final_test_acc"] for r in runs]),
            "final_fourier_score": np.mean([r["final_fourier_score"] for r in runs]),
            "grok_delay":          (np.mean([r["grok_delay"] for r in runs if r["grok_delay"]])
                                    if any(r["grok_delay"] for r in runs) else None),
            "steps":               ref_run["steps"][:min_len],
            "test_acc":            [np.mean([r["test_acc"][i] for r in runs])
                                    for i in range(min_len)],
            "fourier_alignment":   [np.mean([r["fourier_alignment"][i] for r in runs])
                                    for i in range(min_len)],
        }
        avg_results.append(avg)

    plot_results(avg_results, args.out_dir, args.prime)

    print("\n=== SUMMARY ===")
    print(f"{'Noise':<8} {'Grokked':<10} {'Grok Delay':<12} {'Test Acc':<12} {'Fourier Score'}")
    print("-" * 55)
    for r in avg_results:
        print(f"{r['noise_frac']:<8.2f} {str(r['grokked']):<10} "
              f"{str(r['grok_delay']):<12} {r['final_test_acc']:<12.3f} {r['final_fourier_score']:.3f}")


if __name__ == "__main__":
    main()
