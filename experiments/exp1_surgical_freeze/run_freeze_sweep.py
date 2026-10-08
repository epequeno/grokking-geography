"""
Experiment 1 — Surgical Freeze Sweep.

For each component, freeze it at the memorization plateau and measure:
  - Does grokking still occur?
  - If yes: how much is the delay changed?
  - If no: is this component necessary for grokking?

Also runs "minimal sufficient update" experiments:
  - Freeze ALL components except one; measure if grokking occurs.

This is the core novel contribution of Experiment 1.

Run:
    # Freeze sweep (freeze one component at memorization step)
    python experiments/exp1_surgical_freeze/run_freeze_sweep.py --mode freeze_one

    # Minimal sufficient update (only one component can update)
    python experiments/exp1_surgical_freeze/run_freeze_sweep.py --mode freeze_all_except
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import json
import torch
from copy import deepcopy

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager
from src.results import save_result_batch, config_hash


COMPONENTS_TO_TEST = [
    "embedding",
    "unembedding",
    "attn_Q",
    "attn_K",
    "attn_V",
    "attn_O",
    "attn_all",
    "mlp_in",
    "mlp_out",
    "mlp_all",
]


def run_single_experiment(
    model_cfg: TransformerConfig,
    task: ModularAddition,
    freeze_at_step: int,
    freeze_action: str,      # "freeze_one" or "freeze_all_except"
    target_component: str,
    train_cfg_base: TrainConfig,
    out_dir: str,
) -> dict:
    """
    Run one freeze experiment.
    
    freeze_action == "freeze_one":
        - Train normally until freeze_at_step
        - Freeze target_component at that step
        - Continue training; measure if grokking occurs
    
    freeze_action == "freeze_all_except":
        - Freeze everything EXCEPT target_component from step 0
        - Measure if grokking occurs with only this component updating
    """
    model = GrokTransformer(model_cfg)
    fm = FreezeManager(model)

    if freeze_action == "freeze_one":
        schedule = [(freeze_at_step, "freeze", target_component)]
    elif freeze_action == "freeze_all_except":
        # No freeze schedule entries needed — fm.freeze_except(target_component) is
        # called directly before training starts (see below), which freezes everything
        # except the target from step 0. Zero-gradient hooks ensure frozen params
        # receive no updates while optimizer state (Adam moments) stays intact.
        schedule = []
    else:
        raise ValueError(f"Unknown freeze_action: {freeze_action}")

    cfg = deepcopy(train_cfg_base)
    cfg.freeze_schedule = schedule
    cfg.run_name = f"{freeze_action}_{target_component}_@{freeze_at_step}"

    trainer = GrokTrainer(model, task, cfg, freeze_manager=fm)

    if freeze_action == "freeze_all_except":
        # Apply freeze_except before training starts
        fm.freeze_except(target_component)

    metrics = trainer.train()

    result = {
        "freeze_action": freeze_action,
        "target_component": target_component,
        "freeze_at_step": freeze_at_step,
        "grok_step": metrics.grok_step,
        "mem_step": metrics.mem_step,
        "grok_delay": metrics.grok_delay,
        "grokked": metrics.grok_step is not None,
        "final_test_acc": metrics.test_acc[-1] if metrics.test_acc else 0.0,
        "final_train_acc": metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "elapsed_seconds": metrics.elapsed_seconds,
    }

    print(f"\n  [{freeze_action}] {target_component:20s} | "
          f"grokked={result['grokked']} | grok_step={result['grok_step']} | "
          f"delay={result['grok_delay']}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime",         type=int,   default=113)
    parser.add_argument("--train_frac",    type=float, default=0.3)
    parser.add_argument("--steps",         type=int,   default=80_000)
    parser.add_argument("--d_model",       type=int,   default=128)
    parser.add_argument("--n_heads",       type=int,   default=4)
    parser.add_argument("--d_mlp",         type=int,   default=512)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--wd",            type=float, default=1.0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--freeze_step",   type=int,   default=-1,
                        help="Step to apply freeze. -1 = auto-detect memorization step from baseline")
    parser.add_argument("--mode",          type=str,   default="freeze_one",
                        choices=["freeze_one", "freeze_all_except", "both"])
    parser.add_argument("--baseline_json", type=str,   default="",
                        help="Path to baseline results JSON to get freeze_step from mem_step")
    parser.add_argument("--n_seeds",       type=int,   default=5,
                        help="Number of seeds to run per component (default: 5)")
    parser.add_argument("--out_dir",       type=str,   default="results/exp1_freeze_sweep")
    parser.add_argument("--use_wandb",     action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load baseline to get memorization step if not specified
    freeze_step = args.freeze_step
    if freeze_step < 0 and args.baseline_json:
        with open(args.baseline_json) as f:
            baseline = json.load(f)
        freeze_step = baseline.get("mem_step", 5000)
        print(f"Using memorization step from baseline: {freeze_step}")
    elif freeze_step < 0:
        freeze_step = 5000
        print(f"Using default freeze step: {freeze_step}")

    # Task + model config (shared across all experiments)
    task = ModularAddition(p=args.prime, train_frac=args.train_frac, seed=args.seed)

    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=1,
    )

    train_cfg_base = TrainConfig(
        n_steps=args.steps,
        lr=args.lr,
        weight_decay=args.wd,
        log_every=200,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project="grokking-geography",
    )

    modes = ["freeze_one", "freeze_all_except"] if args.mode == "both" else [args.mode]
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    all_results = []
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}  ({args.n_seeds} seeds: {seeds})")
        print(f"{'='*60}")

        mode_results = []
        for seed in seeds:
            # Rebuild task and train config per seed for reproducibility
            task_s = ModularAddition(p=args.prime, train_frac=args.train_frac, seed=seed)
            cfg_s = deepcopy(train_cfg_base)
            cfg_s.seed = seed

            for comp in COMPONENTS_TO_TEST:
                result = run_single_experiment(
                    model_cfg=model_cfg,
                    task=task_s,
                    freeze_at_step=freeze_step,
                    freeze_action=mode,
                    target_component=comp,
                    train_cfg_base=cfg_s,
                    out_dir=args.out_dir,
                )
                result["seed"] = seed
                mode_results.append(result)
                all_results.append(result)

        # Save each mode as a separate immutable artifact
        sweep_config = {
            "experiment": "exp1_freeze_sweep",
            "mode": mode,
            "prime": args.prime,
            "seeds": seeds,
            "n_seeds": args.n_seeds,
            "freeze_step": freeze_step,
            "steps": args.steps,
            "lr": args.lr,
            "wd": args.wd,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_mlp": args.d_mlp,
            "components": COMPONENTS_TO_TEST,
        }
        save_result_batch(
            experiment="exp1_freeze_sweep",
            results=mode_results,
            config=sweep_config,
            run_name=f"{mode}_p{args.prime}_s{args.seed}",
        )

    # Also save the combined legacy format for backward compatibility
    out_path = os.path.join(args.out_dir, f"freeze_sweep_p{args.prime}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table (aggregated over seeds)
    print(f"\n{'='*75}")
    print(f"RESULTS SUMMARY (aggregated over {args.n_seeds} seeds)")
    print(f"{'='*75}")
    print(f"{'Mode':<20} {'Component':<15} {'Grok Rate':<12} {'Mean Delay':<14} {'Mean Test Acc'}")
    print("-" * 80)

    from collections import defaultdict
    import numpy as np

    by_key = defaultdict(list)
    for r in all_results:
        by_key[(r['freeze_action'], r['target_component'])].append(r)

    for (mode, comp), runs in sorted(by_key.items()):
        n_grokked = sum(1 for r in runs if r['grokked'])
        grok_rate = f"{n_grokked}/{len(runs)}"
        delays = [r['grok_delay'] for r in runs if r['grok_delay'] is not None]
        mean_delay = f"{np.mean(delays):.0f} ± {np.std(delays):.0f}" if delays else "NEVER"
        mean_acc = np.mean([r['final_test_acc'] for r in runs])
        print(f"{mode:<20} {comp:<15} {grok_rate:<12} {mean_delay:<14} {mean_acc:.4f}")

    print(f"\nResults saved to: {out_path}")
    print(f"Immutable artifacts saved to: results/exp1_freeze_sweep/")


if __name__ == "__main__":
    main()
