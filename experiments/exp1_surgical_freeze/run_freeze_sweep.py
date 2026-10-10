"""
Experiment 1 — Surgical Freeze Sweep.

For each component, freeze it at the memorization step of THAT run and measure:
  - Does grokking still occur?
  - If yes: how much is the delay changed (paired against the same-seed baseline)?
  - If no: is this component necessary for grokking?

Modes:
  freeze_one         freeze one component at the run's own mem_step
  freeze_one_decay   same, but frozen params still receive weight decay
                     (control: separates "gradient updates needed" from
                     "component must be exempt from weight decay")
  freeze_all_except  freeze everything except one component from step 0
                     (sufficiency; trivially fails for components that cannot
                     memorise alone — read alongside run_joint_sufficiency.py)

Every mode also runs an un-frozen baseline per seed with the identical init and
data, so effects are paired within a seed rather than compared across seeds.

Run:
    uv run python experiments/exp1_surgical_freeze/run_freeze_sweep.py --mode both
"""

import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np

from src.models.transformer import TransformerConfig, build_model
from src.tasks import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager
from src.results import save_result_batch


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

MODES = ["freeze_one", "freeze_one_decay", "freeze_all_except"]


def run_single_experiment(model_cfg, task, mode, target, train_cfg_base) -> dict:
    """
    Run one (mode, component, seed) experiment. target == "baseline" runs no freeze.
    """
    seed = train_cfg_base.seed
    model = build_model(model_cfg, seed)
    fm = FreezeManager(model)

    cfg = deepcopy(train_cfg_base)
    cfg.run_name = f"{mode}_{target}_s{seed}"

    if target == "baseline":
        pass
    elif mode == "freeze_one":
        cfg.freeze_on_mem = [target]
    elif mode == "freeze_one_decay":
        cfg.freeze_on_mem = [target]
        cfg.frozen_params_decay = True
    elif mode == "freeze_all_except":
        # No early stop before the run has had its full budget to find a solution
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    trainer = GrokTrainer(model, task, cfg, freeze_manager=fm)
    if mode == "freeze_all_except" and target != "baseline":
        fm.freeze_except(target)

    metrics = trainer.train()

    result = {
        "freeze_action": mode,
        "target_component": target,
        "seed": seed,
        "grok_step": metrics.grok_step,
        "mem_step": metrics.mem_step,
        "grok_delay": metrics.grok_delay,
        "grokked": metrics.grok_step is not None,
        "freeze_events": metrics.freeze_events,
        "peak_test_acc": metrics.peak_test_acc,
        "final_test_acc": metrics.test_acc[-1] if metrics.test_acc else 0.0,
        "final_train_acc": metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "elapsed_seconds": metrics.elapsed_seconds,
    }

    print(f"\n  [{mode}] {target:12s} s{seed} | grokked={result['grokked']} | "
          f"mem={result['mem_step']} grok={result['grok_step']} | "
          f"peak_test={result['peak_test_acc']:.3f}")
    return result


def summarize(results, seeds):
    print(f"\n{'='*100}")
    print(f"RESULTS SUMMARY ({len(seeds)} seeds; delay ratio = frozen delay / same-seed baseline delay)")
    print(f"{'='*100}")
    print(f"{'Mode':<19}{'Component':<13}{'Grok':<7}{'Mean delay':<16}{'Paired ratio':<16}{'Peak test':<11}{'Final test'}")
    print("-" * 100)

    base = {r["seed"]: r for r in results
            if r["target_component"] == "baseline"}
    by_key = defaultdict(list)
    for r in results:
        if r["target_component"] != "baseline":
            by_key[(r["freeze_action"], r["target_component"])].append(r)

    for (mode, comp), runs in sorted(by_key.items()):
        n = sum(r["grokked"] for r in runs)
        delays = [r["grok_delay"] for r in runs if r["grok_delay"] is not None]
        d = f"{np.mean(delays):.0f} ± {np.std(delays):.0f}" if delays else "NEVER"
        ratios = [r["grok_delay"] / base[r["seed"]]["grok_delay"] for r in runs
                  if r["grok_delay"] is not None
                  and r["seed"] in base and base[r["seed"]]["grok_delay"]]
        rr = f"{np.mean(ratios):.2f}x (n={len(ratios)})" if ratios else "—"
        peak = np.mean([r["peak_test_acc"] for r in runs])
        final = np.mean([r["final_test_acc"] for r in runs])
        print(f"{mode:<19}{comp:<13}{n}/{len(runs):<5}{d:<16}{rr:<16}{peak:<11.3f}{final:.3f}")


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
    parser.add_argument("--mode",          type=str,   default="both",
                        choices=MODES + ["both"],
                        help="'both' = freeze_one + freeze_one_decay (the causal arms)")
    parser.add_argument("--n_seeds",       type=int,   default=5)
    parser.add_argument("--components",    nargs="+",  default=COMPONENTS_TO_TEST)
    parser.add_argument("--out_dir",       type=str,   default="results/exp1_freeze_sweep")
    parser.add_argument("--use_wandb",     action="store_true")
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

    modes = ["freeze_one", "freeze_one_decay"] if args.mode == "both" else [args.mode]
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    all_results = []
    baselines = {}

    for seed in seeds:
        task = ModularAddition(p=args.prime, train_frac=args.train_frac, seed=seed)
        base_cfg = TrainConfig(
            n_steps=args.steps, lr=args.lr, weight_decay=args.wd,
            log_every=200, seed=seed,
            use_wandb=args.use_wandb, wandb_project="grokking-geography",
        )
        # Paired baseline: same seed => same data split and same init
        baselines[seed] = run_single_experiment(model_cfg, task, "freeze_one", "baseline", base_cfg)
        all_results.append(baselines[seed])

        for mode in modes:
            for comp in args.components:
                all_results.append(
                    run_single_experiment(model_cfg, task, mode, comp, base_cfg))

    for mode in modes:
        mode_results = [r for r in all_results
                        if r["freeze_action"] == mode or r["target_component"] == "baseline"]
        save_result_batch(
            experiment="exp1_freeze_sweep",
            results=mode_results,
            config={
                "experiment": "exp1_freeze_sweep",
                "mode": mode,
                "freeze_timing": "per-run mem_step" if mode != "freeze_all_except" else "step 0",
                "prime": args.prime,
                "seeds": seeds,
                "n_seeds": args.n_seeds,
                "steps": args.steps,
                "lr": args.lr,
                "wd": args.wd,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "d_mlp": args.d_mlp,
                "components": args.components,
            },
            run_name=f"{mode}_p{args.prime}_s{args.seed}",
        )

    out_path = os.path.join(args.out_dir, f"freeze_sweep_p{args.prime}_s{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    summarize(all_results, seeds)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
