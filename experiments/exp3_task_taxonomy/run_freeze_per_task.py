"""
Experiment 3 (interventional) — Freeze-one sweep across task types.

For each task and seed:
  1. Run an un-frozen baseline (same seed => same data split and same init).
  2. For each component, rerun with that component frozen at the run's OWN
     memorisation step (TrainConfig.freeze_on_mem), optionally also with the
     weight-decay control arm (--decay_control).

Tasks whose baseline never groks within the budget are skipped for that seed (no
freeze effect can be measured against a baseline that does not grok).

Parity is reported but uninformative: it "groks" at mem_step (delay 0), so there
is no post-memorisation dynamic to intervene on.

Run:
    uv run python experiments/exp3_task_taxonomy/run_freeze_per_task.py --quick
    uv run python experiments/exp3_task_taxonomy/run_freeze_per_task.py --tasks mod_mul s5_compose --n_seeds 3
"""

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np

from src.models.transformer import TransformerConfig, build_model
from src.tasks.modular import ModularAddition, ModularMultiplication
from src.tasks.groups import S5Composition, DihedralComposition
from src.tasks.boolean import XORTask, ParityTask
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager


def build_tasks(seed: int, quick: bool = False):
    """Tasks are rebuilt per seed so the train/test split varies with the seed."""
    p = 97 if quick else 113
    return {
        "mod_add":      ModularAddition(p=p, train_frac=0.3, seed=seed),
        "mod_mul":      ModularMultiplication(p=p, train_frac=0.3, seed=seed),
        "s5_compose":   S5Composition(train_frac=0.3, seed=seed),
        "dihedral_12":  DihedralComposition(n=12, train_frac=0.3, seed=seed),
        "xor_6bit":     XORTask(n_bits=6, train_frac=0.3, seed=seed),
        "parity_8bit":  ParityTask(n_bits=8, train_frac=0.3, seed=seed),
    }


COMPONENTS = [
    "mlp_all", "embedding", "unembedding",
    "attn_all", "attn_Q", "attn_K", "attn_V", "attn_O",
    "mlp_in", "mlp_out",
]


def run_one(task_name, task, component, decay, model_cfg, train_cfg, seed):
    """component == 'baseline' runs without any freeze."""
    model = build_model(model_cfg, seed)
    fm = FreezeManager(model)

    cfg = deepcopy(train_cfg)
    cfg.run_name = f"{task_name}_{component}{'_decay' if decay else ''}_s{seed}"
    if component != "baseline":
        cfg.freeze_on_mem = [component]
        cfg.frozen_params_decay = decay

    t0 = time.time()
    m = GrokTrainer(model, task, cfg, freeze_manager=fm).train()

    return {
        "task":           task_name,
        "component":      component,
        "decay_control":  decay,
        "seed":           seed,
        "grokked":        m.grok_step is not None,
        "grok_step":      m.grok_step,
        "mem_step":       m.mem_step,
        "grok_delay":     m.grok_delay,
        "peak_test_acc":  m.peak_test_acc,
        "final_test":     m.test_acc[-1]  if m.test_acc  else 0.0,
        "final_train":    m.train_acc[-1] if m.train_acc else 0.0,
        "freeze_events":  m.freeze_events,
        "elapsed":        time.time() - t0,
    }


def _key(r):
    return (r["task"], r["component"], r["decay_control"], r["seed"])


def summarize(results, tasks):
    print(f"\n{'='*92}")
    print("FINAL SUMMARY (paired: delay ratio = frozen delay / same-seed baseline delay)")
    print(f"{'='*92}")
    print(f"{'Task':<13}{'Component':<13}{'Decay':<7}{'Grok':<7}{'Paired ratio':<18}{'Peak test':<11}{'Final test'}")
    print("-" * 92)
    base = {(r["task"], r["seed"]): r for r in results if r["component"] == "baseline"}
    for t in tasks:
        for comp in COMPONENTS:
            for decay in (False, True):
                runs = [r for r in results if r["task"] == t and r["component"] == comp
                        and r["decay_control"] == decay]
                if not runs:
                    continue
                n = sum(r["grokked"] for r in runs)
                ratios = [r["grok_delay"] / base[(t, r["seed"])]["grok_delay"] for r in runs
                          if r["grok_delay"] is not None and (t, r["seed"]) in base
                          and base[(t, r["seed"])]["grok_delay"]]
                rr = f"{np.mean(ratios):.2f}x (n={len(ratios)})" if ratios else "—"
                print(f"{t:<13}{comp:<13}{'yes' if decay else 'no':<7}{n}/{len(runs):<5}{rr:<18}"
                      f"{np.mean([r['peak_test_acc'] for r in runs]):<11.3f}"
                      f"{np.mean([r['final_test'] for r in runs]):.3f}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",    nargs="+",  default=None, help="Subset of tasks (default: all)")
    parser.add_argument("--quick",    action="store_true", help="Reduced steps / smaller primes")
    parser.add_argument("--steps",    type=int,   default=120_000)
    parser.add_argument("--n_seeds",  type=int,   default=3)
    parser.add_argument("--d_model",  type=int,   default=128)
    parser.add_argument("--n_heads",  type=int,   default=4)
    parser.add_argument("--d_mlp",    type=int,   default=512)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--wd",       type=float, default=1.0)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--decay_control", action="store_true",
                        help="Also run each freeze with weight decay still applied to frozen params")
    parser.add_argument("--out_dir",  type=str,   default="results/exp3_freeze_per_task")
    args = parser.parse_args()

    if args.quick:
        args.steps = 20_000

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.n_seeds))
    out_path = os.path.join(args.out_dir, f"freeze_per_task_s{args.seed}_n{args.n_seeds}.json")

    results, done = [], set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        done = {_key(r) for r in results}
        print(f"Resuming — {len(results)} results already saved.")

    def save():
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    task_names = None
    for seed in seeds:
        all_tasks = build_tasks(seed, quick=args.quick)
        tasks = {k: v for k, v in all_tasks.items() if args.tasks is None or k in args.tasks}
        task_names = list(tasks)

        for name, task in tasks.items():
            model_cfg = TransformerConfig(
                vocab_size=task.transformer_vocab_size, n_ctx=3, d_model=args.d_model,
                n_heads=args.n_heads, d_head=args.d_model // args.n_heads,
                d_mlp=args.d_mlp, n_layers=1,
            )
            train_cfg = TrainConfig(
                n_steps=args.steps, lr=args.lr, weight_decay=args.wd,
                log_every=500, seed=seed,
            )
            print(f"\n=== {name} seed {seed} (vocab={task.transformer_vocab_size}, "
                  f"group={task.cfg.is_group}, commutative={task.cfg.is_commutative}) ===")

            bkey = (name, "baseline", False, seed)
            if bkey in done:
                baseline = next(r for r in results if _key(r) == bkey)
            else:
                baseline = run_one(name, task, "baseline", False, model_cfg, train_cfg, seed)
                results.append(baseline); done.add(bkey); save()
            print(f"  baseline: grokked={baseline['grokked']} mem={baseline['mem_step']} "
                  f"grok={baseline['grok_step']} peak_test={baseline['peak_test_acc']:.3f}")

            if not baseline["grokked"]:
                print("  baseline did not grok within budget — skipping freeze sweep for this seed.")
                continue

            for comp in COMPONENTS:
                for decay in ([False, True] if args.decay_control else [False]):
                    k = (name, comp, decay, seed)
                    if k in done:
                        continue
                    r = run_one(name, task, comp, decay, model_cfg, train_cfg, seed)
                    results.append(r); done.add(k); save()
                    print(f"  {comp:<12} decay={str(decay):<5} grokked={str(r['grokked']):<5} "
                          f"delay={r['grok_delay']} peak_test={r['peak_test_acc']:.3f}")

    summarize(results, task_names or [])
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
