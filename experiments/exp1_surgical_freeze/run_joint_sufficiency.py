"""
Experiment 1c — Minimal Sufficient Set.

Having established:
  - freeze_one:        only MLP is necessary
  - freeze_all_except: no single component is sufficient alone
                       (MLP alone reaches 12% test; embedding alone 0.1%)

This experiment asks: what is the minimal *joint* set of components that is
sufficient for grokking?

Combinations tested (freeze everything EXCEPT these, from step 0):
  A. [mlp_all, embedding]        — primary prediction: the minimal sufficient pair
  B. [mlp_all, attn_all]         — control: nonlinearity + routing, no Fourier basis
  C. [embedding, attn_all]       — control: Fourier basis + routing, no nonlinearity
  D. [mlp_all, embedding, attn_all] — does attention accelerate the minimal pair?
  E. [mlp_all, embedding, unembedding] — does adding unembedding help?

Predictions:
  A groks (possibly slowly)
  B fails — MLP needs Fourier-structured input
  C fails — need MLP nonlinearity for trig identities
  D groks at similar speed to A (attention already converged by step 700 in baseline)
  E groks, possibly faster than A (unembedding can adapt to the circuit)

Run:
    uv run python experiments/exp1_surgical_freeze/run_joint_sufficiency.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, argparse, time
import torch
from copy import deepcopy

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager


COMBINATIONS = [
    # Primary prediction: minimal sufficient pair
    ("mlp+embed",           ["mlp_all", "embedding"]),
    # Controls: missing one key ingredient each
    ("mlp+attn",            ["mlp_all", "attn_all"]),
    ("embed+attn",          ["embedding", "attn_all"]),
    # Does adding attention to the sufficient pair help?
    ("mlp+embed+attn",      ["mlp_all", "embedding", "attn_all"]),
    # Does adding unembedding help?
    ("mlp+embed+unembed",   ["mlp_all", "embedding", "unembedding"]),
]

# Per-combination step budgets. Controls (expected to fail) get fewer steps
# since failure is clear early. Positive cases get the full budget.
STEP_BUDGETS = {
    "mlp+embed":          80_000,
    "mlp+attn":           20_000,   # will fail quickly — save time
    "embed+attn":         20_000,   # will fail quickly
    "mlp+embed+attn":     40_000,   # should grok faster than mlp+embed
    "mlp+embed+unembed":  40_000,
}


def run_combination(name, keep_unfrozen, task, train_cfg, seed):
    torch.manual_seed(seed)

    # Rebuild model_cfg from task (always same architecture)
    from src.models.transformer import TransformerConfig
    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )
    model = GrokTransformer(model_cfg)
    fm    = FreezeManager(model)

    cfg = deepcopy(train_cfg)
    cfg.run_name = f"joint_{name}_s{seed}"

    trainer = GrokTrainer(model, task, cfg, freeze_manager=fm)
    fm.freeze_except(keep_unfrozen)   # freeze everything else from step 0

    t0 = time.time()
    metrics = trainer.train()

    return {
        "name":           name,
        "keep_unfrozen":  keep_unfrozen,
        "seed":           seed,
        "grokked":        metrics.grok_step is not None,
        "grok_step":      metrics.grok_step,
        "mem_step":       metrics.mem_step,
        "grok_delay":     metrics.grok_delay,
        "final_train_acc": metrics.train_acc[-1] if metrics.train_acc else 0.0,
        "final_test_acc":  metrics.test_acc[-1]  if metrics.test_acc  else 0.0,
        "elapsed_seconds": time.time() - t0,
        # Save learning curves for visualisation
        "steps":          metrics.steps,
        "train_acc":      metrics.train_acc,
        "test_acc":       metrics.test_acc,
        "fourier_alignment": metrics.fourier_alignment,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime",     type=int,   default=113)
    parser.add_argument("--train_frac",type=float, default=0.3)
    parser.add_argument("--steps",     type=int,   default=80_000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--wd",        type=float, default=1.0)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--log_every", type=int,   default=200)
    parser.add_argument("--out_dir",   type=str,   default="results/exp1_joint_sufficiency")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    task = ModularAddition(p=args.prime, train_frac=args.train_frac, seed=args.seed)
    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )
    train_cfg_base = TrainConfig(
        n_steps=args.steps, lr=args.lr, weight_decay=args.wd,
        log_every=args.log_every, seed=args.seed,
    )

    print(f"Joint sufficiency sweep — p={args.prime}, {len(COMBINATIONS)} combinations\n")
    print(f"{'Combination':<24} {'Keep unfrozen':<35} {'Grokked':<9} {'Grok step':<12} {'Test acc'}")
    print("-" * 90)

    all_results = []
    for name, keep in COMBINATIONS:
        print(f"{name:<24} {str(keep):<35} ", end="", flush=True)
        budget = STEP_BUDGETS.get(name, args.steps)
        cfg_for_run = deepcopy(train_cfg_base)
        cfg_for_run.n_steps = budget
        r = run_combination(name, keep, task, cfg_for_run, args.seed)
        all_results.append(r)
        print(f"{str(r['grokked']):<9} {str(r['grok_step']):<12} {r['final_test_acc']:.4f}")

    # Save (strip large arrays for the compact summary)
    summary_path = os.path.join(args.out_dir, f"joint_sufficiency_p{args.prime}_s{args.seed}.json")
    summary = [{k: v for k, v in r.items()
                if k not in ("steps", "train_acc", "test_acc", "fourier_alignment")}
               for r in all_results]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results (with curves) separately
    curves_path = os.path.join(args.out_dir, f"joint_sufficiency_curves_p{args.prime}_s{args.seed}.json")
    with open(curves_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Curves saved to:  {curves_path}")

    print("\n=== SUMMARY ===")
    for r in all_results:
        status = f"grok @ {r['grok_step']}" if r['grokked'] else "NEVER"
        print(f"  {r['name']:<24} keep={r['keep_unfrozen']}  →  {status}  "
              f"(train={r['final_train_acc']:.1%}, test={r['final_test_acc']:.1%})")


if __name__ == "__main__":
    main()
