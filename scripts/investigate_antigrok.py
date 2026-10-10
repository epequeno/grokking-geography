"""
Investigate anti-grokking in S5 2-layer seed 46.

The depth ablation showed this run grokking at step 71,500 and then ending at
3.4% test / 7.4% train. The trainer's default early stop only collects 2000
steps post-grok, so the collapse was not observed directly.

This script re-runs the same configuration with:
  - No early stopping (full 120K steps; stop_after_grok=None)
  - Dense logging (every 100 steps)
  - Peak test accuracy and the step where grokking is lost (RunMetrics)
  - Weight norms tracked for norm-ratio analysis

Usage:
    uv run python scripts/investigate_antigrok.py
"""

import json
import os

from src.models.transformer import TransformerConfig, build_model
from src.tasks.groups import S5Composition
from src.training.trainer import GrokTrainer, TrainConfig

SEED = 46
N_LAYERS = 2
N_STEPS = 120_000
LOG_EVERY = 100


def run_no_early_stop():
    task = S5Composition(train_frac=0.3, seed=SEED)

    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=N_LAYERS,
    )
    model = build_model(model_cfg, SEED)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"S5 {N_LAYERS}-layer, seed {SEED}, {n_params:,} params, {N_STEPS:,} steps")
    print(f"Logging every {LOG_EVERY} steps, NO early stopping\n")

    train_cfg = TrainConfig(
        n_steps=N_STEPS, lr=1e-3, weight_decay=1.0, log_every=LOG_EVERY, seed=SEED,
        stop_after_grok=None,
        run_name=f"antigrok_s5_L{N_LAYERS}_s{SEED}",
    )
    metrics = GrokTrainer(model, task, train_cfg).train()

    print(f"\n{'='*60}\nRESULTS: S5 {N_LAYERS}-layer seed {SEED}\n{'='*60}")
    print(f"Memorisation step:   {metrics.mem_step}")
    print(f"Grokking step:       {metrics.grok_step}")
    print(f"Peak test accuracy:  {metrics.peak_test_acc:.4f} at step {metrics.peak_test_step}")
    print(f"Final test accuracy: {metrics.test_acc[-1]:.4f}")
    print(f"Final train accuracy:{metrics.train_acc[-1]:.4f}")
    print(f"Anti-grokking:       {metrics.grok_lost_step is not None}")
    if metrics.grok_lost_step is not None:
        print(f"Grok lost at step:   {metrics.grok_lost_step} "
              f"({metrics.grok_lost_step - metrics.grok_step} steps after grokking)")
    print(f"Elapsed:             {metrics.elapsed_seconds:.0f}s")

    out_dir = "results/exp5_antigrok_investigation"
    os.makedirs(out_dir, exist_ok=True)
    result = {
        **metrics.to_dict(),
        "task": "s5_compose",
        "n_layers": N_LAYERS,
        "seed": SEED,
        "n_steps": N_STEPS,
        "log_every": LOG_EVERY,
        "early_stopping": False,
    }
    out_path = os.path.join(out_dir, f"antigrok_s5_L{N_LAYERS}_s{SEED}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    run_no_early_stop()
