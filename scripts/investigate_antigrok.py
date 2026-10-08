"""
Investigate anti-grokking in S5 2-layer seed 46.

The depth ablation results show this run grokked at step 71,500 then collapsed 
to 3.4% test / 7.4% train by training end. The trainer's early stopping gives 
only 2000 steps post-grok, so the collapse happened very fast.

This script re-runs the same configuration with:
  - No early stopping (full 120K steps)
  - Dense logging (every 100 steps instead of 500)
  - Full training curves saved
  - Weight norms tracked for norm-ratio analysis

Usage:
    cd grokking-geography  # from the parent directory of this repo
    uv run python scripts/investigate_antigrok.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import torch
import numpy as np

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.groups import S5Composition
from src.training.trainer import GrokTrainer, TrainConfig, RunMetrics


def run_no_early_stop():
    """Re-run S5 2L seed 46 with no early stopping and dense logging."""
    seed = 46
    n_layers = 2
    n_steps = 120_000
    
    task = S5Composition(train_frac=0.3)
    
    model_cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3,
        d_model=128,
        n_heads=4,
        d_head=32,
        d_mlp=512,
        n_layers=n_layers,
    )
    
    torch.manual_seed(seed)
    model = GrokTransformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"S5 2-layer, seed {seed}, {n_params:,} params, {n_steps:,} steps")
    print(f"Logging every 100 steps, NO early stopping\n")
    
    train_cfg = TrainConfig(
        n_steps=n_steps,
        lr=1e-3,
        weight_decay=1.0,
        log_every=100,
        seed=seed,
        run_name=f"antigrok_s5_L2_s{seed}",
    )
    
    # Patch the trainer to disable early stopping
    trainer = GrokTrainer(model, task, train_cfg)
    
    # Monkey-patch: override train() to remove early stopping
    original_train = trainer.train
    
    def train_no_early_stop(callback=None):
        """Modified train loop without early stopping."""
        metrics = RunMetrics()
        cfg = trainer.cfg
        start_time = time.time()
        
        trainer.model.train()
        from tqdm import tqdm
        import torch.nn.functional as F
        
        # Track anti-grokking
        peak_test_acc = 0.0
        peak_test_step = 0
        grok_lost_step = None  # step where test acc drops below threshold after grokking
        
        pbar = tqdm(range(cfg.n_steps), desc="Training (no early stop)")
        
        for step in pbar:
            seqs, lbls = trainer._sample_batch()
            logits = trainer.model(seqs)
            loss = F.cross_entropy(logits, lbls)
            
            trainer.opt.zero_grad()
            loss.backward()
            trainer.opt.step()
            
            if step % cfg.log_every == 0:
                train_acc, test_acc, train_loss, test_loss = trainer.evaluate()
                wn = trainer.model.weight_norms()
                
                W_E = trainer.model.embed.W_E.detach().cpu()
                from src.analysis.fourier import fourier_alignment_score
                fa_score = fourier_alignment_score(W_E, task.cfg.vocab_size)
                
                metrics.steps.append(step)
                metrics.train_acc.append(train_acc)
                metrics.test_acc.append(test_acc)
                metrics.train_loss.append(train_loss)
                metrics.test_loss.append(test_loss)
                metrics.fourier_alignment.append(fa_score)
                for k, v in wn.items():
                    metrics.weight_norms.setdefault(k, []).append(v)
                
                # Detect memorization
                if metrics.mem_step is None and train_acc >= cfg.grok_threshold:
                    metrics.mem_step = step
                
                # Detect grokking
                if metrics.grok_step is None and test_acc >= cfg.grok_threshold:
                    metrics.grok_step = step
                    if metrics.mem_step is not None:
                        metrics.grok_delay = step - metrics.mem_step
                
                # Track anti-grokking
                if test_acc > peak_test_acc:
                    peak_test_acc = test_acc
                    peak_test_step = step
                
                if (metrics.grok_step is not None 
                    and grok_lost_step is None 
                    and test_acc < cfg.grok_threshold
                    and step > metrics.grok_step + 500):
                    grok_lost_step = step
                    print(f"\n⚠️  ANTI-GROKKING DETECTED at step {step}!")
                    print(f"   Grokked at step {metrics.grok_step}, lost at step {step}")
                    print(f"   Peak test acc: {peak_test_acc:.4f} at step {peak_test_step}")
                    print(f"   Current test acc: {test_acc:.4f}")
                
                pbar.set_postfix({
                    "tr": f"{train_acc:.3f}",
                    "te": f"{test_acc:.3f}",
                    "fourier": f"{fa_score:.3f}",
                    "peak_te": f"{peak_test_acc:.3f}",
                })
        
        metrics.elapsed_seconds = time.time() - start_time
        
        # Add anti-grokking metadata
        extra = {
            "peak_test_acc": peak_test_acc,
            "peak_test_step": peak_test_step,
            "grok_lost_step": grok_lost_step,
            "anti_grokking": grok_lost_step is not None,
        }
        
        return metrics, extra
    
    t0 = time.time()
    metrics, extra = train_no_early_stop()
    elapsed = time.time() - t0
    
    # Report
    print(f"\n{'='*60}")
    print(f"RESULTS: S5 2-layer seed {seed}")
    print(f"{'='*60}")
    print(f"Memorisation step:  {metrics.mem_step}")
    print(f"Grokking step:      {metrics.grok_step}")
    print(f"Peak test accuracy: {extra['peak_test_acc']:.4f} at step {extra['peak_test_step']}")
    print(f"Final test accuracy:{metrics.test_acc[-1]:.4f}")
    print(f"Final train accuracy:{metrics.train_acc[-1]:.4f}")
    print(f"Anti-grokking:      {extra['anti_grokking']}")
    if extra['grok_lost_step']:
        print(f"Grok lost at step:  {extra['grok_lost_step']}")
        print(f"Collapse window:    {extra['grok_lost_step'] - metrics.grok_step} steps")
    print(f"Elapsed:            {elapsed:.0f}s")
    
    # Save full curves
    out_dir = "results/exp5_antigrok_investigation"
    os.makedirs(out_dir, exist_ok=True)
    
    result = {
        **metrics.to_dict(),
        **extra,
        "task": "s5_compose",
        "n_layers": n_layers,
        "seed": seed,
        "n_steps": n_steps,
        "log_every": 100,
        "early_stopping": False,
    }
    
    out_path = os.path.join(out_dir, f"antigrok_s5_L2_s{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    run_no_early_stop()
