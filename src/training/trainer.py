"""
Training loop for grokking experiments.

Key features:
  - Dense metric logging (every N steps): train acc, test acc, weight norms per component
  - Fourier alignment score logged as a continuous grokking progress measure
  - Freeze schedule support: trigger freeze/unfreeze at specific steps
  - Grokking detection: auto-detect the grokking step (when test acc crosses threshold)
  - WandB integration (optional)
  - Deterministic seeding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple
from tqdm import tqdm
import time

from src.analysis.fourier import fourier_alignment_score


@dataclass
class TrainConfig:
    n_steps: int = 50_000
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1.0        # Strong WD is key for grokking (Nanda 2023)
    optimizer: str = "adamw"
    log_every: int = 100             # log metrics every N steps
    grok_threshold: float = 0.99    # test acc threshold to call "grokked"
    device: str = "auto"            # "auto", "cuda", "mps", "cpu"
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "grokking-geography"
    run_name: str = ""
    # Freeze schedule: list of (step, action, component)
    # action: "freeze" or "unfreeze"
    # e.g. [(5000, "freeze", "attn_Q"), (10000, "unfreeze", "attn_Q")]
    freeze_schedule: List[Tuple[int, str, str]] = field(default_factory=list)
    # Components to freeze at THIS run's own memorisation step (the first logged
    # step with train_acc >= grok_threshold). Per-run, so no separate baseline
    # run is needed and the freeze lands where the intervention is meaningful.
    # Takes effect from the step after the one at which mem_step was detected.
    freeze_on_mem: List[str] = field(default_factory=list)
    # False (default): frozen params are held exactly constant (no gradient, no
    # weight decay). True: frozen params receive no gradient updates but still
    # decay each step by lr*weight_decay. Comparing the two separates "this
    # component's gradient updates are required" from "this component must be
    # exempt from weight decay".
    frozen_params_decay: bool = False
    # Stop this many steps after grokking is detected. None = run all n_steps
    # (needed to observe post-grok behaviour such as anti-grokking). Early stop
    # never fires before every scheduled freeze/freeze_on_mem event has happened.
    stop_after_grok: Optional[int] = 2000


@dataclass
class RunMetrics:
    """Container for all logged metrics across training."""
    steps: List[int] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    weight_norms: Dict[str, List[float]] = field(default_factory=dict)
    fourier_alignment: List[float] = field(default_factory=list)  # empty for tasks where the DFT metric is not meaningful
    grok_step: Optional[int] = None          # step where test acc crossed threshold
    mem_step: Optional[int] = None           # step where train acc crossed threshold
    grok_delay: Optional[int] = None         # grok_step - mem_step
    peak_test_acc: float = 0.0               # max test acc seen at any logged step
    peak_test_step: Optional[int] = None
    grok_lost_step: Optional[int] = None     # first logged step after grokking where test acc fell back below threshold
    freeze_events: List[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self):
        return {
            "steps": self.steps,
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "weight_norms": self.weight_norms,
            "fourier_alignment": self.fourier_alignment,
            "grok_step": self.grok_step,
            "mem_step": self.mem_step,
            "grok_delay": self.grok_delay,
            "peak_test_acc": self.peak_test_acc,
            "peak_test_step": self.peak_test_step,
            "grok_lost_step": self.grok_lost_step,
            "anti_grokking": self.grok_lost_step is not None,
            "freeze_events": self.freeze_events,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class GrokTrainer:

    def __init__(self, model: nn.Module, task, cfg: TrainConfig, freeze_manager=None):
        self.model = model
        self.task = task
        self.cfg = cfg
        self.fm = freeze_manager
        self.device = _get_device(cfg.device)

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.model = self.model.to(self.device)

        if cfg.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        # Load data
        self.train_seqs, self.train_lbls = task.dataset.get_train()
        self.test_seqs,  self.test_lbls  = task.dataset.get_test()
        self.train_seqs = self.train_seqs.to(self.device)
        self.train_lbls = self.train_lbls.to(self.device)
        self.test_seqs  = self.test_seqs.to(self.device)
        self.test_lbls  = self.test_lbls.to(self.device)

        # Build freeze schedule as a dict: step -> [(action, component)]
        self._freeze_sched: Dict[int, List] = {}
        for step, action, comp in cfg.freeze_schedule:
            self._freeze_sched.setdefault(step, []).append((action, comp))

        if cfg.use_wandb:
            import wandb
            wandb.init(project=cfg.wandb_project, name=cfg.run_name or None)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float, float]:
        """Returns (train_acc, test_acc, train_loss, test_loss)."""
        self.model.eval()

        def _eval(seqs, lbls):
            logits = self.model(seqs)
            loss = F.cross_entropy(logits, lbls).item()
            acc = (logits.argmax(-1) == lbls).float().mean().item()
            return acc, loss

        train_acc, train_loss = _eval(self.train_seqs, self.train_lbls)
        test_acc,  test_loss  = _eval(self.test_seqs,  self.test_lbls)
        self.model.train()
        return train_acc, test_acc, train_loss, test_loss

    def _sample_batch(self):
        idx = torch.randint(len(self.train_seqs), (self.cfg.batch_size,))
        return self.train_seqs[idx], self.train_lbls[idx]

    def train(self, callback: Optional[Callable] = None) -> RunMetrics:
        """
        Run the full training loop.
        
        Args:
            callback: optional fn(step, metrics) called at each log step.
        """
        metrics = RunMetrics()
        cfg = self.cfg
        start_time = time.time()
        last_sched_step = max(self._freeze_sched, default=-1)
        decay = cfg.lr * cfg.weight_decay if cfg.frozen_params_decay else 0.0
        mem_freeze_due = False

        self.model.train()
        pbar = tqdm(range(cfg.n_steps), desc="Training")

        for step in pbar:
            if mem_freeze_due:
                mem_freeze_due = False
                if self.fm is not None:
                    for comp in cfg.freeze_on_mem:
                        self.fm.freeze(comp)
                        metrics.freeze_events.append({
                            "step": step, "action": "freeze", "component": comp,
                            "trigger": "mem_step",
                        })
            # Apply freeze schedule
            if step in self._freeze_sched:
                for action, comp in self._freeze_sched[step]:
                    if self.fm is not None:
                        if action == "freeze":
                            self.fm.freeze(comp)
                        elif action == "unfreeze":
                            self.fm.unfreeze(comp)
                        metrics.freeze_events.append({
                            "step": step, "action": action, "component": comp
                        })
                    # Zero-gradient hooks keep every parameter (and its Adam
                    # moments) in the optimizer; restore_frozen() after the
                    # optimizer step undoes momentum/weight-decay drift.

            # Forward + backward
            seqs, lbls = self._sample_batch()
            logits = self.model(seqs)
            loss = F.cross_entropy(logits, lbls)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Reset frozen params (undoes momentum / weight-decay drift)
            if self.fm is not None and self.fm.has_frozen():
                self.fm.restore_frozen(decay=decay)

            # Logging
            if step % cfg.log_every == 0:
                train_acc, test_acc, train_loss, test_loss = self.evaluate()
                wn = self.model.weight_norms()

                # Fourier alignment is only defined where token index == residue
                # (mod_add). For other tasks the DFT of arbitrary token ids is noise.
                fa_score = None
                if self.task.cfg.fourier_valid:
                    W_E = self.model.embed.W_E.detach().cpu()
                    fa_score = fourier_alignment_score(W_E, self.task.cfg.vocab_size)

                metrics.steps.append(step)
                metrics.train_acc.append(train_acc)
                metrics.test_acc.append(test_acc)
                metrics.train_loss.append(train_loss)
                metrics.test_loss.append(test_loss)
                if fa_score is not None:
                    metrics.fourier_alignment.append(fa_score)
                for k, v in wn.items():
                    metrics.weight_norms.setdefault(k, []).append(v)

                # Detect memorization step
                if metrics.mem_step is None and train_acc >= cfg.grok_threshold:
                    metrics.mem_step = step
                    if cfg.freeze_on_mem:
                        mem_freeze_due = True

                # Detect grokking step
                if metrics.grok_step is None and test_acc >= cfg.grok_threshold:
                    metrics.grok_step = step
                    if metrics.mem_step is not None:
                        metrics.grok_delay = step - metrics.mem_step

                # Peak test accuracy / loss of generalization after grokking
                if test_acc > metrics.peak_test_acc:
                    metrics.peak_test_acc = test_acc
                    metrics.peak_test_step = step
                if (metrics.grok_step is not None and metrics.grok_lost_step is None
                        and test_acc < cfg.grok_threshold and step > metrics.grok_step):
                    metrics.grok_lost_step = step

                postfix = {
                    "tr_acc": f"{train_acc:.3f}",
                    "te_acc": f"{test_acc:.3f}",
                    "grokked": metrics.grok_step is not None,
                }
                if fa_score is not None:
                    postfix["fourier"] = f"{fa_score:.3f}"
                pbar.set_postfix(postfix)

                if cfg.use_wandb:
                    import wandb
                    log = {
                        "step": step,
                        "train_acc": train_acc, "test_acc": test_acc,
                        "train_loss": train_loss, "test_loss": test_loss,
                        **{f"norm/{k}": v for k, v in wn.items()},
                    }
                    if fa_score is not None:
                        log["fourier_alignment"] = fa_score
                    wandb.log(log)

                if callback:
                    callback(step, metrics)

            # Early stop: grokked and post-grok window collected. Never before
            # every scheduled freeze has fired (otherwise the intervention is skipped).
            if (cfg.stop_after_grok is not None
                    and metrics.grok_step is not None
                    and step >= metrics.grok_step + cfg.stop_after_grok
                    and not mem_freeze_due
                    and step > last_sched_step
                    and (not cfg.freeze_on_mem or metrics.mem_step is not None)):
                print(f"\nEarly stop: grokked at step {metrics.grok_step}, +{cfg.stop_after_grok} steps collected.")
                break

        metrics.elapsed_seconds = time.time() - start_time
        return metrics
