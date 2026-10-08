"""
Task registry and dataset base class.
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


@dataclass
class TaskConfig:
    name: str
    vocab_size: int          # number of distinct input/output symbols
    n_ctx: int               # sequence length fed to transformer
    eq_token: int            # index of the '=' token in vocabulary
    description: str = ""
    # Algebraic properties (for taxonomy analysis)
    is_commutative: bool = False
    group_order: Optional[int] = None
    operation: str = ""


class TaskDataset(Dataset):
    """
    Generates all (a, b, =, label) pairs for a grokking task.
    Supports label noise injection.
    """

    def __init__(
        self,
        task_cfg: TaskConfig,
        fn: Callable,                # fn(a, b) -> c
        train_frac: float = 0.3,
        noise_frac: float = 0.0,     # fraction of training labels to corrupt
        seed: int = 42,
    ):
        self.cfg = task_cfg
        self.fn = fn
        self.noise_frac = noise_frac
        rng = np.random.RandomState(seed)

        p = task_cfg.vocab_size  # shorthand (often a prime)
        # Build full dataset: all (a, b) pairs
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        labels = [fn(a, b) for a, b in all_pairs]

        # Shuffle and split
        idx = rng.permutation(len(all_pairs))
        n_train = int(len(all_pairs) * train_frac)
        train_idx = idx[:n_train]
        test_idx  = idx[n_train:]

        self.train_pairs  = [all_pairs[i] for i in train_idx]
        self.train_labels = [labels[i] for i in train_idx]
        self.test_pairs   = [all_pairs[i] for i in test_idx]
        self.test_labels  = [labels[i] for i in test_idx]

        # Apply label noise to training set
        if noise_frac > 0:
            n_corrupt = int(len(self.train_labels) * noise_frac)
            corrupt_idx = rng.choice(len(self.train_labels), n_corrupt, replace=False)
            for i in corrupt_idx:
                # Replace with a random wrong label
                correct = self.train_labels[i]
                wrong = correct
                while wrong == correct:
                    wrong = int(rng.randint(0, p))
                self.train_labels[i] = wrong

        self._build_tensors()

    def _build_tensors(self):
        eq = self.cfg.eq_token

        def to_tensors(pairs, labels):
            seqs = torch.tensor([[a, b, eq] for a, b in pairs], dtype=torch.long)
            lbls = torch.tensor(labels, dtype=torch.long)
            return seqs, lbls

        self.train_seqs, self.train_lbls = to_tensors(self.train_pairs, self.train_labels)
        self.test_seqs,  self.test_lbls  = to_tensors(self.test_pairs,  self.test_labels)

    def get_train(self):
        return self.train_seqs, self.train_lbls

    def get_test(self):
        return self.test_seqs, self.test_lbls

    def __len__(self):
        return len(self.train_seqs)

    def __getitem__(self, idx):
        return self.train_seqs[idx], self.train_lbls[idx]


# ------------------------------------------------------------------ #
#  Registry                                                           #
# ------------------------------------------------------------------ #

_REGISTRY = {}


def register_task(name):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)


def list_tasks():
    return list(_REGISTRY.keys())
