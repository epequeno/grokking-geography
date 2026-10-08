"""
Non-abelian group tasks.
S_5 permutation composition, dihedral group D_n.
"""

import itertools
import numpy as np
from .registry import TaskConfig, TaskDataset, register_task


# ------------------------------------------------------------------ #
#  S5 permutation group                                               #
# ------------------------------------------------------------------ #

def _generate_s5():
    """Generate all 120 permutations of {0,1,2,3,4}."""
    return list(itertools.permutations(range(5)))


def _compose_perms(sigma, tau):
    """(sigma ∘ tau)(i) = sigma(tau(i))"""
    return tuple(sigma[tau[i]] for i in range(len(sigma)))


@register_task("s5_compose")
class S5Composition:
    """
    S_5 permutation composition.
    |S_5| = 120. Non-abelian. Good negative control for modular addition.
    Used in the March 2026 geometric inductive bias paper as negative control.
    Vocab: 120 permutations + eq_token (120)
    """
    def __init__(self, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        perms = _generate_s5()
        self.perm_to_idx = {p: i for i, p in enumerate(perms)}
        self.idx_to_perm = perms
        n = len(perms)  # 120

        self.cfg = TaskConfig(
            name="s5_compose",
            vocab_size=n,
            n_ctx=3,
            eq_token=n,
            is_commutative=False,
            group_order=120,
            operation="compose",
        )

        def fn(a, b):
            pa = self.idx_to_perm[a]
            pb = self.idx_to_perm[b]
            result = _compose_perms(pa, pb)
            return self.perm_to_idx[result]

        self.dataset = TaskDataset(self.cfg, fn, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return 121  # 120 + eq_token


# ------------------------------------------------------------------ #
#  Dihedral group D_n                                                 #
# ------------------------------------------------------------------ #

@register_task("dihedral")
class DihedralComposition:
    """
    Dihedral group D_n (symmetries of regular n-gon).
    |D_n| = 2n. Elements: (rotation r, reflection s) where r in [0,n), s in {0,1}.
    Multiplication: depends on n.
    Non-abelian for n >= 3.
    """
    def __init__(self, n: int = 12, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.n = n
        size = 2 * n
        # Encode element (r, s) as index: r + s*n
        def encode(r, s): return r + s * n
        def decode(idx): return idx % n, idx // n

        def multiply(idx_a, idx_b):
            r1, s1 = decode(idx_a)
            r2, s2 = decode(idx_b)
            # D_n multiplication: (r1,s1)*(r2,s2)
            if s1 == 0:
                r_out = (r1 + r2) % n
                s_out = s2
            else:
                r_out = (r1 - r2) % n
                s_out = 1 - s2
            return encode(r_out, s_out)

        self.cfg = TaskConfig(
            name=f"dihedral_n{n}",
            vocab_size=size,
            n_ctx=3,
            eq_token=size,
            is_commutative=False,
            group_order=size,
            operation="dihedral_mul",
        )

        self.dataset = TaskDataset(self.cfg, multiply, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return 2 * self.n + 1
