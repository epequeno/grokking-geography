"""
Modular arithmetic tasks.
vocab: [0, ..., p-1, eq_token]  where eq_token = p
"""

from .registry import TaskConfig, TaskDataset, register_task


def _make_mod_cfg(p: int, name: str, op: str, commutative: bool,
                  is_group: bool = False, fourier_valid: bool = False,
                  group_order=None) -> TaskConfig:
    return TaskConfig(
        name=name,
        vocab_size=p,
        n_ctx=3,           # [a, b, =]
        eq_token=p,        # '=' is token index p
        is_commutative=commutative,
        is_group=is_group,
        group_order=group_order,
        operation=op,
        n_classes=p,
        fourier_valid=fourier_valid,
    )


@register_task("mod_add")
class ModularAddition:
    """
    f(a, b) = (a + b) mod p
    Standard grokking benchmark (Nanda 2023).
    """
    def __init__(self, p: int = 113, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_add", "+", commutative=True,
                                 is_group=True, fourier_valid=True, group_order=p)
        self.dataset = TaskDataset(self.cfg, lambda a, b: (a + b) % p, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return self.p + 1  # symbols [0..p-1] + eq_token


@register_task("mod_mul")
class ModularMultiplication:
    """
    f(a, b) = (a * b) mod p over the multiplicative group Z_p^* (a, b in 1..p-1).
    Pairs with a=0 or b=0 are excluded: the zero row/column is a degenerate
    shortcut that is not part of the group. Token 0 is never used.
    """
    def __init__(self, p: int = 113, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_mul", "*", commutative=True,
                                 is_group=True, group_order=p - 1)
        self.dataset = TaskDataset(self.cfg, lambda a, b: (a * b) % p, train_frac, noise_frac, seed,
                                   valid_pair=lambda a, b: a != 0 and b != 0)

    @property
    def transformer_vocab_size(self):
        return self.p + 1


@register_task("mod_exp")
class ModularExponentiation:
    """
    f(a, b) = (a ** b) mod p
    Not a group operation (non-commutative, non-associative). Much harder.
    """
    def __init__(self, p: int = 97, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_exp", "**", commutative=False)
        # pow(0, 0, p) == 1 in Python, which matches the convention 0**0 = 1
        def fn(a, b):
            return pow(int(a), int(b), p)
        self.dataset = TaskDataset(self.cfg, fn, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return self.p + 1


@register_task("mod_div")
class ModularDivision:
    """
    f(a, b) = (a / b) mod p  [i.e. a * b^{-1} mod p]
    Not a group operation. Undefined for b = 0: those pairs are excluded from
    both train and test (no placeholder label).
    """
    def __init__(self, p: int = 113, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_div", "/", commutative=False)
        def fn(a, b):
            return (a * pow(int(b), p - 2, p)) % p
        self.dataset = TaskDataset(self.cfg, fn, train_frac, noise_frac, seed,
                                   valid_pair=lambda a, b: b != 0)

    @property
    def transformer_vocab_size(self):
        return self.p + 1
