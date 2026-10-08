"""
Modular arithmetic tasks.
vocab: [0, ..., p-1, eq_token]  where eq_token = p
"""

from .registry import TaskConfig, TaskDataset, register_task


def _make_mod_cfg(p: int, name: str, op: str, commutative: bool) -> TaskConfig:
    return TaskConfig(
        name=name,
        vocab_size=p,
        n_ctx=3,           # [a, b, =]
        eq_token=p,        # '=' is token index p
        is_commutative=commutative,
        group_order=p,
        operation=op,
    )


@register_task("mod_add")
class ModularAddition:
    """
    f(a, b) = (a + b) mod p
    Standard grokking benchmark (Nanda 2023).
    """
    def __init__(self, p: int = 113, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_add", "+", commutative=True)
        self.dataset = TaskDataset(self.cfg, lambda a, b: (a + b) % p, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return self.p + 1  # symbols [0..p-1] + eq_token


@register_task("mod_mul")
class ModularMultiplication:
    """
    f(a, b) = (a * b) mod p
    Note: 0 is degenerate; typically exclude a=0, b=0.
    """
    def __init__(self, p: int = 113, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_mul", "*", commutative=True)
        self.dataset = TaskDataset(self.cfg, lambda a, b: (a * b) % p, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return self.p + 1


@register_task("mod_exp")
class ModularExponentiation:
    """
    f(a, b) = (a ** b) mod p
    Non-commutative. Much harder — good test for grokking geography.
    """
    def __init__(self, p: int = 97, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_exp", "**", commutative=False)
        # Avoid 0**0 ambiguity; define 0**0 = 1
        def fn(a, b):
            if a == 0 and b == 0:
                return 1
            return pow(int(a), int(b), p)
        self.dataset = TaskDataset(self.cfg, fn, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return self.p + 1


@register_task("mod_div")
class ModularDivision:
    """
    f(a, b) = (a / b) mod p  [i.e. a * b^{-1} mod p]
    Only defined for b != 0. Skip b=0 cases.
    """
    def __init__(self, p: int = 113, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.p = p
        self.cfg = _make_mod_cfg(p, "mod_div", "/", commutative=False)
        def fn(a, b):
            if b == 0:
                return 0  # undefined; label 0 (excluded from eval)
            return (a * pow(int(b), p - 2, p)) % p
        self.dataset = TaskDataset(self.cfg, fn, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return self.p + 1
