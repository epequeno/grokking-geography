"""
Boolean / binary string tasks.
XOR, parity. These have known Fourier structures and serve as
comparison points for understanding whether grokking is a Fourier phenomenon.
"""

from .registry import TaskConfig, TaskDataset, register_task


@register_task("xor")
class XORTask:
    """
    f(a, b) = a XOR b (bitwise, treated as integers)
    For n-bit XOR: vocab_size = 2^n
    """
    def __init__(self, n_bits: int = 6, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.n_bits = n_bits
        vocab = 2 ** n_bits
        self.cfg = TaskConfig(
            name=f"xor_{n_bits}bit",
            vocab_size=vocab,
            n_ctx=3,
            eq_token=vocab,
            is_commutative=True,
            group_order=vocab,
            operation="xor",
            description=f"{n_bits}-bit XOR"
        )
        self.dataset = TaskDataset(self.cfg, lambda a, b: a ^ b, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return 2 ** self.n_bits + 1


@register_task("parity")
class ParityTask:
    """
    f(a, b) = parity(a) XOR parity(b)  [i.e. (popcount(a) + popcount(b)) mod 2]
    Binary output. Simple but classic grokking task.
    """
    def __init__(self, n_bits: int = 8, train_frac: float = 0.3, noise_frac: float = 0.0, seed: int = 42):
        self.n_bits = n_bits
        vocab = 2 ** n_bits
        def fn(a, b):
            return (bin(a).count('1') + bin(b).count('1')) % 2
        self.cfg = TaskConfig(
            name=f"parity_{n_bits}bit",
            vocab_size=vocab,
            n_ctx=3,
            eq_token=vocab,
            is_commutative=True,
            group_order=2,
            operation="parity",
            description=f"{n_bits}-bit parity"
        )
        self.dataset = TaskDataset(self.cfg, fn, train_frac, noise_frac, seed)

    @property
    def transformer_vocab_size(self):
        return 2 ** self.n_bits + 1
