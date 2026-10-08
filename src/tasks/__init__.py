"""
Task registry for grokking geography experiments.

Each task is defined by:
  - A function f(a, b) -> c
  - A vocabulary (the inputs/outputs are integers in [0, vocab_size))
  - A sequence format for the transformer input

Standard format: [a, b, =_token] -> predict c at position 2
(Matching Nanda 2023)
"""

from .registry import get_task, list_tasks, TaskDataset
from .modular import ModularAddition, ModularMultiplication, ModularExponentiation, ModularDivision
from .groups import S5Composition, DihedralComposition
from .boolean import XORTask, ParityTask

__all__ = [
    "get_task", "list_tasks", "TaskDataset",
    "ModularAddition", "ModularMultiplication", "ModularExponentiation", "ModularDivision",
    "S5Composition", "DihedralComposition",
    "XORTask", "ParityTask",
]
