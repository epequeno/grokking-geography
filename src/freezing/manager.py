"""
FreezeManager — surgical component freezing API.

Uses zero-gradient hooks to freeze parameters while keeping them in the
optimizer's parameter group. This preserves Adam/AdamW first- and second-order
moment estimates for ALL parameters (frozen and unfrozen), eliminating the
confound of optimizer state resets that would occur if we rebuilt the optimizer.

Usage:
    fm = FreezeManager(model)

    # Freeze everything except MLP
    fm.freeze_all()
    fm.unfreeze("mlp_all")

    # At step 500, freeze attention Q/K
    fm.freeze("attn_Q")
    fm.freeze("attn_K")

    # Restore everything
    fm.unfreeze_all()

Implementation note (2026-03-25):
    Previous implementation set requires_grad=False and rebuilt the optimizer,
    which reset Adam moments for ALL parameters — a confounding variable.
    New implementation uses register_hook() to zero gradients, so the optimizer
    still runs its update rule on frozen params but the zero gradient means
    only the weight decay term applies. To fully freeze (no weight decay either),
    we also cache and restore the parameter data after each optimizer step.
    See GrokTrainer._step_with_freeze() for the restore logic.
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict


class FreezeManager:
    """
    Manages gradient flow for named component groups in GrokTransformer.

    Design:
    - Freezing registers a zero-gradient hook on each parameter and records
      a snapshot of the parameter data. The parameter stays in the optimizer
      (requires_grad remains True) so Adam moments are never disrupted.
    - After each optimizer.step(), the trainer calls restore_frozen() to
      overwrite any weight-decay drift on frozen params with the snapshot.
    - Unfreezing removes the hook and discards the snapshot, allowing
      normal gradient flow and updates to resume with intact optimizer state.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.groups = model.component_groups()
        self._frozen: set = set()
        # Map from parameter id -> hook handle (for removal on unfreeze)
        self._hooks: Dict[int, torch.utils.hooks.RemovableHook] = {}
        # Map from parameter id -> frozen snapshot of param data
        self._snapshots: Dict[int, torch.Tensor] = {}
        # Build reverse map: param id -> param reference (for restore)
        self._param_by_id: Dict[int, nn.Parameter] = {}
        for group_params in self.groups.values():
            for p in group_params:
                self._param_by_id[id(p)] = p

    def _get_params(self, component: str) -> list:
        if component not in self.groups:
            available = list(self.groups.keys())
            raise ValueError(f"Unknown component '{component}'. Available: {available}")
        return self.groups[component]

    @staticmethod
    def _zero_grad_hook(grad):
        """Hook that replaces the gradient with zeros."""
        return torch.zeros_like(grad)

    def freeze(self, component: str):
        """Freeze a named component group using zero-gradient hooks."""
        if component in self._frozen:
            return
        for p in self._get_params(component):
            pid = id(p)
            if pid not in self._hooks:
                # Snapshot current param data so we can restore after optimizer step
                self._snapshots[pid] = p.data.clone()
                # Register hook to zero out gradients
                handle = p.register_hook(self._zero_grad_hook)
                self._hooks[pid] = handle
        self._frozen.add(component)
        print(f"[FreezeManager] Froze: {component}")

    def unfreeze(self, component: str):
        """Unfreeze a named component group by removing hooks."""
        if component not in self._frozen:
            return
        for p in self._get_params(component):
            pid = id(p)
            if pid in self._hooks:
                self._hooks[pid].remove()
                del self._hooks[pid]
            # Discard snapshot — parameter is free to update normally
            self._snapshots.pop(pid, None)
        self._frozen.discard(component)
        print(f"[FreezeManager] Unfroze: {component}")

    def restore_frozen(self, decay: float = 0.0):
        """
        Reset frozen parameters after optimizer.step().

        Call this AFTER optimizer.step(). The optimizer still maintains valid
        moment estimates (they just see zero gradients), but momentum and
        AdamW's decoupled weight decay would otherwise keep moving frozen
        weights.

        Args:
            decay: if 0 (default) frozen params are restored exactly to their
                freeze-time snapshot (no gradient AND no weight decay).
                If > 0 it is the per-step AdamW shrink factor (lr * weight_decay):
                the snapshot is shrunk by (1 - decay) each call, so frozen
                params receive no gradient updates but still feel weight decay.
                This is the control arm that separates "gradient updates to this
                component are required" from "this component must be exempt from
                weight decay".
        """
        for pid, snapshot in self._snapshots.items():
            if decay:
                snapshot.mul_(1.0 - decay)
            self._param_by_id[pid].data.copy_(snapshot)

    def freeze_all(self):
        """Freeze all components."""
        for name in self.groups:
            self.freeze(name)

    def unfreeze_all(self):
        """Unfreeze all components."""
        for name in list(self._frozen):
            self.unfreeze(name)

    def freeze_except(self, keep_unfrozen: Union[str, List[str]]):
        """
        Freeze everything EXCEPT the specified component(s).
        Useful for the 'minimal sufficient update' experiment.
        """
        if isinstance(keep_unfrozen, str):
            keep_unfrozen = [keep_unfrozen]
        self.freeze_all()
        for name in keep_unfrozen:
            self.unfreeze(name)

    def status(self) -> dict:
        """Return current freeze status of all components."""
        return {name: (name in self._frozen) for name in self.groups}

    def frozen_components(self) -> list:
        return sorted(self._frozen)

    def has_frozen(self) -> bool:
        """Return True if any component is currently frozen."""
        return len(self._frozen) > 0

    def trainable_params(self) -> list:
        """Return only the currently trainable (unfrozen) parameters."""
        frozen_ids = set(self._snapshots.keys())
        return [p for p in self.model.parameters()
                if id(p) not in frozen_ids]

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_params())
