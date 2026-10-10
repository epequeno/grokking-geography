"""
Activation patching for causal interventions on GrokTransformer.

Activation patching answers the question: "If I replace the activations at
layer L of a *source* model's forward pass with the activations from a
*target* model's forward pass (on the same or different input), how does the
output change?"

This is the primary tool for establishing *causal* claims in mechanistic
interpretability:

  "Is layer L responsible for grokking?"
  → patch post-grokking activations into pre-grokking model;
    if accuracy improves to post-grokking levels, the answer is yes.

  "Does freezing the embedding prevent the Fourier circuit from forming?"
  → patch a grokked model's embedding activations into the frozen model;
    if accuracy jumps, the embedding is the bottleneck.

Two patching modes:
  - activation_patch: replace a layer's output with values from a different
    forward pass (different model weights OR different input tokens).
  - path_patch: more fine-grained — patch specific attention head outputs or
    MLP outputs rather than the full residual stream. (For future use.)

Usage:
    # Compare pre- and post-grokking checkpoints on the same input
    from src.analysis.activation_patch import ActivationPatcher

    patcher = ActivationPatcher(model_pre, model_post)
    result  = patcher.patch_layer(tokens, patch_layer=0, position=-1)
    print(result.patched_acc)   # accuracy after patching layer 0
    print(result.baseline_acc)  # accuracy of model_pre without patching
    print(result.clean_acc)     # accuracy of model_post (upper bound)
"""

import torch
from dataclasses import dataclass
from typing import Optional, List
import matplotlib.pyplot as plt


@dataclass
class PatchResult:
    """
    Results from a single patching experiment.

    Attributes:
        patch_layer:    which layer was patched (None = no patching)
        position:       which sequence position was patched
        baseline_acc:   accuracy of the corrupted/source model without patching
        clean_acc:      accuracy of the target model (upper bound)
        patched_acc:    accuracy of source model after patching target's activations
        patched_logits: [batch, vocab] logits after patching
        recovery:       (patched_acc - baseline_acc) / (clean_acc - baseline_acc + 1e-8)
                        1.0 = full recovery, 0.0 = no effect
    """
    patch_layer:    Optional[int]
    position:       int
    baseline_acc:   float
    clean_acc:      float
    patched_acc:    float
    patched_logits: torch.Tensor
    recovery:       float


class ActivationPatcher:
    """
    Patches activations from a *clean* (target) model's forward pass into a
    *corrupted* (source) model's forward pass and measures the effect on output.

    Typical usage:
        - source = pre-grokking checkpoint (corrupted / low accuracy)
        - target = post-grokking checkpoint (clean / high accuracy)
        → Identifies which layer, when patched, most restores accuracy.

    The patching is done via PyTorch forward hooks — no model modification needed.
    """

    def __init__(self, source_model, target_model):
        """
        Args:
            source_model: GrokTransformer whose activations will be partially replaced.
            target_model: GrokTransformer providing the replacement activations.
        """
        self.source = source_model
        self.target = target_model
        self.source.eval()
        self.target.eval()

    @torch.no_grad()
    def _get_target_cache(self, tokens: torch.Tensor) -> dict:
        """Run the target model and collect residual stream states per layer."""
        _, cache = self.target(tokens, return_cache=True)
        return cache

    @torch.no_grad()
    def patch_layer(
        self,
        tokens:       torch.Tensor,
        correct_labels: torch.Tensor,
        patch_layer:  int,
        position:     int = -1,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> PatchResult:
        """
        Patch the residual stream at `patch_layer` (after its block's computation)
        with values from the target model's forward pass.

        Args:
            tokens:         [batch, seq] input tokens for the source model
            correct_labels: [batch] correct output token indices
            patch_layer:    which layer's output to replace (0-indexed)
            position:       sequence position to patch (-1 = last / prediction position)
            target_tokens:  tokens for target model (defaults to same as tokens)

        Returns:
            PatchResult with baseline, clean, and patched accuracies.
        """
        if target_tokens is None:
            target_tokens = tokens

        # --- Baseline: source model, no patching ---
        baseline_logits = self.source(tokens)   # [batch, vocab]
        baseline_acc    = _accuracy(baseline_logits, correct_labels)

        # --- Clean: target model, no patching ---
        clean_logits = self.target(target_tokens)
        clean_acc    = _accuracy(clean_logits, correct_labels)

        # --- Get target activations at patch_layer ---
        target_cache = self._get_target_cache(target_tokens)
        patch_values = target_cache[f"block_{patch_layer}_post"]  # [batch, seq, d_model]

        # --- Patch: run source model, replacing block_{patch_layer} output ---
        patched_logits = self._run_with_patch(tokens, patch_layer, patch_values, position)
        patched_acc    = _accuracy(patched_logits, correct_labels)

        recovery = (patched_acc - baseline_acc) / (clean_acc - baseline_acc + 1e-8)

        return PatchResult(
            patch_layer=patch_layer,
            position=position,
            baseline_acc=baseline_acc,
            clean_acc=clean_acc,
            patched_acc=patched_acc,
            patched_logits=patched_logits,
            recovery=float(recovery),
        )

    @torch.no_grad()
    def patch_all_layers(
        self,
        tokens:         torch.Tensor,
        correct_labels: torch.Tensor,
        position:       int = -1,
        target_tokens:  Optional[torch.Tensor] = None,
    ) -> List[PatchResult]:
        """
        Patch each layer in turn and return a list of PatchResults.
        Identifies which layer has the highest causal influence on the output.
        """
        n_layers = self.source.cfg.n_layers
        return [
            self.patch_layer(tokens, correct_labels, layer, position, target_tokens)
            for layer in range(n_layers)
        ]

    @torch.no_grad()
    def patch_embedding(
        self,
        tokens:         torch.Tensor,
        correct_labels: torch.Tensor,
        target_tokens:  Optional[torch.Tensor] = None,
    ) -> PatchResult:
        """
        Patch the embedding output (before any transformer block) with values
        from the target model. Tests whether the embedding representations
        alone are sufficient to recover the target model's accuracy.
        """
        if target_tokens is None:
            target_tokens = tokens

        baseline_logits = self.source(tokens)
        baseline_acc    = _accuracy(baseline_logits, correct_labels)

        clean_logits = self.target(target_tokens)
        clean_acc    = _accuracy(clean_logits, correct_labels)

        # Get target embedding activations
        _, target_cache = self.target(target_tokens, return_cache=True)
        patch_values    = target_cache["embed"]   # [batch, seq, d_model]

        patched_logits = self._run_with_embed_patch(tokens, patch_values)
        patched_acc    = _accuracy(patched_logits, correct_labels)
        recovery = (patched_acc - baseline_acc) / (clean_acc - baseline_acc + 1e-8)

        return PatchResult(
            patch_layer=None,
            position=-1,
            baseline_acc=baseline_acc,
            clean_acc=clean_acc,
            patched_acc=patched_acc,
            patched_logits=patched_logits,
            recovery=float(recovery),
        )

    # ---------------------------------------------------------------------- #
    #  Internal: patching implementations using hooks                         #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _run_with_patch(
        self,
        tokens:       torch.Tensor,
        patch_layer:  int,
        patch_values: torch.Tensor,   # [batch, seq, d_model] — full residual stream
        position:     int,
    ) -> torch.Tensor:
        """Run source model, replacing block_{patch_layer} output with patch_values."""
        model  = self.source
        device = next(model.parameters()).device
        tokens = tokens.to(device)
        patch_values = patch_values.to(device)

        # Build the forward pass manually, substituting at the right layer
        x = model.embed(tokens) + model.pos_embed(tokens)

        for i, block in enumerate(model.blocks):
            x = block(x)
            if i == patch_layer:
                # Patch: replace the full residual stream at all positions,
                # or just the target position if position is specified.
                if position == -1 or position == x.shape[1] - 1:
                    # Patch only the last position (the prediction position)
                    x = x.clone()
                    x[:, -1, :] = patch_values[:, -1, :].to(device)
                else:
                    x = x.clone()
                    x[:, position, :] = patch_values[:, position, :].to(device)

        logits = model.unembed(x[:, -1, :])
        return logits

    @torch.no_grad()
    def _run_with_embed_patch(
        self,
        tokens:       torch.Tensor,
        patch_values: torch.Tensor,   # [batch, seq, d_model]
    ) -> torch.Tensor:
        """Run source model, replacing the embedding output with patch_values."""
        model  = self.source
        device = next(model.parameters()).device
        tokens = tokens.to(device)
        patch_values = patch_values.to(device)

        # Replace embed output entirely
        x = patch_values.to(device)

        for block in model.blocks:
            x = block(x)

        return model.unembed(x[:, -1, :])


# --------------------------------------------------------------------------- #
#  Visualisation                                                                #
# --------------------------------------------------------------------------- #

def plot_patch_results(
    results: List[PatchResult],
    title:   str = "",
    ax:      Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Bar chart of accuracy recovery at each patched layer.
    Recovery of 1.0 = patching this layer fully restores target accuracy.
    Recovery of 0.0 = patching this layer has no effect.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(results) * 1.2), 4))

    layers    = [r.patch_layer for r in results]
    recovery  = [r.recovery for r in results]
    colors    = ["seagreen" if r >= 0.5 else "steelblue" for r in recovery]

    ax.bar(range(len(results)), recovery, color=colors, edgecolor="white")
    ax.axhline(1.0, color="green",  linestyle="--", alpha=0.6, label="Full recovery (target)")
    ax.axhline(0.0, color="gray",   linestyle="--", alpha=0.4, label="No effect (baseline)")

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("Recovery score")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title(title or "Activation Patch: Recovery per Layer")

    # Annotate baseline and clean acc from first result
    if results:
        r0 = results[0]
        ax.text(0.02, 0.97,
                f"Baseline acc: {r0.baseline_acc:.3f} | Clean acc: {r0.clean_acc:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, color="gray")

    ax.legend(fontsize=9)
    return ax


def plot_patch_accuracy_comparison(
    results: List[PatchResult],
    title:   str = "",
    ax:      Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Three-line plot showing baseline, patched, and clean accuracy per layer.
    More interpretable than recovery when clean_acc is not 1.0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(results) * 1.2), 4))

    xs            = list(range(len(results)))
    patched_accs  = [r.patched_acc  for r in results]
    baseline_acc  = results[0].baseline_acc if results else 0.0
    clean_acc     = results[0].clean_acc    if results else 1.0

    ax.plot(xs, patched_accs, color="steelblue", marker="o", label="Patched accuracy")
    ax.axhline(baseline_acc, color="red",   linestyle="--", alpha=0.7, label=f"Source (baseline) = {baseline_acc:.3f}")
    ax.axhline(clean_acc,    color="green", linestyle="--", alpha=0.7, label=f"Target (clean) = {clean_acc:.3f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{r.patch_layer}" for r in results])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or "Activation Patch: Per-Layer Accuracy")
    ax.legend(fontsize=9)
    return ax


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    return (logits.argmax(dim=-1) == labels).float().mean().item()
