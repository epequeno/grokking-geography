"""
Logit lens analysis for GrokTransformer.

The logit lens (Nostalgebraist 2020, adopted in Nanda et al.) projects
each layer's residual stream state through the final unembedding matrix
to see what the model "predicts" at that layer before the computation is
complete. This reveals how early (and at which layer) the model commits
to the correct answer.

For grokking analysis:
- Pre-grokking: early layers commit to memorized answers via a direct
  lookup circuit. The model "knows" the answer from layer 0.
- Post-grokking: early layers show dispersed predictions; commitment to
  the correct answer builds gradually across layers as the Fourier circuit
  computes the modular sum. The final layer sharpens to the correct answer.

This qualitative shift in the logit lens trajectory is a mechanistic
signature of the circuit change.

Usage:
    lens = LogitLens(model)

    # Analyse a single batch
    result = lens.run(tokens)      # shape info in LogitLensResult docstring

    # Track how commitment builds across layers over training
    scores = lens.commitment_score(tokens, correct_labels)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LogitLensResult:
    """
    Holds the per-layer logit lens outputs for a batch.

    Attributes:
        layer_logits:  list of length n_layers, each a [batch, vocab] tensor
                       of logits projected from that layer's residual stream
        layer_probs:   same but softmax'd
        final_logits:  [batch, vocab] logits from the true final layer
        n_layers:      number of transformer layers
        vocab_size:    vocabulary size
    """
    layer_logits: list        # list[Tensor[batch, vocab]]
    layer_probs:  list        # list[Tensor[batch, vocab]]
    final_logits: torch.Tensor
    n_layers:     int
    vocab_size:   int

    def top_k_tokens(self, layer: int, k: int = 5) -> torch.Tensor:
        """Return the top-k token indices at the given layer. Shape: [batch, k]."""
        return self.layer_probs[layer].topk(k, dim=-1).indices

    def correct_token_prob(self, layer: int, correct_labels: torch.Tensor) -> torch.Tensor:
        """
        Probability assigned to the correct answer at the given layer.
        correct_labels: [batch] int tensor of ground-truth token indices.
        Returns: [batch] float tensor.
        """
        probs = self.layer_probs[layer]           # [batch, vocab]
        return probs.gather(1, correct_labels.unsqueeze(1)).squeeze(1)

    def commitment_trajectory(self, correct_labels: torch.Tensor) -> torch.Tensor:
        """
        Probability of the correct answer at each layer, averaged over the batch.
        Returns: [n_layers] tensor — the "commitment curve" across layers.
        """
        return torch.stack([
            self.correct_token_prob(i, correct_labels).mean()
            for i in range(self.n_layers)
        ])


class LogitLens:
    """
    Applies the logit lens to a GrokTransformer at each residual stream position.

    The lens works by projecting intermediate residual stream states at the
    *final sequence position* (the prediction position) through the model's
    unembedding matrix (W_U). This doesn't require any hook infrastructure —
    the model's `return_cache=True` mode already saves the residual stream at
    each layer.
    """

    def __init__(self, model):
        """
        Args:
            model: GrokTransformer instance (any number of layers).
        """
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def run(
        self,
        tokens: torch.Tensor,
        position: int = -1,
    ) -> LogitLensResult:
        """
        Run the logit lens on a batch of token sequences.

        Args:
            tokens:   [batch, seq_len] int tensor
            position: which sequence position to analyse (default -1 = last)

        Returns:
            LogitLensResult with per-layer logits and probabilities.
        """
        _, cache = self.model(tokens, return_cache=True)

        W_U = self.model.unembed.W_U   # [d_model, vocab]
        b_U = self.model.unembed.b_U   # [vocab]
        n_layers = self.model.cfg.n_layers

        layer_logits = []
        layer_probs  = []

        for i in range(n_layers):
            h = cache[f"block_{i}_post"][:, position, :]   # [batch, d_model]
            logits = h @ W_U + b_U                          # [batch, vocab]
            layer_logits.append(logits)
            layer_probs.append(F.softmax(logits, dim=-1))

        final_logits = cache["logits"]

        return LogitLensResult(
            layer_logits=layer_logits,
            layer_probs=layer_probs,
            final_logits=final_logits,
            n_layers=n_layers,
            vocab_size=self.model.cfg.vocab_size,
        )

    @torch.no_grad()
    def commitment_score(
        self,
        tokens: torch.Tensor,
        correct_labels: torch.Tensor,
        position: int = -1,
    ) -> float:
        """
        Scalar summary: mean probability of the correct answer at the final layer,
        as a measure of how confidently the model has committed to the right answer
        by that layer.

        Returns a float in [0, 1]. Higher is more committed.
        """
        result = self.run(tokens, position=position)
        return result.correct_token_prob(self.model.cfg.n_layers - 1, correct_labels).mean().item()

    @torch.no_grad()
    def layer_accuracy(
        self,
        tokens: torch.Tensor,
        correct_labels: torch.Tensor,
        position: int = -1,
    ) -> list:
        """
        Accuracy (top-1 correct) at each layer.
        Returns a list of floats of length n_layers.
        """
        result = self.run(tokens, position=position)
        accs = []
        for i in range(result.n_layers):
            pred = result.layer_probs[i].argmax(dim=-1)
            acc = (pred == correct_labels).float().mean().item()
            accs.append(acc)
        return accs


# --------------------------------------------------------------------------- #
#  Visualisation helpers                                                        #
# --------------------------------------------------------------------------- #

def plot_logit_lens_trajectory(
    result: LogitLensResult,
    correct_labels: torch.Tensor,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the commitment trajectory: probability of the correct answer at each layer,
    averaged over the batch. One line per example is shown faintly; the mean is bold.

    Expected shape:
    - Pre-grokking: flat high curve (model commits from layer 0 via memorisation)
    - Post-grokking: S-shaped curve rising in later layers (Fourier circuit computes
      the answer progressively, committing only at the final layer)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    n_layers = result.n_layers
    layers   = list(range(n_layers))

    # Per-example curves (faint)
    batch_size = result.layer_probs[0].shape[0]
    for b in range(min(batch_size, 20)):
        curve = [result.correct_token_prob(i, correct_labels)[b].item() for i in range(n_layers)]
        ax.plot(layers, curve, color="steelblue", alpha=0.15, linewidth=0.8)

    # Mean curve (bold)
    mean_curve = result.commitment_trajectory(correct_labels).cpu().numpy()
    ax.plot(layers, mean_curve, color="steelblue", linewidth=2.5, label="Mean P(correct)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(correct answer)")
    ax.set_title(title or "Logit Lens: Commitment Trajectory")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.legend()
    return ax


def plot_logit_lens_heatmap(
    result: LogitLensResult,
    correct_labels: torch.Tensor,
    example_idx: int = 0,
    top_k: int = 10,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Heatmap of top-k token probabilities across layers for a single example.
    Rows = layers, columns = top-k tokens (by final-layer probability).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(3, result.n_layers)))

    # Use final-layer top-k as the column set
    final_probs = result.layer_probs[-1][example_idx]           # [vocab]
    top_tokens  = final_probs.topk(top_k).indices.cpu().numpy() # [top_k]

    # Build matrix [n_layers, top_k]
    matrix = np.zeros((result.n_layers, top_k))
    for layer in range(result.n_layers):
        probs_layer = result.layer_probs[layer][example_idx].cpu().numpy()
        matrix[layer] = probs_layer[top_tokens]

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_yticks(range(result.n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(result.n_layers)])
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([str(t) for t in top_tokens], rotation=45, ha="right")

    correct = correct_labels[example_idx].item()
    ax.set_title(title or f"Logit Lens Heatmap (example {example_idx}, correct={correct})")
    return ax
