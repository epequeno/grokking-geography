"""
Fourier analysis of learned representations.

For modular arithmetic tasks over Z_p, grokking is associated with the model
learning Fourier features (Nanda et al. 2023). This module provides tools to:

1. Compute the Fourier transform of the embedding matrix
2. Measure how "Fourier-aligned" the embeddings are (a progress metric)
3. Identify which frequencies dominate (key frequencies = {w_k = 2π*k/p})
4. Track Fourier alignment over training as a grokking progress measure

Key insight from Nanda 2023:
  The embedding matrix W_E learns a map: token a -> (cos(w*a), sin(w*a)) for 
  a small set of frequencies w. This circular structure enables modular addition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def fourier_basis(p: int) -> torch.Tensor:
    """
    Returns the Fourier basis matrix F of shape [p, p].
    F[k, a] = (1/sqrt(p)) * e^{2πi*k*a/p}
    
    In real form: rows alternate cos and sin for k=1...(p-1)//2, plus DC and Nyquist.
    Returns real-valued matrix of shape [p, p].
    """
    freqs = torch.arange(p, dtype=torch.float32)
    F = torch.zeros(p, p)
    F[0] = 1.0 / p**0.5  # DC component

    for k in range(1, (p + 1) // 2):
        F[2*k - 1] = torch.cos(2 * torch.pi * k * freqs / p) * (2/p)**0.5
        if 2*k < p:
            F[2*k] = torch.sin(2 * torch.pi * k * freqs / p) * (2/p)**0.5

    if p % 2 == 0:
        F[-1] = torch.cos(torch.pi * freqs) / p**0.5  # Nyquist

    return F


def fourier_transform_embedding(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """
    Project embedding matrix W_E [p, d_model] into Fourier basis.
    Returns F @ W_E of shape [p, d_model], where each row is a Fourier mode.
    """
    F = fourier_basis(p).to(W_E.device)
    return F @ W_E[:p]  # only use the first p rows (exclude eq_token)


def fourier_power_spectrum(W_E: torch.Tensor, p: int) -> torch.Tensor:
    """
    Compute the power in each Fourier frequency for the embedding matrix.
    Returns a [p]-dimensional vector where entry k is the total power
    (sum of squares) across all d_model dimensions for frequency k.
    
    This is the key diagnostic: a grokked model has most power concentrated
    in a few frequencies.
    """
    F_W = fourier_transform_embedding(W_E, p)
    return (F_W ** 2).sum(dim=-1)  # [p]


def fourier_concentration(W_E: torch.Tensor, p: int, top_k: int = 5) -> float:
    """
    Measure how concentrated the Fourier spectrum is.
    Returns the fraction of total power in the top-k frequencies.
    
    High concentration (close to 1.0) indicates the model has learned
    structured Fourier features — a proxy for grokking progress.
    """
    spectrum = fourier_power_spectrum(W_E, p)
    total = spectrum.sum().item()
    if total < 1e-10:
        return 0.0
    top_k_power = spectrum.topk(top_k).values.sum().item()
    return top_k_power / total


def dominant_frequencies(W_E: torch.Tensor, p: int, top_k: int = 5) -> list:
    """
    Return the top-k dominant Fourier frequencies (as k values).
    These correspond to the "key frequencies" in Nanda 2023.
    """
    spectrum = fourier_power_spectrum(W_E, p)
    top_idx = spectrum.topk(top_k).indices.tolist()
    # Convert Fourier row indices to frequency k values
    # Row 0 = DC, rows 2k-1 and 2k = frequency k (cos/sin pair)
    freqs = []
    for idx in top_idx:
        if idx == 0:
            freqs.append(0)
        elif idx % 2 == 1:
            freqs.append((idx + 1) // 2)
        else:
            freqs.append(idx // 2)
    return sorted(set(freqs))


def plot_fourier_spectrum(W_E: torch.Tensor, p: int, title: str = "", ax=None):
    """Plot the Fourier power spectrum of the embedding matrix."""
    spectrum = fourier_power_spectrum(W_E, p).cpu().numpy()
    freq_idx = np.arange(len(spectrum))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(freq_idx, spectrum, color="steelblue", alpha=0.8)
    ax.set_xlabel("Fourier frequency index")
    ax.set_ylabel("Power (sum of squares)")
    ax.set_title(title or "Fourier Spectrum of Embedding Matrix")
    return ax


# ------------------------------------------------------------------ #
#  Progress measure: Restricted Loss (Nanda 2023 proxy)              #
# ------------------------------------------------------------------ #

def fourier_alignment_score(W_E: torch.Tensor, p: int) -> float:
    """
    Scalar progress measure: fraction of total embedding Fourier power held by
    the 6 strongest Fourier modes (3 cos/sin pairs when the top modes pair up).

    Near `chance_alignment(p, d_model)` for random embeddings (NOT 0), rising
    toward 1 for fully Fourier-structured embeddings. Always compare a measured
    score against that chance floor before calling it structure.

    Only meaningful when token index == residue (mod_add): the DFT is taken over
    token ids.
    """
    top_k = min(6, p // 2)
    return fourier_concentration(W_E, p, top_k=top_k)


def chance_alignment(p: int, d_model: int, n_draws: int = 20, seed: int = 0) -> float:
    """
    Expected fourier_alignment_score of an unstructured embedding: mean over
    `n_draws` i.i.d. Gaussian [p, d_model] matrices (deterministic given `seed`).
    """
    g = torch.Generator().manual_seed(seed)
    scores = [
        fourier_alignment_score(torch.randn(p, d_model, generator=g), p)
        for _ in range(n_draws)
    ]
    return float(sum(scores) / len(scores))


def key_frequency_alignment(
    W_E: torch.Tensor,
    p: int,
    key_freqs: list,
) -> float:
    """
    Nanda-style restricted-loss equivalent: fraction of total embedding power
    concentrated in a *known* set of key frequencies.

    Unlike fourier_alignment_score (which finds the top-k modes dynamically),
    this measures power in pre-specified frequencies — exactly the "restricted
    loss" concept from Nanda et al. 2023. Because the key frequencies are fixed,
    this score can rise *before* the behavioral transition if the model begins
    building circuit structure in those frequencies while still memorising.

    Args:
        W_E:       embedding matrix [p, d_model]
        p:         number of tokens (prime for mod-add)
        key_freqs: list of integer frequency indices k (e.g. [16, 49, 52]
                   for mod-add p=113). Each k contributes two Fourier modes
                   (cos at row 2k-1, sin at row 2k).

    Returns:
        Scalar in [0, 1]: higher = more power in the key frequencies.
    """
    spectrum = fourier_power_spectrum(W_E, p)
    total = spectrum.sum().item()
    if total < 1e-10:
        return 0.0

    key_power = 0.0
    for k in key_freqs:
        if k == 0:
            key_power += spectrum[0].item()
        else:
            cos_idx = 2 * k - 1
            sin_idx = 2 * k
            if cos_idx < len(spectrum):
                key_power += spectrum[cos_idx].item()
            if sin_idx < len(spectrum):
                key_power += spectrum[sin_idx].item()

    return key_power / total
