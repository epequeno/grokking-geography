"""
Figure 1 — Baseline Grokking on Modular Addition.

Three-panel figure:
  (a) Training dynamics: train/test accuracy + Fourier alignment over time,
      with three phases annotated.
  (b) Fourier power spectrum at convergence.
  (c) Component weight norms over time.

Run:
    uv run python notebooks/fig1_baseline.py
    uv run python notebooks/fig1_baseline.py --json results/exp1_baseline/baseline_p113_s42.json
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import torch

from pub_style import apply_style, PAL, save_fig, add_phase_bands, add_phase_vlines, thousands_formatter, label_panel
from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.analysis.fourier import fourier_power_spectrum, dominant_frequencies


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def retrain_for_spectrum(p, seed, n_steps):
    """Quick re-train to get final model weights for spectrum analysis."""
    torch.manual_seed(seed)
    task = ModularAddition(p=p, train_frac=0.3, seed=seed)
    cfg = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )
    model = GrokTransformer(cfg)
    train_cfg = TrainConfig(n_steps=n_steps, lr=1e-3, weight_decay=1.0,
                            log_every=500, seed=seed)
    print(f"  Re-training ({n_steps} steps) for spectrum…", flush=True)
    GrokTrainer(model, task, train_cfg).train()
    return model, task


# ── Panel (a): Training dynamics ─────────────────────────────────────────────

def panel_dynamics(ax, m):
    steps = m["steps"]
    mem_step = m["mem_step"]
    grok_step = m["grok_step"]

    add_phase_bands(ax, mem_step, grok_step, steps[-1])
    add_phase_vlines(ax, mem_step, grok_step)

    ax.plot(steps, m["train_acc"], color=PAL.TRAIN, lw=1.8, label="Train accuracy")
    ax.plot(steps, m["test_acc"],  color=PAL.TEST,  lw=1.8, label="Test accuracy")

    # Fourier alignment on right axis
    ax2 = ax.twinx()
    ax2.plot(steps, m["fourier_alignment"], color=PAL.FOURIER, lw=1.5,
             ls=":", label="Fourier alignment")
    ax2.set_ylabel("Fourier alignment", color=PAL.FOURIER)
    ax2.tick_params(axis="y", colors=PAL.FOURIER)
    ax2.set_ylim(-0.05, 1.08)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PAL.FOURIER)
    ax2.spines["right"].set_linewidth(0.6)

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Training step")
    ax.set_ylim(-0.05, 1.08)
    ax.xaxis.set_major_formatter(thousands_formatter())

    # Annotate delay
    if mem_step and grok_step:
        mid = (mem_step + grok_step) / 2
        ax.annotate("", xy=(grok_step, 0.5), xytext=(mem_step, 0.5),
                    arrowprops=dict(arrowstyle="<->", color="#666666", lw=0.8))
        ax.text(mid, 0.53, f"Δ = {m['grok_delay']:,} steps",
                ha="center", va="bottom", fontsize=7.5, color="#666666")

    # Phase labels
    phase_info = [
        (0, mem_step, "Memorisation", PAL.PHASE_MEM),
        (mem_step, grok_step, "Circuit\nformation", PAL.PHASE_CIRCUIT),
        (grok_step, steps[-1], "Cleanup", PAL.PHASE_CLEANUP),
    ]
    for x0, x1, label, color in phase_info:
        if x0 is not None and x1 is not None:
            ax.text((x0 + x1) / 2, 1.04, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")

    # Merged legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=7.5)

    ax.set_title("Training dynamics")


# ── Panel (b): Fourier spectrum ──────────────────────────────────────────────

def panel_spectrum(ax, model, p, key_freqs):
    W_E = model.embed.W_E.detach().cpu()
    spectrum = fourier_power_spectrum(W_E, p).numpy()
    n = len(spectrum)

    key_indices = set()
    for k in key_freqs:
        key_indices.add(2 * k - 1)
        key_indices.add(2 * k)

    colors = [PAL.ORANGE if i in key_indices else "#DDDDDD" for i in range(n)]

    ax.bar(range(n), spectrum, color=colors, width=1.0, edgecolor="none")

    for k in key_freqs[:3]:
        cos_i = 2 * k - 1
        sin_i = 2 * k
        h = max(spectrum[cos_i], spectrum[sin_i]) if sin_i < n else spectrum[cos_i]
        ax.text((cos_i + sin_i) / 2, h + max(spectrum) * 0.03, f"k={k}",
                ha="center", va="bottom", fontsize=7, color=PAL.ORANGE, fontweight="bold")

    ax.set_xlabel("Fourier mode index")
    ax.set_ylabel("Power (sum of squares)")
    ax.set_title("Fourier spectrum at convergence")
    ax.set_xlim(-1, n)

    legend_items = [
        mpatches.Patch(color=PAL.ORANGE, label=f"Key frequencies"),
        mpatches.Patch(color="#DDDDDD", label="Other modes"),
    ]
    ax.legend(handles=legend_items, fontsize=7.5)


# ── Panel (c): Weight norms ──────────────────────────────────────────────────

def panel_norms(ax, m):
    steps = m["steps"]
    wn = m["weight_norms"]

    add_phase_bands(ax, m["mem_step"], m["grok_step"], steps[-1])
    add_phase_vlines(ax, m["mem_step"], m["grok_step"])

    components = [
        ("embedding",   "Embedding",  PAL.BLUE),
        ("attn_all",    "Attention",  PAL.CYAN),
        ("mlp_all",     "MLP",        PAL.ORANGE),
        ("unembedding", "Unembedding", PAL.GREY),
    ]
    for key, label, color in components:
        if key in wn:
            ax.plot(steps, wn[key], color=color, lw=1.5, label=label)

    ax.set_ylabel("L₂ norm")
    ax.set_xlabel("Training step")
    ax.xaxis.set_major_formatter(thousands_formatter())
    ax.set_title("Component weight norms")
    ax.legend(fontsize=7.5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",  default="results/exp1_baseline/baseline_p113_s42.json")
    parser.add_argument("--out",   default="results/exp1_baseline/fig1_baseline")
    parser.add_argument("--prime", type=int, default=113)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    apply_style()

    print("Loading metrics…")
    m = load_metrics(args.json)

    print("Re-training for spectrum…")
    final_steps = (m["grok_step"] or 3500) + 2000
    model, task = retrain_for_spectrum(args.prime, args.seed, final_steps)
    key_freqs = dominant_frequencies(model.embed.W_E.detach().cpu(), args.prime, top_k=3)
    print(f"  Key frequencies: {key_freqs}")

    # Layout: 1 row × 3 columns; (a) is wider
    fig = plt.figure(figsize=(14, 3.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.50,
                           width_ratios=[1.6, 1.0, 1.0])

    ax_dyn  = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_norm = fig.add_subplot(gs[0, 2])

    panel_dynamics(ax_dyn, m)
    panel_spectrum(ax_spec, model, args.prime, key_freqs)
    panel_norms(ax_norm, m)

    label_panel(ax_dyn,  "a")
    label_panel(ax_spec, "b")
    label_panel(ax_norm, "c")

    fig.suptitle("Baseline grokking: modular addition (p = 113)",
                 fontsize=11, fontweight="bold", y=1.02)

    save_fig(fig, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
