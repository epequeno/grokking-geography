"""
Baseline grokking visualizations.

Produces a single figure with five panels that tell the complete story
of the baseline modular addition run:

  1. The grokking story — train/test accuracy + Fourier alignment over time,
     with three phases annotated.
  2. Fourier alignment as a leading indicator — zoomed view of the
     transition window showing how alignment tracks circuit formation.
  3. Fourier power spectrum at convergence — the clean frequency structure
     the model learned, with key frequencies highlighted.
  4. Component weight norms over time — shows weight decay progressively
     removing the memorisation components during cleanup.
  5. Alignment vs test accuracy correlation — shows these metrics are
     co-varying (not leading/lagging), which motivates key_frequency_alignment
     as a better probe.

Run:
    uv run python notebooks/visualize_baseline.py
    uv run python notebooks/visualize_baseline.py --json results/exp1_baseline/baseline_p113_s42.json
"""

import sys, os


import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.analysis.fourier import fourier_power_spectrum, dominant_frequencies


# ── colour palette ────────────────────────────────────────────────────────────
C_TRAIN    = "#3A86FF"   # blue
C_TEST     = "#FF6B6B"   # coral
C_FOURIER  = "#8338EC"   # purple
C_MEM      = "#FB5607"   # orange-red  (memorisation vline)
C_GROK     = "#06D6A0"   # teal        (grokking vline)

PHASE_ALPHA = 0.07
PHASE_COLORS = {
    "Memorisation": "#FB5607",
    "Circuit formation": "#8338EC",
    "Cleanup": "#06D6A0",
}


def load_metrics(json_path):
    with open(json_path) as f:
        return json.load(f)


def retrain_for_spectrum(p, seed, steps_override):
    """Re-run a short training to get the final model weights for spectrum analysis."""
    # Seed torch before model initialisation so weights match the original run.
    torch.manual_seed(seed)
    task = ModularAddition(p=p, train_frac=0.3, seed=seed)
    cfg  = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )
    model = GrokTransformer(cfg)
    train_cfg = TrainConfig(
        n_steps=steps_override, lr=1e-3, weight_decay=1.0,
        log_every=500,   # log at normal cadence so grokking detection works
        seed=seed,
    )
    print(f"  Re-training for spectrum (steps={steps_override})…", flush=True)
    GrokTrainer(model, task, train_cfg).train()
    return model, task


def add_phase_bands(ax, mem_step, grok_step, max_step):
    """Shade the three grokking phases."""
    regions = [
        (0,         mem_step,  "Memorisation"),
        (mem_step,  grok_step, "Circuit formation"),
        (grok_step, max_step,  "Cleanup"),
    ]
    for x0, x1, label in regions:
        ax.axvspan(x0, x1, color=PHASE_COLORS[label], alpha=PHASE_ALPHA, zorder=0)

    # Phase labels at the top
    for x0, x1, label in regions:
        mid = (x0 + x1) / 2
        ax.text(mid, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, color=PHASE_COLORS[label],
                fontweight="bold", alpha=0.85)


def vlines(ax, mem_step, grok_step):
    ax.axvline(mem_step,  color=C_MEM,  lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.axvline(grok_step, color=C_GROK, lw=1.2, ls="--", alpha=0.7, zorder=2)


# ── panel builders ────────────────────────────────────────────────────────────

def panel_grokking_story(ax, m):
    """Train/test accuracy + Fourier alignment on a dual y-axis."""
    steps     = m["steps"]
    mem_step  = m["mem_step"]
    grok_step = m["grok_step"]
    max_step  = steps[-1]

    add_phase_bands(ax, mem_step, grok_step, max_step)
    vlines(ax, mem_step, grok_step)

    ax.plot(steps, m["train_acc"], color=C_TRAIN, lw=2,   label="Train acc",  zorder=3)
    ax.plot(steps, m["test_acc"],  color=C_TEST,  lw=2,   label="Test acc",   zorder=3)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Training step", fontsize=10)

    # Fourier alignment on right axis
    ax2 = ax.twinx()
    ax2.plot(steps, m["fourier_alignment"], color=C_FOURIER, lw=2,
             ls="-.", label="Fourier alignment", zorder=3)
    ax2.set_ylabel("Fourier alignment", color=C_FOURIER, fontsize=10)
    ax2.tick_params(axis="y", colors=C_FOURIER)
    ax2.set_ylim(-0.05, 1.12)

    # Legend (merge both axes)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="center left", fontsize=8, framealpha=0.9)

    # Annotate delay
    ax.annotate(
        f"delay = {m['grok_delay']} steps",
        xy=(grok_step, 0.99), xytext=(grok_step - 900, 0.78),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.0),
        fontsize=8, color="gray",
    )

    ax.set_title("Grokking: modular addition (p=113)", fontsize=11, fontweight="bold")


def panel_transition_zoom(ax, m):
    """Zoomed into the transition window, showing test acc and alignment rising together."""
    steps = np.array(m["steps"])
    te    = np.array(m["test_acc"])
    fa    = np.array(m["fourier_alignment"])
    mem_step  = m["mem_step"]
    grok_step = m["grok_step"]

    # Window: mem_step - 200 to grok_step + 800
    lo = max(0,          mem_step  - 200)
    hi = min(steps[-1],  grok_step + 800)
    mask = (steps >= lo) & (steps <= hi)

    add_phase_bands(ax, mem_step, grok_step, hi)
    vlines(ax, mem_step, grok_step)

    ax.plot(steps[mask], te[mask], color=C_TEST,    lw=2.5, label="Test acc")
    ax2 = ax.twinx()
    ax2.plot(steps[mask], fa[mask], color=C_FOURIER, lw=2.5, ls="-.", label="Fourier alignment")
    ax2.set_ylabel("Fourier alignment", color=C_FOURIER, fontsize=10)
    ax2.tick_params(axis="y", colors=C_FOURIER)
    ax2.set_ylim(
        fa[mask].min() - 0.02,
        fa[mask].max() + 0.05,
    )

    ax.set_ylabel("Test accuracy", fontsize=10)
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(lo, hi)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title("Transition window (zoomed)", fontsize=11, fontweight="bold")


def panel_fourier_spectrum(ax, model, p, key_freqs):
    """Fourier power spectrum with key frequencies highlighted."""
    W_E      = model.embed.W_E.detach().cpu()
    spectrum = fourier_power_spectrum(W_E, p).numpy()
    n        = len(spectrum)

    # Build per-bar colour: key-freq pairs in orange, noise in light grey
    key_mode_indices = set()
    for k in key_freqs:
        key_mode_indices.add(2 * k - 1)
        key_mode_indices.add(2 * k)

    colors = ["#FF9F1C" if i in key_mode_indices else "#CCCCCC" for i in range(n)]

    ax.bar(range(n), spectrum, color=colors, width=1.0, edgecolor="none")

    # Annotate frequency labels above key pairs
    for k in key_freqs:
        cos_i = 2 * k - 1
        h = max(spectrum[cos_i], spectrum[2 * k])
        ax.text((cos_i + 2 * k) / 2, h + 0.3, f"k={k}",
                ha="center", va="bottom", fontsize=8, color="#CC6600", fontweight="bold")

    ax.set_xlabel("Fourier mode index", fontsize=10)
    ax.set_ylabel("Power (sum of squares)", fontsize=10)
    ax.set_title(f"Fourier spectrum at convergence\n3 key frequency pairs (k={key_freqs})",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-1, n)

    legend_patches = [
        mpatches.Patch(color="#FF9F1C", label=f"Key freqs: k∈{key_freqs}"),
        mpatches.Patch(color="#CCCCCC", label="Other modes (noise)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.9)


def panel_weight_norms(ax, m):
    """Weight norm trajectories for key components."""
    steps    = m["steps"]
    mem_step = m["mem_step"]
    grok_step= m["grok_step"]
    wn       = m["weight_norms"]

    add_phase_bands(ax, mem_step, grok_step, steps[-1])
    vlines(ax, mem_step, grok_step)

    components = [
        ("embedding",   "Token+pos embed",  C_TRAIN,   "-"),
        ("attn_all",    "All attention",     "#FF9F1C",  "-"),
        ("mlp_all",     "All MLP",           C_TEST,    "-"),
        ("unembedding", "Unembedding",       "#555555",  "--"),
    ]
    for key, label, color, ls in components:
        if key in wn:
            ax.plot(steps, wn[key], color=color, lw=2, ls=ls, label=label, zorder=3)

    ax.set_ylabel("L2 norm", fontsize=10)
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_title("Component weight norms\n(weight decay drives cleanup)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)


def panel_alignment_vs_accuracy(ax, m):
    """Fourier alignment vs test accuracy coloured by training step — shows co-variation."""
    steps = np.array(m["steps"])
    te    = np.array(m["test_acc"])
    fa    = np.array(m["fourier_alignment"])

    sc = ax.scatter(fa, te, c=steps, cmap="plasma", s=30, zorder=3, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Training step")

    # Annotate key steps
    for label, step in [("mem", m["mem_step"]), ("grok", m["grok_step"])]:
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - step))
        ax.annotate(
            f"  {label} (step {step})",
            xy=(fa[idx], te[idx]),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            xytext=(fa[idx] + 0.04, te[idx] - 0.08),
        )

    ax.set_xlabel("Fourier alignment score", fontsize=10)
    ax.set_ylabel("Test accuracy", fontsize=10)
    ax.set_title("Alignment vs test accuracy\n(co-varying → use key_frequency_alignment for leading indicator)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(-0.05, 1.05)

    # Diagonal reference
    xs = np.linspace(0, 0.62, 100)
    ax.plot(xs, xs / 0.62, color="gray", lw=0.8, ls=":", alpha=0.5, label="perfect correlation")
    ax.legend(fontsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",  default="results/exp1_baseline/baseline_p113_s42.json")
    parser.add_argument("--out",   default="results/exp1_baseline/baseline_analysis.png")
    parser.add_argument("--prime", type=int, default=113)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading metrics…")
    m = load_metrics(args.json)

    print("Re-training for final model weights (spectrum analysis)…")
    # Run until a few hundred steps past grokking for a clean converged model
    final_steps = (m["grok_step"] or 3500) + 2000
    model, task = retrain_for_spectrum(args.prime, args.seed, final_steps)

    # Always derive key frequencies from the actual re-trained model — don't
    # use the saved JSON value, which came from a different RNG initialisation.
    # (Different seeds can converge to different frequency sets: Zhong et al.)
    key_freqs = dominant_frequencies(
        model.embed.W_E.detach().cpu(), args.prime, top_k=3
    )
    print(f"Key frequencies (this model): {key_freqs}")
    saved_freqs = m.get("key_frequencies")
    if saved_freqs and set(saved_freqs) != set(key_freqs):
        print(f"  Note: saved run used {saved_freqs} — different init → different freqs")

    # ── Layout: 2 rows × 3 cols, with the top-left panel spanning 2 columns ──
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F9F9F9")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.38,
        left=0.06, right=0.97,
        top=0.93,  bottom=0.07,
    )

    ax_story   = fig.add_subplot(gs[0, :2])   # top-left, spans 2 cols
    ax_spectrum= fig.add_subplot(gs[0, 2])    # top-right
    ax_zoom    = fig.add_subplot(gs[1, 0])    # bottom-left
    ax_norms   = fig.add_subplot(gs[1, 1])    # bottom-middle
    ax_corr    = fig.add_subplot(gs[1, 2])    # bottom-right

    for ax in [ax_story, ax_spectrum, ax_zoom, ax_norms, ax_corr]:
        ax.set_facecolor("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#CCCCCC")

    print("Building panels…")
    panel_grokking_story(ax_story,     m)
    panel_fourier_spectrum(ax_spectrum, model, args.prime, key_freqs)
    panel_transition_zoom(ax_zoom,     m)
    panel_weight_norms(ax_norms,       m)
    panel_alignment_vs_accuracy(ax_corr, m)

    fig.suptitle(
        "Baseline Grokking Analysis — mod_add p=113  |  mem=700, grok=3000, delay=2300",
        fontsize=13, fontweight="bold", color="#222222", y=0.975,
    )

    print(f"Saving to {args.out}…")
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Done.")


if __name__ == "__main__":
    main()
