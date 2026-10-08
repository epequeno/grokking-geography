"""
Freeze sweep visualization — Experiment 1.

Three panels:
  1. Delay ratio bar chart — how much does freezing each component slow (or
     block) grokking, relative to the unfrozen baseline.
  2. Causal map — components arranged by their role in the Nanda circuit,
     coloured by causal impact category.
  3. The outlier pair — learning curves reconstructed from the sweep JSON for
     the two most striking results: mlp_all (never groks) vs attn_all (no delay).

Run:
    uv run python notebooks/visualize_freeze_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager

# ── palette ──────────────────────────────────────────────────────────────────
C_NEVER   = "#E63946"   # red     — blocks grokking entirely
C_SEVERE  = "#F4A261"   # orange  — 5–15× delay
C_MAJOR   = "#E9C46A"   # yellow  — 2–5× delay
C_MINOR   = "#A8DADC"   # teal    — 1–2× delay
C_NEUTRAL = "#457B9D"   # blue    — ≤1× (no delay or faster)
C_BASE    = "#2D6A4F"   # green   — baseline reference

BASELINE_DELAY = 2300

COMPONENT_LABELS = {
    "embedding":   "Embedding (W_E)",
    "unembedding": "Unembedding (W_U)",
    "attn_Q":      "Attention Q",
    "attn_K":      "Attention K",
    "attn_V":      "Attention V",
    "attn_O":      "Attention O",
    "attn_all":    "Attention (all)",
    "mlp_in":      "MLP W_in",
    "mlp_out":     "MLP W_out",
    "mlp_all":     "MLP (all)",
}

def delay_color(ratio):
    if ratio is None:    return C_NEVER
    if ratio > 5.0:      return C_SEVERE
    if ratio > 2.0:      return C_MAJOR
    if ratio > 1.0:      return C_MINOR
    return C_NEUTRAL


# ── panel 1: delay ratio bar chart ───────────────────────────────────────────

def panel_delay_bars(ax, results):
    # Sort: never-grokked last, then by delay ratio descending
    def sort_key(r):
        if not r["grokked"]: return float("inf")
        return -(r["grok_delay"] / BASELINE_DELAY)

    sorted_r = sorted(results, key=sort_key)
    names    = [COMPONENT_LABELS[r["target_component"]] for r in sorted_r]
    ratios   = [r["grok_delay"] / BASELINE_DELAY if r["grokked"] else None
                for r in sorted_r]
    colors   = [delay_color(ratio) for ratio in ratios]

    y_pos = list(range(len(names)))

    # Bars
    for i, (ratio, color) in enumerate(zip(ratios, colors)):
        val = ratio if ratio is not None else 0
        ax.barh(i, val, color=color, edgecolor="white", linewidth=0.5, height=0.7)
        if ratio is None:
            ax.text(0.4, i, "NEVER  (4.5% test acc after 80K steps)",
                    va="center", ha="left", fontsize=8.5,
                    color=C_NEVER, fontweight="bold")
        else:
            label = f"{ratio:.1f}×"
            if ratio < 1.0:
                label += "  ← faster than baseline"
            ax.text(val + 0.15, i, label, va="center", ha="left", fontsize=8.5,
                    color="#333333")

    # Baseline vline
    ax.axvline(1.0, color=C_BASE, lw=1.5, ls="--", alpha=0.8, label="Baseline (no freeze)")
    ax.text(1.0, len(names) - 0.4, " baseline", color=C_BASE, fontsize=8, va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("Grokking delay  ×  baseline  (higher = frozen component matters more)", fontsize=9)
    ax.set_xlim(0, 13)
    ax.set_title("Freeze-one sweep: causal impact on grokking delay\n"
                 "(frozen at memorisation step 700; trained for up to 80K steps)",
                 fontsize=11, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color=C_NEVER,  label="Blocks grokking entirely"),
        mpatches.Patch(color=C_SEVERE, label=">5× delay"),
        mpatches.Patch(color=C_MAJOR,  label="2–5× delay"),
        mpatches.Patch(color=C_MINOR,  label="1–2× delay"),
        mpatches.Patch(color=C_NEUTRAL,label="≤1× (no effect / faster)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)


# ── panel 2: circuit anatomy map ─────────────────────────────────────────────

def panel_circuit_map(ax):
    """
    Schematic of the Nanda Fourier multiplication circuit, annotated with
    freeze-sweep findings. Shows which component is responsible for which
    part of the algorithm and what happens when it's frozen.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Circuit anatomy: what each component does\n"
                 "(Nanda et al. algorithm + freeze-sweep causal evidence)",
                 fontsize=11, fontweight="bold")

    def box(x, y, w, h, color, label, sublabel="", text_color="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white", linewidth=1.2, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color, zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.28,
                    sublabel, ha="center", va="center",
                    fontsize=7.5, color=text_color, alpha=0.9, zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#888888",
                                   lw=1.2, mutation_scale=12),
                    zorder=2)

    # Input tokens
    box(0.3, 3.5, 1.4, 1.0, "#555555", "Tokens", "a, b, =", text_color="white")
    arrow(1.7, 4.0, 2.1, 4.0)

    # Embedding
    box(2.1, 3.5, 1.6, 1.0, C_SEVERE, "Embedding", "10× delay if frozen", text_color="white")
    ax.text(2.9, 3.2, "Maps tokens → (cos(ωa), sin(ωa))\nFourier basis must form here",
            ha="center", va="top", fontsize=7, color="#333333", style="italic")
    arrow(3.7, 4.0, 4.1, 4.0)

    # Attention block
    box(4.1, 3.5, 1.8, 1.0, C_NEUTRAL, "Attention", "≈0× delay if frozen", text_color="white")
    ax.text(5.0, 3.2, "Computes sum a+b in Fourier space\n(rotation). Already correct at step 700.",
            ha="center", va="top", fontsize=7, color="#333333", style="italic")
    arrow(5.9, 4.0, 6.3, 4.0)

    # MLP block
    box(6.3, 3.5, 1.8, 1.0, C_NEVER, "MLP", "BLOCKS grokking", text_color="white")
    ax.text(7.2, 3.2, "Frequency selection via nonlinearity\n(trig identity → answer logit).\nMust keep updating.",
            ha="center", va="top", fontsize=7, color="#333333", style="italic")
    arrow(8.1, 4.0, 8.5, 4.0)

    # Unembedding
    box(8.5, 3.5, 1.2, 1.0, C_MAJOR, "Unembed", "3× delay", text_color="white")
    ax.text(9.1, 3.2, "Reads answer\nfrom residual.",
            ha="center", va="top", fontsize=7, color="#333333", style="italic")

    # Summary interpretation boxes
    interpretations = [
        (1.0, 6.5, 3.5, 1.0, C_SEVERE,
         "Embedding: slow bottleneck, not hard block",
         "Without Fourier basis, model finds alternative\nalgorithm — slowly. 10× delay, 98.8% final acc."),
        (1.0, 1.2, 3.5, 1.0, C_NEUTRAL,
         "Attention: already committed at step 700",
         "Freezing all attention slightly SPEEDS grokking.\nMemorisation circuit is in MLP, not attention."),
        (5.2, 6.5, 3.5, 1.0, C_NEVER,
         "MLP: causally necessary",
         "Neither W_in nor W_out alone blocks grokking\n(3× delay each). Together: grokking never fires."),
        (5.2, 1.2, 3.5, 1.0, C_MAJOR,
         "Unembedding: significant but compensable",
         "3× delay. Embedding + attention + MLP can\nbuild internal representation; readout adapts last."),
    ]
    for x, y, w, h, color, title, body in interpretations:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.0, zorder=1,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.12, title,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=delay_color(None if "BLOCKS" in title else
                                   (6.0 if "slow" in title else
                                    (0.5 if "already" in title else 3.0))),
                zorder=2)
        ax.text(x + w/2, y + 0.38, body,
                ha="center", va="center", fontsize=7, color="#333333",
                zorder=2)


# ── panel 3: outlier learning curves ─────────────────────────────────────────

def run_outlier_experiments(baseline_json, prime=113, seed=42):
    """Re-run mlp_all and attn_all freeze experiments to get learning curves."""
    with open(baseline_json) as f:
        b = json.load(f)
    freeze_step = b["mem_step"]   # 700

    task = ModularAddition(p=prime, train_frac=0.3, seed=seed)
    cfg  = TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )

    curves = {}
    for name, target_comp, n_steps in [
        ("mlp_all_frozen",  "mlp_all",  30000),  # won't grok; run long enough to confirm
        ("attn_all_frozen", "attn_all", 10000),  # groks fast
    ]:
        torch_seed = seed
        import torch; torch.manual_seed(torch_seed)
        model = GrokTransformer(cfg)
        fm    = FreezeManager(model)
        train_cfg = TrainConfig(
            n_steps=n_steps, lr=1e-3, weight_decay=1.0, log_every=200, seed=seed,
            freeze_schedule=[(freeze_step, "freeze", target_comp)],
        )
        print(f"  Running {name}…", flush=True)
        metrics = GrokTrainer(model, task, train_cfg, freeze_manager=fm).train()
        curves[name] = {
            "steps":       metrics.steps,
            "train_acc":   metrics.train_acc,
            "test_acc":    metrics.test_acc,
            "grok_step":   metrics.grok_step,
            "mem_step":    metrics.mem_step,
        }
    return curves


def panel_outlier_curves(ax_mlp, ax_attn, curves, baseline_json):
    with open(baseline_json) as f:
        b = json.load(f)

    C_TRAIN = "#3A86FF"
    C_TEST  = "#FF6B6B"

    # ── mlp_all ────────────────────────────────────────────────────────────
    c = curves["mlp_all_frozen"]
    ax_mlp.plot(c["steps"], c["train_acc"], color=C_TRAIN, lw=2,   label="Train acc")
    ax_mlp.plot(c["steps"], c["test_acc"],  color=C_TEST,  lw=2,   label="Test acc")
    ax_mlp.axvline(700, color=C_NEVER, lw=1.5, ls="--", alpha=0.9, label="Freeze MLP (step 700)")
    ax_mlp.axhline(0.99, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax_mlp.set_ylim(-0.05, 1.12)
    ax_mlp.set_xlabel("Training step", fontsize=9)
    ax_mlp.set_ylabel("Accuracy", fontsize=9)
    ax_mlp.set_title("MLP (all) frozen — grokking blocked\n"
                     "Train acc → 1.0; test acc → 4.5% (chance)",
                     fontsize=10, fontweight="bold", color=C_NEVER)
    ax_mlp.legend(fontsize=8)
    ax_mlp.text(0.97, 0.45, "perfect memorisation\nzero generalisation",
                transform=ax_mlp.transAxes, ha="right", va="center",
                fontsize=8.5, color=C_NEVER, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_NEVER, alpha=0.8))

    # ── attn_all ───────────────────────────────────────────────────────────
    c2 = curves["attn_all_frozen"]
    ax_attn.plot(c2["steps"], c2["train_acc"], color=C_TRAIN, lw=2,   label="Train acc")
    ax_attn.plot(c2["steps"], c2["test_acc"],  color=C_TEST,  lw=2,   label="Test acc")
    ax_attn.axvline(700, color=C_NEUTRAL, lw=1.5, ls="--", alpha=0.9, label="Freeze attention (step 700)")

    # Also plot baseline test acc for comparison
    ax_attn.plot(b["steps"], b["test_acc"], color="#AAAAAA", lw=1.5, ls=":",
                 alpha=0.7, label="Baseline test acc (no freeze)")

    if c2["grok_step"]:
        ax_attn.axvline(c2["grok_step"], color=C_NEUTRAL, lw=1.2, ls="-.", alpha=0.7)
        ax_attn.text(c2["grok_step"] + 50, 0.5,
                     f"grok @ {c2['grok_step']}\n(baseline: 3000)",
                     fontsize=8, color=C_NEUTRAL)

    ax_attn.axhline(0.99, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax_attn.set_ylim(-0.05, 1.12)
    ax_attn.set_xlabel("Training step", fontsize=9)
    ax_attn.set_ylabel("Accuracy", fontsize=9)
    ax_attn.set_title("Attention (all) frozen — grokking unimpeded\n"
                      "Delay 2000 steps vs baseline 2300 (slightly faster)",
                      fontsize=10, fontweight="bold", color=C_NEUTRAL)
    ax_attn.legend(fontsize=8)
    ax_attn.text(0.97, 0.45, "attention already\ncommitted at step 700",
                 transform=ax_attn.transAxes, ha="right", va="center",
                 fontsize=8.5, color=C_NEUTRAL, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor=C_NEUTRAL, alpha=0.8))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",     default="results/exp1_freeze_sweep/freeze_sweep_p113_s42.json")
    parser.add_argument("--baseline", default="results/exp1_baseline/baseline_p113_s42.json")
    parser.add_argument("--out",      default="results/exp1_freeze_sweep/freeze_sweep_analysis.png")
    parser.add_argument("--prime",    type=int, default=113)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading results…")
    with open(args.json) as f:
        results = json.load(f)

    print("Re-running outlier experiments for learning curves…")
    curves = run_outlier_experiments(args.baseline, args.prime, args.seed)

    # ── Layout: 2 rows. Top row: bars + circuit map (equal width).
    #            Bottom row: two learning curves.
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#F9F9F9")

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.45, wspace=0.32,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )
    ax_bars  = fig.add_subplot(gs[0, 0])
    ax_map   = fig.add_subplot(gs[0, 1])
    ax_mlp   = fig.add_subplot(gs[1, 0])
    ax_attn  = fig.add_subplot(gs[1, 1])

    for ax in [ax_bars, ax_mlp, ax_attn]:
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#CCCCCC")
    ax_map.set_facecolor("#FAFAFA")

    print("Building panels…")
    panel_delay_bars(ax_bars, results)
    panel_circuit_map(ax_map)
    panel_outlier_curves(ax_mlp, ax_attn, curves, args.baseline)

    fig.suptitle(
        "Exp 1 — Surgical Freeze: Which component updates cause grokking?  "
        "(mod_add p=113, freeze at memorisation step 700)",
        fontsize=13, fontweight="bold", color="#222222", y=0.975,
    )

    print(f"Saving to {args.out}…")
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Done.")


if __name__ == "__main__":
    main()
