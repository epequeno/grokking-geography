"""
Figure 2b — Joint Sufficiency: Which Component Pairs Can Drive Grokking?

Two-panel figure:
  (a) Grok step bar chart for each component combination.
  (b) Learning curves for each combination.

Run:
    uv run python notebooks/fig2b_joint_sufficiency.py
"""

import sys, os


import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from pub_style import apply_style, PAL, save_fig, label_panel, thousands_formatter


COMBO_LABELS = {
    "mlp+embed":         "MLP + Embed",
    "mlp+attn":          "MLP + Attn",
    "embed+attn":        "Embed + Attn",
    "mlp+embed+attn":    "MLP + Embed + Attn",
    "mlp+embed+unembed": "MLP + Embed + Unembed",
}

COMBO_ORDER = [
    "mlp+embed+unembed",
    "mlp+embed",
    "mlp+embed+attn",
    "mlp+attn",
    "embed+attn",
]


# ── Panel (a): Grok step bars ───────────────────────────────────────────────

def panel_bars(ax, results, baseline_grok_step):
    combos = [r for r in results if r["name"] in COMBO_ORDER]
    combos.sort(key=lambda r: COMBO_ORDER.index(r["name"]))

    names = [COMBO_LABELS[r["name"]] for r in combos]
    y = np.arange(len(combos))

    for i, r in enumerate(combos):
        grok_step = r.get("grok_step")
        if grok_step:
            color = PAL.GROK
            ax.barh(i, grok_step, color=color, height=0.55,
                    edgecolor="white", linewidth=0.4)
            ax.text(grok_step + 100, i, f"step {grok_step:,}",
                    va="center", fontsize=7.5, color=PAL.GROK, fontweight="bold")
        else:
            color = PAL.RED
            test_acc = r.get("final_test_acc", 0)
            ax.barh(i, 500, color=color, height=0.55,
                    edgecolor="white", linewidth=0.4, alpha=0.4)
            ax.text(700, i, f"No grok (test: {test_acc:.0%})",
                    va="center", fontsize=7.5, color=PAL.RED, fontweight="bold")

    if baseline_grok_step:
        ax.axvline(baseline_grok_step, color=PAL.GREY, lw=0.8, ls="--", alpha=0.6)
        ax.text(baseline_grok_step + 50, len(combos) - 0.3,
                f"baseline ({baseline_grok_step:,})",
                fontsize=7, color="#666666", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Grokking step")
    ax.xaxis.set_major_formatter(thousands_formatter())
    ax.set_title("Joint sufficiency: which pairs drive grokking?")

    legend = [
        mpatches.Patch(color=PAL.GROK, label="Groks"),
        mpatches.Patch(color=PAL.RED, alpha=0.4, label="Does not grok"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="lower right")


# ── Panel (b): Learning curves ──────────────────────────────────────────────

def panel_curves(ax, curves):
    combo_colors = {
        "mlp+embed+unembed": PAL.BLUE,
        "mlp+embed":         PAL.TEAL,
        "mlp+embed+attn":    PAL.ORANGE,
        "mlp+attn":          PAL.RED,
        "embed+attn":        PAL.GREY,
    }

    for r in sorted(curves, key=lambda r: COMBO_ORDER.index(r["name"])
                    if r["name"] in COMBO_ORDER else 99):
        name = r["name"]
        if name not in COMBO_ORDER:
            continue
        steps = r.get("steps", [])
        test_acc = r.get("test_acc", [])
        if not steps:
            continue

        color = combo_colors.get(name, PAL.GREY)
        ax.plot(steps, test_acc, color=color, lw=1.5, label=COMBO_LABELS[name])

    ax.axhline(0.99, color="#CCCCCC", lw=0.6, ls=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(-0.05, 1.08)
    ax.xaxis.set_major_formatter(thousands_formatter())
    ax.set_title("Learning curves for each combination")
    ax.legend(fontsize=7, loc="center right")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",   default="results/exp1_joint_sufficiency/joint_sufficiency_p113_s42.json")
    parser.add_argument("--curves", default="results/exp1_joint_sufficiency/joint_sufficiency_curves_p113_s42.json")
    parser.add_argument("--out",    default="results/exp1_joint_sufficiency/fig2b_joint_sufficiency")
    args = parser.parse_args()

    apply_style()

    if not os.path.exists(args.json):
        print(f"ERROR: {args.json} not found. Run the joint sufficiency experiment first.")
        return

    print("Loading data…")
    with open(args.json) as f:
        results = json.load(f)

    curves = []
    if os.path.exists(args.curves):
        with open(args.curves) as f:
            curves = json.load(f)

    # Try to get baseline grok step
    baseline_grok = None
    baseline_path = "results/exp1_baseline/baseline_p113_s42.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            b = json.load(f)
        baseline_grok = b.get("grok_step")

    # Layout: 1 × 2
    fig, (ax_bars, ax_curves) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.subplots_adjust(wspace=0.35)

    panel_bars(ax_bars, results, baseline_grok)
    if curves:
        panel_curves(ax_curves, curves)
    else:
        ax_curves.text(0.5, 0.5, "No curves data available",
                       transform=ax_curves.transAxes, ha="center", fontsize=9, color="#999999")

    label_panel(ax_bars, "a")
    label_panel(ax_curves, "b")

    fig.suptitle("Joint sufficiency: minimal component sets for grokking (mod_add p = 113)",
                 fontsize=10, fontweight="bold", y=1.02)

    save_fig(fig, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
