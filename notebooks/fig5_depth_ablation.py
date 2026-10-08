"""
Figure 5 — Depth Ablation: Does Adding Layers Help Hard Tasks Grok?

Two-panel figure:
  (a) Grok rate comparison: 1-layer vs 2-layer for each task.
  (b) Grokking delay comparison (for tasks/depths that grok).

Run:
    uv run python notebooks/fig5_depth_ablation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict

from pub_style import apply_style, PAL, save_fig, label_panel


TASK_LABELS = {
    "mod_add":     "Mod add\n(Z₁₁₃)",
    "s5_compose":  "S₅\ncompose",
    "dihedral_12": "Dihedral\nD₁₂",
}

TASK_ORDER = ["mod_add", "s5_compose", "dihedral_12"]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    # Handle both artifact and legacy format
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def aggregate(runs):
    n = len(runs)
    n_grokked = sum(1 for r in runs if r["grokked"])
    delays = [r["grok_delay"] for r in runs if r["grok_delay"] is not None]
    accs = [r["final_test_acc"] for r in runs]
    return {
        "n": n,
        "n_grokked": n_grokked,
        "grok_rate": n_grokked / n if n > 0 else 0,
        "delay_mean": np.mean(delays) if delays else None,
        "delay_std": np.std(delays) if len(delays) > 1 else 0,
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs) if n > 1 else 0,
        "n_params": runs[0]["n_params"] if runs else 0,
    }


# ── Panel (a): Grok rate comparison ─────────────────────────────────────────

def panel_grok_rate(ax, by_key):
    tasks = [t for t in TASK_ORDER if any((t, l) in by_key for l in [1, 2])]
    layers_tested = sorted(set(l for _, l in by_key.keys()))

    x = np.arange(len(tasks))
    width = 0.35
    layer_colors = {1: PAL.BLUE, 2: PAL.ORANGE}

    for i, n_layers in enumerate(layers_tested):
        offset = (i - (len(layers_tested) - 1) / 2) * width
        rates = []
        labels_text = []
        for t in tasks:
            agg = by_key.get((t, n_layers))
            if agg:
                rates.append(agg["grok_rate"])
                labels_text.append(f"{agg['n_grokked']}/{agg['n']}")
            else:
                rates.append(0)
                labels_text.append("—")

        color = layer_colors.get(n_layers, PAL.GREY)
        bars = ax.bar(x + offset, rates, width, color=color,
                      edgecolor="white", linewidth=0.4,
                      label=f"{n_layers}-layer ({by_key.get((tasks[0], n_layers), {}).get('n_params', '?'):,} params)")

        for xi, (r, lab) in enumerate(zip(rates, labels_text)):
            ax.text(xi + offset, r + 0.03, lab,
                    ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylabel("Grok rate (fraction of seeds)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Grok rate by depth")
    ax.legend(fontsize=7)


# ── Panel (b): Delay comparison ──────────────────────────────────────────────

def panel_delay(ax, by_key):
    tasks = [t for t in TASK_ORDER if any((t, l) in by_key for l in [1, 2])]
    layers_tested = sorted(set(l for _, l in by_key.keys()))

    x = np.arange(len(tasks))
    width = 0.35
    layer_colors = {1: PAL.BLUE, 2: PAL.ORANGE}

    for i, n_layers in enumerate(layers_tested):
        offset = (i - (len(layers_tested) - 1) / 2) * width
        delays = []
        errs = []
        for t in tasks:
            agg = by_key.get((t, n_layers))
            if agg and agg["delay_mean"] is not None:
                delays.append(agg["delay_mean"])
                errs.append(agg["delay_std"])
            else:
                delays.append(0)
                errs.append(0)

        color = layer_colors.get(n_layers, PAL.GREY)
        ax.bar(x + offset, delays, width, yerr=errs, color=color,
               edgecolor="white", linewidth=0.4, capsize=2,
               error_kw={"lw": 0.6, "capthick": 0.5},
               label=f"{n_layers}-layer")

        for xi, d in enumerate(delays):
            if d > 0:
                ax.text(xi + offset, d + max(errs[xi], max(delays) * 0.02),
                        f"{int(d):,}",
                        ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylabel("Mean grokking delay (steps)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.set_title("Grokking delay by depth")
    ax.legend(fontsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="results/exp3_depth_ablation/depth_ablation_results.json")
    parser.add_argument("--out",  default="results/exp3_depth_ablation/fig5_depth_ablation")
    args = parser.parse_args()

    apply_style()

    if not os.path.exists(args.json):
        print(f"ERROR: {args.json} not found. Run the depth ablation experiment first:")
        print("  uv run python experiments/exp3_task_taxonomy/run_depth_ablation.py")
        return

    print("Loading data…")
    results = load_results(args.json)

    # Group by (task, n_layers)
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["task"], r["n_layers"])].append(r)

    by_key = {k: aggregate(v) for k, v in grouped.items()}

    n_seeds = max(a["n"] for a in by_key.values())
    print(f"  {len(results)} runs, {len(by_key)} conditions, up to {n_seeds} seeds")

    # Layout: 1 × 2
    fig, (ax_rate, ax_delay) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.35)

    panel_grok_rate(ax_rate, by_key)
    panel_delay(ax_delay, by_key)

    label_panel(ax_rate, "a")
    label_panel(ax_delay, "b")

    fig.suptitle(f"Depth ablation: 1-layer vs 2-layer (n = {n_seeds} seeds)",
                 fontsize=10, fontweight="bold", y=1.02)

    save_fig(fig, args.out)

    # Summary table
    print(f"\n{'Task':<14} {'Layers':<8} {'Params':<10} {'Grok Rate':<11} "
          f"{'Delay (mean±sd)':<20} {'Test Acc (mean±sd)'}")
    print("-" * 80)
    for (task, n_layers) in sorted(by_key.keys()):
        a = by_key[(task, n_layers)]
        delay_str = f"{a['delay_mean']:.0f} ± {a['delay_std']:.0f}" if a["delay_mean"] else "—"
        print(f"{task:<14} {n_layers:<8} {a['n_params']:<10,} "
              f"{a['n_grokked']}/{a['n']:<9} {delay_str:<20} "
              f"{a['acc_mean']:.3f} ± {a['acc_std']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
