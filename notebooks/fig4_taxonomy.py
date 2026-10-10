"""
Figure 4 — Task Taxonomy: The Grokking Geography Map.

Three-panel figure:
  (a) Task taxonomy bar chart — grok rate and delay by task.
  (b) Component importance heatmap — delay ratio for each (task, component).
  (c) Key contrast: mod_add vs S5 component profiles.

Run:
    uv run python notebooks/fig4_taxonomy.py
"""

import sys, os


import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

from pub_style import apply_style, PAL, save_fig, label_panel


TASK_ORDER = ["mod_add", "mod_mul", "mod_div", "xor_6bit", "xor_5bit",
              "parity_8bit", "mod_exp", "s5_compose", "dihedral_12"]

TASK_LABELS = {
    "mod_add":     "Mod add (Z₁₁₃)",
    "mod_mul":     "Mod mul (Z₁₁₃)",
    "mod_div":     "Mod div (Z₁₁₃)",
    "mod_exp":     "Mod exp (Z₉₇)",
    "s5_compose":  "S₅ compose",
    "dihedral_12": "Dihedral D₁₂",
    "xor_6bit":    "XOR 6-bit",
    "xor_5bit":    "XOR 5-bit",
    "parity_8bit": "Parity 8-bit",
}

COMP_ORDER = ["mlp_all", "mlp_in", "mlp_out", "attn_all", "attn_Q", "attn_K",
              "attn_V", "attn_O", "embedding", "unembedding"]

COMP_LABELS = {
    "mlp_all": "MLP (all)", "mlp_in": "MLP Wᵢₙ", "mlp_out": "MLP Wₒᵤₜ",
    "attn_all": "Attn (all)", "attn_Q": "Attn Q", "attn_K": "Attn K",
    "attn_V": "Attn V", "attn_O": "Attn O",
    "embedding": "Embed", "unembedding": "Unembed",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_taxonomy(path):
    with open(path) as f:
        rows = json.load(f)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_name"]].append(r)
    agg = {}
    for t, rs in by_task.items():
        delays = [r["grok_delay"] for r in rs if r["grok_delay"] is not None]
        accs = [r["final_test_acc"] for r in rs]
        n = len(rs)
        agg[t] = {
            "n": n,
            "n_grokked": sum(1 for r in rs if r["grokked"]),
            "grok_rate": sum(1 for r in rs if r["grokked"]) / n,
            "mean_delay": np.mean(delays) if delays else None,
            "std_delay": np.std(delays) if len(delays) > 1 else 0,
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs) if n > 1 else 0,
            "is_commutative": rs[0]["is_commutative"],
        }
    return agg


def load_freeze_per_task(path):
    with open(path) as f:
        return json.load(f)


# ── Panel (a): Task taxonomy ────────────────────────────────────────────────

def panel_taxonomy(ax, tax):
    tasks = [t for t in TASK_ORDER if t in tax]
    n = len(tasks)
    y = np.arange(n)

    for i, t in enumerate(tasks):
        d = tax[t]
        color = PAL.ABELIAN if d["is_commutative"] else PAL.NONABELIAN
        alpha = 1.0 if d["grok_rate"] > 0 else 0.4

        if d["mean_delay"] is not None and d["mean_delay"] > 0:
            ax.barh(i, d["mean_delay"], color=color, height=0.6,
                    edgecolor="white", linewidth=0.4, alpha=alpha)
            if d["std_delay"] > 0:
                ax.errorbar(d["mean_delay"], i, xerr=d["std_delay"], fmt="none",
                            ecolor="#333333", capsize=2, capthick=0.5, lw=0.5)
            label = f"{d['n_grokked']}/{d['n']}  Δ={int(d['mean_delay']):,}"
            ax.text(d["mean_delay"] + d["std_delay"] + 200, i, label,
                    va="center", fontsize=7, color="#333333")
        else:
            ax.barh(i, 500, color=color, height=0.6,
                    edgecolor=PAL.RED, linewidth=0.8, alpha=0.3)
            ax.text(700, i, f"0/{d['n']} grokked  (test: {d['mean_acc']:.1%})",
                    va="center", fontsize=7, color=PAL.RED, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_xlabel("Mean grokking delay (steps)")
    ax.set_title("Task taxonomy: which tasks grok?")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

    legend = [
        mpatches.Patch(color=PAL.ABELIAN, label="Commutative"),
        mpatches.Patch(color=PAL.NONABELIAN, label="Non-commutative"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="lower right")


# ── Panel (b): Component importance heatmap ──────────────────────────────────

def panel_heatmap(ax, freeze_data):
    by_task = defaultdict(dict)
    for r in freeze_data:
        by_task[r["task"]][r["component"]] = r

    tasks = [t for t in TASK_ORDER if t in by_task]
    comps = COMP_ORDER

    n_tasks = len(tasks)
    n_comps = len(comps)

    arr = np.full((n_tasks, n_comps), np.nan)
    blocked = np.zeros((n_tasks, n_comps), dtype=bool)

    for i, task in enumerate(tasks):
        base = by_task[task].get("baseline")
        base_delay = base["grok_delay"] if base and base["grokked"] else None

        for j, comp in enumerate(comps):
            r = by_task[task].get(comp)
            if r is None:
                continue
            if not r["grokked"]:
                blocked[i, j] = True
                arr[i, j] = 12.0  # sentinel
            elif base_delay and base_delay > 0:
                arr[i, j] = r["grok_delay"] / base_delay

    # Custom colourmap: blue → white → orange → red
    cmap = LinearSegmentedColormap.from_list("delay", [
        (0.0,  PAL.BLUE),
        (0.15, "#FFFFFF"),
        (0.5,  PAL.ORANGE),
        (0.75, PAL.RED),
        (1.0,  "#000000"),
    ])

    display = np.clip(arr, 0, 10)
    masked = np.ma.masked_where(np.isnan(arr), display)

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=10,
                   interpolation="nearest")

    for i in range(n_tasks):
        for j in range(n_comps):
            if blocked[i, j]:
                ax.text(j, i, "X", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
            elif not np.isnan(arr[i, j]):
                val = arr[i, j]
                txt = f"{val:.1f}×"
                color = "white" if val > 4 else "#333333"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=color)

    ax.set_xticks(range(n_comps))
    ax.set_xticklabels([COMP_LABELS[c] for c in comps], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks], fontsize=7)
    ax.set_title("Component freeze impact (delay ratio)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Delay / baseline", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# ── Panel (c): Contrast — mod_add vs S5 ─────────────────────────────────────

def panel_contrast(ax, freeze_data):
    by_task = defaultdict(dict)
    for r in freeze_data:
        by_task[r["task"]][r["component"]] = r

    task_pairs = []
    for task_name, color, label in [
        ("mod_add", PAL.BLUE, "mod_add (abelian)"),
        ("s5_compose", PAL.RED, "S₅ (non-abelian)"),
    ]:
        if task_name not in by_task:
            continue
        base = by_task[task_name].get("baseline")
        base_delay = base["grok_delay"] if base and base["grokked"] else None
        ratios = {}
        for comp in COMP_ORDER:
            r = by_task[task_name].get(comp)
            if r is None:
                ratios[comp] = np.nan
            elif not r["grokked"]:
                ratios[comp] = None
            elif base_delay and base_delay > 0:
                ratios[comp] = r["grok_delay"] / base_delay
            else:
                ratios[comp] = np.nan
        task_pairs.append((task_name, color, label, ratios))

    if len(task_pairs) < 2:
        ax.text(0.5, 0.5, "Need both mod_add and s5_compose\ndata for contrast",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#999999")
        return

    comps = [c for c in COMP_ORDER
             if any(not np.isnan(tp[3].get(c, np.nan) or 0) for tp in task_pairs)]
    y = np.arange(len(comps))
    bar_h = 0.35
    offsets = [-bar_h/2, bar_h/2]

    for (task_name, color, label, ratios), offset in zip(task_pairs, offsets):
        vals = []
        for c in comps:
            v = ratios.get(c)
            if v is None:
                vals.append(0)
            elif np.isnan(v):
                vals.append(0)
            else:
                vals.append(v)

        bars = ax.barh(y + offset, vals, height=bar_h, color=color,
                       edgecolor="white", linewidth=0.3, alpha=0.8, label=label)

        for i, (c, v) in enumerate(zip(comps, vals)):
            ratio = ratios.get(c)
            if ratio is None:
                ax.text(0.3, i + offset, "X", va="center", fontsize=8,
                        color=color, fontweight="bold")
            elif v > 0:
                ax.text(v + 0.1, i + offset, f"{v:.1f}×",
                        va="center", fontsize=6.5, color="#333333")

    ax.axvline(1.0, color="#999999", lw=0.6, ls="--", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([COMP_LABELS[c] for c in comps], fontsize=7.5)
    ax.set_xlabel("Delay / baseline")
    ax.set_title("MLP ↔ Attention role reversal")
    ax.legend(fontsize=7, loc="lower right")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy",  default="results/exp3_taxonomy/taxonomy_results.json")
    parser.add_argument("--freeze",    default="results/exp3_freeze_per_task/freeze_per_task_s42.json")
    parser.add_argument("--out",       default="results/exp3_taxonomy/fig4_taxonomy")
    args = parser.parse_args()

    apply_style()

    has_taxonomy = os.path.exists(args.taxonomy)
    has_freeze = os.path.exists(args.freeze)

    if not has_taxonomy and not has_freeze:
        print("ERROR: No experiment 3 data found. Run experiments first.")
        return

    # Layout: 1 tall left panel + 2 stacked right panels
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.50,
                           width_ratios=[1.0, 1.2])

    ax_tax      = fig.add_subplot(gs[:, 0])   # left, spans both rows
    ax_heatmap  = fig.add_subplot(gs[0, 1])   # top-right
    ax_contrast = fig.add_subplot(gs[1, 1])   # bottom-right

    if has_taxonomy:
        print("Loading taxonomy…")
        tax = load_taxonomy(args.taxonomy)
        panel_taxonomy(ax_tax, tax)
        n_seeds = max(d["n"] for d in tax.values())
    else:
        ax_tax.text(0.5, 0.5, "No taxonomy data", transform=ax_tax.transAxes,
                    ha="center", fontsize=9, color="#999999")
        n_seeds = "?"

    if has_freeze:
        print("Loading freeze-per-task…")
        freeze_data = load_freeze_per_task(args.freeze)
        panel_heatmap(ax_heatmap, freeze_data)
        panel_contrast(ax_contrast, freeze_data)
    else:
        for ax in [ax_heatmap, ax_contrast]:
            ax.text(0.5, 0.5, "No freeze data", transform=ax.transAxes,
                    ha="center", fontsize=9, color="#999999")

    label_panel(ax_tax, "a")
    label_panel(ax_heatmap, "b")
    label_panel(ax_contrast, "c")

    fig.suptitle(f"Task taxonomy and per-task circuit analysis (n = {n_seeds} seeds)",
                 fontsize=10, fontweight="bold", y=1.01)

    save_fig(fig, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
