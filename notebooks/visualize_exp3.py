"""
Exp 3 visualization — Task Taxonomy + Freeze-per-Task.

Three panels:
  1. Task Taxonomy — which tasks grok, grok delay, and group properties.
  2. Component Importance Heatmap — for each task × component, how much
     does freezing at memorisation delay (or block) grokking?
  3. The Contrast Story — delay ratio bars for mod_add vs s5_compose,
     illustrating the MLP ↔ Attention role reversal.

Run:
    uv run python notebooks/visualize_exp3.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── palette ──────────────────────────────────────────────────────────────────
C_NEVER   = "#E63946"
C_SEVERE  = "#F4A261"
C_MAJOR   = "#E9C46A"
C_MINOR   = "#A8DADC"
C_NEUTRAL = "#457B9D"
C_BASE    = "#2D6A4F"

C_ABELIAN     = "#3A86FF"
C_NONABELIAN  = "#E63946"
C_MIXED       = "#FB8500"  # non-abelian but grokked (mod_div)

BG = "#F9F9F9"

TASK_ORDER = ["parity_8bit", "xor_5bit", "xor_6bit", "mod_add", "mod_mul",
              "mod_div", "mod_exp", "s5_compose", "dihedral_12"]

TASK_LABELS = {
    "mod_add":     "Mod Addition\n(Z₁₁₃, abelian)",
    "mod_mul":     "Mod Multiply\n(Z₁₁₃, abelian)",
    "mod_div":     "Mod Division\n(Z₁₁₃, abelian)",
    "mod_exp":     "Mod Exponent\n(Z₉₇, non-comm.)",
    "s5_compose":  "S₅ Compose\n(non-abelian, |G|=120)",
    "dihedral_12": "Dihedral D₁₂\n(non-abelian, |G|=24)",
    "xor_6bit":    "XOR 6-bit\n(Z₂⁶, abelian)",
    "xor_5bit":    "XOR 5-bit\n(Z₂⁵, abelian)",
    "parity_8bit": "Parity 8-bit\n(Z₂, abelian)",
}

COMP_LABELS = {
    "baseline":    "Baseline",
    "mlp_all":     "MLP (all)",
    "embedding":   "Embedding",
    "unembedding": "Unembedding",
    "attn_all":    "Attention (all)",
    "attn_Q":      "Attn Q",
    "attn_K":      "Attn K",
    "attn_V":      "Attn V",
    "attn_O":      "Attn O",
    "mlp_in":      "MLP W_in",
    "mlp_out":     "MLP W_out",
}

COMP_ORDER = ["mlp_all", "mlp_in", "mlp_out", "attn_all", "attn_Q", "attn_K",
              "attn_V", "attn_O", "embedding", "unembedding"]

# ── data loading ─────────────────────────────────────────────────────────────

def load_taxonomy(path):
    with open(path) as f:
        rows = json.load(f)
    # Average over seeds
    by_task = {}
    for r in rows:
        t = r["task_name"]
        by_task.setdefault(t, []).append(r)
    agg = {}
    for t, rs in by_task.items():
        grokked = [r["grokked"] for r in rs]
        delays  = [r["grok_delay"] for r in rs if r["grok_delay"] is not None]
        accs    = [r["final_test_acc"] for r in rs]
        agg[t] = {
            "grokked":       all(grokked),
            "any_grokked":   any(grokked),
            "mean_delay":    np.mean(delays) if delays else None,
            "mean_test_acc": np.mean(accs),
            "is_commutative": rs[0]["is_commutative"],
            "group_order":   rs[0]["group_order"],
            "vocab_size":    rs[0]["vocab_size"],
            "operation":     rs[0]["operation"],
        }
    return agg


def load_freeze(path):
    with open(path) as f:
        return json.load(f)


def build_delay_matrix(freeze_data):
    """
    Returns (tasks, comps, matrix) where matrix[i][j] is the delay ratio
    relative to baseline for task i, component j. None = blocked.
    Special: -1 = baseline never grokked (skip task for ratio).
    """
    # Group by task
    by_task = {}
    for r in freeze_data:
        by_task.setdefault(r["task"], []).append(r)

    tasks = [t for t in TASK_ORDER if t in by_task]
    comps = COMP_ORDER

    matrix = []
    for task in tasks:
        rows = {r["component"]: r for r in by_task[task]}
        base = rows.get("baseline")
        base_delay = base["grok_delay"] if base and base["grokked"] else None

        row = []
        for comp in comps:
            r = rows.get(comp)
            if r is None:
                row.append(np.nan)
            elif not r["grokked"]:
                row.append(None)  # blocked
            elif base_delay is None:
                # baseline didn't grok — can't compute ratio; use raw delay
                row.append(np.nan)
            elif base_delay == 0:
                # parity: grok_delay=0 for baseline; use absolute step as proxy
                row.append(1.0)
            else:
                row.append(r["grok_delay"] / base_delay)
        matrix.append(row)

    return tasks, comps, matrix


# ── panel 1: task taxonomy bar chart ─────────────────────────────────────────

def panel_taxonomy(ax, tax):
    tasks = [t for t in TASK_ORDER if t in tax]
    n = len(tasks)
    y = np.arange(n)

    delays = []
    colors = []
    hatches = []
    for t in tasks:
        d = tax[t]
        delay = d["mean_delay"]
        delays.append(delay if delay is not None else 0)

        if not d["grokked"]:
            colors.append(C_NEVER)
            hatches.append("////")
        elif not d["is_commutative"]:
            colors.append(C_MIXED)
            hatches.append("")
        else:
            colors.append(C_ABELIAN)
            hatches.append("")

    max_delay = max(d for d in delays if d and d > 0)

    for i, (task, delay, color, hatch) in enumerate(zip(tasks, delays, colors, hatches)):
        d = tax[task]
        if d["grokked"]:
            ax.barh(i, delay, color=color, hatch=hatch,
                    edgecolor="white", linewidth=0.6, height=0.65, zorder=3)
            label = f"{int(delay):,} steps  ({d['mean_test_acc']:.1%})"
            ax.text(delay + max_delay * 0.02, i, label,
                    va="center", fontsize=8, color="#333333")
        else:
            ax.barh(i, max_delay * 0.12, color=color, hatch=hatch,
                    edgecolor=C_NEVER, linewidth=1.0, height=0.65, zorder=3)
            ax.text(max_delay * 0.14, i,
                    f"NEVER  ({d['mean_test_acc']:.1%} final test acc)",
                    va="center", fontsize=8, color=C_NEVER, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks], fontsize=8.5)
    ax.set_xlabel("Mean grokking delay (steps, avg over 2 seeds)", fontsize=9)
    ax.set_title("Task taxonomy: which algebraic structures grok?\n"
                 "(all tasks: 1-layer transformer, 30% train split, up to 50K steps)",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, max_delay * 1.45)
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#CCCCCC")

    legend_items = [
        mpatches.Patch(color=C_ABELIAN,    label="Abelian — grokked"),
        mpatches.Patch(color=C_MIXED,      label="Non-commutative — grokked (slower)"),
        mpatches.Patch(color=C_NEVER, hatch="////",
                       label="Failed to grok", edgecolor=C_NEVER),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)


# ── panel 2: component heatmap ───────────────────────────────────────────────

def panel_heatmap(ax, tasks, comps, matrix):
    """
    Heatmap: rows = tasks, cols = components.
    Color encodes delay ratio (log scale). Blocked = deep red with X.
    NaN = grey (task not run for that component).
    """
    n_tasks = len(tasks)
    n_comps = len(comps)

    # Build numeric array: blocked → 999, NaN → NaN
    arr = np.full((n_tasks, n_comps), np.nan)
    blocked = np.zeros((n_tasks, n_comps), dtype=bool)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val is None:
                blocked[i, j] = True
                arr[i, j] = 20.0   # sentinel for color scale
            elif not np.isnan(val):
                arr[i, j] = val

    # Color map: blue (fast/neutral) → white (1×) → orange → red
    cmap_colors = [C_ABELIAN, "#FFFFFF", C_MAJOR, C_SEVERE, C_NEVER]
    cmap_positions = [0.0, 0.2, 0.5, 0.75, 1.0]
    cmap = LinearSegmentedColormap.from_list(
        "delay_map", list(zip(cmap_positions, cmap_colors))
    )

    # Clip display range to [0, 10] for ratio; blocked shown as max
    display = np.clip(arr, 0, 10)
    masked = np.ma.masked_where(np.isnan(arr), display)

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=10,
                   interpolation="nearest")

    # Mark blocked cells
    for i in range(n_tasks):
        for j in range(n_comps):
            if blocked[i, j]:
                ax.text(j, i, "✕", ha="center", va="center",
                        fontsize=13, color="white", fontweight="bold")
            elif not np.isnan(arr[i, j]):
                val = arr[i, j]
                txt = f"{val:.1f}×" if val < 10 else f"{val:.0f}×"
                color = "white" if val > 4 or val < 0.6 else "#333333"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=color)

    ax.set_xticks(range(n_comps))
    ax.set_xticklabels([COMP_LABELS[c] for c in comps],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks], fontsize=8)
    ax.set_title("Component freeze impact across tasks\n"
                 "(cell = grok delay ratio vs baseline; ✕ = grokking blocked)",
                 fontsize=10.5, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Delay ratio (1× = same as baseline)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Horizontal separators between task groups
    # abelian grok | non-abelian grok | failed
    sep_after = []
    task_groups = [TASK_ORDER.index(t) if t in TASK_ORDER else -1 for t in tasks]
    prev_group = None
    for i, t in enumerate(tasks):
        tax_grokked = None  # we don't have taxonomy here; use color from matrix
        # detect transition: parity/xor/mod_add/mod_mul → mod_div → mod_exp/s5/dihedral
        if t == "mod_div":
            sep_after.append(i - 0.5)
        if t == "mod_exp":
            sep_after.append(i - 0.5)

    for y in sep_after:
        ax.axhline(y, color="#888888", linewidth=1.5, linestyle="--", alpha=0.6)


# ── panel 3: contrast bars — mod_add vs s5_compose ───────────────────────────

def panel_contrast(ax_add, ax_s5, freeze_data):
    """Side-by-side delay ratio bars for mod_add vs s5_compose."""

    def get_ratios(task):
        rows = {r["component"]: r for r in freeze_data if r["task"] == task}
        base = rows.get("baseline")
        base_delay = base["grok_delay"] if base and base["grokked"] else None

        ratios = {}
        for comp in COMP_ORDER:
            r = rows.get(comp)
            if r is None:
                ratios[comp] = np.nan
            elif not r["grokked"]:
                ratios[comp] = None   # blocked
            elif base_delay is None or base_delay == 0:
                ratios[comp] = 1.0
            else:
                ratios[comp] = r["grok_delay"] / base_delay
        return ratios

    def draw_bars(ax, task, ratios, title, note):
        comps = [c for c in COMP_ORDER if not np.isnan(ratios.get(c, np.nan) or 0)]
        y = np.arange(len(comps))
        vals   = []
        colors = []
        for c in comps:
            v = ratios[c]
            vals.append(v if v is not None else 0)
            if v is None:
                colors.append(C_NEVER)
            elif v > 5:
                colors.append(C_SEVERE)
            elif v > 2:
                colors.append(C_MAJOR)
            elif v > 1:
                colors.append(C_MINOR)
            else:
                colors.append(C_NEUTRAL)

        bars = ax.barh(y, vals, color=colors, edgecolor="white",
                       linewidth=0.6, height=0.65)

        max_val = max(v for v in vals if v and v > 0) if any(vals) else 10
        for i, (comp, v, col) in enumerate(zip(comps, vals, colors)):
            ratio = ratios[comp]
            if ratio is None:
                ax.text(0.5, i, "BLOCKED",
                        va="center", fontsize=8.5, color=C_NEVER, fontweight="bold")
            else:
                label = f"{ratio:.1f}×"
                ax.text(v + max_val * 0.02, i, label,
                        va="center", fontsize=8, color="#333333")

        ax.axvline(1.0, color=C_BASE, lw=1.5, ls="--", alpha=0.8)
        ax.text(1.02, len(comps) - 0.3, "baseline", color=C_BASE, fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels([COMP_LABELS[c] for c in comps], fontsize=9)
        ax.set_xlabel("Grok delay × baseline", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(0, max_val * 1.5)
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#CCCCCC")
        ax.text(0.97, 0.04, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, style="italic",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                          edgecolor="#CCCCCC", alpha=0.9))

    ratios_add = get_ratios("mod_add")
    ratios_s5  = get_ratios("s5_compose")

    draw_bars(ax_add, "mod_add", ratios_add,
              "mod_add (Z₁₁₃)  — MLP is critical, attention dispensable",
              "Baseline grok delay: 4500 steps\nFreeze at step: 1000 (post-memorisation)")

    draw_bars(ax_s5, "s5_compose", ratios_s5,
              "S₅ compose (non-abelian)  — Attention is critical, MLP dispensable",
              "Baseline grok delay: 47500 steps\nFreeze at step: 1000 (post-memorisation)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    tax_path    = "results/exp3_taxonomy/taxonomy_results.json"
    freeze_path = "results/exp3_freeze_per_task/freeze_per_task_s42.json"
    out_path    = "results/exp3_taxonomy/exp3_analysis.png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Loading data…")
    tax = load_taxonomy(tax_path)
    freeze_data = load_freeze(freeze_path)

    tasks, comps, matrix = build_delay_matrix(freeze_data)

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row 0: taxonomy bar (left, tall) | heatmap (right, tall)
    # Row 1: contrast bars left | contrast bars right
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.10, right=0.97, top=0.92, bottom=0.06,
        height_ratios=[1.1, 1.0],
    )

    ax_tax  = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_add  = fig.add_subplot(gs[1, 0])
    ax_s5   = fig.add_subplot(gs[1, 1])

    print("Panel 1: Task taxonomy…")
    panel_taxonomy(ax_tax, tax)

    print("Panel 2: Heatmap…")
    panel_heatmap(ax_heat, tasks, comps, matrix)

    print("Panel 3: Contrast bars…")
    panel_contrast(ax_add, ax_s5, freeze_data)

    fig.suptitle(
        "Exp 3 — Task Taxonomy & Per-Task Circuit Analysis\n"
        "Abelian groups grok reliably; MLP vs. Attention role reverses across group types",
        fontsize=13, fontweight="bold", color="#222222", y=0.975,
    )

    print(f"Saving to {out_path}…")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Done.")


if __name__ == "__main__":
    main()
