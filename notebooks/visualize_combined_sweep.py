"""
Combined freeze sweep visualization — both experiments together.

Shows the complete causal map for grokking:
  - freeze_one:        necessity test  (which component must keep updating?)
  - freeze_all_except: sufficiency test (which component can carry grokking alone?)

Together these answer: what is the minimal circuit required for grokking?

Run:
    uv run python notebooks/visualize_combined_sweep.py
"""

import sys, os


import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── load freeze_one results from data files ──────────────────────────────────
# Previous version hard-coded these values from a sweep run that was later
# overwritten. Now we load from saved artifacts to ensure reproducibility.

def _load_freeze_one(results_dir="results/exp1_freeze_sweep"):
    """
    Load freeze_one results from saved artifacts.
    
    Tries (in order):
      1. Immutable artifact: latest_freeze_one_p113_s42.json
      2. Legacy combined file: freeze_sweep_p113_s42.json
      3. Falls back to a clearly-marked placeholder with a warning
    """
    import os
    
    # Try immutable artifact first
    artifact_path = os.path.join(results_dir, "latest_freeze_one_p113_s42.json")
    if os.path.exists(artifact_path):
        with open(artifact_path) as f:
            data = json.load(f)
        results = data.get("results", data)  # handle both artifact and legacy format
        if isinstance(results, list):
            return {r["target_component"]: r for r in results}
    
    # Try legacy combined file
    legacy_path = os.path.join(results_dir, "freeze_sweep_p113_s42.json")
    if os.path.exists(legacy_path):
        with open(legacy_path) as f:
            raw = json.load(f)
        freeze_one = {r["target_component"]: r
                      for r in raw if r.get("freeze_action") == "freeze_one"}
        if freeze_one:
            return freeze_one
    
    print("WARNING: No freeze_one results found. Run:")
    print("  uv run python experiments/exp1_surgical_freeze/run_freeze_sweep.py --mode freeze_one")
    print("Visualization will be incomplete.")
    return {}


def _build_freeze_one_display(raw_results, baseline_delay):
    """Convert raw result dicts to the display format used by the plot."""
    display = {}
    for comp, r in raw_results.items():
        delay = r.get("grok_delay")
        ratio = delay / baseline_delay if delay is not None and baseline_delay else None
        display[comp] = dict(
            grokked=r.get("grokked", False),
            delay=delay,
            ratio=ratio,
            test_acc=r.get("final_test_acc", 0.0),
        )
    return display


def _load_baseline_delay(results_dir="results/exp1_baseline"):
    """Load baseline grokking delay from baseline results."""
    import os, glob
    # Try standard baseline file
    for pattern in ["baseline_p113_s42.json", "baseline_p113_*.json"]:
        matches = glob.glob(os.path.join(results_dir, pattern))
        if matches:
            with open(matches[0]) as f:
                data = json.load(f)
            delay = data.get("grok_delay")
            if delay is not None:
                return delay
    print("WARNING: No baseline results found. Using default delay=2300.")
    return 2300


BASELINE_DELAY = _load_baseline_delay()
_raw_freeze_one = _load_freeze_one()
FREEZE_ONE = _build_freeze_one_display(_raw_freeze_one, BASELINE_DELAY)

LABELS = {
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

COMP_ORDER = [
    "mlp_all", "embedding", "unembedding",
    "mlp_out", "mlp_in",
    "attn_O", "attn_V", "attn_Q", "attn_all", "attn_K",
]

# colours
C_BLOCK   = "#E63946"   # hard block (freeze_one: never groks)
C_SEVERE  = "#F4A261"   # 5-15× delay
C_MAJOR   = "#E9C46A"   # 2-5× delay
C_MINOR   = "#A8DADC"   # 1-2× delay
C_NEUTRAL = "#457B9D"   # ≤1×
C_FAIL    = "#CCCCCC"   # freeze_except: grokking fails
C_PARTIAL = "#8338EC"   # freeze_except: partial progress
C_BASE    = "#2D6A4F"


def delay_color(ratio):
    if ratio is None:  return C_BLOCK
    if ratio > 5:      return C_SEVERE
    if ratio > 2:      return C_MAJOR
    if ratio > 1:      return C_MINOR
    return C_NEUTRAL


# ── panel 1: side-by-side bars (necessity vs sufficiency) ────────────────────

def panel_dual_bars(ax_need, ax_suff, freeze_except):
    y   = list(range(len(COMP_ORDER)))
    names = [LABELS[c] for c in COMP_ORDER]

    # ── necessity (freeze_one) ──────────────────────────────────────────────
    for i, comp in enumerate(COMP_ORDER):
        fo    = FREEZE_ONE[comp]
        ratio = fo["ratio"]
        color = delay_color(ratio)
        val   = ratio if ratio is not None else 0
        ax_need.barh(i, val, color=color, height=0.65,
                     edgecolor="white", linewidth=0.5)
        if ratio is None:
            ax_need.text(0.3, i, "NEVER", va="center", fontsize=8.5,
                         color=C_BLOCK, fontweight="bold")
        else:
            tag = f"{ratio:.1f}×"
            if ratio < 1.0:
                tag += " (faster)"
            ax_need.text(val + 0.2, i, tag, va="center",
                         fontsize=8.5, color="#333333")

    ax_need.axvline(1.0, color=C_BASE, lw=1.5, ls="--", alpha=0.8)
    ax_need.text(1.02, len(COMP_ORDER) - 0.3, "baseline",
                 color=C_BASE, fontsize=7.5, va="top")
    ax_need.set_yticks(y)
    ax_need.set_yticklabels(names, fontsize=9)
    ax_need.set_xlim(0, 13)
    ax_need.set_xlabel("Delay × baseline  (higher = this component matters more)", fontsize=8.5)
    ax_need.set_title("NECESSITY TEST\nfreeze this component at step 700;\ndo other components carry grokking?",
                       fontsize=9.5, fontweight="bold", color="#333333")

    legend_items = [
        mpatches.Patch(color=C_BLOCK,   label="Blocks grokking entirely"),
        mpatches.Patch(color=C_SEVERE,  label=">5× delay"),
        mpatches.Patch(color=C_MAJOR,   label="2–5× delay"),
        mpatches.Patch(color=C_MINOR,   label="1–2× delay"),
        mpatches.Patch(color=C_NEUTRAL, label="≤1× (no effect or faster)"),
    ]
    ax_need.legend(handles=legend_items, fontsize=7.5, loc="lower right",
                   framealpha=0.9)

    # ── sufficiency (freeze_all_except) ────────────────────────────────────
    for i, comp in enumerate(COMP_ORDER):
        fe = freeze_except.get(comp, {})
        train = fe.get("final_train_acc", 0)
        test  = fe.get("final_test_acc", 0)

        # Two sub-bars: train (light) and test (saturated)
        bar_color = C_PARTIAL if test > 0.05 else C_FAIL
        ax_suff.barh(i + 0.18, train, height=0.3,
                     color=bar_color, alpha=0.4,
                     edgecolor="white", linewidth=0.3)
        ax_suff.barh(i - 0.18, test, height=0.3,
                     color=bar_color, alpha=0.95,
                     edgecolor="white", linewidth=0.3)

        ax_suff.text(max(train, test) + 0.01, i,
                     f"tr {train:.0%} / te {test:.0%}",
                     va="center", fontsize=7.5, color="#444444")

    ax_suff.axvline(1/113, color="gray", lw=0.8, ls=":",
                    alpha=0.6, label=f"Chance (1/p = {1/113:.1%})")
    ax_suff.set_yticks(y)
    ax_suff.set_yticklabels([""] * len(COMP_ORDER))  # shared y-axis
    ax_suff.set_xlim(0, 0.75)
    ax_suff.set_xlabel("Accuracy after 80K steps (light = train, dark = test)", fontsize=8.5)
    ax_suff.set_title("SUFFICIENCY TEST\nfreeze everything EXCEPT this;\ncan this component alone drive grokking?",
                       fontsize=9.5, fontweight="bold", color="#333333")

    legend_items2 = [
        mpatches.Patch(color=C_PARTIAL, alpha=0.9, label="Test acc > 5% (partial signal)"),
        mpatches.Patch(color=C_FAIL,    alpha=0.9, label="Test acc ≈ chance (failed)"),
    ]
    ax_suff.legend(handles=legend_items2, fontsize=7.5, loc="lower right",
                   framealpha=0.9)


# ── panel 2: necessity × sufficiency scatter ─────────────────────────────────

def panel_scatter(ax, freeze_except):
    """
    Each component plotted as a point:
      x-axis: freeze_one delay ratio (higher = more necessary)
      y-axis: freeze_all_except test accuracy (higher = more sufficient alone)
    Reveals which components are in which quadrant.
    """
    for comp in COMP_ORDER:
        fo    = FREEZE_ONE[comp]
        fe    = freeze_except.get(comp, {})
        ratio = fo["ratio"] if fo["ratio"] is not None else 14  # cap never
        test  = fe.get("final_test_acc", 0)
        label = LABELS[comp]
        color = delay_color(fo["ratio"])

        ax.scatter(ratio, test, color=color, s=120, zorder=4,
                   edgecolors="white", linewidths=0.8)

        # nudge labels to avoid overlap
        nudge = {
            "mlp_all":     (-0.6,  0.008),
            "embedding":   (-0.7, -0.012),
            "attn_all":    ( 0.2,  0.005),
            "attn_K":      ( 0.2,  0.000),
            "mlp_in":      ( 0.2,  0.003),
            "mlp_out":     ( 0.2, -0.005),
            "unembedding": ( 0.2,  0.001),
        }.get(comp, (0.2, 0))
        ax.text(ratio + nudge[0], test + nudge[1],
                label, fontsize=7.5, va="center", color="#333333")

    # Quadrant lines
    ax.axhline(0.05, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(2.0,  color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(14,   color=C_BLOCK, lw=0.8, ls="-.", alpha=0.4)

    # Quadrant labels
    ax.text(0.5,  0.065, "not necessary\nnot sufficient",
            fontsize=7.5, color="gray", alpha=0.7, ha="left")
    ax.text(3.0,  0.065, "necessary\nnot sufficient",
            fontsize=7.5, color="#AA4444", alpha=0.8, ha="left")
    ax.text(0.5,  0.50,  "not necessary\nbut sufficient alone",
            fontsize=7.5, color="gray", alpha=0.7, ha="left")

    ax.set_xlabel("Freeze-one delay ratio  (→ more necessary)", fontsize=9)
    ax.set_ylabel("Freeze-all-except test acc  (→ more sufficient alone)", fontsize=9)
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.01, 0.70)
    ax.set_title("Necessity vs Sufficiency\n(no component is sufficient alone; only MLP is hard-necessary)",
                 fontsize=10, fontweight="bold")

    # "NEVER" annotation for mlp_all
    ax.annotate("NEVER GROKS\nif frozen", xy=(14, 0.121),
                xytext=(11.5, 0.25),
                arrowprops=dict(arrowstyle="->", color=C_BLOCK, lw=1.0),
                fontsize=8, color=C_BLOCK, fontweight="bold")

    # Chance line
    ax.axhline(1/113, color="black", lw=0.6, ls=":", alpha=0.4)
    ax.text(13, 1/113 + 0.005, "chance", fontsize=7, color="gray")


# ── panel 3: summary interpretation ──────────────────────────────────────────

def panel_summary(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("What the two experiments together reveal",
                 fontsize=10, fontweight="bold")

    def card(x, y, w, h, bg, border, title, body, title_color="white"):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.22, title,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=title_color, zorder=3)
        ax.text(x + w/2, y + h/2 - 0.15, body,
                ha="center", va="center", fontsize=7.8,
                color="#222222", zorder=3, linespacing=1.5)

    # Necessity result
    card(0.3, 5.0, 4.2, 2.6,
         "#FDECEA", C_BLOCK,
         "Only MLP is necessary",
         "Freezing mlp_all is the only hard block.\n"
         "Every other component can be frozen at step 700\n"
         "and grokking still fires (with varying delay).\n"
         "Conclusion: the MLP contains the critical\n"
         "nonlinear step that cannot be bypassed.",
         title_color=C_BLOCK)

    # Sufficiency result
    card(5.3, 5.0, 4.2, 2.6,
         "#EEF4FF", "#3A86FF",
         "No component is sufficient alone",
         "When only one component can update\n"
         "(all others frozen at random init),\n"
         "grokking fails for every component.\n"
         "MLP alone makes most progress (12% test)\n"
         "but cannot generate the Fourier basis\n"
         "it needs without the embedding.",
         title_color="#1A56DB")

    # Joint conclusion
    card(1.5, 1.2, 7.0, 3.2,
         "#F0FDF4", C_BASE,
         "Grokking is a coordinated circuit transition",
         "The minimum viable circuit requires joint updates in multiple components.\n"
         "Embedding must form the Fourier basis (circular number representation).\n"
         "MLP must learn the trig-identity products (uses the Fourier basis).\n"
         "These two depend on each other: MLP without Fourier embeddings → 12% test.\n"
         "Fourier embeddings without MLP nonlinearity → 0.1% test.\n\n"
         "Next experiment: freeze_all_except [mlp_all + embedding] together.\n"
         "Prediction: this pair is the minimal sufficient set.",
         title_color=C_BASE)

    # Arrow between the two top cards
    ax.annotate("", xy=(5.3, 6.3), xytext=(4.5, 6.3),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2))
    ax.text(4.9, 6.55, "+", fontsize=14, ha="center", color="#888888")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out = "results/exp1_freeze_sweep/combined_sweep_analysis.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open("results/exp1_freeze_sweep/freeze_sweep_p113_s42.json") as f:
        raw = json.load(f)
    freeze_except = {r["target_component"]: r
                     for r in raw if r["freeze_action"] == "freeze_all_except"}

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor("#F9F9F9")

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.40, wspace=0.32,
                           left=0.06, right=0.97,
                           top=0.93,  bottom=0.06)

    ax_need    = fig.add_subplot(gs[0, 0])
    ax_suff    = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_summary = fig.add_subplot(gs[1, :])

    for ax in [ax_need, ax_suff, ax_scatter]:
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#CCCCCC")
    ax_summary.set_facecolor("#FAFAFA")

    panel_dual_bars(ax_need, ax_suff, freeze_except)
    panel_scatter(ax_scatter, freeze_except)
    panel_summary(ax_summary)

    fig.suptitle(
        "Exp 1 — Complete Causal Map: Necessity + Sufficiency  "
        "(mod_add p=113, freeze at memorisation step 700)",
        fontsize=13, fontweight="bold", y=0.975,
    )

    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
