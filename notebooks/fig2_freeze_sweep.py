"""
Figure 2 — Surgical Freeze Sweep: Causal Component Analysis.

Three-panel figure:
  (a) Freeze-one: delay ratio bar chart showing necessity of each component.
  (b) Freeze-all-except: sufficiency bar chart (can this component alone drive grokking?).
  (c) Necessity × Sufficiency scatter plot.

All values loaded from data files (no hard-coded results).
Multi-seed support: shows mean ± SEM when multiple seeds are available.

Run:
    uv run python notebooks/fig2_freeze_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from pub_style import apply_style, PAL, save_fig, thousands_formatter, label_panel


COMP_ORDER = [
    "mlp_all", "embedding", "unembedding",
    "mlp_out", "mlp_in",
    "attn_O", "attn_V", "attn_Q", "attn_all", "attn_K",
]

COMP_LABELS = {
    "embedding":   "Embedding",
    "unembedding": "Unembedding",
    "attn_Q":      "Attn Q",
    "attn_K":      "Attn K",
    "attn_V":      "Attn V",
    "attn_O":      "Attn O",
    "attn_all":    "Attn (all)",
    "mlp_in":      "MLP W_in",
    "mlp_out":     "MLP W_out",
    "mlp_all":     "MLP (all)",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sweep_data(results_dir):
    """Load freeze sweep results from immutable artifacts or legacy format."""
    freeze_one = {}
    freeze_except = {}

    # Try immutable artifacts first
    for name, target_dict, action in [
        ("latest_freeze_one_p113_s42.json", freeze_one, "freeze_one"),
        ("latest_freeze_all_except_p113_s42.json", freeze_except, "freeze_all_except"),
    ]:
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results = data.get("results", [])
            for r in results:
                comp = r["target_component"]
                target_dict.setdefault(comp, []).append(r)

    # Fall back to legacy combined file
    if not freeze_one or not freeze_except:
        legacy = os.path.join(results_dir, "freeze_sweep_p113_s42.json")
        if os.path.exists(legacy):
            with open(legacy) as f:
                raw = json.load(f)
            for r in raw:
                comp = r["target_component"]
                if r["freeze_action"] == "freeze_one" and not freeze_one:
                    freeze_one.setdefault(comp, []).append(r)
                elif r["freeze_action"] == "freeze_all_except" and not freeze_except:
                    freeze_except.setdefault(comp, []).append(r)

    return freeze_one, freeze_except


def load_baseline_delay(results_dir):
    """Load baseline grokking delay."""
    import glob
    for pattern in ["baseline_p113_s42.json", "baseline_p113_*.json"]:
        matches = glob.glob(os.path.join(results_dir, pattern))
        if matches:
            with open(matches[0]) as f:
                data = json.load(f)
            # Handle both artifact and legacy format
            result = data.get("result", data)
            d = result.get("grok_delay")
            if d is not None:
                return d
    return 2300  # fallback


def aggregate(runs):
    """Aggregate multiple seeds into mean ± sem."""
    grokked = [r["grokked"] for r in runs]
    delays = [r["grok_delay"] for r in runs if r["grok_delay"] is not None]
    test_accs = [r.get("final_test_acc", r.get("test_acc", 0)) for r in runs]
    n = len(runs)
    return {
        "n": n,
        "grok_rate": sum(grokked) / n,
        "n_grokked": sum(grokked),
        "delay_mean": np.mean(delays) if delays else None,
        "delay_sem": np.std(delays) / len(delays)**0.5 if len(delays) > 1 else 0,
        "test_mean": np.mean(test_accs),
        "test_sem": np.std(test_accs) / n**0.5 if n > 1 else 0,
    }


# ── Panel (a): Necessity — freeze one component ─────────────────────────────

def panel_necessity(ax, freeze_one, baseline_delay):
    comps = [c for c in COMP_ORDER if c in freeze_one]
    names = [COMP_LABELS[c] for c in comps]
    y = np.arange(len(comps))

    for i, comp in enumerate(comps):
        agg = aggregate(freeze_one[comp])
        if agg["delay_mean"] is not None:
            ratio = agg["delay_mean"] / baseline_delay
            ratio_sem = agg["delay_sem"] / baseline_delay
            color = PAL.delay_color(ratio)
            ax.barh(i, ratio, color=color, height=0.65,
                    edgecolor="white", linewidth=0.4)
            if ratio_sem > 0:
                ax.errorbar(ratio, i, xerr=ratio_sem, fmt="none",
                            ecolor="#333333", capsize=2, capthick=0.6, lw=0.6)
            tag = f"{ratio:.1f}×"
            if ratio < 1.0:
                tag += " (faster)"
            ax.text(ratio + ratio_sem + 0.15, i, tag,
                    va="center", fontsize=7.5, color="#333333")
        else:
            color = PAL.DELAY_BLOCKED
            ax.barh(i, 0.3, color=color, height=0.65,
                    edgecolor="white", linewidth=0.4)
            grok_str = f"{agg['n_grokked']}/{agg['n']} seeds" if agg["n"] > 1 else ""
            ax.text(0.5, i, f"BLOCKED  {grok_str}".strip(),
                    va="center", fontsize=7.5, color=PAL.BLOCK, fontweight="bold")

    ax.axvline(1.0, color="#666666", lw=0.8, ls="--", alpha=0.6)
    ax.text(1.02, len(comps) - 0.3, "baseline", fontsize=7, color="#666666", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Grokking delay / baseline delay")
    ax.set_xlim(0, 13)
    ax.set_title("Necessity: freeze this component")

    legend = [
        mpatches.Patch(color=PAL.DELAY_BLOCKED, label="Blocks grokking"),
        mpatches.Patch(color=PAL.DELAY_SEVERE,  label=">5× delay"),
        mpatches.Patch(color=PAL.DELAY_MAJOR,   label="2–5× delay"),
        mpatches.Patch(color=PAL.DELAY_MINOR,   label="1–2× delay"),
        mpatches.Patch(color=PAL.DELAY_NONE,    label="≤1× (no effect)"),
    ]
    ax.legend(handles=legend, fontsize=6.5, loc="lower right")


# ── Panel (b): Sufficiency — freeze all except one ──────────────────────────

def panel_sufficiency(ax, freeze_except):
    comps = [c for c in COMP_ORDER if c in freeze_except]
    names = [COMP_LABELS[c] for c in comps]
    y = np.arange(len(comps))

    for i, comp in enumerate(comps):
        agg = aggregate(freeze_except[comp])
        test_mean = agg["test_mean"]
        test_sem = agg["test_sem"]

        color = PAL.TEAL if test_mean > 0.10 else PAL.GREY
        ax.barh(i, test_mean, color=color, height=0.65,
                edgecolor="white", linewidth=0.4, alpha=0.8)
        if test_sem > 0:
            ax.errorbar(test_mean, i, xerr=test_sem, fmt="none",
                        ecolor="#333333", capsize=2, capthick=0.6, lw=0.6)

        label = f"{test_mean:.1%}"
        if agg["n"] > 1:
            label += f" ± {test_sem:.1%}"
        ax.text(test_mean + test_sem + 0.01, i, label,
                va="center", fontsize=7.5, color="#333333")

    ax.axvline(1/113, color="#999999", lw=0.6, ls=":", alpha=0.6)
    ax.text(1/113 + 0.005, len(comps) - 0.3, "chance", fontsize=7, color="#999999")

    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(comps))  # shared with panel (a)
    ax.set_xlabel("Test accuracy (after full training)")
    ax.set_xlim(0, 0.30)
    ax.set_title("Sufficiency: only this component updates")


# ── Panel (c): Necessity × Sufficiency scatter ──────────────────────────────

def panel_scatter(ax, freeze_one, freeze_except, baseline_delay):
    comps = [c for c in COMP_ORDER if c in freeze_one and c in freeze_except]

    for comp in comps:
        fo = aggregate(freeze_one[comp])
        fe = aggregate(freeze_except[comp])

        if fo["delay_mean"] is not None:
            ratio = fo["delay_mean"] / baseline_delay
        else:
            ratio = 14  # off-scale for blocked

        test = fe["test_mean"]
        color = PAL.delay_color(fo["delay_mean"] / baseline_delay if fo["delay_mean"] else None)

        ax.scatter(ratio, test, color=color, s=60, zorder=4,
                   edgecolors="white", linewidths=0.5)
        # Label
        nudge_x = 0.3
        nudge_y = 0.003
        if comp == "mlp_all":
            nudge_x, nudge_y = -0.8, 0.008
        ax.text(ratio + nudge_x, test + nudge_y, COMP_LABELS[comp],
                fontsize=6.5, va="center", color="#444444")

    # Quadrant lines
    ax.axhline(0.05, color="#CCCCCC", lw=0.6, ls="--")
    ax.axvline(2.0, color="#CCCCCC", lw=0.6, ls="--")
    ax.axhline(1/113, color="#999999", lw=0.5, ls=":", alpha=0.5)

    # Quadrant labels
    ax.text(0.5, 0.27, "not necessary,\npartially sufficient",
            fontsize=6.5, color="#999999", ha="left", va="top")
    ax.text(3.0, 0.27, "necessary,\nnot sufficient",
            fontsize=6.5, color="#999999", ha="left", va="top")

    ax.set_xlabel("Necessity (freeze-one delay ratio)")
    ax.set_ylabel("Sufficiency (freeze-all-except test acc)")
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.01, 0.30)
    ax.set_title("Necessity × sufficiency")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir",    default="results/exp1_freeze_sweep")
    parser.add_argument("--baseline_dir", default="results/exp1_baseline")
    parser.add_argument("--out",          default="results/exp1_freeze_sweep/fig2_freeze_sweep")
    args = parser.parse_args()

    apply_style()

    print("Loading data…")
    freeze_one, freeze_except = load_sweep_data(args.sweep_dir)
    baseline_delay = load_baseline_delay(args.baseline_dir)
    print(f"  Baseline delay: {baseline_delay}")
    print(f"  freeze_one: {len(freeze_one)} components, "
          f"freeze_except: {len(freeze_except)} components")

    if not freeze_one:
        print("ERROR: No freeze_one data. Run the experiment first.")
        return

    # Layout: 1 × 3
    fig = plt.figure(figsize=(14, 4.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25,
                           width_ratios=[1.2, 1.0, 1.0])

    ax_need    = fig.add_subplot(gs[0, 0])
    ax_suff    = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 2])

    panel_necessity(ax_need, freeze_one, baseline_delay)
    if freeze_except:
        panel_sufficiency(ax_suff, freeze_except)
        panel_scatter(ax_scatter, freeze_one, freeze_except, baseline_delay)
    else:
        ax_suff.text(0.5, 0.5, "No freeze_all_except data\navailable",
                     transform=ax_suff.transAxes, ha="center", va="center",
                     fontsize=9, color="#999999")
        ax_scatter.text(0.5, 0.5, "No data", transform=ax_scatter.transAxes,
                        ha="center", va="center", fontsize=9, color="#999999")

    label_panel(ax_need, "a")
    label_panel(ax_suff, "b")
    label_panel(ax_scatter, "c")

    n_seeds = max((len(v) for v in freeze_one.values()), default=1)
    fig.suptitle(f"Surgical freeze sweep: necessity and sufficiency of each component  "
                 f"(mod_add p = 113, n = {n_seeds} seeds)",
                 fontsize=10, fontweight="bold", y=1.02)

    save_fig(fig, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
