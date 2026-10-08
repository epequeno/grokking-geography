"""
Exp2 — Label noise visualisation (multi-seed, extended runs).

Loads from all three result directories and deduplicates to the longest
run per (noise, seed) pair.

Run:
    uv run python notebooks/visualize_label_noise.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict

# ── data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def load_all():
    """Load all runs, deduplicate to longest per (noise, seed)."""
    raw = (load_json("results/exp2_label_noise/noise_sweep_curves_p113.json") +
           load_json("results/exp2_label_noise_ext/noise_sweep_curves_p113.json") +
           load_json("results/exp2_label_noise_ext_mid/noise_sweep_curves_p113.json"))

    best = {}
    for r in raw:
        key = (round(r["noise_frac"], 2), r["seed"])
        if key not in best or len(r["steps"]) > len(best[key]["steps"]):
            best[key] = r

    results = sorted(best.values(), key=lambda r: (r["noise_frac"], r["seed"]))
    by_noise = defaultdict(list)
    for r in results:
        by_noise[round(r["noise_frac"], 2)].append(r)
    return results, by_noise


# ── colour helpers ────────────────────────────────────────────────────────────

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
CMAP = plt.cm.RdYlGn_r

def noise_color(n):
    return CMAP(n / 0.40)

# Regime colours for background shading
R_CLEAN   = "#06D6A0"
R_CORRUPT = "#F4A261"
R_FAIL    = "#E63946"


# ── summary stats ─────────────────────────────────────────────────────────────

def summarise(by_noise):
    rows = []
    for noise in NOISE_LEVELS:
        runs = by_noise.get(noise, [])
        if not runs:
            continue
        tests    = [r["test_acc"][-1]        for r in runs]
        fouriers = [r["fourier_alignment"][-1] for r in runs]
        rows.append({
            "noise":      noise,
            "n":          len(runs),
            "grokked":    sum(1 for r in runs if r["grokked"]),
            "test_mean":  np.mean(tests),
            "test_std":   np.std(tests),
            "fa_mean":    np.mean(fouriers),
            "fa_std":     np.std(fouriers),
        })
    return rows


# ── panels ────────────────────────────────────────────────────────────────────

def panel_test_acc(ax, rows):
    noises = [r["noise"] for r in rows]
    means  = [r["test_mean"] for r in rows]
    stds   = [r["test_std"]  for r in rows]

    ax.fill_between(noises, [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                    alpha=0.18, color="#888888", zorder=1)
    ax.plot(noises, means, color="#333333", lw=1.5, zorder=2)

    for r in rows:
        color = noise_color(r["noise"])
        marker = "*" if r["grokked"] == r["n"] else "o"
        ax.scatter(r["noise"], r["test_mean"], color=color, s=110, zorder=4,
                   marker=marker, edgecolors="white", linewidths=0.7)

    # Regime shading
    ax.axvspan(-0.01, 0.02,  color=R_CLEAN,   alpha=0.08, zorder=0)
    ax.axvspan(0.02,  0.22,  color=R_CORRUPT,  alpha=0.08, zorder=0)
    ax.axvspan(0.22,  0.43,  color=R_FAIL,     alpha=0.06, zorder=0)

    ax.axhline(0.99, color=R_CLEAN, lw=0.9, ls="--", alpha=0.7, label="Grokking threshold")
    ax.axhline(1/113, color="gray", lw=0.7, ls=":", alpha=0.5, label="Chance")

    ax.text(0.01, 0.50, "Clean\ngrokking", ha="center", fontsize=8,
            color=R_CLEAN, fontweight="bold", transform=ax.get_xaxis_transform(),
            va="bottom")
    ax.text(0.12, 0.50, "Corrupted\nfixed point", ha="center", fontsize=8,
            color=R_CORRUPT, fontweight="bold", transform=ax.get_xaxis_transform(),
            va="bottom")
    ax.text(0.33, 0.50, "Circuit\nfailure", ha="center", fontsize=8,
            color=R_FAIL, fontweight="bold", transform=ax.get_xaxis_transform(),
            va="bottom")

    ax.set_xlabel("Label noise fraction", fontsize=10)
    ax.set_ylabel("Final test accuracy (clean test set)", fontsize=10)
    ax.set_title("Test accuracy vs label noise  (n=3 seeds, error = ±1 std)\n"
                 "★ = all seeds grokked", fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.02, 0.43)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")


def panel_fourier_acc(ax, rows):
    noises  = [r["noise"]   for r in rows]
    fa_mean = [r["fa_mean"] for r in rows]
    fa_std  = [r["fa_std"]  for r in rows]
    te_mean = [r["test_mean"] for r in rows]

    ax.scatter(fa_mean, te_mean,
               c=[noise_color(n) for n in noises], s=110, zorder=4,
               edgecolors="white", linewidths=0.7)

    # Label each point
    for r in rows:
        ax.annotate(f"{r['noise']:.0%}",
                    (r["fa_mean"], r["test_mean"]),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=8, color="#333333")

    ax.axhline(0.99, color=R_CLEAN, lw=0.9, ls="--", alpha=0.7, label="Grokking threshold")
    ax.axhline(1/113, color="gray", lw=0.7, ls=":", alpha=0.5, label="Chance")

    ax.set_xlabel("Final Fourier alignment score", fontsize=10)
    ax.set_ylabel("Final test accuracy", fontsize=10)
    ax.set_title("Test accuracy vs Fourier alignment\n"
                 "(both measured at end of run; labelled by noise %)",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0.05, 0.78)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=8.5, framealpha=0.9)


def panel_learning_curves(ax, by_noise):
    """Individual runs coloured by noise level."""
    for noise in NOISE_LEVELS:
        runs = by_noise.get(noise, [])
        color = noise_color(noise)
        lw    = 2.2 if noise <= 0.05 else 1.2
        alpha = 0.95 if noise == 0.0 else 0.70

        for i, r in enumerate(runs):
            label = f"noise={noise:.0%}" if i == 0 else None
            ax.plot(r["steps"], r["test_acc"],
                    color=color, lw=lw, alpha=alpha, label=label)

    ax.axhline(0.99, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Test accuracy (clean test set)", fontsize=10)
    ax.set_title("Learning curves — all seeds and noise levels\n"
                 "(0.05–0.20 extended to 200K steps)",
                 fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.9, loc="upper left")


def panel_corrupted_fixed_points(ax, by_noise):
    """
    Show the 0.05–0.20 cases with mean ± std band, highlighting the
    corrupted-fixed-point regime: circuit forms but test acc stabilises
    below the clean-grokking threshold.
    """
    for noise in [0.05, 0.10, 0.15, 0.20]:
        runs  = by_noise.get(noise, [])
        color = noise_color(noise)

        # Truncate all to the shortest run for averaging
        min_len = min(len(r["test_acc"]) for r in runs)
        steps_ref = runs[0]["steps"][:min_len]
        mat   = np.array([r["test_acc"][:min_len] for r in runs])
        mean  = mat.mean(0)
        std   = mat.std(0)

        ax.fill_between(steps_ref, mean - std, mean + std,
                        alpha=0.18, color=color)
        ax.plot(steps_ref, mean, color=color, lw=2,
                label=f"noise={noise:.0%}  (final: {mean[-1]:.0%}±{std[-1]:.0%})")

    ax.axhline(0.99, color=R_CLEAN, lw=1.0, ls="--", alpha=0.7, label="Clean grokking threshold")
    ax.axhline(1/113, color="gray", lw=0.7, ls=":", alpha=0.5, label="Chance")

    ax.text(0.55, 0.78,
            "None of these noise levels\ngrok after 200K steps.\n"
            "Each settles at a stable\n'corrupted fixed point'\ndetermined by the noise level.",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                      edgecolor=R_CORRUPT, alpha=0.92))

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Test accuracy (mean ± 1 std across seeds)", fontsize=10)
    ax.set_title("Corrupted fixed points at 5–20% noise (200K steps)\n"
                 "Circuit forms but test accuracy stabilises below 99%",
                 fontsize=10.5, fontweight="bold", color=R_CORRUPT)
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.set_ylim(-0.05, 1.08)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out = "results/exp2_label_noise/noise_analysis_multiseed.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    results, by_noise = load_all()
    rows = summarise(by_noise)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#F9F9F9")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.33,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    ax_sum, ax_fa, ax_all, ax_cfp = axes

    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#CCCCCC")

    panel_test_acc(ax_sum, rows)
    panel_fourier_acc(ax_fa, rows)
    panel_learning_curves(ax_all, by_noise)
    panel_corrupted_fixed_points(ax_cfp, by_noise)

    # Shared noise colour legend
    patches = [mpatches.Patch(color=noise_color(n), label=f"{n:.0%}")
               for n in NOISE_LEVELS]
    fig.legend(handles=patches, title="Noise level",
               loc="upper center", ncol=len(patches), fontsize=8.5,
               title_fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.975))

    fig.suptitle(
        "Exp 2 — Grokking Under Label Noise  "
        "(mod_add p=113, n=3 seeds, 200K steps for 0–20% noise)",
        fontsize=13, fontweight="bold", y=0.998,
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {out}")

    # Print summary table
    print()
    print(f"{'Noise':>6}  {'n':>3}  {'Grokked':>8}  {'Test acc':>14}  {'Fourier':>10}  {'Regime'}")
    print("-" * 62)
    for r in rows:
        regime = ("Clean grokking" if r["grokked"] == r["n"]
                  else "Corrupted fixed pt" if r["noise"] <= 0.20
                  else "Circuit failure")
        print(f"{r['noise']:>6.2f}  {r['n']:>3}  {r['grokked']}/{r['n']:>6}  "
              f"{r['test_mean']:>6.1%} ± {r['test_std']:>4.1%}  "
              f"{r['fa_mean']:>6.3f} ± {r['fa_std']:.3f}  {regime}")


if __name__ == "__main__":
    main()
