"""
Figure 3 — Grokking Under Label Noise.

Four-panel figure:
  (a) Final test accuracy vs noise fraction (mean ± std across seeds).
  (b) Fourier alignment vs noise — does the circuit form at all?
  (c) Learning curves coloured by noise level.
  (d) Corrupted fixed points: mean ± std band for the 5–20% noise regime.

Data is loaded from all available result directories and deduplicated.

Run:
    uv run python notebooks/fig3_label_noise.py
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


# ── Data loading ─────────────────────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def load_all():
    """Load noise sweep results from all result directories, deduplicate to longest run."""
    dirs = [
        "results/exp2_label_noise",
        "results/exp2_label_noise_ext",
        "results/exp2_label_noise_ext_mid",
    ]
    raw = []
    for d in dirs:
        raw.extend(load_json(os.path.join(d, "noise_sweep_curves_p113.json")))

    # Deduplicate: keep longest run per (noise, seed)
    best = {}
    for r in raw:
        key = (round(r["noise_frac"], 3), r["seed"])
        if key not in best or len(r.get("steps", [])) > len(best[key].get("steps", [])):
            best[key] = r

    results = sorted(best.values(), key=lambda r: (r["noise_frac"], r["seed"]))
    by_noise = defaultdict(list)
    for r in results:
        by_noise[round(r["noise_frac"], 3)].append(r)
    return results, by_noise


def summarise(by_noise):
    rows = []
    for noise in sorted(by_noise.keys()):
        runs = by_noise[noise]
        tests = [r["test_acc"][-1] for r in runs if r.get("test_acc")]
        fouriers = [r["fourier_alignment"][-1] for r in runs if r.get("fourier_alignment")]
        rows.append({
            "noise": noise,
            "n": len(runs),
            "n_grokked": sum(1 for r in runs if r.get("grokked", False)),
            "test_mean": np.mean(tests) if tests else 0,
            "test_std": np.std(tests) if tests else 0,
            "fa_mean": np.mean(fouriers) if fouriers else 0,
            "fa_std": np.std(fouriers) if fouriers else 0,
        })
    return rows


# ── Panel (a): Test accuracy vs noise ────────────────────────────────────────

def panel_test_acc(ax, rows):
    noises = [r["noise"] for r in rows]
    means = [r["test_mean"] for r in rows]
    stds = [r["test_std"] for r in rows]

    ax.fill_between(noises, [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                    alpha=0.15, color=PAL.BLUE)
    ax.plot(noises, means, color=PAL.BLUE, lw=1.5, marker="o", ms=4, zorder=3)

    # Mark grok/no-grok boundary
    for r in rows:
        if r["n_grokked"] == r["n"]:
            marker = "*"
            ms = 8
        elif r["n_grokked"] > 0:
            marker = "D"
            ms = 5
        else:
            marker = "o"
            ms = 4
        ax.scatter(r["noise"], r["test_mean"], color=PAL.noise_color(r["noise"]),
                   marker=marker, s=ms**2, zorder=5, edgecolors="white", linewidths=0.4)

    ax.axhline(0.99, color=PAL.GROK, lw=0.7, ls="--", alpha=0.6, label="Grok threshold")
    ax.axhline(1/113, color="#999999", lw=0.5, ls=":", alpha=0.5, label="Chance (1/113)")

    ax.set_xlabel("Label noise fraction")
    ax.set_ylabel("Final test accuracy (clean test set)")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.02, max(noises) + 0.02)
    ax.set_title("Test accuracy vs noise")
    ax.legend(fontsize=7)


# ── Panel (b): Fourier alignment vs noise ────────────────────────────────────

def panel_fourier(ax, rows):
    noises = [r["noise"] for r in rows]
    fa_means = [r["fa_mean"] for r in rows]
    fa_stds = [r["fa_std"] for r in rows]

    ax.fill_between(noises, [m - s for m, s in zip(fa_means, fa_stds)],
                             [m + s for m, s in zip(fa_means, fa_stds)],
                    alpha=0.15, color=PAL.PURPLE)
    ax.plot(noises, fa_means, color=PAL.PURPLE, lw=1.5, marker="s", ms=4, zorder=3)

    for r in rows:
        ax.scatter(r["noise"], r["fa_mean"], color=PAL.noise_color(r["noise"]),
                   s=25, zorder=5, edgecolors="white", linewidths=0.4)

    ax.set_xlabel("Label noise fraction")
    ax.set_ylabel("Final Fourier alignment")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, max(noises) + 0.02)
    ax.set_title("Fourier circuit formation vs noise")


# ── Panel (c): Learning curves ───────────────────────────────────────────────

def panel_curves(ax, by_noise):
    noise_levels = sorted(by_noise.keys())
    max_noise = max(noise_levels) if noise_levels else 0.5

    for noise in noise_levels:
        runs = by_noise[noise]
        color = PAL.noise_color(noise, max_frac=max_noise)
        lw = 1.8 if noise <= 0.05 else 1.0
        alpha = 0.9 if noise == 0.0 else 0.65

        for i, r in enumerate(runs):
            if not r.get("steps"):
                continue
            label = f"{noise:.0%}" if i == 0 else None
            ax.plot(r["steps"], r["test_acc"],
                    color=color, lw=lw, alpha=alpha, label=label)

    ax.axhline(0.99, color="#CCCCCC", lw=0.6, ls=":")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(-0.05, 1.08)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.set_title("Learning curves by noise level")
    ax.legend(fontsize=6, ncol=2, title="Noise", title_fontsize=7)


# ── Panel (d): Corrupted fixed points ────────────────────────────────────────

def panel_fixed_points(ax, by_noise):
    target_noises = [n for n in sorted(by_noise.keys()) if 0.04 <= n <= 0.21]
    if not target_noises:
        ax.text(0.5, 0.5, "No data for 5–20% noise",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#999999")
        return

    max_noise = max(target_noises) if target_noises else 0.2

    for noise in target_noises:
        runs = by_noise[noise]
        if not runs or not runs[0].get("steps"):
            continue
        color = PAL.noise_color(noise, max_frac=max_noise)
        min_len = min(len(r["test_acc"]) for r in runs)
        steps = runs[0]["steps"][:min_len]
        mat = np.array([r["test_acc"][:min_len] for r in runs])
        mean = mat.mean(0)
        std = mat.std(0)

        ax.fill_between(steps, mean - std, mean + std, alpha=0.12, color=color)
        ax.plot(steps, mean, color=color, lw=1.5,
                label=f"{noise:.0%} → {mean[-1]:.0%}")

    ax.axhline(0.99, color=PAL.GROK, lw=0.7, ls="--", alpha=0.5)
    ax.axhline(1/113, color="#999999", lw=0.5, ls=":", alpha=0.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Test accuracy (mean ± 1 std)")
    ax.set_ylim(-0.05, 1.08)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.set_title("Corrupted fixed points (5–20% noise)")
    ax.legend(fontsize=6.5, title="Noise → final acc", title_fontsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/exp2_label_noise/fig3_label_noise")
    args = parser.parse_args()

    apply_style()

    print("Loading data…")
    results, by_noise = load_all()
    rows = summarise(by_noise)

    if not rows:
        print("ERROR: No label noise data found. Run the experiment first.")
        return

    n_seeds = max(r["n"] for r in rows)
    print(f"  {len(results)} runs across {len(by_noise)} noise levels, up to {n_seeds} seeds")

    # Layout: 2 × 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    panel_test_acc(axes[0, 0], rows)
    panel_fourier(axes[0, 1], rows)
    panel_curves(axes[1, 0], by_noise)
    panel_fixed_points(axes[1, 1], by_noise)

    label_panel(axes[0, 0], "a")
    label_panel(axes[0, 1], "b")
    label_panel(axes[1, 0], "c")
    label_panel(axes[1, 1], "d")

    fig.suptitle(f"Grokking under label noise (mod_add p = 113, n = {n_seeds} seeds)",
                 fontsize=10, fontweight="bold", y=1.01)

    save_fig(fig, args.out)

    # Print summary table
    print(f"\n{'Noise':>6}  {'n':>3}  {'Grok':>6}  {'Test acc':>14}  {'Fourier':>14}")
    print("-" * 55)
    for r in rows:
        print(f"{r['noise']:>6.2f}  {r['n']:>3}  "
              f"{r['n_grokked']}/{r['n']:>4}  "
              f"{r['test_mean']:>6.1%} ± {r['test_std']:>4.1%}  "
              f"{r['fa_mean']:>6.3f} ± {r['fa_std']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
