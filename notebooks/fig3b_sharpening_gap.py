"""
Fig 3b — The "Sharpening Gap": Fourier alignment vs. test accuracy under noise.

Key insight: At 5–20% noise, the Fourier algorithm assembles (alignment 0.46–0.51)
but test accuracy stays low (31–92%). This gap quantifies how much algorithmic 
structure exists that hasn't been "sharpened" into generalisation — connecting our 
Exp2 results to the emerging "grokking as sharpening" narrative (Swaroop 2603.23784, 
He et al. 2601.09049).

Usage:
    cd grokking-geography  # from the parent directory of this repo
    uv run python notebooks/fig3b_sharpening_gap.py
"""

import sys, os


import json
import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, PAL, save_fig, label_panel

apply_style()


def load_noise_data():
    """Load noise sweep curves and compute per-noise-level summaries."""
    # Load both standard and extended runs
    with open("results/exp2_label_noise/noise_sweep_curves_p113.json") as f:
        standard = json.load(f)
    with open("results/exp2_label_noise_ext_mid/noise_sweep_curves_p113.json") as f:
        extended = json.load(f)
    
    # Use extended runs for 10/15/20% noise, standard for everything else
    ext_fracs = {item['noise_frac'] for item in extended}
    
    all_runs = []
    for item in standard:
        if item['noise_frac'] not in ext_fracs:
            all_runs.append(item)
    all_runs.extend(extended)
    
    # Group by noise fraction
    from collections import defaultdict
    by_noise = defaultdict(list)
    for item in all_runs:
        by_noise[item['noise_frac']].append(item)
    
    summaries = []
    for nf in sorted(by_noise.keys()):
        runs = by_noise[nf]
        test_accs = [r['final_test_acc'] for r in runs]
        fourier_scores = [r['final_fourier_score'] for r in runs]
        summaries.append({
            'noise_frac': nf,
            'test_acc_mean': np.mean(test_accs),
            'test_acc_std': np.std(test_accs),
            'fourier_mean': np.mean(fourier_scores),
            'fourier_std': np.std(fourier_scores),
            'n_seeds': len(runs),
            'runs': runs,
        })
    
    return summaries


def make_figure(summaries):  # noqa: C901
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))
    
    noise_fracs = [s['noise_frac'] for s in summaries]
    test_means = [s['test_acc_mean'] for s in summaries]
    test_stds = [s['test_acc_std'] for s in summaries]
    fourier_means = [s['fourier_mean'] for s in summaries]
    fourier_stds = [s['fourier_std'] for s in summaries]
    
    noise_pct = [nf * 100 for nf in noise_fracs]
    
    # Normalise Fourier to [0,1] scale relative to clean baseline for visual comparison
    clean_fourier = fourier_means[0]
    fourier_norm = [fm / clean_fourier for fm in fourier_means]
    fourier_norm_std = [fs / clean_fourier for fs in fourier_stds]
    
    # ── Panel (a): Normalised Fourier vs test accuracy ──
    # Plot on the SAME scale so the visual gap is meaningful
    ax1.errorbar(noise_pct, test_means, yerr=test_stds,
                 color=PAL.TEST, marker='o', ms=5, lw=1.5, capsize=3,
                 label="Test accuracy", zorder=3)
    ax1.errorbar(noise_pct, fourier_norm, yerr=fourier_norm_std,
                 color=PAL.FOURIER, marker='s', ms=5, lw=1.5, capsize=3,
                 label="Fourier alignment\n(normalised to clean)", zorder=3)
    
    # Shade the gap region at 15-20% noise where the story is clearest
    for i in range(len(noise_fracs)):
        nf = noise_fracs[i]
        if 0.12 < nf < 0.25 and fourier_norm[i] > test_means[i]:
            ax1.fill_between(
                [noise_pct[i] - 1.5, noise_pct[i] + 1.5],
                test_means[i], fourier_norm[i],
                color=PAL.FOURIER, alpha=0.15, zorder=1
            )
    
    ax1.set_xlabel("Label noise (%)")
    ax1.set_ylabel("Score")
    ax1.set_xlim(-2, 52)
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax1.set_title("Algorithmic structure vs. generalisation")
    label_panel(ax1, "a")
    
    # Add regime annotations
    ax1.axvspan(0, 3, color=PAL.GROK, alpha=0.07, zorder=0)
    ax1.axvspan(3, 25, color=PAL.PHASE_MEM, alpha=0.07, zorder=0)
    ax1.axvspan(25, 52, color=PAL.BLOCK, alpha=0.07, zorder=0)
    
    ax1.text(1.5, -0.02, "Grok", fontsize=6.5, ha='center', color=PAL.GROK, fontstyle='italic')
    ax1.text(14, -0.02, "Corrupted fixed point", fontsize=6.5, ha='center', color='#996633', fontstyle='italic')
    ax1.text(38, -0.02, "Circuit failure", fontsize=6.5, ha='center', color=PAL.BLOCK, fontstyle='italic')
    
    # Annotate the sharpening gap
    ax1.annotate(
        "Sharpening\ngap",
        xy=(17.5, 0.78), fontsize=7, ha='center', color=PAL.FOURIER,
        fontstyle='italic', fontweight='bold',
    )
    
    # ── Panel (b): Fourier trajectory divergence under noise ──
    # Show Fourier alignment TIME SERIES for 0%, 10%, 20%, 30% to see 
    # how structure builds then stalls
    target_noises = [0.0, 0.10, 0.20, 0.30]
    colors_ts = [PAL.GROK, PAL.CYAN, PAL.ORANGE, PAL.RED]
    
    # Get time series data — use seed 42 for each
    all_runs_flat = []
    with open("results/exp2_label_noise/noise_sweep_curves_p113.json") as f:
        all_runs_flat.extend(json.load(f))
    with open("results/exp2_label_noise_ext_mid/noise_sweep_curves_p113.json") as f:
        all_runs_flat.extend(json.load(f))
    
    for nf, col in zip(target_noises, colors_ts):
        # Find seed 42 run (prefer extended if available)
        candidates = [r for r in all_runs_flat if abs(r['noise_frac'] - nf) < 0.001 and r['seed'] == 42]
        if not candidates:
            continue
        # Use the one with longest steps
        run = max(candidates, key=lambda r: len(r.get('steps', [])))
        steps = run.get('steps', [])
        fa = run.get('fourier_alignment', [])
        if steps and fa:
            steps_k = [s/1000 for s in steps]
            ax2.plot(steps_k, fa, color=col, lw=1.3, label=f"{nf*100:.0f}% noise")
    
    ax2.set_xlabel("Training step (×1000)")
    ax2.set_ylabel("Fourier alignment")
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, 0.8)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax2.set_title("Circuit assembly under noise")
    label_panel(ax2, "b")
    
    # Annotate: structure forms but saturates
    ax2.annotate(
        "Structure forms\nbut stalls",
        xy=(50, 0.49), xytext=(60, 0.65),
        fontsize=7, ha='center',
        arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
    )
    
    plt.tight_layout()
    return fig


def main():
    summaries = load_noise_data()
    
    print("Noise level summaries:")
    print(f"{'Noise':>6} {'Test Acc':>10} {'Fourier':>10} {'Gap':>8}")
    print("-" * 40)
    
    clean_fourier = summaries[0]['fourier_mean']
    for s in summaries:
        nf = s['noise_frac']
        norm_fourier = s['fourier_mean'] / clean_fourier
        gap = norm_fourier - s['test_acc_mean']
        print(f"{nf*100:5.0f}% {s['test_acc_mean']:10.3f} {s['fourier_mean']:10.3f} {gap:8.3f}")
    
    fig = make_figure(summaries)
    save_fig(fig, "results/exp2_label_noise/fig3b_sharpening_gap")
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
