"""
Fig 6 — Recurrent Circuit Instability in Non-Abelian Grokking.

S5 2-layer seed 46 shows a pattern not seen in abelian tasks: the circuit
forms, collapses, and re-forms multiple times before stabilising. Six collapse
events in 27K steps, with recovery times of 100–800 steps.

This is NOT simple anti-grokking (permanent collapse). It's recurrent 
instability — the non-abelian circuit is assembled in a metastable
configuration that is vulnerable to perturbation by ongoing training dynamics.

The original Exp5 result (final_test=3.4%) was an artifact of the trainer's
early stopping window (grok_step + 2000) catching the first collapse event.

Usage:
    cd grokking-geography  # from the parent directory of this repo
    uv run python notebooks/fig6_circuit_instability.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pub_style import apply_style, PAL, save_fig, label_panel, thousands_formatter

apply_style()


def load_data():
    with open("results/exp5_antigrok_investigation/antigrok_s5_L2_s46.json") as f:
        return json.load(f)


def make_figure(d):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})

    steps = np.array(d['steps'])
    test_acc = np.array(d['test_acc'])
    train_acc = np.array(d['train_acc'])
    fourier = np.array(d['fourier_alignment'])

    steps_k = steps / 1000

    # ── Panel (a): Test & train accuracy ──
    ax1.plot(steps_k, train_acc, color=PAL.TRAIN, lw=0.8, alpha=0.5, label="Train acc")
    ax1.plot(steps_k, test_acc, color=PAL.TEST, lw=1.2, label="Test acc")
    
    # Shade collapse events
    collapse_steps = [94000, 96200, 104600, 110500, 114300, 117800]
    for cs in collapse_steps:
        ax1.axvline(cs/1000, color=PAL.RED, lw=0.5, alpha=0.4, ls=':')
    
    # Mark the grokking threshold
    ax1.axhline(0.99, color=PAL.GREY, lw=0.5, ls='--', alpha=0.5)
    ax1.text(120.5, 0.99, "99%", fontsize=6, color=PAL.GREY, va='center')
    
    # Mark the original early stopping window
    grok_step = d['grok_step']  # 93400
    early_stop = grok_step + 2000  # 95400
    ax1.axvspan(grok_step/1000, early_stop/1000, color=PAL.RED, alpha=0.08, zorder=0)
    ax1.annotate("Original early\nstop window",
                 xy=((grok_step + 1000)/1000, 0.15), fontsize=6.5,
                 ha='center', color=PAL.RED, fontstyle='italic')
    
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(-0.05, 1.08)
    ax1.legend(loc="center left", framealpha=0.9, fontsize=7)
    ax1.set_title("S₅ 2-layer seed 46: recurrent circuit instability", fontsize=10)
    label_panel(ax1, "a")
    
    # Count collapses annotation
    ax1.text(0.98, 0.55, f"6 collapse events\n6 recoveries\nFinal: 100% stable",
             transform=ax1.transAxes, fontsize=7, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=PAL.GREY, alpha=0.9))

    # ── Panel (b): Fourier alignment ──
    ax2.plot(steps_k, fourier, color=PAL.FOURIER, lw=1.0)
    ax2.fill_between(steps_k, 0, fourier, color=PAL.FOURIER, alpha=0.1)
    
    for cs in collapse_steps:
        ax2.axvline(cs/1000, color=PAL.RED, lw=0.5, alpha=0.4, ls=':')
    
    ax2.set_xlabel("Training step (×1000)")
    ax2.set_ylabel("Fourier alignment")
    ax2.set_ylim(0, 0.45)
    ax2.set_xlim(85, 121)
    label_panel(ax2, "b")
    
    # Annotate the sawtooth pattern
    ax2.text(0.98, 0.85, "Fourier alignment shows\nsawtooth recovery pattern",
             transform=ax2.transAxes, fontsize=7, ha='right', va='top',
             color=PAL.FOURIER, fontstyle='italic')

    plt.tight_layout()
    return fig


def main():
    d = load_data()
    
    # Print summary
    steps = d['steps']
    test_acc = d['test_acc']
    
    # Detect collapses
    collapses = []
    for i in range(1, len(test_acc)):
        if test_acc[i] < 0.98 and test_acc[i-1] >= 0.98:
            collapses.append({
                'step': steps[i],
                'test_drop': test_acc[i] - test_acc[i-1],
                'test_val': test_acc[i],
            })
    
    # Detect recoveries
    recoveries = []
    for i in range(1, len(test_acc)):
        if test_acc[i] >= 0.99 and test_acc[i-1] < 0.99 and i > 1:
            # Find the preceding collapse
            recoveries.append({
                'step': steps[i],
            })
    
    print("=== RECURRENT CIRCUIT INSTABILITY SUMMARY ===")
    print(f"Task: S5 composition, 2 layers, seed 46")
    print(f"Total steps: {steps[-1]:,}")
    print(f"First grok: step {d['grok_step']:,}")
    print(f"Total collapses: {len(collapses)}")
    print(f"Total recoveries: {len(recoveries)}")
    print(f"Final state: test={test_acc[-1]:.4f} (stable)")
    print()
    
    print("Collapse events:")
    for i, c in enumerate(collapses):
        # Find recovery step
        rec_step = None
        for r in recoveries:
            if r['step'] > c['step']:
                rec_step = r['step']
                break
        rec_time = f"{rec_step - c['step']} steps" if rec_step else "pending"
        print(f"  #{i+1}: step {c['step']:>7,} | test → {c['test_val']:.3f} | "
              f"recovery in {rec_time}")
    
    fig = make_figure(d)
    save_fig(fig, "results/exp5_antigrok_investigation/fig6_circuit_instability")
    plt.close(fig)


if __name__ == "__main__":
    main()
