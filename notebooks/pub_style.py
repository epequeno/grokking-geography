"""
Shared publication style for all Grokking Geography figures.

Design principles:
  - Clean, minimal chrome (no chartjunk)
  - Consistent palette across all figures
  - Accessible colours (colourblind-safe where possible)
  - LaTeX-style serif fonts for axis labels/titles via mathtext
  - Figures sized for 2-column paper format (≤7" wide) or full-width (≤14" wide)
  - 300 DPI output, PDF preferred for vector quality

Usage:
    from pub_style import apply_style, PAL, save_fig
    apply_style()
    fig, ax = plt.subplots()
    ...
    save_fig(fig, "results/exp1_baseline/fig1_baseline")
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# Palette — inspired by Okabe-Ito (colourblind-safe) with some additions
# ═════════════════════════════════════════════════════════════════════════════

class PAL:
    """Centralised colour palette."""
    # Primary data colours
    BLUE    = "#0077BB"   # train accuracy, abelian tasks
    ORANGE  = "#EE7733"   # test accuracy (warm, high-contrast vs blue)
    PURPLE  = "#AA3377"   # Fourier alignment
    TEAL    = "#009988"   # grokking marker / success
    RED     = "#CC3311"   # blocked / failure
    GREY    = "#BBBBBB"   # baselines, references

    # Extended palette for multi-line plots
    YELLOW  = "#EE3377"
    CYAN    = "#33BBEE"
    BLACK   = "#000000"

    # Semantic aliases
    TRAIN   = BLUE
    TEST    = ORANGE
    FOURIER = PURPLE
    GROK    = TEAL
    BLOCK   = RED
    BASELINE = GREY

    # Phase shading
    PHASE_MEM     = "#EE7733"
    PHASE_CIRCUIT = "#AA3377"
    PHASE_CLEANUP = "#009988"

    # Delay severity scale (for freeze experiments)
    DELAY_NONE    = "#0077BB"   # ≤1× — no effect
    DELAY_MINOR   = "#33BBEE"   # 1–2×
    DELAY_MAJOR   = "#EE7733"   # 2–5×
    DELAY_SEVERE  = "#CC3311"   # >5×
    DELAY_BLOCKED = "#000000"   # never groks

    # Noise level colourmap
    NOISE_CMAP = "YlOrRd"

    # Task type colours
    ABELIAN    = "#0077BB"
    NONABELIAN = "#CC3311"

    @staticmethod
    def delay_color(ratio):
        """Map a delay ratio to its severity colour."""
        if ratio is None:     return PAL.DELAY_BLOCKED
        if ratio > 5.0:       return PAL.DELAY_SEVERE
        if ratio > 2.0:       return PAL.DELAY_MAJOR
        if ratio > 1.0:       return PAL.DELAY_MINOR
        return PAL.DELAY_NONE

    @staticmethod
    def noise_color(frac, max_frac=0.5):
        """Map a noise fraction to a colour from the noise colourmap."""
        cmap = plt.get_cmap(PAL.NOISE_CMAP)
        return cmap(frac / max_frac)


# ═════════════════════════════════════════════════════════════════════════════
# Style configuration
# ═════════════════════════════════════════════════════════════════════════════

def apply_style():
    """Apply publication-quality matplotlib RC settings."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,

        # Axes
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#666666",
        "axes.facecolor": "#FFFFFF",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,

        # Figure
        "figure.facecolor": "#FFFFFF",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Math text
        "mathtext.fontset": "dejavuserif",
    })


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_fig(fig, path_stem, formats=("pdf", "png")):
    """
    Save figure in multiple formats.
    path_stem: e.g. "results/exp1/fig1_baseline" (no extension)
    """
    p = Path(path_stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = p.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        print(f"  Saved: {out}")


def add_phase_bands(ax, mem_step, grok_step, max_step, alpha=0.06):
    """Shade the three grokking phases (memorisation / circuit formation / cleanup)."""
    if mem_step is not None:
        ax.axvspan(0, mem_step, color=PAL.PHASE_MEM, alpha=alpha, zorder=0)
    if mem_step is not None and grok_step is not None:
        ax.axvspan(mem_step, grok_step, color=PAL.PHASE_CIRCUIT, alpha=alpha, zorder=0)
    if grok_step is not None:
        ax.axvspan(grok_step, max_step, color=PAL.PHASE_CLEANUP, alpha=alpha, zorder=0)


def add_phase_vlines(ax, mem_step, grok_step):
    """Add dashed vertical lines at memorisation and grokking steps."""
    if mem_step is not None:
        ax.axvline(mem_step, color=PAL.PHASE_MEM, lw=0.8, ls="--", alpha=0.6, zorder=2)
    if grok_step is not None:
        ax.axvline(grok_step, color=PAL.GROK, lw=0.8, ls="--", alpha=0.6, zorder=2)


def thousands_formatter():
    """Axis formatter that shows e.g. '10K' instead of '10000'."""
    return mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")


def label_panel(ax, letter, x=-0.12, y=1.08):
    """Add a bold panel label (a), (b), etc."""
    ax.text(x, y, f"({letter})", transform=ax.transData if False else ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")
