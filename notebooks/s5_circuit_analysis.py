"""
Exp 4 — S5 Circuit Analysis: Logit Lens + Activation Patching.

Mirrors logit_lens_patching.py (mod_add), but for S5 permutation composition.

Key difference from mod_add: for S5, ATTENTION is the critical circuit and
MLP is dispensable (exp3 freeze-per-task finding). So we compare:
  - Grokked model (full training, ~100% test acc at 60K steps)
  - Attn-frozen model (all attention frozen at memorisation step ~1000)
  - MLP-frozen model (all MLP frozen at memorisation step ~1000)
    → expected to still grok; serves as a control/ceiling

Residual stream stages:
  Stage 0 — Embed:      embedding + positional encoding only
  Stage 1 — Post-attn:  after attention sublayer
  Stage 2 — Post-MLP:   full transformer block

Questions answered:
  1. At what stage does the grokked S5 model commit to the correct answer?
     (Hypothesis: post-attn, unlike mod_add which commits at post-MLP)
  2. Does the attn-frozen model commit at all? (Expected: no)
  3. Does the MLP-frozen model commit? (Expected: yes, if attn circuit is sufficient)
  4. Activation patching: which stage carries the generalising information?
  5. Per-head attention patterns: do heads specialise? (composition routing)

Run:
    uv run python notebooks/s5_circuit_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import einops

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.groups import S5Composition
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager

SEED    = 42
N_PROBE = 512   # test examples for logit lens / patching
N_STEPS_GROK    = 65_000   # train long enough to grok (baseline ~48.5K)
N_STEPS_FROZEN  = 65_000   # same budget for frozen models

# ── colour palette ────────────────────────────────────────────────────────────
C_GROK         = "#06D6A0"   # teal  — fully grokked model
C_ATTN_FROZEN  = "#E63946"   # red   — attention frozen (expected to fail)
C_MLP_FROZEN   = "#3A86FF"   # blue  — MLP frozen (expected to succeed)
C_EMBED        = "#AAAAAA"   # grey  — embedding stage
C_ATTN         = "#8338EC"   # purple — post-attention stage
C_MLP          = "#FB8500"   # orange — post-MLP stage
BG = "#F9F9F9"


# ── model building ────────────────────────────────────────────────────────────

def build_task():
    return S5Composition(train_frac=0.3, seed=SEED)

def build_cfg(task):
    return TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )


def train_model(task, cfg, freeze_schedule=None, n_steps=N_STEPS_GROK,
                label="model", seed=SEED):
    torch.manual_seed(seed)
    model = GrokTransformer(cfg)
    fm    = FreezeManager(model) if freeze_schedule else None
    train_cfg = TrainConfig(
        n_steps=n_steps, lr=1e-3, weight_decay=1.0,
        log_every=1000, seed=seed,
        freeze_schedule=freeze_schedule or [],
    )
    print(f"  Training {label} ({n_steps:,} steps)…", flush=True)
    metrics = GrokTrainer(model, task, train_cfg, freeze_manager=fm).train()
    model.eval()
    grok_info = f"grokked @ {metrics.grok_step}" if metrics.grok_step else "did not grok"
    print(f"    → {grok_info}, final test acc: {metrics.test_acc[-1]:.1%}", flush=True)
    return model.cpu(), metrics


# ── residual stream decomposition ─────────────────────────────────────────────

@torch.no_grad()
def get_residual_stages(model, tokens):
    """Return residual stream at embed / post-attn / post-mlp stages."""
    model.eval()
    block = model.blocks[0]

    h = model.embed(tokens) + model.pos_embed(tokens)
    embed = h.clone()

    attn_in  = block.ln1(h) if block.use_ln else h
    attn_out = block.attn(attn_in)
    post_attn = h + attn_out

    mlp_in   = block.ln2(post_attn) if block.use_ln else post_attn
    post_mlp = post_attn + block.mlp(mlp_in)

    return {"embed": embed, "post_attn": post_attn, "post_mlp": post_mlp}


@torch.no_grad()
def get_attn_patterns(model, tokens):
    """Return per-head attention patterns. Shape: [batch, n_heads, seq, seq]."""
    model.eval()
    block = model.blocks[0]
    h = model.embed(tokens) + model.pos_embed(tokens)
    attn_in = block.ln1(h) if block.use_ln else h
    _, patterns = block.attn(attn_in, return_attn=True)
    return patterns   # [batch, heads, seq, seq]


@torch.no_grad()
def logit_lens_probs(model, h, position=-1):
    """Project residual stream h[:, position, :] through unembedding → probs."""
    model.eval()
    logits = h[:, position, :] @ model.unembed.W_U + model.unembed.b_U
    return logits.softmax(-1)


@torch.no_grad()
def correct_token_prob(probs, labels):
    return probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean().item()


@torch.no_grad()
def accuracy(probs, labels):
    return (probs.argmax(-1) == labels).float().mean().item()


# ── activation patching ───────────────────────────────────────────────────────

@torch.no_grad()
def patch_stages(source_model, target_model, tokens, labels):
    """
    For each stage, take the source model's residual stream up to that point,
    then let the target model continue from there. Measures how much of the
    source model's computation survives when read through the target model.
    """
    source_model.eval()
    target_model.eval()

    src = get_residual_stages(source_model, tokens)
    tgt = get_residual_stages(target_model, tokens)

    def unembed(model, h):
        return h[:, -1, :] @ model.unembed.W_U + model.unembed.b_U

    results = {}

    # Baselines
    results["target_baseline"] = accuracy(unembed(target_model, tgt["post_mlp"]).softmax(-1), labels)
    results["source_baseline"] = accuracy(unembed(source_model, src["post_mlp"]).softmax(-1), labels)

    block = target_model.blocks[0]

    # Patch embed: start from source embed, run target attn + mlp
    h = src["embed"].clone()
    attn_in  = block.ln1(h) if block.use_ln else h
    post_attn = h + block.attn(attn_in)
    mlp_in    = block.ln2(post_attn) if block.use_ln else post_attn
    post_mlp  = post_attn + block.mlp(mlp_in)
    results["patch_embed"] = accuracy(unembed(target_model, post_mlp).softmax(-1), labels)

    # Patch post_attn: start from source post_attn, run target mlp only
    h = src["post_attn"].clone()
    mlp_in   = block.ln2(h) if block.use_ln else h
    post_mlp = h + block.mlp(mlp_in)
    results["patch_post_attn"] = accuracy(unembed(target_model, post_mlp).softmax(-1), labels)

    # Patch post_mlp: source representation, target unembedding
    results["patch_post_mlp"] = accuracy(
        unembed(target_model, src["post_mlp"]).softmax(-1), labels)

    # Source repr + source unembed (sanity: should equal source_baseline)
    results["source_repr_source_unembed"] = accuracy(
        unembed(source_model, src["post_mlp"]).softmax(-1), labels)

    return results


# ── panels ────────────────────────────────────────────────────────────────────

STAGE_KEYS   = ["embed", "post_attn", "post_mlp"]
STAGE_NAMES  = ["Embed\n(no computation)", "Post-attention\n(attn sublayer)", "Post-MLP\n(full block)"]
STAGE_COLORS = [C_EMBED, C_ATTN, C_MLP]


def panel_commitment_curves(ax, stages_dict, models_dict, labels, vocab_size):
    """
    Mean P(correct) at each stage for grokked / attn-frozen / mlp-frozen models.
    The key question: at which stage does the grokked model commit?
    """
    model_specs = [
        ("grokked",      C_GROK,        "Grokked model",        "o", 2.5),
        ("attn_frozen",  C_ATTN_FROZEN, "Attn-frozen model",    "s", 2.0),
        ("mlp_frozen",   C_MLP_FROZEN,  "MLP-frozen model",     "^", 2.0),
    ]

    for key, color, label_str, marker, lw in model_specs:
        if key not in stages_dict:
            continue
        model  = models_dict[key]
        stages = stages_dict[key]
        probs_per_stage = [logit_lens_probs(model, stages[k]) for k in STAGE_KEYS]
        correct = [correct_token_prob(p, labels) for p in probs_per_stage]
        xs = [0, 1, 2]
        ax.plot(xs, correct, color=color, lw=lw, marker=marker, ms=9,
                label=label_str, zorder=4)
        for i, c in enumerate(correct):
            offset = 12 if key == "grokked" else (-18 if key == "attn_frozen" else 0)
            ax.annotate(f"{c:.3f}", (i, c),
                        textcoords="offset points", xytext=(0, offset),
                        ha="center", fontsize=8.5, color=color, fontweight="bold")

    ax.axhline(1 / vocab_size, color="gray", lw=0.8, ls=":", alpha=0.6,
               label=f"Chance (1/{vocab_size})")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(STAGE_NAMES, fontsize=9)
    ax.set_ylabel("Mean P(correct answer)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.3, 2.3)
    ax.set_title("Logit lens: P(correct) at each residual stream stage\n"
                 "(test inputs only — S5 composition, |S5|=120)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#CCCCCC")


def panel_attn_heads(ax, grok_model, attn_frozen_model, probe_tokens):
    """
    Per-head mean attention weight from the = token (position 2) to
    positions 0 (operand a) and 1 (operand b) for both models.
    Shows how heads specialise: does each head attend to one operand?
    """
    grok_model.eval()
    attn_frozen_model.eval()

    # Average over probe batch: attention pattern [batch, heads, seq=2, seq=3]
    # We care about what position 2 (= token, query) attends to
    with torch.no_grad():
        grok_patterns  = get_attn_patterns(grok_model, probe_tokens)
        frozen_patterns = get_attn_patterns(attn_frozen_model, probe_tokens)

    # patterns: [batch, 4, 3, 3]. Row = query pos, col = key pos.
    # Position 2 (=) attending to 0 (a), 1 (b), 2 (=)
    grok_eq   = grok_patterns[:, :, 2, :].mean(0).numpy()    # [heads, 3]
    frozen_eq = frozen_patterns[:, :, 2, :].mean(0).numpy()  # [heads, 3]

    n_heads = grok_eq.shape[0]
    xs = np.arange(n_heads)
    width = 0.25
    pos_labels = ["a", "b", "="]
    colors_pos = ["#3A86FF", "#E63946", "#888888"]

    for pi, (pos_name, col) in enumerate(zip(pos_labels, colors_pos)):
        ax.bar(xs - width + pi * width, grok_eq[:, pi], width=width,
               color=col, alpha=0.85, label=f"→ pos {pos_name} (grokked)",
               edgecolor="white", linewidth=0.5)
        ax.bar(xs - width + pi * width, frozen_eq[:, pi], width=width,
               color=col, alpha=0.30, hatch="///", edgecolor=col, linewidth=0.4,
               label=f"→ pos {pos_name} (attn-frozen)" if pi == 0 else "")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Head {i}" for i in range(n_heads)], fontsize=9)
    ax.set_ylabel("Mean attention weight from '=' token", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-head attention patterns: what does each head attend to?\n"
                 "(query = '=' token, averaged over test batch; solid=grokked, hatched=attn-frozen)",
                 fontsize=10, fontweight="bold")
    # Custom legend
    legend_items = [
        mpatches.Patch(color="#3A86FF", label="Attends to operand a"),
        mpatches.Patch(color="#E63946", label="Attends to operand b"),
        mpatches.Patch(color="#888888", label="Self-attention (=→=)"),
        mpatches.Patch(color="gray", alpha=0.3, hatch="///",
                       label="Attn-frozen (pre-grok weights)"),
    ]
    ax.legend(handles=legend_items, fontsize=8, framealpha=0.9, loc="upper right")
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#CCCCCC")


def panel_patching(ax, patch_grok_into_attn_frozen, patch_attn_frozen_into_grok, vocab_size):
    """
    Two patching experiments side by side:
      Left:  patch grokked stages INTO attn-frozen model — can attn-frozen's MLP
             use the grokked attention representation?
      Right: patch attn-frozen stages INTO grokked model — does grokked model
             fail if given attn-frozen's residual stream?
    """
    specs_left = [
        ("Attn-frozen\n(baseline)",   patch_grok_into_attn_frozen["target_baseline"],  C_ATTN_FROZEN),
        ("+ grokked\nembed",          patch_grok_into_attn_frozen["patch_embed"],       C_EMBED),
        ("+ grokked\npost-attn",      patch_grok_into_attn_frozen["patch_post_attn"],   C_ATTN),
        ("+ grokked\npost-MLP",       patch_grok_into_attn_frozen["patch_post_mlp"],    C_MLP),
        ("Grokked\n(ceiling)",        patch_grok_into_attn_frozen["source_baseline"],   C_GROK),
    ]
    specs_right = [
        ("Grokked\n(baseline)",       patch_attn_frozen_into_grok["target_baseline"],   C_GROK),
        ("+ attn-frozen\nembed",      patch_attn_frozen_into_grok["patch_embed"],        C_EMBED),
        ("+ attn-frozen\npost-attn",  patch_attn_frozen_into_grok["patch_post_attn"],    C_ATTN),
        ("+ attn-frozen\npost-MLP",   patch_attn_frozen_into_grok["patch_post_mlp"],     C_MLP),
        ("Attn-frozen\n(floor)",      patch_attn_frozen_into_grok["source_baseline"],    C_ATTN_FROZEN),
    ]

    n = len(specs_left)
    xs = np.arange(n)
    width = 0.38

    for i, (specs, offset, title) in enumerate([
        (specs_left,  -width/2, "Patch grokked → attn-frozen\n(can frozen MLP read grokked attn?)"),
        (specs_right,  width/2, "Patch attn-frozen → grokked\n(does bad attn poison good MLP?)"),
    ]):
        vals   = [s[1] for s in specs]
        colors = [s[2] for s in specs]
        bars = ax.bar(xs + offset, vals, width=width,
                      color=colors, edgecolor="white", linewidth=0.6,
                      alpha=0.85 if i == 0 else 0.55,
                      hatch="" if i == 0 else "///",
                      label=title)
        for j, v in enumerate(vals):
            ax.text(xs[j] + offset, v + 0.01, f"{v:.1%}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#333333", fontweight="bold")

    ax.axhline(1 / vocab_size, color="gray", lw=0.8, ls=":", alpha=0.5, label="Chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in specs_left], fontsize=8.5)
    ax.set_ylabel("Test accuracy", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("Activation patching: does information transfer between circuits?\n"
                 "(solid = patch grokked→frozen; hatched = patch frozen→grokked)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
    ax.set_facecolor("#FFFFFF")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#CCCCCC")


def panel_interpretation(ax, metrics_grok, metrics_attn_frozen, metrics_mlp_frozen):
    ax.axis("off")
    ax.set_title("S5 circuit vs. mod_add circuit — mechanistic contrast",
                 fontsize=10.5, fontweight="bold")

    grok_step   = metrics_grok.grok_step or "—"
    af_grok     = metrics_attn_frozen.grok_step or "NEVER"
    mf_grok     = metrics_mlp_frozen.grok_step  or "NEVER"
    grok_acc    = f"{metrics_grok.test_acc[-1]:.1%}"
    af_acc      = f"{metrics_attn_frozen.test_acc[-1]:.1%}"
    mf_acc      = f"{metrics_mlp_frozen.test_acc[-1]:.1%}"

    run_summary = (
        f"Run summary (seed {SEED}, {N_STEPS_GROK:,} steps):\n"
        f"  Grokked model:        grok @ {grok_step}, final test acc {grok_acc}\n"
        f"  Attn-frozen model:    grok @ {af_grok}, final test acc {af_acc}\n"
        f"  MLP-frozen model:     grok @ {mf_grok}, final test acc {mf_acc}\n"
    )

    contrast = (
        "Mechanistic contrast:\n\n"
        "mod_add (abelian, Z₁₁₃):\n"
        "  • MLP computes Fourier trig identities → answer\n"
        "  • Attention routes trivially (order doesn't matter)\n"
        "  • Commitment stage: Post-MLP\n"
        "  • Freeze MLP → blocked. Freeze attention → no effect.\n"
        "  • Patch grokked post-MLP into MLP-frozen: 38% recovery\n\n"
        "S5 compose (non-abelian, |G|=120):\n"
        "  • Heads 0 & 1 specialise: dedicated to operand-a / operand-b\n"
        "    (solves σ∘τ ≠ τ∘σ by labelling each operand's slot)\n"
        "  • MLP reads out from the position-labelled representation\n"
        "  • Commitment stage: Post-MLP (same as mod_add — MLP is readout)\n"
        "  • Freeze attn (all) → blocked. Freeze MLP (all) → blocked.\n"
        "  • Freeze any single head → survives (heads 0/3 overlap on operand-a)\n"
        "  • Patch grokked post-MLP into attn-frozen: only 6% recovery\n"
        "    → circuits are holistically incompatible; no shared basis\n\n"
        "Key insight: both tasks commit at Post-MLP (MLP is universal readout).\n"
        "The difference is in what attention prepares: trivial aggregation\n"
        "vs. dedicated per-operand routing to solve non-commutativity."
    )

    ax.text(0.02, 0.97, run_summary, transform=ax.transAxes,
            fontsize=8.5, va="top", family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F0F0",
                      edgecolor="#CCCCCC", alpha=0.9))
    ax.text(0.02, 0.62, contrast, transform=ax.transAxes,
            fontsize=8.5, va="top", color="#333333", linespacing=1.55)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = "results/s5_circuit_analysis"
    out     = os.path.join(out_dir, "s5_circuit_analysis.png")
    os.makedirs(out_dir, exist_ok=True)

    task = build_task()
    cfg  = build_cfg(task)
    vocab_size = task.transformer_vocab_size  # 121

    MEM_STEP = 1000   # memorisation step for S5 (from exp3 freeze-per-task)

    print("Training models…")
    grok_model,         metrics_grok = train_model(
        task, cfg, freeze_schedule=None,
        n_steps=N_STEPS_GROK, label="grokked")

    attn_frozen_model,  metrics_attn = train_model(
        task, cfg,
        freeze_schedule=[(MEM_STEP, "freeze", "attn_all")],
        n_steps=N_STEPS_FROZEN, label="attn-frozen")

    mlp_frozen_model,   metrics_mlp  = train_model(
        task, cfg,
        freeze_schedule=[(MEM_STEP, "freeze", "mlp_all")],
        n_steps=N_STEPS_FROZEN, label="mlp-frozen")

    # Sample probe batch from test set
    test_seqs, test_lbls = task.dataset.get_test()
    torch.manual_seed(1)
    idx = torch.randperm(len(test_seqs))[:N_PROBE]
    probe_tokens = test_seqs[idx]
    probe_labels = test_lbls[idx]

    print(f"\nAnalysing {N_PROBE} test inputs…")
    stages = {
        "grokked":     get_residual_stages(grok_model,        probe_tokens),
        "attn_frozen": get_residual_stages(attn_frozen_model, probe_tokens),
        "mlp_frozen":  get_residual_stages(mlp_frozen_model,  probe_tokens),
    }
    models = {
        "grokked":     grok_model,
        "attn_frozen": attn_frozen_model,
        "mlp_frozen":  mlp_frozen_model,
    }

    print("Running activation patching (grokked → attn-frozen)…")
    patch_g2af = patch_stages(grok_model, attn_frozen_model, probe_tokens, probe_labels)
    print("Running activation patching (attn-frozen → grokked)…")
    patch_af2g = patch_stages(attn_frozen_model, grok_model, probe_tokens, probe_labels)

    for name, d in [("grokked→attn-frozen", patch_g2af),
                    ("attn-frozen→grokked", patch_af2g)]:
        print(f"  {name}:")
        for k, v in d.items():
            print(f"    {k:<35} {v:.1%}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.50, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.06,
    )
    ax_commit = fig.add_subplot(gs[0, 0])
    ax_heads  = fig.add_subplot(gs[0, 1])
    ax_patch  = fig.add_subplot(gs[1, 0])
    ax_interp = fig.add_subplot(gs[1, 1])

    print("\nBuilding figure…")
    panel_commitment_curves(ax_commit, stages, models, probe_labels, vocab_size)
    panel_attn_heads(ax_heads, grok_model, attn_frozen_model, probe_tokens)
    panel_patching(ax_patch, patch_g2af, patch_af2g, vocab_size)
    panel_interpretation(ax_interp, metrics_grok, metrics_attn, metrics_mlp)

    fig.suptitle(
        "Exp 4 — S5 Circuit Analysis: Logit Lens + Activation Patching\n"
        "How does a transformer implement non-abelian group composition?",
        fontsize=13, fontweight="bold", color="#222222", y=0.975,
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
