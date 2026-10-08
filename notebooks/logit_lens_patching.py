"""
Logit lens + activation patching analysis.

Compares the grokked model (full training, 100% test accuracy) against
the MLP-frozen model (MLP frozen at memorisation step, 4.5% test accuracy)
across three residual-stream checkpoints:

  Stage 0 — Embed:      embedding + positional encoding only, no computation
  Stage 1 — Post-attn:  after attention sublayer (linear routing done, MLP not yet applied)
  Stage 2 — Post-MLP:   after full transformer block (= final representation before unembedding)

Questions answered:
  1. At what stage does the grokked model commit to the correct answer?
  2. Does the MLP-frozen model commit at all on test inputs?
  3. If we surgically replace the MLP output in the frozen model with the
     grokked model's MLP output, does accuracy recover?

Run:
    uv run python notebooks/logit_lens_patching.py
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

from src.models.transformer import GrokTransformer, TransformerConfig
from src.tasks.modular import ModularAddition
from src.training.trainer import GrokTrainer, TrainConfig
from src.freezing.manager import FreezeManager

PRIME   = 113
SEED    = 42
N_PROBE = 256     # test examples to analyse

# ── colour palette ────────────────────────────────────────────────────────────
C_GROK   = "#06D6A0"
C_FROZEN = "#E63946"
C_EMBED  = "#AAAAAA"
C_ATTN   = "#8338EC"
C_MLP    = "#3A86FF"


# ── model training ────────────────────────────────────────────────────────────

def build_model_cfg(task):
    return TransformerConfig(
        vocab_size=task.transformer_vocab_size,
        n_ctx=3, d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=1,
    )


def train_grokked(task, seed=SEED):
    torch.manual_seed(seed)
    model = GrokTransformer(build_model_cfg(task))
    cfg   = TrainConfig(n_steps=5_000, lr=1e-3, weight_decay=1.0,
                        log_every=200, seed=seed)
    print("  Training grokked model (full, 5K steps)…", flush=True)
    GrokTrainer(model, task, cfg).train()
    model.eval()
    return model.cpu()   # move to CPU for consistent analysis


def train_mlp_frozen(task, seed=SEED):
    """Freeze MLP at memorisation step (~700), train for 5K steps total."""
    torch.manual_seed(seed)
    model = GrokTransformer(build_model_cfg(task))
    fm    = FreezeManager(model)
    cfg   = TrainConfig(
        n_steps=5_000, lr=1e-3, weight_decay=1.0,
        log_every=200, seed=seed,
        freeze_schedule=[(700, "freeze", "mlp_all")],
    )
    print("  Training MLP-frozen model (5K steps, MLP frozen at step 700)…", flush=True)
    GrokTrainer(model, task, cfg, freeze_manager=fm).train()
    model.eval()
    return model.cpu()   # move to CPU for consistent analysis


# ── residual stream decomposition ─────────────────────────────────────────────

@torch.no_grad()
def get_residual_stages(model: GrokTransformer, tokens: torch.Tensor) -> dict:
    """
    Run a 1-layer transformer and return the residual stream at three stages:
      'embed'     — after token + positional embedding (no computation)
      'post_attn' — after attention sublayer (pre-MLP)
      'post_mlp'  — after full block (= post-MLP)
    All shapes: [batch, seq, d_model].
    """
    model.eval()
    block = model.blocks[0]

    # Stage 0: embedding
    h = model.embed(tokens) + model.pos_embed(tokens)
    embed = h.clone()

    # Stage 1: attention sublayer only
    attn_in  = block.ln1(h) if block.use_ln else h
    attn_out = block.attn(attn_in)
    post_attn = h + attn_out

    # Stage 2: full block (attention + MLP)
    mlp_in   = block.ln2(post_attn) if block.use_ln else post_attn
    post_mlp = post_attn + block.mlp(mlp_in)

    return {"embed": embed, "post_attn": post_attn, "post_mlp": post_mlp}


@torch.no_grad()
def logit_lens_probs(model: GrokTransformer,
                     h: torch.Tensor,
                     position: int = -1) -> torch.Tensor:
    """Project residual stream h at `position` through unembedding. Returns probs [batch, vocab]."""
    model.eval()
    logits = h[:, position, :] @ model.unembed.W_U + model.unembed.b_U
    return logits.softmax(-1)


@torch.no_grad()
def correct_token_prob(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean probability assigned to the correct label."""
    return probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean().item()


@torch.no_grad()
def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return (probs.argmax(-1) == labels).float().mean().item()


# ── activation patching ───────────────────────────────────────────────────────

@torch.no_grad()
def patch_mlp_output(frozen_model: GrokTransformer,
                     grokked_model: GrokTransformer,
                     tokens: torch.Tensor,
                     labels: torch.Tensor) -> dict:
    """
    Patch grokked model's residual stream at various stages into frozen model.
    Returns a dict with accuracy after each patch type.
    """
    frozen_model.eval()
    grokked_model.eval()

    # Get grokked model stages
    grok_stages   = get_residual_stages(grokked_model, tokens)
    frozen_stages = get_residual_stages(frozen_model,  tokens)

    def unembed_logits(model, h):
        return h[:, -1, :] @ model.unembed.W_U + model.unembed.b_U

    results = {}

    # Baseline: frozen model unpatched
    results["frozen_baseline"] = accuracy(
        unembed_logits(frozen_model, frozen_stages["post_mlp"]).softmax(-1), labels)

    # Baseline: grokked model unpatched
    results["grokked_baseline"] = accuracy(
        unembed_logits(grokked_model, grok_stages["post_mlp"]).softmax(-1), labels)

    # Patch 1: replace embed in frozen model with grokked embed, recompute rest
    # (swap just the embedding representation, run frozen model's attn+mlp on top)
    block = frozen_model.blocks[0]
    h = grok_stages["embed"].clone()   # grokked embedding
    attn_in  = block.ln1(h) if block.use_ln else h
    attn_out = block.attn(attn_in)
    post_attn = h + attn_out
    mlp_in    = block.ln2(post_attn) if block.use_ln else post_attn
    post_mlp  = post_attn + block.mlp(mlp_in)
    results["patch_embed"] = accuracy(
        unembed_logits(frozen_model, post_mlp).softmax(-1), labels)

    # Patch 2: replace post_attn in frozen model with grokked post_attn,
    # recompute MLP from frozen model
    h = grok_stages["post_attn"].clone()
    mlp_in   = block.ln2(h) if block.use_ln else h
    post_mlp = h + block.mlp(mlp_in)
    results["patch_post_attn"] = accuracy(
        unembed_logits(frozen_model, post_mlp).softmax(-1), labels)

    # Patch 3: replace post_mlp in frozen model with grokked post_mlp
    # (this bypasses frozen MLP entirely — most direct test)
    h = grok_stages["post_mlp"].clone()
    results["patch_post_mlp"] = accuracy(
        unembed_logits(frozen_model, h).softmax(-1), labels)

    # Patch 4: use frozen model's unembedding to read grokked model's post_mlp
    # (tests whether frozen unembedding can read the grokked representation)
    results["frozen_unembed_grokked_repr"] = accuracy(
        unembed_logits(frozen_model, grok_stages["post_mlp"]).softmax(-1), labels)

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

STAGE_NAMES  = ["Embed\n(no computation)", "Post-attention\n(linear routing)", "Post-MLP\n(full circuit)"]
STAGE_KEYS   = ["embed", "post_attn", "post_mlp"]
STAGE_COLORS = [C_EMBED, C_ATTN, C_MLP]


def panel_commitment_curves(ax, grok_stages, frozen_stages, grok_model, frozen_model, labels):
    """
    Mean P(correct) at each stage for both models on the probe batch.
    The grokked model's curve should rise steeply at Post-MLP.
    The frozen model's curve should stay flat near chance.
    """
    grok_probs_per_stage   = [logit_lens_probs(grok_model,   grok_stages[k])   for k in STAGE_KEYS]
    frozen_probs_per_stage = [logit_lens_probs(frozen_model, frozen_stages[k]) for k in STAGE_KEYS]

    grok_correct   = [correct_token_prob(p, labels) for p in grok_probs_per_stage]
    frozen_correct = [correct_token_prob(p, labels) for p in frozen_probs_per_stage]

    xs = [0, 1, 2]
    ax.plot(xs, grok_correct,   color=C_GROK,   lw=2.5, marker="o", ms=9,
            label="Grokked model", zorder=4)
    ax.plot(xs, frozen_correct, color=C_FROZEN,  lw=2.5, marker="s", ms=9,
            label="MLP-frozen model", zorder=4)

    ax.axhline(1/PRIME, color="gray", lw=0.8, ls=":", alpha=0.6, label=f"Chance (1/{PRIME})")

    for i, (sc, col) in enumerate(zip(grok_correct, STAGE_COLORS)):
        ax.annotate(f"{sc:.3f}", (i, sc), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=C_GROK, fontweight="bold")
    for i, sc in enumerate(frozen_correct):
        ax.annotate(f"{sc:.3f}", (i, sc), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9, color=C_FROZEN)

    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_NAMES, fontsize=9)
    ax.set_ylabel("Mean P(correct answer)", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Logit lens: P(correct) at each residual stream stage\n"
                 "(test inputs only — neither model has seen these)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.3, 2.3)


def panel_token_distributions(ax, grok_stages, frozen_stages,
                               grok_model, frozen_model, tokens, labels):
    """
    For a single probe example: show the top-5 token probabilities at each stage
    for both models. Bar chart with correct answer highlighted.
    """
    idx = 0   # first probe example
    correct_label = labels[idx].item()

    n_stages  = len(STAGE_KEYS)
    bar_width = 0.35
    top_k     = 6

    # Get top-6 tokens by grokked post_mlp probability (to fix the x-axis)
    grok_probs_pm = logit_lens_probs(grok_model, grok_stages["post_mlp"])[idx]
    top_tokens    = grok_probs_pm.topk(top_k).indices.tolist()
    if correct_label not in top_tokens:
        top_tokens[-1] = correct_label   # ensure correct is shown

    xs = np.arange(top_k)
    offsets = np.linspace(-0.5, 0.5, n_stages * 2)

    for si, (key, col) in enumerate(zip(STAGE_KEYS, STAGE_COLORS)):
        grok_probs_s   = logit_lens_probs(grok_model,   grok_stages[key])[idx].numpy()
        frozen_probs_s = logit_lens_probs(frozen_model, frozen_stages[key])[idx].numpy()

        gv = [grok_probs_s[t]   for t in top_tokens]
        fv = [frozen_probs_s[t] for t in top_tokens]

        off_g = -0.30 + si * 0.20
        off_f =  0.00 + si * 0.20

        bars_g = ax.bar(xs + off_g, gv, width=0.18, color=col,
                        alpha=0.85, label=f"Grokked: {STAGE_NAMES[si].split(chr(10))[0]}",
                        edgecolor="white", linewidth=0.4)
        bars_f = ax.bar(xs + off_f, fv, width=0.18, color=col,
                        alpha=0.35, hatch="///", edgecolor=col, linewidth=0.4,
                        label=f"Frozen: {STAGE_NAMES[si].split(chr(10))[0]}")

    # Highlight the correct answer column
    ax.axvspan(top_tokens.index(correct_label) - 0.5,
               top_tokens.index(correct_label) + 0.5,
               color="gold", alpha=0.15, zorder=0, label=f"Correct = {correct_label}")

    a, b = tokens[idx, 0].item(), tokens[idx, 1].item()
    ax.set_title(f"Token distribution for a single test input: {a} + {b} = {correct_label} (mod {PRIME})\n"
                 f"Grokked = solid, Frozen = hatched  |  Stages: grey=embed, purple=post-attn, blue=post-mlp",
                 fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"tok {t}" + (" ← ✓" if t == correct_label else "")
                        for t in top_tokens], fontsize=8.5)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_xlabel("Token", fontsize=9)
    ax.legend(fontsize=7.5, ncol=3, framealpha=0.9, loc="upper right")


def panel_activation_patching(ax, patch_results):
    """
    Bar chart: test accuracy after patching each stage from grokked → frozen model.
    Shows the causal contribution of each residual stream stage.
    """
    items = [
        ("Frozen\n(baseline)",          patch_results["frozen_baseline"],      C_FROZEN),
        ("Patch:\ngrokked embed",        patch_results["patch_embed"],          C_EMBED),
        ("Patch:\ngrokked post-attn",   patch_results["patch_post_attn"],      C_ATTN),
        ("Patch:\ngrokked post-MLP",    patch_results["patch_post_mlp"],       C_MLP),
        ("Frozen W_U +\ngrokked repr",  patch_results["frozen_unembed_grokked_repr"], "#FF9F1C"),
        ("Grokked\n(ceiling)",           patch_results["grokked_baseline"],     C_GROK),
    ]
    labels_plot = [x[0] for x in items]
    values      = [x[1] for x in items]
    colors      = [x[2] for x in items]

    bars = ax.bar(range(len(items)), values, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.65)

    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333")

    ax.axhline(patch_results["frozen_baseline"],  color=C_FROZEN, lw=1.0, ls="--", alpha=0.5)
    ax.axhline(patch_results["grokked_baseline"], color=C_GROK,   lw=1.0, ls="--", alpha=0.5)
    ax.axhline(1/PRIME, color="gray", lw=0.8, ls=":", alpha=0.5, label="Chance")

    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels_plot, fontsize=9)
    ax.set_ylabel("Test accuracy", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("Activation patching: which stage carries the generalising information?\n"
                 "(replace frozen model's activations with grokked model's at each stage)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9)


def panel_interpretation(ax):
    """Text summary of what the three panels show."""
    ax.axis("off")
    ax.set_title("Mechanistic interpretation", fontsize=10.5, fontweight="bold")

    findings = [
        (C_EMBED,  "Embed stage",
         "Both models start from the same place: near-chance predictions from token\n"
         "embeddings alone. The Fourier basis (if it formed) is latent in the weights\n"
         "but doesn't directly predict the answer — it provides structure for computation."),
        (C_ATTN,   "Post-attention stage",
         "Attention linearly combines the two inputs' Fourier coordinates. The grokked\n"
         "model's attention has learned to place both tokens' representations at the same\n"
         "position, enabling the MLP to act on them jointly. Still not sufficient alone."),
        (C_MLP,    "Post-MLP stage",
         "The MLP computes the trig-identity products (cos·cos − sin·sin) from the\n"
         "Fourier coordinates supplied by attention. This is where commitment jumps.\n"
         "The frozen model can't do this — its MLP was locked before the Fourier\n"
         "circuit had time to form."),
        (C_GROK,   "Patching result",
         "Patching grokked post-MLP activations into the frozen model asks: can the\n"
         "frozen unembedding read off the grokked representation? High recovery = yes,\n"
         "the unembedding is compatible. The frozen-W_U + grokked-repr bar isolates\n"
         "whether the frozen readout can decode the grokked circuit's output."),
    ]

    y = 0.95
    for color, stage, text in findings:
        ax.text(0.01, y, f"● {stage}", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", color=color, va="top")
        ax.text(0.01, y - 0.06, text, transform=ax.transAxes,
                fontsize=8.5, color="#333333", va="top", linespacing=1.5)
        y -= 0.28


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out = "results/logit_lens_patching/analysis.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    task = ModularAddition(p=PRIME, train_frac=0.3, seed=SEED)

    print("Training models…")
    grok_model   = train_grokked(task)
    frozen_model = train_mlp_frozen(task)

    # Evaluate to confirm
    test_seqs, test_lbls = task.dataset.get_test()
    with torch.no_grad():
        grok_model.eval(); frozen_model.eval()
        grok_acc   = (grok_model(test_seqs).argmax(-1) == test_lbls).float().mean().item()
        frozen_acc = (frozen_model(test_seqs).argmax(-1) == test_lbls).float().mean().item()
    print(f"  Grokked model test acc:    {grok_acc:.1%}")
    print(f"  MLP-frozen model test acc: {frozen_acc:.1%}")

    # Sample probe batch from test set
    torch.manual_seed(1)
    idx          = torch.randperm(len(test_seqs))[:N_PROBE]
    probe_tokens = test_seqs[idx]
    probe_labels = test_lbls[idx]

    print(f"\nAnalysing {N_PROBE} test inputs…")
    grok_stages   = get_residual_stages(grok_model,   probe_tokens)
    frozen_stages = get_residual_stages(frozen_model, probe_tokens)

    print("Running activation patching…")
    patch_results = patch_mlp_output(frozen_model, grok_model, probe_tokens, probe_labels)
    print("  Patch results:")
    for k, v in patch_results.items():
        print(f"    {k:<35} {v:.1%}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor("#F9F9F9")

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_commit  = fig.add_subplot(gs[0, 0])
    ax_tokens  = fig.add_subplot(gs[0, 1])
    ax_patch   = fig.add_subplot(gs[1, 0])
    ax_interp  = fig.add_subplot(gs[1, 1])

    for ax in [ax_commit, ax_tokens, ax_patch]:
        ax.set_facecolor("#FFFFFF")
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#CCCCCC")

    print("\nBuilding figure…")
    panel_commitment_curves(ax_commit, grok_stages, frozen_stages,
                            grok_model, frozen_model, probe_labels)
    panel_token_distributions(ax_tokens, grok_stages, frozen_stages,
                               grok_model, frozen_model, probe_tokens, probe_labels)
    panel_activation_patching(ax_patch, patch_results)
    panel_interpretation(ax_interp)

    fig.suptitle(
        "Mechanistic Analysis: Logit Lens + Activation Patching  "
        f"(mod_add p={PRIME}, {N_PROBE} test inputs)",
        fontsize=13, fontweight="bold", y=0.975,
    )

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
