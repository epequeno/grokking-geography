"""
Tiny transformer for grokking research.

Design goals:
  - Full access to individual component parameters (Q, K, V, O, W_in, W_out, E, U)
  - Named parameter groups for the freeze/unfreeze API
  - Standard 1-layer grokking setup matching Nanda et al. 2023
  - Optional: multi-layer for taxonomy experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    vocab_size: int          # |Z_p| + special tokens (e.g. "=")
    n_ctx: int               # sequence length (typically 3: a op b =)
    d_model: int = 128
    d_head: int = 32
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    act_fn: str = "relu"     # "relu" or "gelu"
    use_ln: bool = False     # layer norm (False matches Nanda 2023 baseline)


class Embed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) / cfg.d_model**0.5)

    def forward(self, tokens):
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) / cfg.d_model**0.5)

    def forward(self, tokens):
        return self.W_pos[:tokens.shape[1]]


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) / cfg.d_model**0.5)
        self.W_K = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) / cfg.d_model**0.5)
        self.W_V = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) / cfg.d_model**0.5)
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) / cfg.d_head**0.5)
        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x, return_attn=False):
        # x: [batch, seq, d_model]
        q = einops.einsum(x, self.W_Q, "b s d, h d dh -> b s h dh") + self.b_Q
        k = einops.einsum(x, self.W_K, "b s d, h d dh -> b s h dh") + self.b_K
        v = einops.einsum(x, self.W_V, "b s d, h d dh -> b s h dh") + self.b_V

        attn_scores = einops.einsum(q, k, "b sq h dh, b sk h dh -> b h sq sk")
        attn_scores = attn_scores / self.cfg.d_head**0.5
        attn_pattern = attn_scores.softmax(dim=-1)

        z = einops.einsum(attn_pattern, v, "b h sq sk, b sk h dh -> b sq h dh")
        out = einops.einsum(z, self.W_O, "b s h dh, h dh d -> b s d") + self.b_O

        if return_attn:
            return out, attn_pattern
        return out


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_in  = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) / cfg.d_model**0.5)
        self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) / cfg.d_mlp**0.5)
        self.b_in  = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))
        self.act = F.relu if cfg.act_fn == "relu" else F.gelu

    def forward(self, x):
        pre = x @ self.W_in + self.b_in
        post = self.act(pre)
        return post @ self.W_out + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp  = MLP(cfg)
        self.use_ln = cfg.use_ln
        if cfg.use_ln:
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x, return_attn=False):
        residual = x
        attn_in = self.ln1(x) if self.use_ln else x
        if return_attn:
            attn_out, attn_pattern = self.attn(attn_in, return_attn=True)
        else:
            attn_out = self.attn(attn_in)
        x = residual + attn_out

        residual = x
        mlp_in = self.ln2(x) if self.use_ln else x
        x = residual + self.mlp(mlp_in)

        if return_attn:
            return x, attn_pattern
        return x


class Unembed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(cfg.d_model, cfg.vocab_size) / cfg.d_model**0.5)
        self.b_U = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, x):
        return x @ self.W_U + self.b_U


class GrokTransformer(nn.Module):
    """
    Tiny transformer for grokking experiments.
    
    Sequence format: [a, op_token, b, eq_token] -> predicts result at last position
    (Following Nanda 2023: sequence is [a, b, =] and we predict at =)
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed   = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, tokens, return_cache=False):
        """
        tokens: [batch, seq_len] int tensor
        returns: logits at the last position [batch, vocab_size]
        """
        x = self.embed(tokens) + self.pos_embed(tokens)

        cache = {"embed": x.detach()} if return_cache else None

        attn_patterns = []
        for i, block in enumerate(self.blocks):
            if return_cache:
                x, attn_pattern = block(x, return_attn=True)
                cache[f"block_{i}_attn"] = attn_pattern.detach()
                cache[f"block_{i}_post"] = x.detach()
                attn_patterns.append(attn_pattern)
            else:
                x = block(x)

        logits = self.unembed(x[:, -1, :])  # predict at last token position

        if return_cache:
            cache["logits"] = logits.detach()
            return logits, cache
        return logits

    # ------------------------------------------------------------------ #
    #  Named parameter groups for the freeze API                          #
    # ------------------------------------------------------------------ #

    def component_groups(self) -> dict:
        """
        Returns a dict mapping component name -> list of parameters.
        Use this with the FreezeManager to freeze/unfreeze specific parts.
        
        Components:
          "embedding"        : token + positional embeddings
          "unembedding"      : W_U, b_U
          "attn_Q"           : all W_Q across layers
          "attn_K"           : all W_K across layers
          "attn_V"           : all W_V across layers
          "attn_O"           : all W_O across layers
          "attn_all"         : all attention params
          "mlp_in"           : all W_in across layers
          "mlp_out"          : all W_out across layers
          "mlp_all"          : all MLP params
          "layer_{i}_attn"   : attention params for layer i
          "layer_{i}_mlp"    : MLP params for layer i
        """
        groups = {
            "embedding":   [self.embed.W_E, self.pos_embed.W_pos],
            "unembedding": [self.unembed.W_U, self.unembed.b_U],
            "attn_Q": [], "attn_K": [], "attn_V": [], "attn_O": [],
            "mlp_in": [], "mlp_out": [],
            "attn_all": [], "mlp_all": [],
        }
        for i, block in enumerate(self.blocks):
            attn_params = [block.attn.W_Q, block.attn.b_Q,
                           block.attn.W_K, block.attn.b_K,
                           block.attn.W_V, block.attn.b_V,
                           block.attn.W_O, block.attn.b_O]
            mlp_params  = [block.mlp.W_in, block.mlp.b_in,
                           block.mlp.W_out, block.mlp.b_out]

            groups[f"layer_{i}_attn"] = attn_params
            groups[f"layer_{i}_mlp"]  = mlp_params
            groups["attn_all"].extend(attn_params)
            groups["mlp_all"].extend(mlp_params)
            groups["attn_Q"].extend([block.attn.W_Q, block.attn.b_Q])
            groups["attn_K"].extend([block.attn.W_K, block.attn.b_K])
            groups["attn_V"].extend([block.attn.W_V, block.attn.b_V])
            groups["attn_O"].extend([block.attn.W_O, block.attn.b_O])
            groups["mlp_in"].extend([block.mlp.W_in, block.mlp.b_in])
            groups["mlp_out"].extend([block.mlp.W_out, block.mlp.b_out])

        return groups

    def weight_norms(self) -> dict:
        """Returns L2 norm of each component group. Useful for tracking grokking progress."""
        groups = self.component_groups()
        return {
            name: sum(p.norm().item()**2 for p in params)**0.5
            for name, params in groups.items()
            if params
        }
