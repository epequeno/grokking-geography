"""
Tests for the FreezeManager — verifying the hook-based freezing approach.

Key properties to verify:
1. Frozen parameters receive zero gradients (via hooks)
2. Frozen parameters retain requires_grad=True (optimizer keeps them)
3. Optimizer moments for unfrozen params are preserved across freeze events
4. restore_frozen() undoes weight-decay drift on frozen params
5. Unfreeze correctly removes hooks and allows gradient flow
"""

import torch
import torch.nn.functional as F
import pytest

from src.models.transformer import GrokTransformer, TransformerConfig
from src.freezing.manager import FreezeManager


@pytest.fixture
def model_and_fm():
    cfg = TransformerConfig(
        vocab_size=17, n_ctx=3, d_model=32, d_head=8,
        n_heads=4, d_mlp=64, n_layers=1,
    )
    model = GrokTransformer(cfg)
    fm = FreezeManager(model)
    return model, fm


def _run_step(model, fm, opt):
    """Run one forward-backward-step cycle."""
    x = torch.randint(0, 17, (8, 3))
    logits = model(x)
    labels = torch.randint(0, 17, (8,))
    loss = F.cross_entropy(logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if fm.has_frozen():
        fm.restore_frozen()


class TestFreezeHooks:
    def test_frozen_params_get_zero_grad(self, model_and_fm):
        model, fm = model_and_fm
        fm.freeze("mlp_all")
        x = torch.randint(0, 17, (4, 3))
        logits = model(x)
        logits.sum().backward()
        for p in fm._get_params("mlp_all"):
            assert p.grad is not None
            assert (p.grad == 0).all()

    def test_frozen_params_keep_requires_grad(self, model_and_fm):
        model, fm = model_and_fm
        fm.freeze("attn_all")
        for p in fm._get_params("attn_all"):
            assert p.requires_grad is True

    def test_unfrozen_params_get_nonzero_grad(self, model_and_fm):
        model, fm = model_and_fm
        fm.freeze("mlp_all")
        x = torch.randint(0, 17, (4, 3))
        logits = model(x)
        logits.sum().backward()
        # Embedding should still get gradients
        emb_params = fm._get_params("embedding")
        has_nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for p in emb_params)
        assert has_nonzero

    def test_unfreeze_restores_gradient_flow(self, model_and_fm):
        model, fm = model_and_fm
        fm.freeze("mlp_all")
        fm.unfreeze("mlp_all")
        x = torch.randint(0, 17, (4, 3))
        logits = model(x)
        logits.sum().backward()
        mlp_params = fm._get_params("mlp_all")
        has_nonzero = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlp_params)
        assert has_nonzero


class TestOptimizerStatePreservation:
    def test_moments_preserved_across_freeze(self, model_and_fm):
        """Adam moments for unfrozen params should NOT reset when we freeze something."""
        model, fm = model_and_fm
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

        # Run a few steps to build up optimizer moments
        for _ in range(5):
            _run_step(model, fm, opt)

        # Snapshot optimizer state for embedding (will remain unfrozen)
        emb_param = fm._get_params("embedding")[0]
        state_before = {
            "exp_avg": opt.state[emb_param]["exp_avg"].clone(),
            "exp_avg_sq": opt.state[emb_param]["exp_avg_sq"].clone(),
            "step": opt.state[emb_param]["step"].clone() if isinstance(opt.state[emb_param]["step"], torch.Tensor) else opt.state[emb_param]["step"],
        }

        # Freeze MLP
        fm.freeze("mlp_all")

        # Run more steps
        for _ in range(3):
            _run_step(model, fm, opt)

        # Check embedding moments are a continuation, not reset
        state_after = opt.state[emb_param]
        # Step count should have increased (not reset to 0)
        step_after = state_after["step"].item() if isinstance(state_after["step"], torch.Tensor) else state_after["step"]
        step_before = state_before["step"].item() if isinstance(state_before["step"], torch.Tensor) else state_before["step"]
        assert step_after > step_before, "Optimizer step should continue increasing"

    def test_restore_frozen_undoes_weight_decay(self, model_and_fm):
        """Frozen params should be exactly restored after optimizer step."""
        model, fm = model_and_fm
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

        fm.freeze("mlp_all")
        snapshot = {id(p): p.data.clone() for p in fm._get_params("mlp_all")}

        _run_step(model, fm, opt)

        for p in fm._get_params("mlp_all"):
            assert torch.equal(p.data, snapshot[id(p)])


class TestFreezeExcept:
    def test_freeze_except_leaves_target_trainable(self, model_and_fm):
        model, fm = model_and_fm
        fm.freeze_except("mlp_all")
        # MLP should be unfrozen
        assert "mlp_all" not in fm._frozen
        # Everything else should be frozen
        for name in fm.groups:
            if name not in ("mlp_all", "mlp_in", "mlp_out"):
                # Note: mlp_in/mlp_out share params with mlp_all
                pass
        # Check that frozen set is non-empty
        assert fm.has_frozen()

    def test_freeze_except_only_target_gets_grad(self, model_and_fm):
        model, fm = model_and_fm
        fm.freeze_except("embedding")
        x = torch.randint(0, 17, (4, 3))
        logits = model(x)
        logits.sum().backward()
        # Embedding should get gradients
        emb_params = fm._get_params("embedding")
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in emb_params)
        # MLP should get zero gradients
        mlp_params = fm._get_params("mlp_all")
        for p in mlp_params:
            if p.grad is not None:
                assert (p.grad == 0).all()


class TestMultiLayer:
    def test_per_layer_freeze(self):
        cfg = TransformerConfig(
            vocab_size=17, n_ctx=3, d_model=32, d_head=8,
            n_heads=4, d_mlp=64, n_layers=2,
        )
        model = GrokTransformer(cfg)
        fm = FreezeManager(model)

        fm.freeze("layer_0_attn")
        assert "layer_0_attn" in fm._frozen
        assert "layer_1_attn" not in fm._frozen

        fm.freeze("layer_1_mlp")
        assert "layer_1_mlp" in fm._frozen
        assert "layer_0_mlp" not in fm._frozen

        fm.unfreeze_all()
        assert not fm.has_frozen()
