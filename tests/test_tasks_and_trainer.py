"""Behavioural tests for dataset construction, seeding, and trainer freeze logic."""

import torch

from src.freezing.manager import FreezeManager
from src.models.transformer import TransformerConfig, build_model
from src.tasks import (
    ModularAddition, ModularDivision, ModularMultiplication, ParityTask, S5Composition,
)
from src.training.trainer import GrokTrainer, TrainConfig


def _tiny_cfg(task):
    return TransformerConfig(
        vocab_size=task.transformer_vocab_size, n_ctx=3,
        d_model=16, d_head=4, n_heads=4, d_mlp=32, n_layers=1,
    )


# ---------------------------------------------------------------- datasets

class TestDatasetSplitAndNoise:
    def test_split_is_disjoint_and_complete(self):
        ds = ModularAddition(p=11, train_frac=0.3, seed=1).dataset
        train, test = set(ds.train_pairs), set(ds.test_pairs)
        assert not train & test
        assert len(train) + len(test) == 11 * 11
        assert len(train) == int(121 * 0.3)

    def test_noise_corrupts_exact_fraction_and_leaves_test_clean(self):
        clean = ModularAddition(p=13, train_frac=0.5, seed=3).dataset
        noisy = ModularAddition(p=13, train_frac=0.5, noise_frac=0.2, seed=3).dataset
        assert noisy.train_pairs == clean.train_pairs  # same split
        assert noisy.test_labels == clean.test_labels  # test never corrupted
        wrong = sum(a != b for a, b in zip(noisy.train_labels, clean.train_labels))
        assert wrong == int(len(clean.train_labels) * 0.2)

    def test_noise_stays_inside_label_space(self):
        # parity labels are {0, 1}; corrupted labels must not leave that set
        ds = ParityTask(n_bits=4, train_frac=0.5, noise_frac=0.5, seed=0).dataset
        assert set(ds.train_labels) <= {0, 1}
        clean = ParityTask(n_bits=4, train_frac=0.5, seed=0).dataset
        flipped = [a != b for a, b in zip(ds.train_labels, clean.train_labels)]
        assert sum(flipped) == int(len(flipped) * 0.5)

    def test_mod_div_excludes_undefined_pairs(self):
        ds = ModularDivision(p=13, seed=0).dataset
        pairs = ds.train_pairs + ds.test_pairs
        assert all(b != 0 for _, b in pairs)
        assert len(pairs) == 13 * 12

    def test_mod_mul_is_the_group_without_zero(self):
        ds = ModularMultiplication(p=13, seed=0).dataset
        pairs = ds.train_pairs + ds.test_pairs
        assert all(a != 0 and b != 0 for a, b in pairs)
        labels = ds.train_labels + ds.test_labels
        assert 0 not in labels  # Z_p^* is closed

    def test_s5_composition_is_associative_and_noncommutative(self):
        t = S5Composition(seed=0)
        f = t.dataset.fn
        assert t.cfg.is_group and not t.cfg.is_commutative
        for a, b, c in [(3, 17, 90), (5, 61, 118), (44, 2, 99)]:
            assert f(f(a, b), c) == f(a, f(b, c))
        assert any(f(a, b) != f(b, a) for a in range(10) for b in range(10))

    def test_only_mod_add_enables_fourier_metric(self):
        assert ModularAddition(p=7).cfg.fourier_valid
        assert not ModularMultiplication(p=7).cfg.fourier_valid
        assert not S5Composition().cfg.fourier_valid


# ----------------------------------------------------------------- seeding

class TestSeedDeterminism:
    def test_build_model_independent_of_prior_rng_state(self):
        cfg = _tiny_cfg(ModularAddition(p=7))
        torch.manual_seed(123)
        torch.randn(50)  # disturb global RNG as an earlier run would
        a = build_model(cfg, seed=5)
        torch.manual_seed(999)
        b = build_model(cfg, seed=5)
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert torch.equal(pa, pb)

    def test_different_seeds_differ(self):
        cfg = _tiny_cfg(ModularAddition(p=7))
        a, b = build_model(cfg, 1), build_model(cfg, 2)
        assert not torch.equal(a.embed.W_E, b.embed.W_E)

    def test_training_run_is_reproducible(self):
        task = ModularAddition(p=7, seed=0)
        cfg = TrainConfig(n_steps=30, log_every=10, seed=7, device="cpu", batch_size=16)
        runs = []
        for _ in range(2):
            torch.manual_seed(1000 + len(runs))  # global RNG differs between runs
            m = GrokTrainer(build_model(_tiny_cfg(task), 7), task, cfg).train()
            runs.append(m.train_loss)
        assert runs[0] == runs[1]


# ---------------------------------------------------------- trainer freezing

def _train(task, **cfg_kw):
    model = build_model(_tiny_cfg(task), 0)
    fm = FreezeManager(model)
    base = dict(n_steps=200, log_every=10, seed=0, device="cpu", batch_size=32,
                grok_threshold=0.0, stop_after_grok=None)
    base.update(cfg_kw)
    metrics = GrokTrainer(model, task, TrainConfig(**base), freeze_manager=fm).train()
    return model, fm, metrics


class TestTrainerFreeze:
    def test_freeze_on_mem_fires_after_mem_step_and_holds_params(self):
        task = ModularAddition(p=7, seed=0)
        model = build_model(_tiny_cfg(task), 0)
        fm = FreezeManager(model)
        cfg = TrainConfig(n_steps=200, log_every=10, seed=0, device="cpu", batch_size=32,
                          grok_threshold=0.0, stop_after_grok=None, freeze_on_mem=["mlp_all"])
        m = GrokTrainer(model, task, cfg, freeze_manager=fm).train()
        # threshold 0.0 => memorisation detected at the first logged step (0)
        assert m.mem_step == 0
        ev = [e for e in m.freeze_events if e["trigger"] == "mem_step"]
        assert [e["step"] for e in ev] == [1]
        assert fm.frozen_components() == ["mlp_all"]

    def test_frozen_params_exactly_constant_without_decay(self):
        task = ModularAddition(p=7, seed=0)
        _, fm, _ = _train(task, freeze_on_mem=["mlp_all"], weight_decay=1.0, lr=1e-2)
        assert fm._snapshots  # something was actually frozen
        for pid, s in fm._snapshots.items():
            assert torch.equal(fm._param_by_id[pid].data, s)

    def test_decay_control_shrinks_frozen_params_without_gradients(self):
        task = ModularAddition(p=7, seed=0)
        lr, wd, steps = 1e-2, 1.0, 200
        model, fm, _ = _train(task, freeze_on_mem=["mlp_all"], weight_decay=wd, lr=lr,
                              frozen_params_decay=True, n_steps=steps)
        model_nd, fm_nd, _ = _train(task, freeze_on_mem=["mlp_all"], weight_decay=wd, lr=lr,
                                    frozen_params_decay=False, n_steps=steps)
        def norm(m):
            return m.component_groups()["mlp_in"][0].norm().item()

        # same init & batches => identical until freeze; afterwards only decay differs
        assert norm(model) < norm(model_nd)
        # freeze lands before step 1; decay is applied after each of steps 1..steps-1
        frozen_steps = steps - 1
        ratio = norm(model) / norm(model_nd)
        assert abs(ratio - (1 - lr * wd) ** frozen_steps) < 1e-4

    def test_early_stop_never_precedes_scheduled_freeze(self):
        task = ModularAddition(p=7, seed=0)
        # grok_threshold 0 => groks at step 0; stop_after_grok=5 would stop at 5,
        # but the freeze at step 100 must still happen first.
        _, fm, m = _train(task, freeze_schedule=[(100, "freeze", "attn_all")],
                          stop_after_grok=5, n_steps=300)
        assert any(e["step"] == 100 and e["action"] == "freeze" for e in m.freeze_events)
        assert fm.frozen_components() == ["attn_all"]

    def test_early_stop_without_schedule_stops_promptly(self):
        task = ModularAddition(p=7, seed=0)
        _, _, m = _train(task, stop_after_grok=5, n_steps=300)
        assert m.steps[-1] < 20

    def test_fourier_metric_only_logged_for_mod_add(self):
        _, _, m = _train(ModularAddition(p=7, seed=0), n_steps=20)
        assert len(m.fourier_alignment) == len(m.steps) > 0
        _, _, m2 = _train(ModularMultiplication(p=7, seed=0), n_steps=20)
        assert m2.fourier_alignment == [] and len(m2.steps) > 0

    def test_peak_and_grok_loss_tracking(self):
        task = ModularAddition(p=7, seed=0)
        _, _, m = _train(task, n_steps=100, grok_threshold=0.0)
        assert m.peak_test_acc == max(m.test_acc)
        assert m.test_acc[m.steps.index(m.peak_test_step)] == m.peak_test_acc
        assert m.grok_lost_step is None  # threshold 0.0 can never be lost
