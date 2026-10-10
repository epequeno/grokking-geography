"""Tests for the immutable result artifact system."""

import pytest

from src.results import save_result, save_result_batch, load_results, load_latest, config_hash


@pytest.fixture
def tmp_results(tmp_path):
    """Provide a temporary results root."""
    return tmp_path / "results"


class TestConfigHash:
    def test_deterministic(self):
        c = {"a": 1, "b": [1, 2, 3]}
        assert config_hash(c) == config_hash(c)

    def test_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_different_configs_different_hash(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})


class TestSaveLoad:
    def test_save_creates_file(self, tmp_results):
        path = save_result("test_exp", {"x": 1}, {"seed": 42}, results_root=tmp_results)
        assert path.exists()

    def test_save_creates_symlink(self, tmp_results):
        save_result("test_exp", {"x": 1}, {"seed": 42}, run_name="my_run", results_root=tmp_results)
        link = tmp_results / "test_exp" / "latest_my_run.json"
        assert link.exists()

    def test_load_returns_artifact(self, tmp_results):
        save_result("test_exp", {"x": 1}, {"seed": 42}, results_root=tmp_results)
        results = load_results("test_exp", results_root=tmp_results)
        assert len(results) == 1
        assert results[0]["result"]["x"] == 1
        assert results[0]["config"]["seed"] == 42

    def test_multiple_saves_dont_overwrite(self, tmp_results):
        """Two saves with same config at different times create separate files."""
        import time
        save_result("test_exp", {"x": 1}, {"seed": 42}, results_root=tmp_results)
        time.sleep(1.1)  # ensure different timestamp
        save_result("test_exp", {"x": 2}, {"seed": 42}, results_root=tmp_results)
        results = load_results("test_exp", results_root=tmp_results)
        assert len(results) == 2

    def test_filter_fn(self, tmp_results):
        save_result("test_exp", {"x": 1}, {"seed": 42}, results_root=tmp_results)
        save_result("test_exp", {"x": 2}, {"seed": 99}, results_root=tmp_results)
        results = load_results(
            "test_exp",
            filter_fn=lambda r: r["config"]["seed"] == 42,
            results_root=tmp_results,
        )
        assert len(results) == 1
        assert results[0]["config"]["seed"] == 42


class TestBatchSave:
    def test_batch_save(self, tmp_results):
        results = [{"comp": "mlp", "grokked": True}, {"comp": "attn", "grokked": False}]
        save_result_batch("test_exp", results, {"mode": "freeze_one"}, results_root=tmp_results)
        loaded = load_results("test_exp", results_root=tmp_results)
        assert len(loaded) == 1
        assert len(loaded[0]["results"]) == 2


class TestLoadLatest:
    def test_load_latest(self, tmp_results):
        save_result("test_exp", {"x": 1}, {"seed": 42}, run_name="my_run", results_root=tmp_results)
        latest = load_latest("test_exp", "my_run", results_root=tmp_results)
        assert latest is not None
        assert latest["result"]["x"] == 1
