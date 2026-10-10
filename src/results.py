"""
Immutable result artifact management.

Every experiment run produces a result file with:
  - A config hash derived from all relevant hyperparameters
  - A timestamp
  - The full config dict embedded in the result

Results are written to content-addressed paths so they are never overwritten:
    results/<experiment>/<config_hash>_<timestamp>.json

A latest symlink is maintained for convenience:
    results/<experiment>/latest_<run_name>.json -> <config_hash>_<timestamp>.json

Usage:
    from src.results import save_result, load_results, config_hash

    result = {"grokked": True, "grok_step": 3200, ...}
    config = {"prime": 113, "seed": 42, "mode": "freeze_one", ...}
    save_result("exp1_freeze_sweep", result, config, run_name="freeze_one_mlp_all")

    # Load all results for an experiment
    all_results = load_results("exp1_freeze_sweep")

    # Load results matching a config filter
    results = load_results("exp1_freeze_sweep", filter_fn=lambda r: r["config"]["seed"] == 42)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional


RESULTS_ROOT = Path(__file__).parent.parent / "results"


def config_hash(config: dict) -> str:
    """
    Deterministic hash of a config dict.
    Produces a short (12-char) hex string suitable for filenames.
    """
    # Sort keys recursively for determinism
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _timestamp() -> str:
    """UTC timestamp string for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_result(
    experiment: str,
    result: dict,
    config: dict,
    run_name: str = "",
    results_root: Optional[Path] = None,
) -> Path:
    """
    Save a result artifact with embedded config and content-addressed filename.

    Args:
        experiment: Experiment directory name (e.g. "exp1_freeze_sweep")
        result: The result data dict
        config: The full config dict (embedded in the saved file for provenance)
        run_name: Optional human-readable name for the latest symlink
        results_root: Override the default results root directory

    Returns:
        Path to the saved file
    """
    root = Path(results_root) if results_root else RESULTS_ROOT
    exp_dir = root / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    chash = config_hash(config)
    ts = _timestamp()

    # Build the artifact
    artifact = {
        "_meta": {
            "config_hash": chash,
            "timestamp": ts,
            "run_name": run_name,
            "experiment": experiment,
        },
        "config": config,
        "result": result,
    }

    # Write to content-addressed path (never overwrites)
    filename = f"{chash}_{ts}.json"
    filepath = exp_dir / filename
    with open(filepath, "w") as f:
        json.dump(artifact, f, indent=2)

    # Maintain a latest symlink for convenience
    if run_name:
        link_name = f"latest_{run_name}.json"
    else:
        link_name = f"latest_{chash}.json"
    link_path = exp_dir / link_name
    # Remove old symlink if it exists
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(filename)

    print(f"[results] Saved: {filepath}")
    return filepath


def save_result_batch(
    experiment: str,
    results: List[dict],
    config: dict,
    run_name: str = "",
    results_root: Optional[Path] = None,
) -> Path:
    """
    Save a batch of results (e.g. a sweep) as a single artifact.
    Same immutability guarantees as save_result.
    """
    root = Path(results_root) if results_root else RESULTS_ROOT
    exp_dir = root / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)

    chash = config_hash(config)
    ts = _timestamp()

    artifact = {
        "_meta": {
            "config_hash": chash,
            "timestamp": ts,
            "run_name": run_name,
            "experiment": experiment,
            "n_results": len(results),
        },
        "config": config,
        "results": results,
    }

    filename = f"{chash}_{ts}.json"
    filepath = exp_dir / filename
    with open(filepath, "w") as f:
        json.dump(artifact, f, indent=2)

    if run_name:
        link_name = f"latest_{run_name}.json"
    else:
        link_name = f"latest_{chash}.json"
    link_path = exp_dir / link_name
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(filename)

    print(f"[results] Saved batch ({len(results)} results): {filepath}")
    return filepath


def load_results(
    experiment: str,
    filter_fn: Optional[Callable[[dict], bool]] = None,
    results_root: Optional[Path] = None,
) -> List[dict]:
    """
    Load all result artifacts for an experiment.

    Args:
        experiment: Experiment directory name
        filter_fn: Optional filter applied to each artifact dict
        results_root: Override the default results root directory

    Returns:
        List of artifact dicts (each has _meta, config, result/results keys)
    """
    root = Path(results_root) if results_root else RESULTS_ROOT
    exp_dir = root / experiment

    if not exp_dir.exists():
        return []

    artifacts = []
    for f in sorted(exp_dir.glob("*.json")):
        # Skip symlinks (latest_*.json)
        if f.is_symlink():
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Only load files with our _meta marker
            if "_meta" not in data:
                continue
            if filter_fn is None or filter_fn(data):
                artifacts.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return artifacts


def load_latest(
    experiment: str,
    run_name: str,
    results_root: Optional[Path] = None,
) -> Optional[dict]:
    """Load the latest result for a given run name."""
    root = Path(results_root) if results_root else RESULTS_ROOT
    link = root / experiment / f"latest_{run_name}.json"
    if not link.exists():
        return None
    with open(link) as f:
        return json.load(f)
