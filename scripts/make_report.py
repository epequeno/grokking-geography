"""
Generate the README "Findings" tables directly from results/*.json.

Every number in the README is produced here, so prose cannot drift from data.

Usage:
    uv run python scripts/make_report.py            # print markdown to stdout
    uv run python scripts/make_report.py > findings.md
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

R = "results"


def _load(path):
    with open(path) as f:
        return json.load(f)


def _fmt_delay(runs):
    d = [r["grok_delay"] for r in runs if r.get("grok_delay") is not None]
    return f"{np.mean(d):.0f} ± {np.std(d):.0f}" if d else "—"


def freeze_sweep(path_glob=f"{R}/exp1_freeze_sweep_v2/*.json"):
    rows = []
    for p in sorted(glob.glob(path_glob)):
        d = _load(p)
        if isinstance(d, list):
            rows.extend(d)
    if not rows:
        return "_No v2 freeze-sweep results found._\n"

    base = {r["seed"]: r for r in rows if r["target_component"] == "baseline"}
    seeds = sorted(base)
    out = [f"Seeds: {seeds}. Freeze lands the step after each run's own `mem_step`. "
           "`ratio` = frozen delay / same-seed baseline delay (paired). "
           "`decay` rows keep weight decay on frozen params (gradient-only freeze).\n"]
    n_b = sum(b["grokked"] for b in base.values())
    out.append(f"Baseline: grokked {n_b}/{len(base)}; mean delay {_fmt_delay(list(base.values()))}.\n")
    out.append("| Component | Arm | Grokked | Paired delay ratio (mean, n) | Peak test acc | Final test acc |")
    out.append("|---|---|---|---|---|---|")

    by = defaultdict(list)
    for r in rows:
        if r["target_component"] != "baseline":
            by[(r["target_component"], r["freeze_action"])].append(r)
    for (comp, mode), runs in sorted(by.items()):
        n = sum(r["grokked"] for r in runs)
        ratios = [r["grok_delay"] / base[r["seed"]]["grok_delay"] for r in runs
                  if r["grok_delay"] is not None and r["seed"] in base and base[r["seed"]]["grok_delay"]]
        rr = f"{np.mean(ratios):.2f}x (n={len(ratios)})" if ratios else "—"
        arm = {"freeze_one": "no decay", "freeze_one_decay": "decay on"}.get(mode, mode)
        out.append(f"| {comp} | {arm} | {n}/{len(runs)} | {rr} | "
                   f"{np.mean([r['peak_test_acc'] for r in runs]):.3f} | "
                   f"{np.mean([r['final_test_acc'] for r in runs]):.3f} |")
    return "\n".join(out) + "\n"


def noise_sweep():
    p = f"{R}/exp2_label_noise/noise_sweep_p113.json"
    if not os.path.exists(p):
        return "_No noise-sweep results found._\n"
    rows = _load(p)
    by = defaultdict(list)
    for r in rows:
        by[r["noise_frac"]].append(r)
    out = ["| Noise | Seeds | Grokked | Final test acc (mean ± sd) | Final train acc | Fourier score |",
           "|---|---|---|---|---|---|"]
    for nf, runs in sorted(by.items()):
        acc = [r["final_test_acc"] for r in runs]
        out.append(f"| {nf:.0%} | {len(runs)} | {sum(r['grokked'] for r in runs)}/{len(runs)} | "
                   f"{np.mean(acc):.3f} ± {np.std(acc):.3f} | "
                   f"{np.mean([r['final_train_acc'] for r in runs]):.3f} | "
                   f"{np.mean([r['final_fourier_score'] for r in runs]):.3f} |")
    out.append("\nFourier score of an unstructured random embedding (chance floor, p=113, d_model=128): "
               "see `chance_alignment` in `src/analysis/fourier.py`.")
    return "\n".join(out) + "\n"


def taxonomy():
    rows = []
    for p in sorted(glob.glob(f"{R}/exp3_taxonomy_v2/*/taxonomy_results.json")):
        rows.extend(_load(p))
    src = "v2 (fixed tasks)"
    if not rows:
        p = f"{R}/exp3_taxonomy/taxonomy_results.json"
        if not os.path.exists(p):
            return "_No taxonomy results found._\n"
        rows, src = _load(p), "v1 (pre-fix tasks; mod_mul/mod_div include degenerate pairs)"
    by = defaultdict(list)
    for r in rows:
        by[r["task_name"]].append(r)
    out = [f"Source: {src}.\n",
           "| Task | Group? | Commutative | Grokked | Delay (mean ± sd) | Peak test | Final test |",
           "|---|---|---|---|---|---|---|"]
    for t, runs in by.items():
        g = runs[0].get("is_group", "n/a")
        peak = np.mean([r.get("peak_test_acc", r["final_test_acc"]) for r in runs])
        out.append(f"| {t} | {g} | {runs[0]['is_commutative']} | "
                   f"{sum(r['grokked'] for r in runs)}/{len(runs)} | {_fmt_delay(runs)} | "
                   f"{peak:.3f} | {np.mean([r['final_test_acc'] for r in runs]):.3f} |")
    return "\n".join(out) + "\n"


def depth():
    p = f"{R}/exp3_depth_ablation/depth_ablation_results.json"
    if not os.path.exists(p):
        return "_No depth-ablation results found._\n"
    by = defaultdict(list)
    for r in _load(p):
        by[(r["task"], r["n_layers"])].append(r)
    out = ["| Task | Layers | Grokked | Delay (mean ± sd) | Final test (mean) | Collapsed after grok |",
           "|---|---|---|---|---|---|"]
    for (t, L), runs in sorted(by.items()):
        # a run that grokked but ended with test acc < 0.99 lost its solution
        collapsed = sum(1 for r in runs if r["grokked"] and r["final_test_acc"] < 0.9)
        out.append(f"| {t} | {L} | {sum(r['grokked'] for r in runs)}/{len(runs)} | {_fmt_delay(runs)} | "
                   f"{np.mean([r['final_test_acc'] for r in runs]):.3f} | {collapsed} |")
    return "\n".join(out) + "\n"


def joint():
    p = f"{R}/exp1_joint_sufficiency/joint_sufficiency_p113_s42.json"
    if not os.path.exists(p):
        return "_No joint-sufficiency results found._\n"
    out = ["| Trainable set | Grokked | Grok step | Final test acc |", "|---|---|---|---|"]
    for r in _load(p):
        out.append(f"| {r['name']} | {r['grokked']} | {r['grok_step']} | {r['final_test_acc']:.3f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print("### Exp 1 — freeze sweep (mod_add, p=113)\n" + freeze_sweep())
    print("### Exp 1 — joint sufficiency (single seed)\n" + joint())
    print("### Exp 2 — label noise\n" + noise_sweep())
    print("### Exp 3 — taxonomy\n" + taxonomy())
    print("### Exp 3 — depth ablation\n" + depth())
