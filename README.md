# Grokking Geography

Grokking is a delayed generalization phenomenon in neural network training. A model memorizes the training set (reaching ~100% train accuracy) long before it generalizes (test accuracy stays near random). Then, after thousands of additional gradient steps, test accuracy suddenly jumps to near-perfect — the model "groks" the task. First reported by Power et al. (2022) on modular arithmetic, grokking challenges the standard train/validation paradigm: the model appears to be overfitting, but is actually undergoing a slower phase of circuit formation that eventually generalizes.

Subsequent work has described *what* happens during grokking — Nanda et al. (2023) showed that the model forms a Fourier basis to represent modular addition; Zhong et al. (2023) characterized the "clock" circuit. But these are *post-hoc descriptions* of the grokked state. They tell us what the circuit looks like, not which components *cause* the transition.

This project asks three interventional questions:

1. **Which transformer component updates matter for grokking?** We freeze individual components (MLP, attention, embeddings) at each run's own memorization step and measure whether grokking still occurs, paired against a same-seed baseline. A control arm keeps weight decay on the frozen parameters, to separate "this component's gradient updates are needed" from "this component must be exempt from weight decay".
2. **How robust is grokking to label noise?** We train with 5–50% corrupted training labels and track test accuracy on a clean test set, plus a Fourier-structure score compared against its chance floor.
3. **What task properties predict where grokking occurs?** We map grokking behavior across a battery of tasks (group and non-group operations, commutative and non-commutative) and across depth (1 vs 2 layers).

All experiments use a small transformer (d_model=128, 4 heads, d_mlp=512, 1 layer unless stated) trained with AdamW (lr 1e-3, **weight decay 1.0**, as in Nanda et al. 2023) on algorithmic tasks with a 30% train split. Weight decay is part of the setup; nothing here is "unregularized". Related work observing memorization-to-generalization transitions in LLM pretraining exists (arXiv:2506.21551), but this repo does not test whether anything here transfers to that setting.


## Project Structure

```
grokking-geography/
├── src/
│   ├── models/          # Tiny transformer + component-level access
│   ├── tasks/           # Task registry: modular arithmetic, permutations, XOR, etc.
│   ├── training/        # Training loop with freeze hooks + dense metric logging
│   ├── freezing/        # Component freeze/unfreeze API
│   └── analysis/        # Circuit analysis: Fourier probes, logit lens, activation patching
├── experiments/
│   ├── exp1_surgical_freeze/   # Which components are necessary/sufficient?
│   ├── exp2_label_noise/       # Grokking under label noise
│   └── exp3_task_taxonomy/     # Grokking geography map
├── notebooks/           # Visualization + analysis
├── results/             # Saved runs (JSON + figures, tracked in git)
└── scripts/             # Report generator (make_report.py), anti-grokking investigation
```

## Setup

```bash
uv sync            # add `--extra wandb` for --use_wandb
uv run pytest
```

## Quick start

Scripts are run as files; `src` is importable through the editable install.

```bash
# Baseline grokking run (modular addition, no freezing)
uv run python experiments/exp1_surgical_freeze/run_baseline.py

# Freeze sweep: per-run mem_step freeze, paired baseline, weight-decay control arm
uv run python experiments/exp1_surgical_freeze/run_freeze_sweep.py --mode both --n_seeds 5

# Joint sufficiency (which trainable sets can grok on their own)
uv run python experiments/exp1_surgical_freeze/run_joint_sufficiency.py

# Label noise sweep
uv run python experiments/exp2_label_noise/run_noise_sweep.py --noise_levels 0.0 0.05 0.1 0.2 0.4

# Task taxonomy, depth ablation, per-task freeze sweep
uv run python experiments/exp3_task_taxonomy/run_taxonomy.py
uv run python experiments/exp3_task_taxonomy/run_depth_ablation.py
uv run python experiments/exp3_task_taxonomy/run_freeze_per_task.py --decay_control

# Anti-grokking investigation (S5, 2 layers, seed 46, no early stop)
uv run python scripts/investigate_antigrok.py

# Regenerate every table in "Findings" from results/*.json
uv run python scripts/make_report.py
```

## Related work

Citation IDs below were copied from earlier notes and have **not** been re-verified; check them before citing.

**Foundational:**
- Power et al. 2022 (arXiv:2201.02177) — original grokking on modular arithmetic
- Nanda et al. 2023 (arXiv:2301.05217) — Fourier/"clock" circuit for modular addition; the training setup used here
- Zhong et al. 2023 (arXiv:2306.12644) — Clock vs Pizza
- Prieto et al. 2025 — softmax collapse / numerical stability in grokking

**Component roles in grokking (adjacent to Exp 1):**
- arXiv:2609.17571 — component attribution across width/head counts (attribution-based, complementary to freezing)
- arXiv:2609.18078 — transfer acceleration and trajectory stability governed by distinct component roles
- arXiv:2607.04333 — representational priors and grokking delay
- arXiv:2602.18523 — multi-task grokking; reports a "holographic" (distributed) grokked solution, which should be reconciled with any claim that one component is necessary

**Label noise (Exp 2):**
- Doshi et al. 2023 (arXiv:2310.13061) — grokking on corrupted modular arithmetic with regularization. That paper's setup and success criterion differ from the 99%-test-accuracy threshold used here, so the two are not directly comparable; this repo does not reproduce their setting.
- arXiv:2605.18022 — generalization structure forms latently under heavy noise but is masked by memorization

**Non-abelian groups / S5 (Exp 3):**
- Stander et al. 2023 (arXiv:2312.06581) — reverse-engineers networks that grokked S5/S6 arithmetic
- arXiv:2603.05228 — uses S5 as a negative control for an architectural prior
- arXiv:2603.29262 — parsimony / Singular Learning Theory view of grokking
- arXiv:2506.21551 — memorization-to-generalization transition in LLM pretraining

---

## Findings

All tables below are generated by `uv run python scripts/make_report.py` from `results/`. Settings: AdamW, lr 1e-3, **weight decay 1.0**, batch 512, grok threshold 0.99 test accuracy, test/train splits seeded per run. "Grokked" means test accuracy reached 0.99 within the step budget; a run that does not reach it may still have high test accuracy (see the peak/final columns).

### Exp 1 — Freeze sweep (mod_add, p=113, 5 seeds, 40K-step budget)

Each run freezes one component at its own memorization step (`freeze_on_mem`) and is compared against a same-seed baseline with identical init and data split. The budget is ~10× the mean baseline delay; a run that "does not grok" here may still grok with a longer budget (the earlier 80K-step sweep is not directly comparable, see below).

Two arms:
- **no decay** — frozen parameters are held exactly constant (no gradient, no weight decay).
- **decay on** — frozen parameters get no gradient but still shrink by `lr·wd` per step. This is a *different* intervention, not a "fixed" version of the first: with wd=1.0 an un-trained parameter loses ~63% of its norm every 1000 steps, so it tests whether the circuit tolerates its frozen weights being shrunk, not whether it tolerates having them fixed.

Baseline: grokked 5/5; mean delay 3720 ± 1211 steps.

| Component | Arm | Grokked | Paired delay ratio (mean, n) | Peak test acc | Final test acc |
|---|---|---|---|---|---|
| attn_K | no decay | 5/5 | 1.05x (n=5) | 1.000 | 1.000 |
| attn_K | decay on | 5/5 | 0.93x (n=5) | 1.000 | 1.000 |
| attn_O | no decay | 5/5 | 1.11x (n=5) | 1.000 | 1.000 |
| attn_O | decay on | 5/5 | 0.94x (n=5) | 0.999 | 0.999 |
| attn_Q | no decay | 5/5 | 0.99x (n=5) | 1.000 | 1.000 |
| attn_Q | decay on | 5/5 | 0.91x (n=5) | 1.000 | 0.999 |
| attn_V | no decay | 5/5 | 1.09x (n=5) | 1.000 | 1.000 |
| attn_V | decay on | 5/5 | 0.92x (n=5) | 0.999 | 0.998 |
| attn_all | no decay | 5/5 | 1.36x (n=5) | 1.000 | 1.000 |
| attn_all | decay on | 0/5 | — | 0.576 | 0.008 |
| embedding | no decay | 3/5 | 11.00x (n=3) | 0.827 | 0.823 |
| embedding | decay on | 0/5 | — | 0.396 | 0.008 |
| mlp_all | no decay | 2/5 | 3.83x (n=2) | 0.417 | 0.416 |
| mlp_all | decay on | 2/5 | 5.40x (n=2) | 0.984 | 0.972 |
| mlp_in | no decay | 5/5 | 1.61x (n=5) | 1.000 | 1.000 |
| mlp_in | decay on | 2/5 | 1.17x (n=2) | 0.979 | 0.967 |
| mlp_out | no decay | 5/5 | 1.82x (n=5) | 1.000 | 1.000 |
| mlp_out | decay on | 2/5 | 1.29x (n=2) | 0.956 | 0.941 |
| unembedding | no decay | 4/5 | 3.25x (n=4) | 0.995 | 0.992 |
| unembedding | decay on | 0/5 | — | 0.856 | 0.009 |

What the data supports:
- **Individual attention matrices (Q, K, V, O) are dispensable after memorization:** 5/5 grok at ~1× delay in both arms. Freezing all of attention without decay costs ~1.4× delay but still groks 5/5.
- **The MLP and embedding are where freezing hurts.** Freezing the whole MLP blocked grokking in 3/5 seeds within the budget (and delayed it ~3.8× in the other 2); freezing the embedding blocked 2/5 and delayed the rest ~11×. This is a strong *slowdown/blocking* effect on these seeds, not "necessity": the earlier claim that MLP freezing blocks grokking "entirely" is not reproduced.
- **Results are strongly arm-dependent.** Shrinking the frozen weights of all of attention, of the embedding, or of the unembedding by weight decay (decay-on arm) prevented grokking in 0/5 seeds each, while holding them fixed allowed 5/5, 3/5 and 4/5 respectively. The "no decay" arm therefore cannot be read as "gradient updates to this component are required" without also stating that the frozen weights were exempt from decay.
- Not tested: freezing at other times, unfreezing later, or whether blocked runs would grok with more steps.

**Legacy sweep.** `results/exp1_freeze_sweep/` also holds the earlier 80K-step sweep. Its `freeze_one` runs froze at step 5000, long after the ~600–800 step memorization point, and in most runs grokking had already begun by then, so those results are not evidence about the memorization-point intervention.

**Joint sufficiency (single seed 42, from the earlier code; not re-run).** Trainable set → grokked: mlp+embed yes (step 7000); mlp+embed+unembed yes (step 1600); mlp+attn no; embed+attn no (train acc never reached 0.99); mlp+embed+attn no within its 40K budget (test 0.60). With one seed and unequal step budgets per row these are suggestive only. The one-component-trainable (`freeze_all_except`) mode is trivially uninformative for components that cannot memorize alone, and is no longer in the default sweep.

### Exp 2 — Label noise (mod_add, p=113, 3 seeds, 80K steps; results from the earlier code, which is unchanged for this experiment)

Training labels corrupted by a fixed fraction (always to a different valid label); test labels clean.

| Noise | Seeds | Grokked | Final test acc (mean ± sd) | Final train acc | Fourier score |
|---|---|---|---|---|---|
| 0% | 3 | 3/3 | 0.999 ± 0.001 | 1.000 | 0.646 |
| 5% | 3 | 0/3 | 0.916 ± 0.015 | 1.000 | 0.501 |
| 10% | 3 | 0/3 | 0.751 ± 0.094 | 0.987 | 0.475 |
| 15% | 3 | 0/3 | 0.574 ± 0.071 | 1.000 | 0.490 |
| 20% | 3 | 0/3 | 0.293 ± 0.130 | 1.000 | 0.446 |
| 30% | 3 | 0/3 | 0.014 ± 0.007 | 0.969 | 0.169 |
| 40% | 3 | 0/3 | 0.009 ± 0.001 | 0.993 | 0.117 |
| 50% | 3 | 0/3 | 0.007 ± 0.001 | 0.997 | 0.092 |

- No noisy run reaches the 99% threshold, but test accuracy at 5% noise is 92%: the model generalizes substantially without meeting the grokking criterion. Whether there is a sharp transition at low noise is not established by this table; use the learning curves (`fig3`, `fig3b`).
- The Fourier score is the fraction of embedding Fourier power in the top-6 modes. The chance floor for a random Gaussian embedding at this size is **0.067** (`chance_alignment`). Scores of 0.45–0.50 at 5–20% noise are about 7× chance, so partial Fourier structure is present; at ≥30% noise the score falls toward chance (0.09–0.17) as test accuracy goes to ~0.
- Weight decay is 1.0 throughout, so this is not an unregularized setting and is not a test of the "regularization helps" claim in the literature.
- 3 seeds; no confidence intervals.

### Exp 3 — Task taxonomy (1 layer, 3 seeds, 120K-step budget; re-run after task fixes)

Task fixes applied before this run: `mod_mul` is the multiplicative group (pairs with 0 excluded), `mod_div` excludes b=0 pairs instead of labeling them 0, the Fourier metric is only computed for `mod_add`, and the train/test split now varies with the seed.

| Task | Group? | Commutative | Grokked | Delay (mean ± sd) | Peak test | Final test |
|---|---|---|---|---|---|---|
| mod_add | True | True | 3/3 | 4200 ± 1558 | 0.999 | 0.999 |
| mod_mul | True | True | 3/3 | 3733 ± 1360 | 1.000 | 1.000 |
| mod_exp | False | False | 0/3 | — | 0.794 | 0.788 |
| mod_div | False | False | 3/3 | 8333 ± 3185 | 1.000 | 1.000 |
| s5_compose | True | False | 2/3 | 83100 ± 33500 | 0.671 | 0.671 |
| dihedral_12 | True | False | 0/3 | — | 0.092 | 0.080 |
| xor_6bit | True | True | 3/3 | 1600 ± 589 | 1.000 | 1.000 |
| xor_5bit | True | True | 3/3 | 4200 ± 2833 | 0.999 | 0.999 |
| parity_8bit | False | True | 3/3 | 0 ± 0 | 1.000 | 1.000 |

- **Commutativity and "group-ness" do not separate grokking from non-grokking.** `mod_div` is non-commutative and not a group yet groks 3/3; `s5_compose` and `dihedral_12` are groups yet grok slowly or not at all within budget. The pattern is better described as "some tasks need much longer than 120K steps (or more capacity)" than as an abelian/non-abelian split.
- **S5 groks at 1 layer** (2/3 here, 3/5 in the depth ablation), but only after about 44K–118K steps. "Non-abelian groups fail at 1 layer" was an artifact of a short step limit.
- `mod_exp` is not a group operation; 0/3 here (an earlier 5-seed run, pre-fix, had 1/5 grokking at ~92K steps).
- `parity_8bit` has delay 0: it generalizes when it memorizes, so there is no delayed transition to study.
- Different tasks have different train-set sizes and vocabularies, so delays are not strictly comparable across tasks.

### Exp 3 — Depth ablation (5 seeds, 120K steps; earlier code, not re-run)

| Task | Layers | Grokked | Delay (mean ± sd) | Final test (mean) | Collapsed after grok |
|---|---|---|---|---|---|
| dihedral_12 | 1 | 0/5 | — | 0.074 | 0 |
| dihedral_12 | 2 | 0/5 | — | 0.047 | 0 |
| mod_add | 1 | 5/5 | 3600 ± 970 | 1.000 | 0 |
| mod_add | 2 | 5/5 | 9600 ± 7262 | 0.986 | 0 |
| s5_compose | 1 | 3/5 | 60500 ± 17949 | 0.603 | 0 |
| s5_compose | 2 | 5/5 | 62900 ± 22103 | 0.807 | 1 |

Caveat: these runs used the same dataset split (seed 42) for every seed, so seed variance here reflects model init and batch order only. One S5 2-layer run (seed 46) grokked at step 71.5K and then lost generalization (final test 3.4%); `scripts/investigate_antigrok.py` re-runs that configuration with no early stop and records the collapse (`results/exp5_antigrok_investigation/`). Depth does not obviously help: 2 layers groks S5 5/5 but not faster, and does nothing for dihedral.

### Not re-run after the fixes

- **Per-task freeze sweep** (`run_freeze_per_task.py`): the script was rewritten (multiple seeds, per-run `mem_step` freeze, same-seed baseline, optional decay control), but the full run takes hours and has not been executed. The earlier single-seed (seed 42) table of component roles for mod_add vs S5, including the "role reversal" claim, is **withdrawn**: it came from one seed per task, froze at a fixed step 1000, and counted runs that reached 0.97–0.985 test accuracy as "blocked".
- **Depth ablation, joint sufficiency, noise sweep:** produced with earlier code; the changes since do not alter their logic, but they have not been regenerated.
- **Figures** in `results/*/fig*.{png,pdf}` and `VISUALIZATION-PLAN.md` were produced from the earlier results.
- **Logit-lens / activation-patching and S5 circuit analysis** (`notebooks/logit_lens_patching.py`, `notebooks/s5_circuit_analysis.py`): no output is checked in, so no claim is made here.

## Known limitations

- 3–5 seeds per condition, no confidence intervals or significance tests.
- One architecture size, one train fraction (30%), one weight-decay value.
- The 99% test-accuracy threshold turns a continuous curve into a binary "grokked" label; peak and final test accuracy are reported alongside for that reason.
- Freezing (even the no-decay arm) is not a clean causal "ablation of learning": it fixes the weights *and* removes them from weight decay, and the results differ between the two (see Exp 1).
