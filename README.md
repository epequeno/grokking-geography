# Grokking Geography

Grokking is a delayed generalization phenomenon in neural network training. A model memorizes the training set (reaching ~100% train accuracy) long before it generalizes (test accuracy stays near random). Then, after thousands of additional gradient steps, test accuracy suddenly jumps to near-perfect — the model "groks" the task. First reported by Power et al. (2022) on modular arithmetic, grokking challenges the standard train/validation paradigm: the model appears to be overfitting, but is actually undergoing a slower phase of circuit formation that eventually generalizes.

Subsequent work has described *what* happens during grokking — Nanda et al. (2023) showed that the model forms a Fourier basis to represent modular addition; Zhong et al. (2023) characterized the "clock" circuit. But these are *post-hoc descriptions* of the grokked state. They tell us what the circuit looks like, not which components *cause* the transition.

This project asks three interventional questions:

1. **Which transformer component updates actually *cause* grokking?** We freeze individual components (MLP, attention, embeddings) at the memorization step and measure whether grokking still occurs. This is a causal intervention, not a description.
2. **Is grokking robust to label noise?** If grokking reflects genuine structure discovery, it should survive a small amount of noise. We test whether the sharp transition and the Fourier circuit persist under 5–40% corrupted labels.
3. **What task properties predict where grokking occurs?** We map grokking behavior across a taxonomy of tasks (abelian vs non-abelian groups, commutative vs non-commutative operations) and ask whether the component roles discovered in Exp 1 generalize or reverse.

All experiments use a tiny 1-layer transformer (full-width, no regularization) trained on algorithmic tasks with a 30% train split — the regime where grokking is most pronounced. While grokking was first observed in small models, recent work has identified analogous memorization-to-generalization transitions in LLM pretraining (arXiv:2506.21551), making the mechanism questions studied here relevant at scale.

**Research questions:**
1. Which transformer components are necessary/sufficient for the grokking transition? (surgical freezing)
2. Is grokking robust to label noise? Does it still discover the true circuit?
3. What task structural properties predict whether/when grokking occurs?

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
├── results/             # Saved runs (gitignored except metadata)
└── scripts/             # Sweep launchers
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Baseline grokking run (modular addition, no freezing)
python experiments/exp1_surgical_freeze/run_baseline.py

# Surgical freeze sweep
python experiments/exp1_surgical_freeze/run_freeze_sweep.py --task mod_add --prime 113

# Label noise sweep
python experiments/exp2_label_noise/run_noise_sweep.py --noise_levels 0.0 0.05 0.1 0.2 0.4

# Task taxonomy sweep
python experiments/exp3_task_taxonomy/run_taxonomy.py

# S5 circuit analysis (logit lens + activation patching)
python notebooks/s5_circuit_analysis.py
```

## Key Papers

**Foundational:**
- Power et al. 2022 (arXiv:2201.02177) — original grokking on modular arithmetic
- Nanda et al. 2023 (arXiv:2301.05217) — mechanistic interpretability: Fourier/Clock circuit for modular addition
- Zhong et al. 2023 (arXiv:2306.12644) — Clock vs Pizza: characterizing the grokked circuit
- Prieto et al. 2025 — Softmax Collapse / numerical stability in grokking

**Causal component roles (closest to Exp 1 & 4):**
- Where Grokking Happens: Distributed Utility and Fourier Recoding — arXiv:2609.17571 (Sep 2026). Component attribution across width/head counts; attribution-based complement to our causal freezing.
- Component Roles in Grokking Transfer and Stability — arXiv:2609.18078 (Sep 2026). Transfer acceleration and trajectory stability governed by distinct component roles; relates to our Exp 3 role reversal.
- Structure-Specific Representational Priors Causally Control the Grokking Delay — arXiv:2607.04333 (Jul 2026). Grokking delay is causally the time to form the right representational structure; backs our Exp 1 framing.

**Label noise (Exp 2):**
- Doshi et al. 2023 (arXiv:2310.13061) — grokking on corrupted modular arithmetic is robust *with regularization* (weight decay/dropout/BatchNorm force ignoring corruption). Direct counterpart to our no-regularization fragility result.
- Unveiling Memorization-Generalization Coexistence — arXiv:2605.18022 (May 2026). Two-layer nets under heavy noise: generalization structure forms latently but is masked by memorization; recoverable with frequency-based extraction even at 80% noise.

**Non-abelian groups / S5 (Exp 3 & 4):**
- Stander et al. 2023 (arXiv:2312.06581) — fully reverse-engineers FCNs that grokked S5/S6 arithmetic, discovering subgroup structure via cosets. Direct prior work for our S5 result.

**Related — adjacent but non-overlapping:**
- Geometry of Multi-Task Grokking — arXiv:2602.18523 (Xu, Feb 2026). Grokked solution is *holographic* (incompressible, sensitive to orthogonal perturbation). Tensions with our "single component is causally necessary" claim — worth reconciling.
- Geometric Inductive Bias of Grokking — arXiv:2603.05228 (Yildirim, Mar 2026). Uses S5 as a *negative control*: L2-normalized residual stream eliminates grokking on Z_p but does not accelerate S5, suggesting the architectural prior must align with the task's intrinsic symmetries. Direct support for our abelian/non-abelian split.
- Grokking From Abstraction to Intelligence — arXiv:2603.29262 (Mar 2026). Reframes grokking as parsimony-driven manifold collapse (Singular Learning Theory); contrasting mechanism story.
- Grokking in LLM Pretraining — arXiv:2506.21551 (ICLR 2026). Memorization→generalization transition observable from MoE routing pathways in real LLM pretraining.

## Novel Contributions


1. **Causal/interventional freezing study** — first to ask which component updates *cause* the grokking transition
2. **Label noise mechanistic analysis** — existing work is purely descriptive; we do circuit-level analysis
3. **Task taxonomy** — systematic map of task structure → grokking behavior
4. **S5 circuit analysis** — mechanistic investigation of how transformers implement non-abelian group composition (attention-dominant circuit, opposed to the MLP-dominant Fourier circuit for abelian groups)

---

## Findings

### Exp 1 — Surgical Freeze (mod_add, p=113)
**Q: Which component updates *cause* the grokking transition?**

- **MLP is causally necessary**: freezing MLP (all) at the memorisation step blocks grokking entirely (test acc → 4.5%). Freezing W_in or W_out individually delays by 1.7–1.6×.
- **Attention is causally dispensable**: freezing all attention heads at memorisation step has *no effect* on grok delay (slightly faster). The attention circuit is already committed at step ~700.
- **Embedding is a bottleneck, not a blocker**: freeze delays grokking ~10× but the model eventually finds an alternative — suggests the Fourier basis is preferred but not required.
- **Unembedding**: 3× delay — the readout layer adapts last.
- Logit lens confirms: the grokked model commits to the correct answer only at the Post-MLP stage; attention alone is insufficient.

**Output**: `results/exp1_freeze_sweep/`, `results/logit_lens_patching/`

---

### Exp 2 — Label Noise (mod_add, p=113)
**Q: Is grokking robust to label noise? Does the model still find the true circuit?**

- Grokking is **shockingly fragile**: even 5% label noise (191 corrupted labels out of 3830) prevents grokking across all 3 seeds. Test accuracy degrades gracefully (91.6% → 78.9% → 27.9% at 5/10/20%) but the sharp grokking transition disappears.
- **Fourier alignment score** drops from 0.62 (clean) to 0.44–0.46 at 5–15% noise, suggesting the model partially discovers the Fourier structure but cannot complete the circuit under noise pressure.
- At ≥25% noise, both Fourier score and test accuracy collapse to near-zero.

**Prior work**: Doshi et al. 2023 (arXiv:2310.13061) showed grokking is robust to label noise *with regularization* (weight decay, dropout, BatchNorm force the model to ignore corruption). Our no-regularization setup isolates the fragile baseline — without these pressures, 5% noise kills the transition. See also arXiv:2605.18022 (May 2026), which shows the generalization structure forms latently under heavy noise but is masked by memorization.

**Output**: `results/exp2_label_noise/`, `results/exp2_label_noise_ext/`

---

### Exp 3 — Task Taxonomy
**Q: What algebraic/structural properties of a task predict whether/when grokking occurs?**

**Taxonomy** (all tasks: 1-layer transformer, 30% train split, 2 seeds, ≤50K steps):

| Task | Structure | Groks? | Mean Delay |
|---|---|---|---|
| Parity 8-bit | Z₂ (abelian) | ✅ | 0 steps (immediate) |
| XOR 5/6-bit | Z₂ⁿ (abelian) | ✅ | ~1800–2400 steps |
| Mod Add | Z₁₁₃ (abelian) | ✅ | ~2700 steps |
| Mod Mul | Z₁₁₃ (abelian) | ✅ | ~2500 steps |
| Mod Div | Z₁₁₃ (abelian) | ✅ | ~9000 steps (slower) |
| Mod Exp | Z₉₇ (non-comm.) | ❌ | — (96.5% / 66.7%) |
| S₅ Compose | S₅ (non-abelian) | ❌ | — (~0.5% test acc) |
| Dihedral D₁₂ | D₁₂ (non-abelian) | ❌ | — (~5% test acc) |

**Key finding**: abelian groups grok reliably; non-abelian groups fail at 1 layer.

**Component freeze per task** (freeze at memorisation step, seed 42) revealed a **role reversal**:

| Component | mod_add | s5_compose |
|---|---|---|
| MLP (all) | ❌ BLOCKED | ❌ BLOCKED |
| Attention (all) | ✅ no effect | ❌ BLOCKED |
| Attn Q/K/V/O individually | ✅ no effect | ✅ grokked (~1.1× delay) |
| Embedding | ❌ BLOCKED | ❌ BLOCKED |
| Unembedding | 2.4× delay | ❌ BLOCKED |

For S₅ compose, *all* components are more tightly coupled — no single component is dispensable. Individual attention heads can compensate for each other, but removing any full subsystem blocks grokking.

**Note**: S₅ baseline grokked at 48,500 steps with seed 42, suggesting the taxonomy 50K-step limit may have been too short for seeds 42/43 in the run_taxonomy job — but the freeze-per-task baseline confirmed it does grok.

**Output**: `results/exp3_taxonomy/`, `results/exp3_freeze_per_task/`

**Prior work**: Stander et al. 2023 (arXiv:2312.06581) fully reverse-engineered one-hidden-layer FCNs that grokked S5 and S6 arithmetic, discovering that circuits decompose via the group's subgroup/coset structure. Consistent with our observation that S5 *does* grok (at ~48.5K steps with seed 42), just past the 50K taxonomy limit.

---

### Exp 4 — S5 Circuit Analysis *(in progress)*
**Q: What is the attention circuit doing for S₅ that MLP cannot? How does it differ mechanistically from the mod_add MLP/Fourier circuit?**

See: `notebooks/s5_circuit_analysis.py`, output: `results/s5_circuit_analysis/`
