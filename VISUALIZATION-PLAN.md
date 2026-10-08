# Visualization Plan — ggplot Redesign

## Narrative Arc

The paper tells a four-act story:

1. **What is grokking?** (fig1) — establish the phenomenon
2. **What causes it?** (fig2) — causal intervention via freezing
3. **How fragile is it?** (fig3) — label noise breaks the transition
4. **Where does it work/fail?** (fig4–6) — task structure predicts behavior; non-abelian is fundamentally different

Each figure should advance the narrative. Current figures are comprehensive but some panels are diagnostic clutter that dilutes the main point.

---

## Per-Figure Guidance

### Fig 1 — Baseline Grokking

**Express:** The three-phase phenomenon — memorization (train≈100%, test≈0%), circuit formation (sudden test jump), cleanup (loss decreases). The sharpness of the transition is the visual hook.

**Current:** 3 panels (dynamics, Fourier spectrum, weight norms).

**Recommended: 2 panels.**
- **(a) Train/test accuracy with phase shading** — this is the money shot. The gap between train and test, the sudden jump. Shade the three phases with `annotate("rect")`.
- **(b) Fourier alignment score over training steps** — shows *when* the circuit forms. Already in saved data (`fourier_alignment` array).

**Drop:** Weight norms (secondary diagnostic), retrain-for-spectrum panel (not reproducible from saved artifacts).

---

### Fig 2 — Surgical Freeze

**Express:** MLP is causally necessary (freezing blocks grokking). Attention is dispensable (freezing has no effect). This is the novel contribution — no prior work does causal interventions.

**Current:** 3 panels (necessity bars, sufficiency bars, scatter).

**Recommended: 2 panels.**
- **(a) Necessity** — grouped bar or dot plot showing grok delay ratio (frozen/baseline) per component. Color by whether it blocked grokking (red) or not (blue). The story is the contrast: MLP bar goes off the chart, attention bars are at ~1.0.
- **(b) Necessity × sufficiency scatter** — this is the figure people will remember. X = necessity (delay ratio when frozen), Y = sufficiency (delay ratio when all-but-this frozen). Components cluster into quadrants. Annotate quadrants: "dispensable" (low X, high Y), "necessary" (high X, low Y).

**Drop:** Sufficiency bar chart — the scatter encodes the same information more compactly.

---

### Fig 3 — Label Noise

**Express:** Grokking is shockingly fragile. 5% noise (191 corrupted labels) kills the sharp transition. The model still learns (accuracy degrades gracefully) but the phase transition disappears.

**Current:** 4 panels.

**Recommended: 2 panels.**
- **(a) Learning curves** — test accuracy over steps, one line per noise level, colored by noise gradient (blue→red). The strongest visual: sharp jump at 0% noise smoothing into gradual rise at 5-20%. Add horizontal dashed line at grokking threshold (99% test acc). Annotate "5% noise kills the transition."
- **(b) Summary scatter** — final test accuracy (Y) vs noise fraction (X), with point shape/size encoding whether it grokked. Overlay Fourier alignment score as a second series. Shows correlation: noise kills Fourier alignment, which kills grokking.

**Drop:** Fixed points panel (save for supplementary or notebook exploration).

---

### Fig 4 — Task Taxonomy

**Express:** (1) Abelian groups grok, non-abelian fail at 1 layer. (2) Component roles reverse — attention is dispensable for abelian but necessary for non-abelian.

**Current:** 3 panels (taxonomy bars, heatmap, contrast).

**Recommended: 2 panels.**
- **(a) Taxonomy summary** — bar chart or table-like visual showing grok delay per task, colored by abelian/non-abelian. Abelian tasks have bars, non-abelian have "no grok" markers. Establishes the geography.
- **(b) Component heatmap** — tasks on Y, components on X, color = grok delay ratio (green=no effect, red=blocked). The role-reversal figure. Annotate key cells: mod_add × attention = green, S5 × attention = red.

**Drop:** Contrast panel — the heatmap already shows the contrast.

---

### Fig 5 — Depth Ablation

**Express:** 2 layers helps non-abelian tasks grok. Bridge between "non-abelian fails at 1 layer" and "S5 does grok with more capacity."

**Current:** 2 panels. Already lean. Keep.

- **(a) Grok rate** — grouped bars (1-layer vs 2-layer) per task. S5/D12 go from 0% to >0%.
- **(b) Delay comparison** — only for tasks that grokked at both depths.

---

### Fig 6 — Circuit Instability

**Express:** Non-abelian circuits are metastable — they form, collapse, and re-form. Qualitatively different from the abelian case (form once, stable forever).

**Current:** 2 panels. Well-conceptualized.

- **(a) Test/train accuracy timeline** with collapse events marked (shaded vertical bands rather than arrows/labels to reduce clutter).
- **(b) Fourier alignment timeline** — shows the circuit forming and collapsing at the same points.

---

## Cross-Figure Considerations

**Color consistency:** Same color for "grokked" (green/blue) and "not grokked" (red/gray) across all figures. Okabe-Ito palette — keep consistent.

**Phase shading:** Three-phase shading (memorization/circuit formation/cleanup) appears in fig1, could appear in fig3 and fig6. Same colors across all figures.

**Figure count:** 6 figures for what could be a 4-figure paper. Consider:
- Merging fig5 (depth ablation) into fig4 (taxonomy) as an additional row — "1-layer vs 2-layer" is a natural extension of the taxonomy.
- Fig6 (circuit instability) could be supplementary if space is tight.

**ggplot specifics:**
- `patchwork` or `cowplot` for multi-panel composition
- `annotate("rect", xmin=, xmax=, ymin=, ymax=, alpha=0.1)` for phase shading
- `scale_color_gradientn()` for heatmaps
- `geom_hline()` / `geom_vline()` for thresholds
- `theme_classic()` as the base, then strip remaining elements

---

## Data Files (for CSV export)

| Dataset | JSON path | Key fields for plotting |
|---------|-----------|------------------------|
| Baseline time-series | `results/exp1_baseline/baseline_p113_s42.json` | steps, train_acc, test_acc, fourier_alignment, grok_step, mem_step |
| Freeze sweep | `results/exp1_freeze_sweep/freeze_sweep_p113_s42.json` | freeze_action, target_component, grok_delay, grokked, final_test_acc |
| Joint sufficiency | `results/exp1_joint_sufficiency/joint_sufficiency_p113_s42.json` | keep_unfrozen, grok_delay, grokked |
| Joint sufficiency curves | `results/exp1_joint_sufficiency/joint_sufficiency_curves_p113_s42.json` | name, steps, train_acc, test_acc |
| Noise sweep | `results/exp2_label_noise/noise_sweep_p113.json` | noise_frac, grokked, final_test_acc, final_fourier_score |
| Noise sweep curves | `results/exp2_label_noise/noise_sweep_curves_p113.json` | noise_frac, steps, test_acc, train_acc, fourier_alignment |
| Taxonomy | `results/exp3_taxonomy/taxonomy_results.json` | task_name, grokked, grok_delay, is_commutative, group_order |
| Freeze per task | `results/exp3_freeze_per_task/freeze_per_task_s42.json` | task, component, grokked, grok_delay, final_test |
| Depth ablation | `results/exp3_depth_ablation/depth_ablation_results.json` | task, n_layers, grokked, grok_delay, final_test_acc |
| Antigrok time-series | `results/exp5_antigrok_investigation/antigrok_s5_L2_s46.json` | steps, train_acc, test_acc, fourier_alignment, grok_step |
