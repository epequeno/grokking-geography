#!/usr/bin/env Rscript
# figures.r — Grokking Geography figure set, ggplot2 redesign
#
# Implements VISUALIZATION-PLAN.md. Run from the repo root:
#     Rscript notebooks/figures.r
#
# Figures land next to their experiment output, e.g.
#     results/exp1_baseline/fig1_baseline.{pdf,png}
#
# NOTE ON FIG 2 — the plan's spec for this figure does not match the saved
# artifacts. See the block comment above fig2() for what the data actually
# supports and why the panel was rebuilt.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(scales)
})

# resolve repo root whether invoked as `Rscript notebooks/figures.r` or from inside
if (!dir.exists("results") && dir.exists("../results")) setwd("..")
if (!dir.exists("results")) stop("run from the repo root (no results/ found)")

source("notebooks/gg_style.r")

W2 <- 7.0   # two panels side by side
H <- 2.9

# =============================================================================
# Fig 1 — baseline grokking
#   (a) train/test accuracy with three-phase shading — the money shot
#   (b) Fourier alignment — shows *when* the circuit forms
#   Dropped per plan: weight norms (secondary), retrain-for-spectrum (not
#   reproducible from saved artifacts).
# =============================================================================
fig1 <- function() {
  b <- baseline()
  steps <- unlist(b$steps)
  df <- data.frame(
    step  = steps,
    train = unlist(b$train_acc),
    test  = unlist(b$test_acc)
  )
  mem <- sc(b$mem_step); grok <- sc(b$grok_step); mx <- max(steps)

  long <- df |>
    pivot_longer(c(train, test), names_to = "metric", values_to = "acc") |>
    mutate(metric = factor(metric, c("train", "test"),
                           c("train accuracy", "test accuracy")))

  p_a <- ggplot(long, aes(step, acc, colour = metric)) +
    phase_rects(mem, grok, mx, 0, 1) +
    geom_line(linewidth = 0.9) +
    geom_vline(xintercept = mem, linetype = "22", linewidth = 0.4,
               colour = PAL$PHASE_MEM) +
    geom_vline(xintercept = grok, linetype = "22", linewidth = 0.4,
               colour = PAL$GROK) +
    annotate("text", x = grok, y = 0.42, hjust = -0.08, size = 2.5,
             family = "serif", colour = PAL$GROK,
             label = sprintf("grok step %s", comma(grok))) +
    annotate("text", x = mem, y = 0.90, hjust = -0.08, size = 2.5,
             family = "serif", colour = PAL$PHASE_MEM,
             label = sprintf("mem. %s", comma(mem))) +
    # phase key: the shading carries the paper's three-act structure, so it
    # needs a legend or the reader has to guess what the bands mean
    annotate("text", x = mem / 2, y = 1.10, size = 2.2, family = "serif",
             colour = "grey35", label = "memorisation") +
    annotate("text", x = (mem + grok) / 2, y = 1.10, size = 2.2, family = "serif",
             colour = "grey35", label = "circuit formation") +
    annotate("text", x = (grok + mx) / 2, y = 1.10, size = 2.2, family = "serif",
             colour = "grey35", label = "cleanup") +
    scale_colour_manual(values = c("train accuracy" = PAL$TRAIN,
                                   "test accuracy"  = PAL$TEST), name = NULL) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = "K")) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1.18), expand = c(0.01, 0)) +
    labs(x = "training step", y = "accuracy") +
    theme_pub() +
    theme(legend.position = c(0.5, 0.24))

  fourier <- data.frame(step = steps, fa = unlist(b$fourier_alignment))
  p_b <- ggplot(fourier, aes(step, fa)) +
    phase_rects(mem, grok, mx, 0, max(fourier$fa) * 1.12) +
    geom_line(linewidth = 0.9, colour = PAL$FOURIER) +
    geom_vline(xintercept = grok, linetype = "22", linewidth = 0.4,
               colour = PAL$GROK) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = "K")) +
    scale_y_continuous(limits = c(0, max(fourier$fa) * 1.12),
                       expand = c(0.01, 0)) +
    labs(x = "training step", y = "Fourier alignment") +
    theme_pub()

  save_fig((p_a | p_b) + tag_ab(),
           "results/exp1_baseline/fig1_baseline", W2, H)
}

# =============================================================================
# Fig 2 — surgical freeze:  WHAT THE DATA ACTUALLY SHOWS
#
# The plan asks for (a) a necessity bar chart "MLP bar goes off the chart,
# attention bars at ~1.0" with blocked components in red, and (b) a
# necessity x sufficiency scatter with quadrants.
#
# Neither is supported by results/exp1_freeze_sweep/:
#
#   freeze_one   (freeze ONE component)  -> 50/50 runs GROK, 5/5 for every one
#                                           of the 10 components. Delay ratios
#                                           0.59-2.37, i.e. no component is
#                                           individually necessary. There is no
#                                           red bar to draw.
#   freeze_all_except (freeze all but one) -> 0/50 runs grok, for EVERY
#                                           component. Leaving one component
#                                           trainable always fails, so this
#                                           contrast is uninformative.
#
# The informative result is the joint sufficiency sets, where {mlp,embed}
# groks at 1.93x and adding attention to it BLOCKS. So (a) shows the flat
# freeze_one distribution (that absence is the finding) and (b) shows the
# sufficiency sets. Panel (a) is deliberately a dot plot of all 5 seeds so the
# spread is visible rather than hidden behind a mean bar.
# =============================================================================
fig2 <- function() {
  base_delay <- sc(baseline()$grok_delay)

  fs <- read_flat("results/exp1_freeze_sweep/freeze_sweep_p113_s42.json") |>
    filter(freeze_action == "freeze_one") |>
    mutate(ratio = grok_delay / base_delay,
           component = factor(target_component, levels = c(
             "embedding", "unembedding", "mlp_in", "mlp_out", "mlp_all",
             "attn_Q", "attn_K", "attn_V", "attn_O", "attn_all")))

  # order components by mean ratio so the plot reads as a ranking
  ord <- fs |> group_by(component) |> summarise(m = mean(ratio), .groups = "drop") |>
    arrange(m) |> pull(component)
  fs <- fs |> mutate(component = factor(component, ord))

  p_a <- ggplot(fs, aes(ratio, component)) +
    geom_vline(xintercept = 1, linetype = "22", linewidth = 0.4,
               colour = "grey50") +
    geom_point(size = 1.5, colour = PAL$TRAIN, alpha = 0.75,
               position = position_jitter(height = 0.16, width = 0, seed = 1)) +
    stat_summary(fun = mean, geom = "point", shape = 124, size = 3.2,
                 colour = "grey15") +
    annotate("text", x = 1, y = 10.6, hjust = 1.05, vjust = 0.4, size = 2.3,
             family = "serif", colour = "grey30",
             label = "baseline delay = 1x") +
    scale_x_continuous(limits = c(0, 3)) +
    labs(x = "grok delay ratio (frozen / baseline)",
         y = NULL,
         title = "Freeze one component: every run still groks") +
    theme_pub() +
    theme(plot.title = element_text(size = 8)) +
    coord_cartesian(clip = "off")

  js <- read_flat("results/exp1_joint_sufficiency/joint_sufficiency_p113_s42.json") |>
    mutate(
      ratio = grok_delay / base_delay,
      status = ifelse(grokked, "grokked", "no grok"),
      label = ifelse(grokked, sprintf("%.2fx", ratio), "BLOCKED"),
      keep = factor(name, name[order(grokked, ratio, decreasing = TRUE)])
    )

  p_b <- ggplot(js, aes(ratio, keep, colour = status)) +
    geom_segment(aes(x = 0, xend = ratio, yend = keep), linewidth = 0.7,
                 na.rm = TRUE) +
    geom_point(size = 2.6, na.rm = TRUE) +
    geom_text(aes(label = label), hjust = -0.25, size = 2.6, family = "serif",
              colour = "grey20", na.rm = TRUE) +
    # A blocked set has NO ratio, so it cannot be drawn as a point on this axis.
    # Without an explicit marker the row renders empty and reads as missing data
    # rather than as the result. Anchor them at x = 0 with a label.
    geom_text(data = filter(js, !grokked), aes(x = 0, y = keep),
              label = "BLOCKED", hjust = -0.06, size = 2.6, family = "serif",
              colour = PAL$BLOCK, fontface = "bold") +
    scale_x_continuous(limits = c(0, 5.4)) +
    GROK_COLOUR +
    labs(x = "grok delay ratio (frozen / baseline)", y = NULL,
         title = "Freeze all but these: attention blocks the transition") +
    theme_pub() +
    theme(legend.position = "none",
          plot.title = element_text(size = 8)) +
    coord_cartesian(clip = "off")

  save_fig((p_a | p_b) + tag_ab(),
           "results/exp1_freeze_sweep/fig2_freeze_sweep", W2, 3.1)
}

# =============================================================================
# Fig 3 — label noise
#   (a) test-accuracy curves per noise level, blue->red gradient
#   (b) final test accuracy + final Fourier score vs noise fraction
#   Dropped per plan: fixed-points panel.
# =============================================================================
fig3 <- function() {
  raw <- read_raw("results/exp2_label_noise/noise_sweep_curves_p113.json")

  curves <- lapply(raw, function(r) {
    data.frame(
      step = unlist(r$steps),
      test = unlist(r$test_acc),
      train = unlist(r$train_acc),
      noise = as.numeric(sc(r$noise_frac)),
      grokked = isTRUE(sc(r$grokked, FALSE))
    )
  })
  curves <- do.call(rbind, curves)
  curves$noise_lab <- factor(sprintf("%d%%", round(curves$noise * 100)),
                             sprintf("%d%%", sort(unique(round(curves$noise * 100)))))

  noise_grad <- c("#0072B2", "#56B4E9", "#009E73", "#F0E442",
                  "#E69F00", "#D55E00", "#CC3311", "#7F0000")

  p_a <- ggplot(curves, aes(step, test, colour = noise, group = noise_lab)) +
    geom_hline(yintercept = 0.99, linetype = "22", linewidth = 0.4,
               colour = "grey45") +
    annotate("text", x = Inf, y = 0.99, hjust = 1.02, vjust = -0.5, size = 2.4,
             family = "serif", colour = "grey35", label = "grokking threshold (99%)") +
    geom_line(linewidth = 0.7, alpha = 0.9) +
    scale_colour_gradientn(colours = noise_grad, name = "label noise",
                           labels = percent_format(accuracy = 1)) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = "K")) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1.05), expand = c(0.01, 0)) +
    labs(x = "training step", y = "test accuracy") +
    theme_pub() +
    # outside the panel: an in-panel colourbar sat on top of the 5-20% traces
    theme(legend.position = "right",
          legend.key.height = unit(30, "pt"))

  sw <- read_flat("results/exp2_label_noise/noise_sweep_p113.json") |>
    mutate(noise_lab = factor(sprintf("%d%%", round(noise_frac * 100)),
                              sprintf("%d%%", sort(unique(round(noise_frac * 100)))))) |>
    group_by(noise_frac, noise_lab) |>
    summarise(final_test_acc = mean(final_test_acc),
              final_fourier = mean(final_fourier_score),
              frac_grokked = mean(grokked), .groups = "drop")

  # two series on one panel: accuracy (left axis) and Fourier score (right axis)
  k <- max(sw$final_test_acc) / max(sw$final_fourier)
  sw_long <- rbind(
    data.frame(noise_frac = sw$noise_frac, value = sw$final_test_acc,
               series = "final test accuracy"),
    data.frame(noise_frac = sw$noise_frac, value = sw$final_fourier * k,
               series = "final Fourier score")
  )

  p_b <- ggplot(sw_long, aes(noise_frac, value, colour = series)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.7) +
    scale_colour_manual(values = c("final test accuracy" = PAL$TEST,
                                   "final Fourier score" = PAL$FOURIER),
                        name = NULL) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1.05), expand = c(0.01, 0),
                       sec.axis = sec_axis(~ . / k, name = "final Fourier score")) +
    labs(x = "label noise fraction", y = "final test accuracy") +
    theme_pub() +
    theme(legend.position = c(0.62, 0.86))

  save_fig((p_a | p_b) + tag_ab(),
           "results/exp2_label_noise/fig3_label_noise", W2, H)
}

# =============================================================================
# Fig 4 — task taxonomy
#   (a) grok delay per task, coloured abelian / non-abelian, no-grok marked
#   (b) task x component heatmap of freeze effect — the role reversal
#   Dropped per plan: contrast panel (the heatmap carries it).
# =============================================================================
fig4 <- function() {
  tx <- read_flat("results/exp3_taxonomy/taxonomy_results.json") |>
    mutate(kind = ifelse(is_commutative, "abelian", "non-abelian"))

  summ <- tx |>
    group_by(task_name, kind) |>
    summarise(n = n(), n_grok = sum(grokked),
              mean_delay = mean(grok_delay[grokked], na.rm = TRUE),
              .groups = "drop") |>
    mutate(
      # Three distinct outcomes, and the log axis can only carry one of them:
      #   - mean() of an empty set is NaN        -> task never grokked
      #   - parity_8bit groks 5/5 at delay == 0  -> log10(0) = -Inf, dropped
      # Both would vanish silently, so both get an anchor row and a label.
      status = case_when(
        n_grok == 0     ~ "never groks",
        mean_delay == 0 ~ "0 (immediate)",
        TRUE            ~ "delayed"
      ),
      floor_y = min(mean_delay[n_grok > 0 & mean_delay > 0], na.rm = TRUE) * 0.45,
      plot_y  = ifelse(status == "delayed", mean_delay, floor_y),
      # keep these short: the floor rows are adjacent, and long strings
      # ("0 (immediate)", "never groks") overprint each other
      label   = ifelse(status == "delayed", sprintf("%d/%d", n_grok, n),
                       ifelse(status == "never groks", "", "0"))
    )

  kind_cols <- c("abelian" = PAL$ABELIAN, "non-abelian" = PAL$NONABEL)
  ord <- summ |> arrange(kind, status, mean_delay) |> pull(task_name)
  summ <- summ |> mutate(task_name = factor(task_name, unique(ord)))

  p_a <- ggplot(summ, aes(task_name, plot_y, fill = kind)) +
    geom_col(data = filter(summ, status == "delayed"), width = 0.68, colour = NA) +
    geom_col(data = filter(summ, status == "0 (immediate)"), width = 0.68,
             fill = "grey72", colour = NA) +
    geom_point(data = filter(summ, status == "never groks"), shape = 4,
               size = 2.6, colour = PAL$BLOCK, stroke = 1.2) +
    geom_text(data = filter(summ, status != "never groks"),
              aes(label = label), vjust = -0.6, size = 2.3, family = "serif",
              colour = "grey20") +
    scale_fill_manual(values = kind_cols, name = NULL) +
    scale_y_log10(labels = label_number(scale = 1e-3, suffix = "K"),
                  breaks = c(1e3, 1e4, 1e5),
                  # headroom so the count label above the tallest bar (mod_exp
                  # at ~89K) is not clipped by the panel edge
                  expand = expansion(mult = c(0.02, 0.16))) +
    labs(x = NULL, y = "mean grok delay (steps, log)",
         caption = "\u00d7 = never grokked;  0 = grokked at step 0") +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position = "top",
          legend.margin = margin(0, 0, 0, 0),
          plot.caption = element_text(size = 7, colour = "grey40", hjust = 0))

  # --- (b) heatmap from the per-task freeze runs ---
  fp <- read_flat("results/exp3_freeze_per_task/freeze_per_task_s42.json")
  base_delay <- fp |> filter(component == "baseline") |>
    select(task, base = grok_delay)

  hm <- fp |>
    filter(component != "baseline") |>
    left_join(base_delay, by = "task") |>
    mutate(
      blocked   = !grokked,
      # parity_8bit groks at step 0, so its baseline delay is 0 and every ratio
      # against it is undefined. That is not the same as "no effect": flag it
      # separately, otherwise the row renders as tile-less NaN text.
      undefined = !blocked & (is.na(base) | base <= 0 | is.na(grok_delay)),
      ratio     = ifelse(blocked | undefined, NA_real_, grok_delay / base),
      ratio_c   = pmin(ratio, 4),
      cell      = ifelse(blocked, "blocked",
                         ifelse(undefined, "undefined", "delay"))
    )

  fn_ord <- read_flat("results/exp3_taxonomy/taxonomy_results.json") |>
    group_by(task_name) |> summarise(c = first(is_commutative), .groups = "drop")
  task_ord <- hm |> distinct(task) |> pull(task)
  task_ord <- c(task_ord[task_ord %in% fn_ord$task_name[!fn_ord$c]],
                task_ord[task_ord %in% fn_ord$task_name[fn_ord$c]])
  comp_ord <- c("embedding", "unembedding", "mlp_in", "mlp_out", "mlp_all",
                "attn_Q", "attn_K", "attn_V", "attn_O", "attn_all")
  hm <- hm |> mutate(task = factor(task, task_ord),
                     component = factor(component, comp_ord))

  grad <- c("#009E73", "#F0E442", "#E69F00", "#D55E00", "#CC3311")

  p_b <- ggplot(hm, aes(component, task)) +
    geom_tile(aes(fill = ratio_c), colour = "white", linewidth = 0.5) +
    # blocked cells have no ratio at all, so folding them into the gradient
    # would invent a value. Draw them as their own dark layer with a label.
    geom_tile(data = filter(hm, cell == "undefined"), fill = "grey86",
              colour = "white", linewidth = 0.5) +
    geom_tile(data = filter(hm, blocked), fill = PAL$BLOCK,
              colour = "white", linewidth = 0.5) +
    geom_text(data = filter(hm, blocked), label = "\u00d7", size = 3,
              colour = "white", family = "serif") +
    geom_text(data = filter(hm, cell == "undefined"), label = "n/a", size = 2,
              colour = "grey40", family = "serif") +
    geom_text(data = filter(hm, cell == "delay"),
              aes(label = sprintf("%.1f", ratio)),
              size = 2.1, family = "serif", colour = "grey15") +
    scale_fill_gradientn(colours = grad, name = "delay\nratio",
                         limits = c(0, 4), na.value = "transparent",
                         breaks = c(1, 2, 3, 4)) +
    labs(x = NULL, y = NULL) +
    theme_pub() +
    theme(axis.text.x = element_text(angle = 40, hjust = 1),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          legend.position = "right",
          legend.key.height = unit(26, "pt"))

  save_fig((p_a | p_b) + tag_ab(),
           "results/exp3_taxonomy/fig4_taxonomy", W2, 3.2)
}

# =============================================================================
# Fig 5 — depth ablation (plan: already lean, keep 2 panels)
#   (a) grok rate by task, 1-layer vs 2-layer
#   (b) delay comparison for tasks that grokked at both depths
# =============================================================================
fig5 <- function() {
  da <- read_flat("results/exp3_depth_ablation/depth_ablation_results.json") |>
    mutate(depth = factor(paste0(n_layers, "-layer"),
                          c("1-layer", "2-layer")))

  rate <- da |>
    group_by(task, depth) |>
    summarise(n = n(), n_grok = sum(grokked), .groups = "drop") |>
    mutate(grok_rate = n_grok / n)

  depth_cols <- c("1-layer" = PAL$GREY, "2-layer" = PAL$TRAIN)

  p_a <- ggplot(rate, aes(task, grok_rate, fill = depth)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.64, colour = NA) +
    geom_text(aes(label = sprintf("%d/%d", n_grok, n)),
              position = position_dodge(width = 0.72), vjust = -0.45,
              size = 2.4, family = "serif", colour = "grey20") +
    scale_fill_manual(values = depth_cols, name = NULL) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1.18), expand = c(0, 0)) +
    labs(x = NULL, y = "grok rate") +
    theme_pub() +
    # at c(0.8, 0.85) the legend sat on the mod_add 2-layer "5/5" label
    theme(legend.position = "top",
          legend.margin = margin(0, 0, 0, 0))

  both <- da |>
    filter(grokked) |>
    group_by(task, depth) |>
    summarise(mean_delay = mean(grok_delay), .groups = "drop") |>
    group_by(task) |>
    filter(n() == 2) |>
    ungroup()

  p_b <- if (nrow(both) == 0) {
    ggplot() +
      annotate("text", x = 0, y = 0, size = 3, family = "serif",
               colour = "grey40",
               label = "no task grokked at both depths") +
      theme_void()
  } else {
    ggplot(both, aes(task, mean_delay, fill = depth)) +
      geom_col(position = position_dodge(width = 0.72), width = 0.64,
               colour = NA) +
      scale_fill_manual(values = depth_cols, name = NULL) +
      scale_y_continuous(labels = label_number(scale = 1e-3, suffix = "K")) +
      labs(x = NULL, y = "mean grok delay (steps)") +
      theme_pub() +
      theme(legend.position = "none")
  }

  save_fig((p_a | p_b) + tag_ab(),
           "results/exp3_depth_ablation/fig5_depth_ablation", W2, H)
}

# =============================================================================
# Fig 6 — circuit instability (non-abelian, S5, 2 layers)
#   (a) train/test accuracy with grokked intervals shaded as bands
#   (b) Fourier alignment timeline — collapses at the same points
#   Bands replace arrows/labels per the plan, to cut clutter.
# =============================================================================
fig6 <- function() {
  a <- read_raw("results/exp5_antigrok_investigation/antigrok_s5_L2_s46.json")
  steps <- unlist(a$steps)
  df <- data.frame(step = steps, train = unlist(a$train_acc),
                   test = unlist(a$test_acc),
                   fa = unlist(a$fourier_alignment))

  # Collapse events, not the grokked stretches. Shading every interval where
  # test >= 0.9 fragments the band wherever accuracy briefly dips, which reads
  # as a rendering artifact. The plan asks for the collapse events themselves:
  # dips below threshold AFTER the circuit has first formed.
  up <- df$test >= 0.9
  r <- rle(up)
  ends <- cumsum(r$lengths); starts <- ends - r$lengths + 1
  first_grok <- if (any(r$values)) starts[which(r$values)[1]] else NA_integer_
  keep <- !r$values & !is.na(first_grok) & starts > first_grok
  bands <- data.frame(xmin = df$step[starts[keep]],
                      xmax = df$step[ends[keep]],
                      at = (df$step[starts[keep]] + df$step[ends[keep]]) / 2)
  cat(sprintf("   fig6: first grok at step %s; %d collapse event(s)\n",
              comma(df$step[first_grok]), nrow(bands)))

  # The plan asked for shaded bands here. Each collapse is a single 100-step
  # sample out of 120,000, so a band 0.08% wide renders as an invisible
  # hairline; widening it to something visible would overstate the collapse
  # duration. A collapse is a moment, not an interval, so mark it with a line.
  collapse_layer <- if (nrow(bands)) {
    geom_vline(data = bands, inherit.aes = FALSE, aes(xintercept = at),
               colour = PAL$BLOCK, linetype = "22", linewidth = 0.5)
  } else NULL

  long <- df |>
    select(step, train, test) |>
    pivot_longer(c(train, test), names_to = "metric", values_to = "acc") |>
    mutate(metric = factor(metric, c("train", "test"),
                           c("train accuracy", "test accuracy")))

  p_a <- ggplot(long, aes(step, acc, colour = metric)) +
    collapse_layer +
    # train accuracy is logged every 100 of 120k steps: at a single linewidth it
    # renders as a solid comb that hides the test curve. Draw it thin.
    geom_line(data = filter(long, metric == "train accuracy"),
              linewidth = 0.3, alpha = 0.75) +
    geom_line(data = filter(long, metric == "test accuracy"), linewidth = 0.8) +
    annotate("text", x = min(df$step), y = 1.12, hjust = 0, size = 2.2,
             family = "serif", colour = PAL$BLOCK,
             label = sprintf("red = circuit collapse (%d events)", nrow(bands))) +
    scale_colour_manual(values = c("train accuracy" = PAL$TRAIN,
                                   "test accuracy" = PAL$TEST), name = NULL) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = "K")) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       limits = c(0, 1.22), expand = c(0.01, 0)) +
    labs(x = "training step", y = "accuracy") +
    theme_pub() +
    theme(legend.position = c(0.28, 0.34))

  p_b <- ggplot(df, aes(step, fa)) +
    collapse_layer +
    geom_line(linewidth = 0.7, colour = PAL$FOURIER) +
    scale_x_continuous(labels = label_number(scale = 1e-3, suffix = "K")) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.12))) +
    labs(x = "training step", y = "Fourier alignment") +
    theme_pub()

  save_fig((p_a | p_b) + tag_ab(),
           "results/exp5_antigrok_investigation/fig6_circuit_instability", W2, H)
}

# =============================================================================
main <- function() {
  cat("grokking-geography figures (ggplot2)\n")
  figs <- list(fig1 = fig1, fig2 = fig2, fig3 = fig3,
               fig4 = fig4, fig5 = fig5, fig6 = fig6)
  for (nm in names(figs)) {
    cat(sprintf("\n--- %s ---\n", nm))
    figs[[nm]]()
  }
  cat("\ndone\n")
}

if (sys.nframe() == 0L) main()
