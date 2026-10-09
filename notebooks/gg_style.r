# gg_style.r — shared ggplot2 style for Grokking Geography figures
#
# Mirrors the semantics of notebooks/pub_style.py (the matplotlib equivalent) so
# the R and Python figures stay visually consistent. Okabe-Ito colourblind-safe
# palette, serif, minimal chrome, 300 DPI PNG + vector PDF.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(scales)
})

# ---- palette, matched to pub_style.py PAL -----------------------------------
PAL <- list(
  TRAIN    = "#0077BB",  # train accuracy
  TEST     = "#EE7733",  # test accuracy
  FOURIER  = "#AA3377",  # fourier alignment
  GROK     = "#009988",  # grokking marker / success
  BLOCK    = "#CC3311",  # blocked / never groks
  GREY     = "#BBBBBB",  # baseline, reference
  ABELIAN  = "#0077BB",
  NONABEL  = "#CC3311",
  PHASE_MEM     = "#EE7733",
  PHASE_CIRCUIT = "#AA3377",
  PHASE_CLEANUP = "#009988"
)

# discrete "grokked / blocked" used consistently across every figure
GROK_COLS <- c("grokked" = PAL$GROK, "no grok" = PAL$BLOCK)
GROK_FILL <- scale_fill_manual(values = GROK_COLS, name = NULL)
GROK_COLOUR <- scale_colour_manual(values = GROK_COLS, name = NULL)

theme_pub <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = "serif") +
    theme(
      axis.line        = element_line(linewidth = 0.4, colour = "#666666"),
      axis.ticks       = element_line(linewidth = 0.4, colour = "#666666"),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1, colour = "grey20"),
      legend.key.size  = unit(9, "pt"),
      legend.text      = element_text(size = base_size - 1),
      legend.background = element_blank(),
      plot.tag         = element_text(size = base_size + 2, face = "bold"),
      plot.title       = element_text(size = base_size + 1),
      strip.background = element_blank(),
      strip.text       = element_text(face = "bold", size = base_size - 1)
    )
}

# shadowed by patchwork::plot_annotation tags; (a)/(b) style
tag_ab <- function() plot_annotation(
  tag_levels = "a", tag_prefix = "(", tag_suffix = ")"
)

# ---- phase shading: memorisation / circuit formation / cleanup --------------
# Rectangles need finite y bounds, so callers pass the panel's y range.
phase_rects <- function(mem_step, grok_step, max_step, ymin, ymax, alpha = 0.09) {
  layers <- list()
  if (!is.null(mem_step))
    layers <- c(layers, annotate("rect", xmin = -Inf, xmax = mem_step,
                                 ymin = ymin, ymax = ymax,
                                 fill = PAL$PHASE_MEM, alpha = alpha))
  if (!is.null(mem_step) && !is.null(grok_step))
    layers <- c(layers, annotate("rect", xmin = mem_step, xmax = grok_step,
                                 ymin = ymin, ymax = ymax,
                                 fill = PAL$PHASE_CIRCUIT, alpha = alpha))
  if (!is.null(grok_step))
    layers <- c(layers, annotate("rect", xmin = grok_step, xmax = max_step,
                                 ymin = ymin, ymax = ymax,
                                 fill = PAL$PHASE_CLEANUP, alpha = alpha))
  layers
}

phase_key <- function(mem_step, grok_step, x = 0) {
  # textual key for the shaded bands, placed in the top-left of the panel
  list(
    annotate("text", x = x, y = Inf, hjust = 0, vjust = 1.6, size = 2.4,
             colour = "grey35", family = "serif",
             label = sprintf("memorisation (0–%s)", comma(mem_step))),
    annotate("text", x = mem_step, y = Inf, hjust = 0, vjust = 1.6, size = 2.4,
             colour = "grey35", family = "serif",
             label = sprintf("circuit formation (%s–%s)", comma(mem_step), comma(grok_step))),
    annotate("text", x = grok_step, y = Inf, hjust = 0, vjust = 1.6, size = 2.4,
             colour = "grey35", family = "serif", label = "cleanup")
  )
}

# ---- IO ---------------------------------------------------------------------
# simplifyVector = FALSE everywhere: these JSON files mix scalars, vectors and
# ragged nested lists, and auto-simplification produces list-columns that are
# worse than unpacking by hand.
read_raw <- function(path) {
  if (!file.exists(path)) stop("missing input: ", path, call. = FALSE)
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

read_flat <- function(path) {
  if (!file.exists(path)) stop("missing input: ", path, call. = FALSE)
  as.data.frame(jsonlite::fromJSON(path, simplifyVector = TRUE),
                stringsAsFactors = FALSE)
}

# unlist a possibly-1-element JSON field to a single value
sc <- function(x, default = NA) {
  if (is.null(x)) return(default)
  v <- unlist(x, use.names = FALSE)
  if (!length(v)) return(default)
  v[[1]]
}

# ---- output -----------------------------------------------------------------
OUT_DIRS <- character(0)

save_fig <- function(p, stem, width, height) {
  dir.create(dirname(stem), showWarnings = FALSE, recursive = TRUE)
  ggsave(paste0(stem, ".pdf"), p, width = width, height = height, device = "pdf")
  ggsave(paste0(stem, ".png"), p, width = width, height = height, dpi = 300)
  cat(sprintf("  saved: %s.{pdf,png}  (%.1f x %.1f in)\n", stem, width, height))
  OUT_DIRS <<- c(OUT_DIRS, stem)
}

# ---- shared constants derived from the baseline -----------------------------
# Baseline grok delay is the denominator for every "delay ratio" in the paper.
BASELINE_JSON <- "results/exp1_baseline/baseline_p113_s42.json"
baseline <- function() read_raw(BASELINE_JSON)
