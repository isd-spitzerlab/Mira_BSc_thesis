#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  match_index <- match(flag, args)
  if (is.na(match_index)) return(default)
  if (match_index == length(args)) stop("Missing value for ", flag)
  args[[match_index + 1]]
}

infile <- get_arg("--input", NULL)
out_pdf <- get_arg("--output", NULL)

alpha <- as.numeric(get_arg("--alpha", "0.05"))
fc_cutoff <- as.numeric(get_arg("--fc-cutoff", "2"))
min_niches_per_sample <- as.numeric(get_arg("--min-niches-per-sample", "10"))
show_fdr_text <- as.logical(get_arg("--show-fdr-text", "FALSE"))

if (is.null(infile)) {
  stop("Missing required argument: --input")
}

if (is.null(out_pdf)) {
  stop("Missing required argument: --output")
}

dir.create(dirname(out_pdf), recursive = TRUE, showWarnings = FALSE)

con <- read.csv(infile)

# Ensure expected columns exist
if (!"p_adj" %in% names(con)) stop("Column 'p_adj' not found in input file.")
if (!"contrast" %in% names(con)) stop("Column 'contrast' not found in input file.")
if (!"cell_type" %in% names(con)) stop("Column 'cell_type' not found in input file.")
if (!"brain_area" %in% names(con)) stop("Column 'brain_area' not found in input file.")
if (!"radius" %in% names(con)) stop("Column 'radius' not found in input file.")

if (!"estimate" %in% names(con)) con$estimate <- NA_real_
if (!"ratio" %in% names(con)) con$ratio <- NA_real_
if (!"fold_change" %in% names(con)) con$fold_change <- NA_real_

parts <- strsplit(as.character(con$contrast), " / ", fixed = TRUE)
con$g1 <- vapply(parts, function(x) if (length(x) >= 1) x[[1]] else NA_character_, character(1))
con$g2 <- vapply(parts, function(x) if (length(x) >= 2) x[[2]] else NA_character_, character(1))

con <- con %>%
  mutate(
    sig = case_when(
      is.na(p_adj) ~ "",
      p_adj <= 0.001 ~ "***",
      p_adj <= 0.01  ~ "**",
      p_adj <= 0.05  ~ "*",
      TRUE ~ ""
    ),
    dir = case_when(
      is.na(p_adj) | p_adj > alpha ~ 0,
      !is.na(ratio) & ratio > 1 ~  1,
      !is.na(ratio) & ratio < 1 ~ -1,
      TRUE ~ 0
    ),
    contrast_clean = str_replace_all(contrast, "\\s*/\\s*", " vs "),
    lab = ifelse(
      show_fdr_text & sig != "",
      paste0(sig, "\nFDR=", formatC(p_adj, format = "g", digits = 2)),
      sig
    )
  )

con_sig <- con %>%
  filter(!is.na(p_adj), p_adj <= alpha) %>%
  filter(!is.na(fold_change), fold_change >= fc_cutoff) %>%
  filter(dir != 0) %>%
  filter(!str_detect(
    as.character(cell_type),
    regex("^(ECs?|Tanycytes|Microglia|Neurons_Other|Neurons_Granule_Immature)$",
          ignore_case = TRUE)
  )) %>%
  mutate(
    radius_num = suppressWarnings(as.numeric(as.character(radius))),
    radius_f = factor(radius_num, levels = sort(unique(radius_num))),
    cell_type = factor(as.character(cell_type), levels = sort(unique(as.character(cell_type))))
  )

p <- ggplot(con_sig, aes(x = radius_f, y = cell_type, fill = factor(dir))) +
  geom_tile(color = "grey85", linewidth = 0.25) +
  facet_grid(brain_area ~ contrast_clean) +
  scale_fill_manual(
    values = c(`-1` = "#2b6cb0", `1` = "#c53030"),
    breaks = c("-1", "1"),
    labels = c(
      "Right of contrast > Left of contrast",
      "Left of contrast > Right of contrast"
    ),
    name = "Direction"
  ) +
  labs(
    x = "Radius (µm)",
    y = "Cell type",
    title = paste0(
      "EC type niches: significant contrasts\n",
      "(FDR <= ", alpha,
      ", fold change >= ", fc_cutoff,
      ", number of EC niches per sample >= ", min_niches_per_sample, ")"
    )
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    panel.grid = element_blank(),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(out_pdf, p, width = 18, height = 12, device = cairo_pdf)

message("Saved: ", out_pdf)
message("Retained contrasts in filtered input after removing EC/ECs rows: ", nrow(con_sig))
message(
  "Filtering used: FDR <= ", alpha,
  ", fold_change >= ", fc_cutoff,
  ", and >= ", min_niches_per_sample,
  " niches per sample for each EC subtype."
)