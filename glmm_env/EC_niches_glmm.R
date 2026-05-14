#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(glmmTMB)
  library(emmeans)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(stringr)
  library(MASS)   
  library(grid)
})

`%||%` <- function(a, b) if (!is.null(a) && !is.na(a) && nzchar(a)) a else b

parse_args <- function(x) {
  # supports: --key=value, --key value, and short flags -i value -o value
  out <- list()
  i <- 1
  while (i <= length(x)) {
    tok <- x[[i]]
    
    # short flags
    if (tok %in% c("-i", "-o")) {
      key <- if (tok == "-i") "infile" else "outdir"
      val <- if (i + 1 <= length(x)) x[[i + 1]] else NA_character_
      out[[key]] <- val
      i <- i + 2
      next
    }
    
    # long flags
    if (startsWith(tok, "--")) {
      tok2 <- sub("^--", "", tok)
      if (grepl("=", tok2, fixed = TRUE)) {
        kv <- strsplit(tok2, "=", fixed = TRUE)[[1]]
        out[[kv[[1]]]] <- kv[[2]]
        i <- i + 1
      } else {
        key <- tok2
        val <- if (i + 1 <= length(x) && !startsWith(x[[i + 1]], "--")) x[[i + 1]] else "TRUE"
        out[[key]] <- val
        i <- i + 2
      }
    } else {
      i <- i + 1
    }
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

# Required
infile <- args$infile %||% NA_character_
outdir <- args$outdir %||% NA_character_

if (is.na(infile) || is.na(outdir)) {
  stop(
    "Usage:\n",
    "  Rscript R/ec_niches_glmm.R --infile <niches_long_glmm.csv> --outdir <results_dir>\n",
    "  Rscript R/ec_niches_glmm.R -i <csv> -o <results_dir>\n\n",
    "Example:\n",
    "  Rscript R/ec_niches_glmm.R -i results/niches_long_glmm.csv -o results/glmm\n"
  )
}
if (!file.exists(infile)) stop("Input file does not exist: ", infile)

# default settings
min_rows_per_fit   <- as.integer(args$min_rows %||% 30)
bracket_alpha      <- as.numeric(args$bracket_alpha %||% 0.05)
facet_ncol         <- as.integer(args$facet_ncol %||% 4)
use_log10_y        <- tolower(args$log10_y %||% "false") %in% c("true","t","1","yes","y")
pdf_width          <- as.numeric(args$pdf_width %||% 15)
pdf_height         <- as.numeric(args$pdf_height %||% 10)
base_size          <- as.numeric(args$base_size %||% 11)
fc_cutoff          <- as.numeric(args$fc_cutoff %||% 2)
min_niches_per_sample <- as.integer(args$min_niches_per_sample %||% 10)
label_with_effect  <- tolower(args$label_with_effect %||% "true") %in% c("true","t","1","yes","y")

# bracket spacing multipliers
bracket_lane_step  <- as.numeric(args$bracket_lane_step %||% 0.25)
bracket_base_pad   <- as.numeric(args$bracket_base_pad  %||% 0.20)


fdr_grouping <- c("brain_area", "radius")


dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "plots_pdf"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "tables"), recursive = TRUE, showWarnings = FALSE)

message("Input:  ", infile)
message("Output: ", outdir)
message("Params: min_rows=", min_rows_per_fit,
        " alpha=", bracket_alpha,
        " facet_ncol=", facet_ncol,
        " log10_y=", use_log10_y,
        " fc_cutoff=", fc_cutoff,
        " radii=from input file")

# Reduce thread oversubscription on HPC
Sys.setenv(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

df_all <- read.csv(infile)
required_cols <- c("EC_subtypes","sample","EC_cell_ID","brain_area","radius","cell_type","cell_count")
missing <- setdiff(required_cols, colnames(df_all))
if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse=", "))

df_all$cell_type   <- gsub("-", "_", df_all$cell_type)
df_all$sample      <- as.factor(df_all$sample)
df_all$EC_subtypes <- as.factor(df_all$EC_subtypes)
df_all$cell_type   <- as.factor(df_all$cell_type)
df_all$brain_area  <- as.factor(df_all$brain_area)
df_all$EC_cell_ID  <- as.factor(df_all$EC_cell_ID)
df_all$cell_count  <- as.integer(df_all$cell_count)

rvals <- sort(unique(as.integer(as.character(df_all$radius))))
df_all$radius <- factor(as.integer(as.character(df_all$radius)),
                        levels = rvals,
                        ordered = TRUE)
lvl <- levels(df_all$EC_subtypes)

# Area offset (density per area)
df_all$neigh_area <- pi * (as.numeric(as.character(df_all$radius))^2)
df_all$log_area_offset <- log(df_all$neigh_area)

message("Total rows: ", nrow(df_all))
message("Brain areas: ", paste(levels(df_all$brain_area), collapse = ", "))
message("Cell types: ", length(levels(df_all$cell_type)))
message("Samples: ", length(levels(df_all$sample)))
message("EC_subtypes levels: ", paste(levels(df_all$EC_subtypes), collapse = ", "))


# Raw means colours
ec_cols <- c(aECs="#D55E00", capECs="#0072B2", vECs="#009E73")

check_min_niches_per_sample <- function(dat, min_niches_per_sample = 10) {
  required_subtypes <- c("aECs", "vECs", "capECs")
  cutoff <- min_niches_per_sample

  counts <- dat %>%
    count(sample, EC_subtypes, name = "n_niches") %>%
    mutate(EC_subtypes = as.character(EC_subtypes)) %>%
    tidyr::complete(
      sample,
      EC_subtypes = required_subtypes,
      fill = list(n_niches = 0)
    )

  failing <- counts %>%
    filter(n_niches < .env$cutoff) %>%
    arrange(sample, EC_subtypes)

  list(
    ok = nrow(failing) == 0,
    counts = counts,
    failing = failing
  )
}

standardize_emm_ci <- function(emm_df) {
  candidates_lower <- c("lower.CL", "asymp.LCL", "lower", "LCL")
  candidates_upper <- c("upper.CL", "asymp.UCL", "upper", "UCL")

  low_col <- candidates_lower[candidates_lower %in% names(emm_df)][1]
  up_col  <- candidates_upper[candidates_upper %in% names(emm_df)][1]

  if (is.na(low_col) || is.na(up_col)) {
    emm_df$lower <- NA_real_
    emm_df$upper <- NA_real_
  } else {
    emm_df$lower <- emm_df[[low_col]]
    emm_df$upper <- emm_df[[up_col]]
  }
  emm_df
}

parse_contrast <- function(x) {
  # emmeans::pairs() yields contrasts "aECs / capECs"
  s <- stringr::str_trim(as.character(x))
  parts <- stringr::str_split(s, "\\s*[/\\-\\u2013\\u2014]\\s*", n = 2)

  g1 <- vapply(parts, function(z) if (length(z) >= 1) z[[1]] else NA_character_, character(1))
  g2 <- vapply(parts, function(z) if (length(z) >= 2) z[[2]] else NA_character_, character(1))

  df <- data.frame(g1 = g1, g2 = g2, stringsAsFactors = FALSE)
}

format_bracket_label <- function(sig, q, fold_change = NA_real_, abs_density_diff = NA_real_, include_effect = TRUE) {
  label <- paste0(sig, " FDR=", formatC(q, format = "g", digits = 2))

  if (!isTRUE(include_effect)) {
    return(label)
  }

  fc_txt <- ifelse(
    is.finite(fold_change),
    paste0("\nFC=", formatC(fold_change, format = "f", digits = 2)),
    ""
  )

  delta_txt <- ifelse(
    is.finite(abs_density_diff),
    paste0(
      ifelse(nchar(fc_txt) > 0, ", ", "\n"),
      "Δ=", formatC(abs_density_diff, format = "f", digits = 1)
    ),
    ""
  )

  paste0(label, fc_txt, delta_txt)
}

assign_lanes <- function(xmin, xmax) {
  lane <- rep(NA_integer_, length(xmin))
  ok <- which(!is.na(xmin) & !is.na(xmax))
  if (length(ok) == 0) return(lane)

  o <- ok[order(xmin[ok], xmax[ok])]
  lane_end <- numeric(0)

  for (i in o) {
    placed <- FALSE
    if (length(lane_end) > 0) {
      for (k in seq_along(lane_end)) {
        if (xmin[i] > lane_end[k]) {
          lane[i] <- k
          lane_end[k] <- xmax[i]
          placed <- TRUE
          break
        }
      }
    }
    if (!placed) {
      lane_end <- c(lane_end, xmax[i])
      lane[i] <- length(lane_end)
    }
  }
  lane
}

fit_one <- function(dat) {
  if (nrow(dat) < min_rows_per_fit) {
    return(list(
      ok = FALSE,
      reason = paste0("Too few total rows: ", nrow(dat), " < ", min_rows_per_fit)
    ))
  }

  if (length(unique(dat$EC_subtypes)) < 2) {
    return(list(
      ok = FALSE,
      reason = "Fewer than 2 EC_subtypes present"
    ))
  }

  niche_check <- check_min_niches_per_sample(dat, min_niches_per_sample)
  if (!niche_check$ok) {
    fail_txt <- niche_check$failing %>%
      mutate(msg = paste0(sample, ":", EC_subtypes, "=", n_niches)) %>%
      pull(msg) %>%
      paste(collapse = "; ")

    return(list(
      ok = FALSE,
      reason = paste0(
        "Failed min_niches_per_sample filter (< ", min_niches_per_sample, "): ",
        fail_txt
      ),
      failing_counts = niche_check$failing,
      all_counts = niche_check$counts
    ))
  }

  mod <- NULL
  fit_type <- NA_character_

  try({
    mod <- glmmTMB(
      cell_count ~ EC_subtypes + (1 | sample) + offset(log_area_offset),
      data = dat,
      family = nbinom2()
    )
    fit_type <- "glmmTMB_nbinom2"
  }, silent = TRUE)

  if (is.null(mod)) {
    try({
      mod <- glm.nb(
        cell_count ~ EC_subtypes + offset(log_area_offset),
        data = dat
      )
      fit_type <- "glm_nb"
    }, silent = TRUE)
  }

  if (is.null(mod)) {
    return(list(
      ok = FALSE,
      reason = "Model fitting failed for both glmmTMB and glm.nb"
    ))
  }

  emm <- tryCatch({
    emmeans(mod, ~ EC_subtypes,
            at = list(log_area_offset = 0),
            type = "response") |> as.data.frame()
  }, error = function(e) NULL)

  if (is.null(emm) || nrow(emm) == 0) {
    return(list(
      ok = FALSE,
      reason = "emmeans failed or returned no rows"
    ))
  }

  emm <- standardize_emm_ci(emm)
  emm$response <- emm$response * 1e6
  emm$lower    <- emm$lower    * 1e6
  emm$upper    <- emm$upper    * 1e6

  con <- tryCatch({
    pairs(
      emmeans(
        mod,
        ~ EC_subtypes,
        at = list(log_area_offset = 0),
        type = "response"
      )
    ) |> as.data.frame()
  }, error = function(e) data.frame())

  list(
    ok = TRUE,
    mod = mod,
    fit_type = fit_type,
    emm = emm,
    con = con
  )
}

areas <- levels(df_all$brain_area)
radii <- levels(df_all$radius)
celltypes <- levels(df_all$cell_type)

fits <- list()
filtered_out <- list()

for (a in areas) {
  for (r in radii) {
    for (ct in celltypes) {
      dat <- df_all %>% filter(brain_area == a, radius == r, cell_type == ct)
      res <- fit_one(dat)

      if (isTRUE(res$ok)) {
        fits[[length(fits) + 1]] <- list(
          brain_area = a,
          radius = r,
          cell_type = ct,
          res = res
        )
      } else {
        filtered_out[[length(filtered_out) + 1]] <- list(
          brain_area = a,
          radius = r,
          cell_type = ct,
          n_rows = nrow(dat),
          n_samples = dplyr::n_distinct(dat$sample),
          n_ec_subtypes = dplyr::n_distinct(dat$EC_subtypes),
          reason = res$reason
        )
      }
    }
  }
}

if (length(fits) == 0) stop("No successful fits at all.")

filtered_out_df <- bind_rows(filtered_out)

if (nrow(filtered_out_df) > 0) {
  write.csv(
    filtered_out_df,
    file.path(outdir, "tables", "Filtered_out_model_subsets.csv"),
    row.names = FALSE
  )

  message("Filtered out subsets: ", nrow(filtered_out_df))
  print(filtered_out_df %>% count(reason, sort = TRUE))
} else {
  message("No subsets were filtered out before fitting.")
}

#results + FDR

emm_all <- purrr::map_dfr(fits, function(x) {
  out <- x$res$emm
  out$brain_area <- x$brain_area
  out$radius     <- x$radius
  out$cell_type  <- x$cell_type
  out$fit_type   <- x$res$fit_type
  out
})

con_all <- purrr::map_dfr(fits, function(x) {
  out <- x$res$con
  if (nrow(out) == 0) return(out)
  out$brain_area <- x$brain_area
  out$radius     <- x$radius
  out$cell_type  <- x$cell_type
  out$fit_type   <- x$res$fit_type
  out
})

contrast_audit_raw <- con_all %>%
  count(contrast, sort = TRUE)

write.csv(
  contrast_audit_raw,
  file.path(outdir, "tables", "Contrast_audit_raw_emmeans_labels.csv"),
  row.names = FALSE
)

message("Raw contrast labels from emmeans:")
print(contrast_audit_raw)            

if (nrow(con_all) > 0) {
  con_all <- con_all %>%
    mutate(
      sig = case_when(
        p.value <= 0.0001 ~ "****",
        p.value <= 0.001  ~ "***",
        p.value <= 0.01   ~ "**",
        p.value <= 0.05   ~ "*",
        TRUE              ~ "ns"
      )
    ) %>%
    group_by(across(all_of(fdr_grouping))) %>%
    mutate(p_adj = p.adjust(p.value, method = "BH")) %>%
    ungroup()

  tmp <- parse_contrast(con_all$contrast)
  con_all$g1 <- str_trim(tmp$g1)
  con_all$g2 <- str_trim(tmp$g2)

contrast_audit_parsed <- con_all %>%
  mutate(pair = paste(g1, g2, sep = " / ")) %>%
  count(pair, sort = TRUE)

write.csv(
  contrast_audit_parsed,
  file.path(outdir, "tables", "Contrast_audit_parsed_pairs.csv"),
  row.names = FALSE
)

message("Parsed contrast pairs:")
print(contrast_audit_parsed)

  # drop anything that still didn't parse
  con_all <- con_all %>% filter(!is.na(g1), !is.na(g2))

  emm_lookup <- emm_all %>%
    dplyr::select(brain_area, radius, cell_type, EC_subtypes, response) %>%
    dplyr::rename(group = EC_subtypes, density = response)

  con_all <- con_all %>%
    left_join(
      emm_lookup %>% dplyr::rename(g1 = group, density_g1 = density),
      by = c("brain_area", "radius", "cell_type", "g1")
    ) %>%
    left_join(
      emm_lookup %>% dplyr::rename(g2 = group, density_g2 = density),
      by = c("brain_area", "radius", "cell_type", "g2")
    ) %>%
    mutate(
      fold_change = pmax(density_g1, density_g2) / pmax(pmin(density_g1, density_g2), 1e-8),
      log2FC = log2(fold_change),
      abs_density_diff = abs(density_g1 - density_g2)
    )
} else {
  con_all$p_adj <- numeric(0)
  con_all$g1 <- character(0)
  con_all$g2 <- character(0)
  con_all$density_g1 <- numeric(0)
  con_all$density_g2 <- numeric(0)
  con_all$fold_change <- numeric(0)
  con_all$log2FC <- numeric(0)
  con_all$abs_density_diff <- numeric(0)
}

cat("Total contrasts tested:", nrow(con_all), "\n")
cat("Nominal p<=0.05:", sum(con_all$p.value <= 0.05, na.rm=TRUE), "\n")
cat("Min p:", min(con_all$p.value, na.rm=TRUE), "\n")

write.csv(
  con_all,
  file.path(outdir, "tables", "Pairwise_contrasts_FDR_within_area_radius_celltype_with_effect_sizes.csv"),
  row.names = FALSE
)



con_sig_fdr <- con_all %>%
  filter(!is.na(p_adj), p_adj <= bracket_alpha)

con_sig <- con_sig_fdr %>%
  filter(
    !is.na(fold_change),
    fold_change >= fc_cutoff
  )

message("Significant comparisons (FDR <= ", bracket_alpha, "): ", nrow(con_sig_fdr))
message("Significant comparisons after effect-size filters: ", nrow(con_sig))

write.csv(
  con_sig_fdr,
  file.path(outdir, "tables", paste0("Significant_contrasts_FDR_le_", bracket_alpha, "_before_effect_size_filter.csv")),
  row.names = FALSE
)

write.csv(
  con_sig,
  file.path(
    outdir,
    "tables",
    paste0(
      "Significant_contrasts_FDR_le_", bracket_alpha,
      "_FC_ge_", fc_cutoff,
      "_min_niches_per_sample_ge_", min_niches_per_sample,
      ".csv"
    )
  ),
  row.names = FALSE
)

dir.create(file.path(outdir, "plots_samplemeans_faceted"), recursive = TRUE, showWarnings = FALSE)

# use only significant subsets
sig_subsets <- con_sig %>%
  distinct(brain_area, radius, cell_type)

if (nrow(sig_subsets) > 0) {

  # raw data restricted to significant brain_area x radius x cell_type subsets
  plot_df <- df_all %>%
    semi_join(sig_subsets, by = c("brain_area", "radius", "cell_type")) %>%
    mutate(
      EC_subtypes = factor(as.character(EC_subtypes), levels = lvl),
      radius_num = as.numeric(as.character(radius))
    ) %>%
    group_by(cell_type, brain_area, radius, radius_num, EC_subtypes, sample) %>%
    summarise(
      sample_mean_count = mean(cell_count, na.rm = TRUE),
      n_niches = n(),
      .groups = "drop"
    )

  # sample colors
  sample_levels <- sort(unique(as.character(plot_df$sample)))
  sample_cols <- setNames(grDevices::hcl.colors(length(sample_levels), "Dark 3"), sample_levels)

  # significance text per subset
  sig_labels <- con_sig %>%
    arrange(cell_type, brain_area, radius, p_adj, contrast) %>%
    mutate(
      sig_txt = paste0(
        contrast,
        " | FDR=", formatC(p_adj, format = "g", digits = 2),
        " | FC=", formatC(fold_change, format = "f", digits = 2)
      )
    ) %>%
    group_by(cell_type, brain_area, radius) %>%
    summarise(
      sig_label = paste(sig_txt, collapse = "\n"),
      .groups = "drop"
    )

  # one PDF per cell type
  sig_celltypes <- sort(unique(as.character(plot_df$cell_type)))

  for (ct in sig_celltypes) {

    dat_ct <- plot_df %>%
      filter(cell_type == ct)

    if (nrow(dat_ct) == 0) next

    # keep panel order stable
    dat_ct <- dat_ct %>%
      mutate(
        brain_area = factor(as.character(brain_area), levels = areas),
        EC_subtypes = factor(as.character(EC_subtypes), levels = lvl),
        radius = factor(as.character(radius), levels = as.character(rvals), ordered = TRUE)
      )

    # panel annotations showing which radii/areas were significant
    p_ct <- ggplot(dat_ct, aes(x = radius, y = sample_mean_count)) +
      geom_boxplot(
        aes(group = radius),
        width = 0.7,
        outlier.shape = NA,
        fill = "grey90",
        colour = "grey70",
        linewidth = 0.35
      ) +
      geom_point(
        aes(colour = sample),
        position = position_jitter(width = 0.12, height = 0),
        size = 1.8,
        alpha = 0.95
      ) +
      facet_grid(EC_subtypes ~ brain_area, scales = "free_y") +
      scale_colour_manual(values = sample_cols, name = "Sample") +
      labs(
        x = "Radius (µm)",
        y = "Mean cell count per niche",
        title = ct
      ) +
      theme_classic(base_size = base_size) +
        theme(
          plot.title = element_text(face = "bold", hjust = 0.5),
          axis.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1),
          strip.text = element_text(face = "bold"),
          strip.background = element_rect(fill = "white", colour = "black", linewidth = 0.4),
          panel.border = element_rect(fill = NA, colour = "black", linewidth = 0.4),
          panel.grid = element_blank(),
          legend.position = "right"
        )

  
    if (use_log10_y) {
      p_ct <- p_ct + scale_y_log10()
    }

    out_ct <- file.path(
      outdir,
      "plots_samplemeans_faceted",
      paste0("SampleMean_boxplot_faceted_", ct, ".pdf")
    )

    ggsave(out_ct, p_ct, width = 16, height = 8, device = cairo_pdf)
  }
}
                  
lvl <- levels(df_all$EC_subtypes)
xmap <- setNames(seq_along(lvl), lvl)

facet_y <- emm_all %>%
  dplyr::group_by(brain_area, radius, cell_type) %>%
  dplyr::summarise(
    ymin_raw = suppressWarnings(min(c(response, lower), na.rm = TRUE)),
    ymax_raw = suppressWarnings(max(c(response, upper), na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    ymin = ifelse(is.finite(ymin_raw), ymin_raw, 0),
    ymax = ifelse(is.finite(ymax_raw), ymax_raw, 0),
    yrng = ymax - ymin,
    yrng = ifelse(!is.finite(yrng) | yrng <= 0, 0, yrng),
    yrng = pmax(yrng, 0.25 * pmax(ymax, 1), 10)
  ) %>%
  dplyr::select(brain_area, radius, cell_type, ymin, ymax, yrng)

brackets <- con_sig %>%
  mutate(
    g1 = str_trim(g1),
    g2 = str_trim(g2),
    x1n = unname(xmap[g1]),
    x2n = unname(xmap[g2]),
    xminn = pmin(x1n, x2n),
    xmaxn = pmax(x1n, x2n),
    xmid  = (x1n + x2n) / 2,
    label = format_bracket_label(sig, p_adj, fold_change, abs_density_diff, include_effect = label_with_effect)
  ) %>%
  left_join(facet_y, by = c("brain_area", "radius", "cell_type")) %>%
  filter(!is.na(xminn), !is.na(xmaxn), !is.na(ymax), !is.na(yrng)) %>%
  group_by(brain_area, radius, cell_type) %>%
  arrange(p_adj, xminn, xmaxn, .by_group = TRUE) %>%
  mutate(lane = assign_lanes(xminn, xmaxn)) %>%
  ungroup() %>%
  filter(!is.na(lane)) %>%
  mutate(
    n_lanes = max(lane, na.rm = TRUE),
    
    min_lane_step = 30,   # vertical distance between brackets
    min_base_pad  = 25,   # distance from data to first bracket
    min_tick      = 6,
    min_text_gap  = 10,
    
    lane_step_eff = pmax(bracket_lane_step * yrng, min_lane_step),
    base_pad_eff  = pmax(bracket_base_pad  * yrng, min_base_pad),
    
    # extra expansion if many lanes
    lane_step_eff = lane_step_eff * ifelse(n_lanes >= 8, 2.0,
                                           ifelse(n_lanes >= 6, 1.6,
                                                  ifelse(n_lanes >= 4, 1.3, 1.1))),
    
    y      = ymax + base_pad_eff + lane_step_eff * (lane - 1),
    
    tick   = pmax(0.04 * yrng, 0.02 * ymax, min_tick),
    
    y_text = y + pmax(0.08 * yrng, 0.04 * ymax, min_text_gap)
  )

pub_theme <- theme_classic(base_size = base_size) +
  theme(
    plot.title = element_text(face = "bold", size = base_size + 2, hjust = 0.5),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    strip.text = element_text(face = "bold", size = base_size - 1),
    strip.background = element_rect(fill = "white", colour = "black", linewidth = 0.4),
    panel.border = element_rect(fill = NA, colour = "black", linewidth = 0.4),
    panel.spacing = grid::unit(0.8, "lines"),
    legend.position = "none",
    plot.margin = margin(25, 25, 10, 10)
  )

set.seed(1)

for (a in areas) {
  for (r in radii) {

    emm_ar <- filter(emm_all, brain_area == a, radius == r)
    outfile <- file.path(outdir, "plots_pdf", paste0("ECniche_", a, "_radius_", r, "um.pdf"))

    if (nrow(emm_ar) == 0) {
      p0 <- ggplot() +
        theme_void(base_size = base_size) +
        ggtitle(paste0("EC niche | ", a, " | radius ", r, " µm")) +
        annotate("text", x = 0, y = 0,
                 label = "No successful fits for any cell type",
                 size = 5)
      ggsave(outfile, p0, width = pdf_width, height = pdf_height, device = cairo_pdf)
      next
    }

    # numeric x for model estimates
    emm_ar <- emm_ar %>%
      mutate(
        EC_subtypes = factor(EC_subtypes, levels = lvl),
        xn = unname(xmap[as.character(EC_subtypes)])
      ) %>%
      filter(is.finite(xn))

    # raw overlay: per-sample mean density
    scale_fac <- 1e6
    
    raw_sample_means <- df_all %>%
      filter(brain_area == a, radius == r) %>%
      mutate(
        EC_subtypes = factor(EC_subtypes, levels = lvl),
        raw_density = (cell_count / neigh_area) * scale_fac
      ) %>%
      group_by(cell_type, EC_subtypes, sample) %>%
      summarise(mean_density = mean(raw_density, na.rm = TRUE), .groups = "drop") %>%
      mutate(xn = unname(xmap[as.character(EC_subtypes)])) %>%
      filter(is.finite(xn))

    br_ar <- brackets %>% filter(brain_area == a, radius == r)

    # DEBUG LINE (per plot)
    message("Plot ", a, " radius ", r, " brackets: ", nrow(br_ar))

    p <- ggplot(emm_ar, aes(x = xn, y = response)) +

      geom_blank(
        data = br_ar,
        aes(x = xmid, y = y_text + pmax(0.08 * yrng, 0.04 * ymax)),
        inherit.aes = FALSE
      ) +

      # raw sample means
      geom_point(
        data = raw_sample_means,
        aes(x = xn, y = mean_density, colour = EC_subtypes),
        inherit.aes = FALSE,
        position = position_jitter(width = 0.10, height = 0),
        alpha = 0.90,
        size = 1.6
      ) +

      # model estimate + CI
      geom_pointrange(
        aes(ymin = lower, ymax = upper),
        linewidth = 0.35,
        colour = "black",
        na.rm = TRUE
      ) +

      facet_wrap(~ cell_type, scales = "free_y", ncol = facet_ncol) +

      scale_colour_manual(values = ec_cols, guide = "none") +
      scale_x_continuous(breaks = seq_along(lvl), labels = lvl) +

      labs(
        x = "EC subtype",
        y = "Cell density (cells / mm²)",
        title = paste0("EC niche | ", a, " | radius ", r, " µm")
      ) +

      pub_theme +
      coord_cartesian(clip = "off") +
      scale_y_continuous(expand = expansion(mult = c(0.02, 0.5)))

    # Brackets
    if (nrow(br_ar) > 0) {
      p <- p +
        geom_segment(
          data = br_ar,
          aes(x = xminn, xend = xmaxn, y = y, yend = y),
          inherit.aes = FALSE,
          linewidth = 0.35,
          colour = "black"
        ) +
        geom_segment(
          data = br_ar,
          aes(x = xminn, xend = xminn, y = y, yend = y - tick),
          inherit.aes = FALSE,
          linewidth = 0.35,
          colour = "black"
        ) +
        geom_segment(
          data = br_ar,
          aes(x = xmaxn, xend = xmaxn, y = y, yend = y - tick),
          inherit.aes = FALSE,
          linewidth = 0.35,
          colour = "black"
        ) +
        geom_text(
          data = br_ar,
          aes(x = xmid, y = y_text, label = label),
          inherit.aes = FALSE,
          size = 2.8,
          lineheight = 0.9,
          vjust = 0,
          colour = "black"
        )
    }

    if (use_log10_y) p <- p + scale_y_log10()

    ggsave(outfile, p, width = pdf_width, height = pdf_height, device = "pdf")
  }
}


