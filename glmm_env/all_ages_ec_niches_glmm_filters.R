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
  out <- list()
  i <- 1
  while (i <= length(x)) {
    tok <- x[[i]]

    if (tok %in% c("-i", "-o")) {
      key <- if (tok == "-i") "infile" else "outdir"
      val <- if (i + 1 <= length(x)) x[[i + 1]] else NA_character_
      out[[key]] <- val
      i <- i + 2
      next
    }

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

infile <- args$infile %||% NA_character_
outdir <- args$outdir %||% NA_character_

if (is.na(infile) || is.na(outdir)) {
  stop(
    "Usage:\n",
    "  Rscript R/all_ages_ec_niches_glmm.R --infile <csv> --outdir <results_dir>\n",
    "  Rscript R/all_ages_ec_niches_glmm.R -i <csv> -o <results_dir>\n"
  )
}
if (!file.exists(infile)) stop("Input file does not exist: ", infile)

# optional parameters
min_rows_per_fit      <- as.integer(args$min_rows %||% 30)
bracket_alpha         <- as.numeric(args$bracket_alpha %||% 0.05)
facet_ncol            <- as.integer(args$facet_ncol %||% 4)
use_log10_y           <- tolower(args$log10_y %||% "false") %in% c("true", "t", "1", "yes", "y")
pdf_width             <- as.numeric(args$pdf_width %||% 15)
pdf_height            <- as.numeric(args$pdf_height %||% 10)
base_size             <- as.numeric(args$base_size %||% 11)
fc_cutoff             <- as.numeric(args$fc_cutoff %||% 2)
min_niches_per_sample <- as.integer(args$min_niches_per_sample %||% 10)
label_with_effect     <- tolower(args$label_with_effect %||% "true") %in% c("true", "t", "1", "yes", "y")
baseline_age          <- suppressWarnings(as.numeric(args$baseline_age %||% 3))
bracket_lane_step     <- as.numeric(args$bracket_lane_step %||% 0.25)
bracket_base_pad      <- as.numeric(args$bracket_base_pad %||% 0.20)

# BH-FDR grouping
fdr_grouping <- c("brain_area", "radius")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "plots_pdf"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "plots_samplemeans_faceted"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(outdir, "tables"), recursive = TRUE, showWarnings = FALSE)

message("Input:  ", infile)
message("Output: ", outdir)
message(
  "Params: min_rows=", min_rows_per_fit,
  " alpha=", bracket_alpha,
  " facet_ncol=", facet_ncol,
  " log10_y=", use_log10_y,
  " fc_cutoff=", fc_cutoff,
  " min_niches_per_sample=", min_niches_per_sample,
  " baseline_age=", baseline_age,
  " radii=from input file"
)

Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")

#load and prep data
df_all <- read.csv(infile)

required_cols <- c(
  "EC_subtypes", "sample", "EC_cell_ID", "brain_area",
  "radius", "cell_type", "cell_count", "age_months"
)
missing <- setdiff(required_cols, colnames(df_all))
if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))

df_all$cell_type   <- gsub("-", "_", df_all$cell_type)
df_all$sample      <- as.factor(df_all$sample)
df_all$EC_subtypes <- as.factor(df_all$EC_subtypes)
df_all$cell_type   <- as.factor(df_all$cell_type)
df_all$brain_area  <- as.factor(df_all$brain_area)
df_all$EC_cell_ID  <- as.factor(df_all$EC_cell_ID)
df_all$cell_count  <- as.integer(df_all$cell_count)

# keep age numeric for baseline comparisons; factor version for model/emmeans
df_all$age_months_num <- suppressWarnings(as.numeric(as.character(df_all$age_months)))
if (any(is.na(df_all$age_months_num))) {
  stop("Could not coerce age_months to numeric for all rows.")
}

age_levels_num <- sort(unique(df_all$age_months_num))
df_all$age_months <- factor(df_all$age_months_num, levels = age_levels_num, ordered = TRUE)

rvals <- sort(unique(as.integer(as.character(df_all$radius))))
df_all$radius <- factor(as.integer(as.character(df_all$radius)), levels = rvals, ordered = TRUE)

lvl <- levels(df_all$EC_subtypes)
areas <- levels(df_all$brain_area)

df_all$neigh_area <- pi * (as.numeric(as.character(df_all$radius))^2)
df_all$log_area_offset <- log(df_all$neigh_area)

message("Total rows: ", nrow(df_all))
message("Brain areas: ", paste(levels(df_all$brain_area), collapse = ", "))
message("Cell types: ", length(levels(df_all$cell_type)))
message("Samples: ", length(levels(df_all$sample)))
message("Age levels: ", paste(levels(df_all$age_months), collapse = ", "))
message("EC_subtypes levels: ", paste(levels(df_all$EC_subtypes), collapse = ", "))

ec_cols <- c(aECs = "#D55E00", capECs = "#0072B2", vECs = "#009E73")


# helper functions for stat analysis

log_msg <- function(...) message(sprintf(...))

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

format_bracket_label <- function(sig, q, fold_change = NA_real_, include_effect = TRUE) {
  label <- paste0(sig, " FDR=", formatC(q, format = "g", digits = 2))
  if (!isTRUE(include_effect)) return(label)

  fc_txt <- ifelse(
    is.finite(fold_change),
    paste0("\nFC=", formatC(fold_change, format = "f", digits = 2)),
    ""
  )

  paste0(label, fc_txt)
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

# fit one brain_area x radius x cell_type subset
# biological contrast: age vs baseline = 3 months within each EC subtype
fit_one <- function(dat, tag) {
  if (nrow(dat) < min_rows_per_fit) {
    log_msg("[SKIP] %s too few rows: %d < %d", tag, nrow(dat), min_rows_per_fit)
    return(NULL)
  }

  if (dplyr::n_distinct(dat$age_months) < 2) {
    log_msg("[SKIP] %s only %d age levels", tag, dplyr::n_distinct(dat$age_months))
    return(NULL)
  }

  if (dplyr::n_distinct(dat$EC_subtypes) < 1) {
    log_msg("[SKIP] %s no EC subtypes", tag)
    return(NULL)
  }

  niche_check <- check_min_niches_per_sample(dat, min_niches_per_sample)
  if (!niche_check$ok) {
    fail_txt <- niche_check$failing %>%
      mutate(msg = paste0(sample, ":", EC_subtypes, "=", n_niches)) %>%
      pull(msg) %>%
      paste(collapse = "; ")

    log_msg("[SKIP] %s failed min_niches_per_sample (< %d): %s",
            tag, min_niches_per_sample, fail_txt)
    return(NULL)
  }

  baseline_chr <- as.character(baseline_age)
  if (!(baseline_chr %in% levels(dat$age_months))) {
    log_msg("[SKIP] %s baseline age %s not present", tag, baseline_chr)
    return(NULL)
  }

    dat$age_months <- factor(as.character(dat$age_months), levels = levels(dat$age_months))
    dat$age_months <- stats::relevel(dat$age_months, ref = baseline_chr)

  mod <- NULL
  fit_type <- NA_character_

  try({
    mod <- glmmTMB(
      cell_count ~ age_months * EC_subtypes + (1 | sample) + offset(log_area_offset),
      data = dat,
      family = nbinom2()
    )
    fit_type <- "glmmTMB_nbinom2"
  }, silent = TRUE)

  if (is.null(mod)) {
    try({
      mod <- glm.nb(
        cell_count ~ age_months * EC_subtypes + offset(log_area_offset),
        data = dat
      )
      fit_type <- "glm_nb"
    }, silent = TRUE)
  }

  if (is.null(mod)) {
    log_msg("[SKIP] %s model failed", tag)
    return(NULL)
  }

  emm <- tryCatch({
    emmeans(
      mod,
      ~ age_months | EC_subtypes,
      at = list(log_area_offset = 0),
      type = "response"
    ) |> as.data.frame()
  }, error = function(e) NULL)

  if (is.null(emm) || nrow(emm) == 0) {
    log_msg("[SKIP] %s emmeans failed", tag)
    return(NULL)
  }

  emm <- standardize_emm_ci(emm)
  emm$response <- emm$response * 1e6
  emm$lower    <- emm$lower * 1e6
  emm$upper    <- emm$upper * 1e6

  con <- tryCatch({
    pairs(
      emmeans(
        mod,
        ~ age_months | EC_subtypes,
        at = list(log_area_offset = 0),
        type = "response"
      ),
      reverse = FALSE
    ) |> as.data.frame()
  }, error = function(e) data.frame())

  list(mod = mod, fit_type = fit_type, emm = emm, con = con)
}


#FIT ALL SUBSETS

radii <- levels(df_all$radius)
celltypes <- levels(df_all$cell_type)

fits <- list()
filtered_out <- list()

for (a in areas) {
  for (r in radii) {
    for (ct in celltypes) {
      dat <- df_all %>% filter(brain_area == a, radius == r, cell_type == ct)
      tag <- paste(a, r, ct, sep = " | ")
      res <- fit_one(dat, tag)

      if (!is.null(res)) {
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
          n_age_levels = dplyr::n_distinct(dat$age_months)
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
}

# results + FDR

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

if (nrow(con_all) > 0) {
  # keep only contrasts vs baseline within each EC subtype
  con_all <- con_all %>%
    mutate(
      contrast = as.character(contrast),
      baseline_chr = as.character(baseline_age)
    ) %>%
    tidyr::extract(
      contrast,
      into = c("g1", "g2"),
      regex = "^\\s*([^/\\-–—]+?)\\s*[/\\-–—]\\s*([^/\\-–—]+?)\\s*$",
      remove = FALSE
    ) %>%
    mutate(
      g1 = str_trim(g1),
      g2 = str_trim(g2)
    ) %>%
    filter(!is.na(g1), !is.na(g2)) %>%
    filter(g1 == baseline_chr | g2 == baseline_chr) %>%
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

  contrast_audit_raw <- con_all %>% count(contrast, sort = TRUE)
  write.csv(
    contrast_audit_raw,
    file.path(outdir, "tables", "Contrast_audit_raw_emmeans_labels.csv"),
    row.names = FALSE
  )

  emm_lookup <- emm_all %>%
    dplyr::select(brain_area, radius, cell_type, EC_subtypes, age_months, response) %>%
    dplyr::rename(age_group = age_months, density = response)

  con_all <- con_all %>%
    left_join(
      emm_lookup %>% dplyr::rename(g1 = age_group, density_g1 = density),
      by = c("brain_area", "radius", "cell_type", "EC_subtypes", "g1")
    ) %>%
    left_join(
      emm_lookup %>% dplyr::rename(g2 = age_group, density_g2 = density),
      by = c("brain_area", "radius", "cell_type", "EC_subtypes", "g2")
    ) %>%
    mutate(
      fold_change = pmax(density_g1, density_g2) / pmax(pmin(density_g1, density_g2), 1e-8),
      log2FC = log2(fold_change),
      age_vs_baseline = ifelse(g1 == baseline_chr, paste0(g2, " vs ", g1), paste0(g1, " vs ", g2))
    )
} else {
  con_all$p_adj <- numeric(0)
  con_all$g1 <- character(0)
  con_all$g2 <- character(0)
  con_all$density_g1 <- numeric(0)
  con_all$density_g2 <- numeric(0)
  con_all$fold_change <- numeric(0)
  con_all$log2FC <- numeric(0)
  con_all$age_vs_baseline <- character(0)
}

cat("Total contrasts tested:", nrow(con_all), "\n")
cat("Nominal p<=0.05:", sum(con_all$p.value <= 0.05, na.rm = TRUE), "\n")
cat("Min p:", min(con_all$p.value, na.rm = TRUE), "\n")

write.csv(
  con_all,
  file.path(outdir, "tables", "Pairwise_age_vs_baseline_FDR_within_area_radius_celltype_with_effect_sizes.csv"),
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
message("Significant comparisons after FC filter: ", nrow(con_sig))

write.csv(
  con_sig_fdr,
  file.path(outdir, "tables", paste0("Significant_age_vs_baseline_FDR_le_", bracket_alpha, "_before_FC_filter.csv")),
  row.names = FALSE
)

write.csv(
  con_sig,
  file.path(
    outdir,
    "tables",
    paste0(
      "Significant_age_vs_baseline_FDR_le_", bracket_alpha,
      "_FC_ge_", fc_cutoff,
      "_min_niches_per_sample_ge_", min_niches_per_sample,
      ".csv"
    )
  ),
  row.names = FALSE
)

#Plots of samples means for significant results only
# ----------------------------
if (nrow(con_sig) > 0) {
  sig_subsets <- con_sig %>%
    distinct(brain_area, radius, cell_type)

  plot_df <- df_all %>%
    semi_join(sig_subsets, by = c("brain_area", "radius", "cell_type")) %>%
    mutate(
      EC_subtypes = factor(as.character(EC_subtypes), levels = lvl),
      radius_num = as.numeric(as.character(radius))
    ) %>%
    group_by(cell_type, brain_area, radius, radius_num, EC_subtypes, sample) %>%
    summarise(
      sample_mean_count = mean(cell_count, na.rm = TRUE),
      .groups = "drop"
    )

  sample_levels <- sort(unique(as.character(plot_df$sample)))
  sample_cols <- setNames(grDevices::hcl.colors(length(sample_levels), "Dark 3"), sample_levels)

  sig_celltypes <- sort(unique(as.character(plot_df$cell_type)))

  for (ct in sig_celltypes) {
    dat_ct <- plot_df %>%
      filter(cell_type == ct) %>%
      mutate(
        brain_area = factor(as.character(brain_area), levels = areas),
        EC_subtypes = factor(as.character(EC_subtypes), levels = lvl),
        radius = factor(as.character(radius), levels = as.character(rvals), ordered = TRUE)
      )

    if (nrow(dat_ct) == 0) next

    p_ct <- ggplot(dat_ct, aes(x = radius, y = sample_mean_count)) +
      geom_boxplot(
        aes(group = radius),
        width = 0.7,
        outlier.shape = NA,
        fill = "white",
        colour = "black",
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

# placement of brackets indicating significance

x_age <- levels(df_all$age_months)
xmap <- setNames(seq_along(x_age), x_age)

facet_y <- emm_all %>%
  dplyr::group_by(brain_area, radius, cell_type, EC_subtypes) %>%
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
  dplyr::select(brain_area, radius, cell_type, EC_subtypes, ymin, ymax, yrng)

brackets <- con_sig %>%
  mutate(
    x1n = unname(xmap[g1]),
    x2n = unname(xmap[g2]),
    xminn = pmin(x1n, x2n),
    xmaxn = pmax(x1n, x2n),
    xmid  = (x1n + x2n) / 2,
    label = format_bracket_label(sig, p_adj, fold_change, include_effect = label_with_effect)
  ) %>%
  left_join(facet_y, by = c("brain_area", "radius", "cell_type", "EC_subtypes")) %>%
  filter(!is.na(xminn), !is.na(xmaxn), !is.na(ymax), !is.na(yrng)) %>%
  group_by(brain_area, radius, cell_type, EC_subtypes) %>%
  arrange(p_adj, xminn, xmaxn, .by_group = TRUE) %>%
  mutate(lane = assign_lanes(xminn, xmaxn)) %>%
  ungroup() %>%
  filter(!is.na(lane)) %>%
  mutate(
    min_lane_step = 30,
    min_base_pad  = 25,
    min_tick      = 6,
    min_text_gap  = 10,
    lane_step_eff = pmax(bracket_lane_step * yrng, min_lane_step),
    base_pad_eff  = pmax(bracket_base_pad * yrng, min_base_pad),
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
    emm_ar <- emm_all %>% filter(brain_area == a, radius == r)
    outfile <- file.path(outdir, "plots_pdf", paste0("ECniche_age_", a, "_radius_", r, "um.pdf"))

    if (nrow(emm_ar) == 0) {
      p0 <- ggplot() +
        theme_void(base_size = base_size) +
        ggtitle(paste0("EC niche age comparison | ", a, " | radius ", r, " µm")) +
        annotate("text", x = 0, y = 0, label = "No successful fits for any cell type", size = 5)
      ggsave(outfile, p0, width = pdf_width, height = pdf_height, device = cairo_pdf)
      next
    }

    emm_ar <- emm_ar %>%
      mutate(
        age_months = factor(as.character(age_months), levels = x_age, ordered = TRUE),
        xn = unname(xmap[as.character(age_months)]),
        EC_subtypes = factor(as.character(EC_subtypes), levels = lvl)
      ) %>%
      filter(is.finite(xn))

    raw_sample_means <- df_all %>%
      filter(brain_area == a, radius == r) %>%
      group_by(cell_type, EC_subtypes, age_months, sample) %>%
      summarise(mean_count = mean(cell_count, na.rm = TRUE), .groups = "drop") %>%
      mutate(
        age_months = factor(as.character(age_months), levels = x_age, ordered = TRUE),
        xn = unname(xmap[as.character(age_months)]),
        EC_subtypes = factor(as.character(EC_subtypes), levels = lvl)
      ) %>%
      filter(is.finite(xn))

    br_ar <- brackets %>% filter(brain_area == a, radius == r)

    p <- ggplot(emm_ar, aes(x = xn, y = response)) +
      geom_blank(
        data = br_ar,
        aes(x = xmid, y = y_text + pmax(0.08 * yrng, 0.04 * ymax)),
        inherit.aes = FALSE
      ) +
      geom_point(
        data = raw_sample_means,
        aes(x = xn, y = mean_count, colour = EC_subtypes),
        inherit.aes = FALSE,
        position = position_jitter(width = 0.10, height = 0),
        alpha = 0.90,
        size = 1.6
      ) +
      geom_pointrange(
        aes(ymin = lower, ymax = upper),
        linewidth = 0.35,
        colour = "black",
        na.rm = TRUE
      ) +
      facet_grid(EC_subtypes ~ cell_type, scales = "free_y") +
      scale_colour_manual(values = ec_cols, guide = "none") +
      scale_x_continuous(breaks = seq_along(x_age), labels = x_age) +
      labs(
        x = "Age (months)",
        y = "Cell density (cells / mm²)",
        title = paste0("EC niche age comparison | ", a, " | radius ", r, " µm")
      ) +
      pub_theme +
      coord_cartesian(clip = "off") +
      scale_y_continuous(expand = expansion(mult = c(0.02, 0.5)))

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

    ggsave(outfile, p, width = pdf_width, height = pdf_height, device = cairo_pdf)
  }
}