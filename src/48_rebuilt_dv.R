#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — the dependent-variable source comparison.
#
# One W per level (the proposed W: `weights.proposed.aggregated` and
# `weights.proposed.disaggregated`, with the configured tie-break sort), one
# specification (ln_land_value ~ ln_plot_area + ln_<measure>, SARAR by GMM, het
# per config), one exclusion order (the submitted one). The only thing that
# moves is where the dependent variable comes from:
#
#   frozen   -- `Preco` / `Puni` as shipped in the frozen layers, i.e. what the
#               submitted tables were computed on. No supplied script builds
#               them from the supplementary CSVs.
#   rebuilt  -- the column stage 1 builds from the raw ITBI + registry files
#               under the one formula of reports/stage1_dependent_variable.md
#               section 2 (`dependent_variable.rebuilt_path`).
#
# The question: if the revision uses the rebuilt variable so that a reader can
# regenerate every table from the supplied records, how much does that move the
# numbers, and does it move any conclusion?
#
# What it produces, per level:
#   1. The exclusion funnel on each source, side by side.
#   2. Table 3 in full (mean, median, relative range, CV and the Pareto ratio)
#      for the unit value and the five centralities on each analysis sample,
#      with deltas. The centrality rows move only because the sample moves --
#      the centralities themselves are frozen inputs and are never recomputed
#      here.
#   3. Table 4's five Pearson correlations of ln(land value) with ln(measure),
#      with deltas, and the Spearman rank correlations beside them.
#   4. Five SARAR fits per source: beta, SE, p, standardized beta by
#      re-estimation on z-scored covariates, lambda, rho, the error-bound flag,
#      pseudo-R2.
#   5. Spatial block CV (revision.spatial_cv) on each source.
#   6. One comparison table per level: per measure, frozen vs rebuilt on beta,
#      standardized beta, p, the sign/significance verdict and the CV RMSE,
#      plus an explicit "verdict change?" column.
#
# W is held fixed across the comparison, so every line is a difference between
# two runs of one specification. The rows on the source the sweep uses
# (`dependent_variable.proposed_source`) are checked against the matching rows
# of outputs/tables/stage4_w_choice_sweep.csv when that file exists
# (`make w-choice`).
#
# COMPUTE. compute.max_workers forked workers via parallel::mclapply; R here
# links the reference single-threaded BLAS, so one worker is one core.
#
# Run:  Rscript src/48_rebuilt_dv.R            (or `make rebuilt-dv`)
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet); library(parallel)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "rebuilt_dv")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
SOURCES  <- c("frozen", "rebuilt")
DV       <- ctx$cfg$dependent_variable
W_SPEC   <- list(disaggregated = ctx$cfg$weights$proposed$disaggregated,
                 aggregated    = ctx$cfg$weights$proposed$aggregated)
BOUND    <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
BEPS     <- 1e-6
LAM_MAX  <- 1.0
ALPHA    <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
NWORK    <- max(1L, min(as.integer(ctx$cfg$compute$max_workers),
                        as.integer(ctx$cfg$compute$host_ceiling)))
CTRL     <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
frm      <- function(m) as.formula(sprintf("ln_land_value ~ %sln_%s", CTRL, m))

log_msg(ctx, sprintf("W: disaggregated '%s' (k = %s), aggregated '%s' (k = %s); tie-break '%s'",
                     W_SPEC$disaggregated, ctx$cfg$weights[[W_SPEC$disaggregated]]$k,
                     W_SPEC$aggregated, ctx$cfg$weights[[W_SPEC$aggregated]]$k,
                     ctx$cfg$weights$identical_points$handling))
log_msg(ctx, sprintf("workers = %d; het = %s",
                     NWORK, isTRUE(ctx$cfg$models$het)))

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c") log(x + ctx$cfg$transforms$log_plus_c) else log(x)
add_logs <- function(D) {
  for (v in c("land_value", MEASURES, "plot_area")) D[[paste0("ln_", v)]] <- lg(D[[v]])
  D
}
canon <- function(L, level) {
  map <- ctx$cfg$measures$columns[[level]]
  for (cn in names(map)) {
    srcn <- map[[cn]]
    if (!is.null(L[[srcn]]) && srcn != cn) L[[cn]] <- L[[srcn]]
  }
  L
}
# Table 3 in full. The first three are the scored statistics, identical to
# descriptives() in src/30_validation_gate.R; the median and the Hierarchy
# column (the Pareto ratio of Sec. 4.4: the sum of the top 20 % of values over
# the sum of the other 80 %, ceil(0.2 n) top records, the definition of
# src/51_table3_hierarchy.py) complete the table on this script's samples.
TOP_SHARE <- as.numeric(ctx$cfg$validation$hierarchy_top_share)
pareto_ratio <- function(x) {
  x <- sort(x, decreasing = TRUE)
  k <- ceiling(TOP_SHARE * length(x))
  sum(x[seq_len(k)]) / sum(x[-seq_len(k)])
}
descriptives <- function(x) {
  c(mean = mean(x), relative_range = (max(x) - min(x)) / mean(x),
    cv = sd(x) / mean(x) * 100, median = stats::median(x),
    pareto_ratio = pareto_ratio(x))
}
target_of <- function(block, level, m) {
  b <- ctx$cfg$validation[[block]][[level]]
  if (is.null(b) || is.null(b[[m]])) NA_real_ else as.numeric(b[[m]])
}
# apply_exclusions() writes into ctx$counts; take the funnel straight off that
# rather than recounting it, so the report cannot drift from the log.
funnel_of <- function(expr) {
  mark <- nrow(ctx$counts)
  out  <- force(expr)
  list(out = out, steps = ctx$counts[seq.int(mark + 1L, nrow(ctx$counts)), ])
}

# =============================================================================
# 1. Load both levels, both sources
# =============================================================================

# --- disaggregated: one layer, the dependent variable swapped in -------------
# Stage 1 writes one row per frozen-layer record, in layer order, and carries
# the layer's own `id`, so the join is by identifier and checkable rather than
# positional.
load_disaggregated <- function() {
  L <- st_read(cfg_path(ctx, "frozen", "cents_disaggregated"), quiet = TRUE)
  if (is.na(st_crs(L)$epsg) || st_crs(L)$epsg != ctx$cfg$repro$crs_epsg)
    L <- st_transform(L, ctx$cfg$repro$crs_epsg)
  if (isTRUE(st_is_longlat(L))) stop("layer is in geographic coordinates", call. = FALSE)

  p  <- file.path(REPO, DV$rebuilt_path)
  if (!file.exists(p))
    stop("stage-1 output missing: ", p, " -- run `make dv` first", call. = FALSE)
  RB <- read.csv(p, stringsAsFactors = FALSE)
  if (nrow(RB) != nrow(L) || !all(RB$id == L$id))
    stop(sprintf("%s is not id-aligned with the frozen layer (%d vs %d rows)",
                 DV$rebuilt_path, nrow(RB), nrow(L)), call. = FALSE)
  d_frozen <- max(abs(RB$Preco_frozen - L$Preco), na.rm = TRUE)
  if (d_frozen > 1e-6)
    stop("the stage-1 file's `Preco_frozen` does not match the layer's `Preco`",
         call. = FALSE)
  # The rebuild does not touch the plot area (plot_area_source: geolayer),
  # so the comparison really is one variable moving and nothing else.
  d_area <- max(abs(RB$Terreno - L$Terreno), na.rm = TRUE)
  log_msg(ctx, sprintf("disaggregated: %d records, id-aligned; rebuilt vs frozen `Preco` max |delta| = %.4g, `Terreno` max |delta| = %.4g",
                       nrow(L), max(abs(RB$Preco - L$Preco), na.rm = TRUE), d_area))
  rel <- abs(RB$Preco - RB$Preco_frozen) / abs(RB$Preco_frozen)
  ex  <- sum(rel <= 1e-6, na.rm = TRUE)
  w1  <- sum(rel <= 0.01, na.rm = TRUE)
  log_msg(ctx, sprintf("disaggregated: the rebuild reproduces the frozen `Preco` exactly on %d of %d records (%.2f%%), within 1%% on %d (%.2f%%)",
                       ex, nrow(RB), 100 * ex / nrow(RB), w1, 100 * w1 / nrow(RB)))
  AGREE <<- list(n = nrow(RB), exact = ex, within1 = w1)

  mk <- function(src) {
    Z <- L
    if (src == "rebuilt") { Z$Preco <- RB$Preco; Z$Puni <- RB$Puni }
    Z <- canon(Z, "disaggregated")
    log_msg(ctx, sprintf("-- funnel [disaggregated / %s]", src))
    f <- funnel_of(apply_exclusions(ctx, Z, "disaggregated"))
    E <- f$out
    list(source = src, level = "disaggregated",
         D = add_logs(st_drop_geometry(E)),
         coords = st_coordinates(E)[, 1:2], steps = f$steps, id = E$id)
  }
  setNames(lapply(SOURCES, mk), SOURCES)
}

# --- aggregated: the frozen 500 m layer vs stage 2's rebuild of it -----------
# Same 821 cells, same geometry, same frozen centralities; only `Puni_mean`
# differs, because stage 2 re-averages the rebuilt address-level values into
# the cells under the same plain-cell-mean rule.
load_aggregated <- function() {
  srcs <- ctx$cfg$revision$maup$sources
  keys <- vapply(srcs, function(s) as.character(s$key), character(1))
  rec  <- srcs[[match("500m_headline", keys)]]
  if (is.null(rec)) stop("revision.maup.sources has no '500m_headline' record",
                         call. = FALSE)

  FZ <- st_read(cfg_path(ctx, "frozen", "cents_aggregated"), quiet = TRUE)
  RB <- st_read(file.path(REPO, rec$path), quiet = TRUE)
  for (nm in c("FZ", "RB")) {
    A <- get(nm)
    if (is.na(st_crs(A)$epsg) || st_crs(A)$epsg != ctx$cfg$repro$crs_epsg)
      assign(nm, st_transform(A, ctx$cfg$repro$crs_epsg))
  }
  if (nrow(FZ) != nrow(RB) || !all(FZ$id == RB$cell_id))
    stop("the rebuilt 500 m layer is not cell-aligned with the frozen one",
         call. = FALSE)
  # `id` is the tie-break key at this level; the rebuild names it cell_id.
  RB$id <- RB$cell_id
  cmap <- ctx$cfg$measures$columns$aggregated
  dmax <- max(vapply(MEASURES, function(m)
    max(abs(FZ[[cmap[[m]]]] - RB[[cmap[[m]]]]), na.rm = TRUE), numeric(1)))
  log_msg(ctx, sprintf("aggregated: %d cells, id-aligned; the five frozen centrality columns agree to %.3g",
                       nrow(FZ), dmax))
  log_msg(ctx, sprintf("aggregated: rebuilt vs frozen `Puni_mean` max |delta| = %.4g on %d cells with a value",
                       max(abs(RB$Puni_mean - RB$Puni_mean_frozen), na.rm = TRUE),
                       sum(!is.na(RB$Puni_mean))))

  mk <- function(src) {
    Z <- if (src == "frozen") FZ else RB
    Z <- canon(Z, "aggregated")
    log_msg(ctx, sprintf("-- funnel [aggregated / %s]", src))
    f <- funnel_of(apply_exclusions(ctx, Z, "aggregated"))
    E <- f$out
    list(source = src, level = "aggregated",
         D = add_logs(st_drop_geometry(E)),
         coords = suppressWarnings(
           st_coordinates(st_point_on_surface(st_geometry(E)))[, 1:2]),
         steps = f$steps, id = E$id)
  }
  setNames(lapply(SOURCES, mk), SOURCES)
}

AGREE <- NULL   # filled by load_disaggregated(): how often the rebuild matches
DATA <- list(disaggregated = load_disaggregated(), aggregated = load_aggregated())

# =============================================================================
# 2. Funnel, Table 3, Table 4
# =============================================================================

funnel_rows <- list(); desc_rows <- list(); cor_rows <- list()
for (lv in names(DATA)) for (src in SOURCES) {
  d <- DATA[[lv]][[src]]
  st <- d$steps
  for (i in seq_len(nrow(st)))
    funnel_rows[[length(funnel_rows) + 1]] <- data.frame(
      level = lv, source = src, step = st$step[i], n = st$n[i],
      delta = st$delta[i], note = st$note[i], stringsAsFactors = FALSE)
  tgt_n <- ctx$cfg$exclusions$targets[[paste0(lv, "_n")]]
  log_msg(ctx, sprintf("%s / %-7s : N = %d (submitted %d, delta %+d)",
                       lv, src, nrow(d$D), tgt_n, nrow(d$D) - tgt_n))

  # the unit value (the dependent variable, in levels) and the five measures
  for (m in c("unit_value", MEASURES)) {
    ds <- descriptives(d$D[[if (m == "unit_value") "land_value" else m]])
    for (stat in names(ds))
      desc_rows[[length(desc_rows) + 1]] <- data.frame(
        level = lv, source = src, measure = m, statistic = stat,
        value = as.numeric(ds[[stat]]),
        submitted = if (m == "unit_value") NA_real_ else
          target_of(if (stat == "mean") "mean_value" else stat, lv, m),
        stringsAsFactors = FALSE)
    if (m == "unit_value") next
    r <- cor(d$D$ln_land_value, d$D[[paste0("ln_", m)]],
             use = "complete.obs", method = "pearson")
    # Spearman is rank-based, so logs or levels give the same value.
    rs <- cor(d$D$land_value, d$D[[m]], use = "complete.obs", method = "spearman")
    cor_rows[[length(cor_rows) + 1]] <- data.frame(
      level = lv, source = src, measure = m, pearson_r = r, spearman = rs,
      submitted = target_of("correlations", lv, m), n = nrow(d$D),
      stringsAsFactors = FALSE)
  }
}
funnel <- do.call(rbind, funnel_rows)
# `delta` on a "loaded" row is the carry-over from the previous block's last
# count (ctx$counts is one running log), which means nothing here -- blank it.
funnel$delta[grepl(": loaded$", funnel$step)] <- NA_integer_
desc   <- do.call(rbind, desc_rows)
corr   <- do.call(rbind, cor_rows)

wide_delta <- function(df, valcol) {
  key <- setdiff(names(df), c("source", valcol, "n"))
  fz <- df[df$source == "frozen", ]; rb <- df[df$source == "rebuilt", ]
  rb <- rb[match(do.call(paste, fz[key]), do.call(paste, rb[key])), ]
  out <- fz[key]
  out$frozen  <- fz[[valcol]]
  out$rebuilt <- rb[[valcol]]
  out$delta   <- out$rebuilt - out$frozen
  out$rel_delta <- out$delta / abs(out$frozen)
  out
}
desc_cmp <- wide_delta(desc, "value")
corr_cmp <- wide_delta(corr[, c("level", "source", "measure", "pearson_r", "submitted")],
                       "pearson_r")
# Spearman beside Pearson, both sources (Table 4 reports the rebuilt one).
sp_of <- function(src) {
  x <- corr[corr$source == src, ]
  x$spearman[match(paste(corr_cmp$level, corr_cmp$measure), paste(x$level, x$measure))]
}
corr_cmp$spearman_frozen  <- sp_of("frozen")
corr_cmp$spearman_rebuilt <- sp_of("rebuilt")

for (lv in names(DATA)) {
  s <- desc_cmp[desc_cmp$level == lv & desc_cmp$measure != "unit_value", ]
  log_msg(ctx, sprintf("Table 3 centralities (%s): max |delta| = %.4g, max |relative delta| = %.2f%% (%s %s)",
                       lv, max(abs(s$delta)), 100 * max(abs(s$rel_delta)),
                       s$measure[which.max(abs(s$rel_delta))],
                       s$statistic[which.max(abs(s$rel_delta))]))
  s <- corr_cmp[corr_cmp$level == lv, ]
  log_msg(ctx, sprintf("Table 4 (%s): max |delta r| = %.4f (%s: %.4f -> %.4f)",
                       lv, max(abs(s$delta)), s$measure[which.max(abs(s$delta))],
                       s$frozen[which.max(abs(s$delta))],
                       s$rebuilt[which.max(abs(s$delta))]))
}

# =============================================================================
# 3. The five SARAR fits per (level, source)
# =============================================================================
#
# Same body as one_fit() in src/40_w_choice.R: standardized beta by
# re-estimation on z-scored covariates, and the error-bound flag.
one_fit <- function(dat, W, m, lv, src, spec) {
  form <- frm(m); bn <- paste0("ln_", m)
  fit  <- fit_spreg(ctx, form, dat, W, "sarar")
  if (is.null(fit)) return(NULL)
  z  <- sarar_row(fit)
  ab <- abs(abs(z$co[["rho"]]) - BOUND) < BEPS    # sphet bounds only rho
  dz <- dat
  dz$ln_plot_area <- as.numeric(scale(dat$ln_plot_area))
  dz[[bn]]        <- as.numeric(scale(dat[[bn]]))
  fz <- fit_spreg(ctx, form, dz, W, "sarar")
  bstd <- sstd <- NA_real_
  if (!is.null(fz)) { zz <- sarar_row(fz); bstd <- zz$co[[bn]]; sstd <- zz$se[[bn]] }
  data.frame(
    level = lv, source = src, w_spec = spec,
    k = ctx$cfg$weights[[spec]]$k %||% NA_integer_,
    n = nrow(dat), measure = m,
    beta_area = z$co[["ln_plot_area"]], se_area = z$se[["ln_plot_area"]],
    p_area = z$p[["ln_plot_area"]],
    beta_measure = z$co[[bn]], se_measure = z$se[[bn]], p_measure = z$p[[bn]],
    beta_std = bstd, se_std = sstd, beta_std_analytic = z$co[[bn]] * sd(dat[[bn]]),
    lambda_lag = z$co[["lambda"]], se_lambda = z$se[["lambda"]],
    rho_error = z$co[["rho"]], se_rho = z$se[["rho"]],
    error_at_bound = as.integer(ab),
    lambda_nonstationary = as.integer(z$co[["lambda"]] >= LAM_MAX),
    pseudo_r2 = pseudo_r2(fit, dat$ln_land_value), stringsAsFactors = FALSE)
}

Ws <- list()
for (lv in names(DATA)) for (src in SOURCES)
  Ws[[paste(lv, src)]] <- suppressWarnings(
    build_weights(ctx, DATA[[lv]][[src]]$coords, spec_name = W_SPEC[[lv]]))

jobs <- expand.grid(lv = names(DATA), src = SOURCES, m = MEASURES,
                    stringsAsFactors = FALSE)
t0  <- Sys.time()
res <- mclapply(seq_len(nrow(jobs)), function(i) {
  j <- jobs[i, ]
  one_fit(DATA[[j$lv]][[j$src]]$D, Ws[[paste(j$lv, j$src)]], j$m, j$lv, j$src,
          W_SPEC[[j$lv]])
}, mc.cores = min(NWORK, nrow(jobs)), mc.preschedule = FALSE)
bad <- vapply(res, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(bad)) log_msg(ctx, sprintf("!! %d of %d fits returned nothing", sum(bad), length(res)))
fits <- do.call(rbind, res[!bad])
log_msg(ctx, sprintf("%d of %d SARAR fits in %.0f s wall on %d workers", sum(!bad),
                     nrow(jobs), as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     min(NWORK, nrow(jobs))))
fits <- fits[order(match(fits$level, names(DATA)), match(fits$source, SOURCES),
                   match(fits$measure, MEASURES)), ]

# --- the reproduction check: the rows on the sweep's source must be its rows --
sw <- file.path(REPO, ctx$cfg$paths$outputs, "tables", "stage4_w_choice_sweep.csv")
repro_note <- NULL
if (file.exists(sw)) {
  S <- read.csv(sw, stringsAsFactors = FALSE)
  ref <- list(disaggregated = "knn_k8", aggregated = "knn_k6")
  msgs <- character(0)
  for (lv in names(ref)) {
    A <- fits[fits$level == lv & fits$source == DV$proposed_source, ]
    B <- S[S$level == lv & S$w_spec == ref[[lv]], ]
    if (nrow(B) != length(MEASURES) || !nrow(A)) next
    B <- B[match(A$measure, B$measure), ]
    d <- max(abs(c(A$beta_measure - B$beta_measure, A$beta_std - B$beta_std,
                   A$rho_error - B$rho_error, A$lambda_lag - B$lambda_lag,
                   A$pseudo_r2 - B$pseudo_r2)))
    msgs <- c(msgs, sprintf("%s vs %s: max |delta| = %.3g%s", lv, ref[[lv]], d,
                            if (d == 0) " (BIT-IDENTICAL)" else ""))
    log_msg(ctx, sprintf("reproduction check, %s source %s", DV$proposed_source,
                         msgs[length(msgs)]))
  }
  repro_note <- paste(msgs, collapse = "; ")
}

# =============================================================================
# 4. Spatial block CV (revision.spatial_cv)
# =============================================================================
#
# Seeded k-means blocks on the coordinates, W rebuilt inside every training set
# (nothing leaks through W), trend-only prediction with the intercept
# recalibrated on the training residual mean, RMSE in log units, OLS and two
# benchmarks: the training-block mean, and OLS on log plot area alone. The random k-fold comparison is a property of the CV scheme, not
# of the dependent variable's source, and is not repeated here.
cvc   <- ctx$cfg$revision$spatial_cv
NB    <- as.integer(cvc$n_blocks)
if (!identical(cvc$prediction, "trend_only"))
  stop("only revision.spatial_cv.prediction = 'trend_only' is implemented", call. = FALSE)
rmse  <- function(e) sqrt(mean(e^2, na.rm = TRUE))

cv_rows <- list()
for (lv in names(DATA)) for (src in SOURCES) {
  d <- DATA[[lv]][[src]]; D <- d$D; coords <- d$coords; spec <- W_SPEC[[lv]]
  set.seed(as.integer(ctx$cfg$repro$seed))       # blocks must be reproducible
  blk <- kmeans(coords, centers = NB, nstart = 25, iter.max = 100)$cluster
  log_msg(ctx, sprintf("%s / %s block CV: %d k-means blocks, sizes %s", lv, src, NB,
                       paste(as.integer(table(blk)), collapse = " ")))

  t0 <- Sys.time()
  fold_res <- mclapply(seq_len(NB), function(b) {
    te <- which(blk == b); tr <- setdiff(seq_len(nrow(D)), te)
    keep <- tr[connected_idx(ctx, coords[tr, , drop = FALSE], spec)]
    dtr  <- D[keep, , drop = FALSE]
    Wtr  <- suppressWarnings(build_weights(ctx, coords[keep, , drop = FALSE],
                                           spec_name = spec))
    out <- list(y = D$ln_land_value[te])
    for (m in MEASURES) {
      form <- frm(m)
      Xte  <- model.matrix(form, data = D[te, , drop = FALSE])
      Xtr  <- model.matrix(form, data = dtr)
      f    <- fit_spreg(ctx, form, dtr, Wtr, "sarar")
      o    <- lm(form, data = dtr)
      out[[m]] <- if (is.null(f)) {
        list(recal = rep(NA_real_, length(te)),
             ols = as.numeric(predict(o, newdata = D[te, , drop = FALSE])),
             lambda = NA_real_)
      } else {
        bb <- sarar_row(f)$co[colnames(Xte)]
        p  <- as.numeric(Xte %*% bb)
        list(recal = p + (mean(dtr$ln_land_value) - mean(as.numeric(Xtr %*% bb))),
             ols = as.numeric(predict(o, newdata = D[te, , drop = FALSE])),
             lambda = sarar_row(f)$co[["lambda"]])
      }
    }
    out
  }, mc.cores = min(NWORK, NB), mc.preschedule = FALSE)

  ok <- !vapply(fold_res, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
  if (any(!ok)) log_msg(ctx, sprintf("!! %d of %d %s/%s CV folds failed",
                                     sum(!ok), NB, lv, src))
  log_msg(ctx, sprintf("%s / %s CV: %d folds in %.0f s wall", lv, src, sum(ok),
                       as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  y <- unlist(lapply(fold_res[ok], `[[`, "y"))
  for (m in MEASURES) {
    pr <- unlist(lapply(fold_res[ok], function(r) r[[m]]$recal))
    po <- unlist(lapply(fold_res[ok], function(r) r[[m]]$ols))
    lm_ <- vapply(fold_res[ok], function(r) r[[m]]$lambda, numeric(1))
    cv_rows[[length(cv_rows) + 1]] <- data.frame(
      level = lv, source = src, w_spec = spec, measure = m,
      n_folds = sum(ok), n_pred = length(y), prediction = "trend_only",
      rmse_sarar_trend_recal = rmse(y - pr), rmse_ols = rmse(y - po),
      mean_lambda_train = mean(lm_, na.rm = TRUE), stringsAsFactors = FALSE)
  }
  nullr <- local({
    p <- numeric(nrow(D))
    for (b in unique(blk)) { te <- which(blk == b); p[te] <- mean(D$ln_land_value[-te]) }
    rmse(D$ln_land_value - p)
  })
  cv_rows[[length(cv_rows) + 1]] <- data.frame(
    level = lv, source = src, w_spec = spec, measure = "null_training_block_mean",
    n_folds = NB, n_pred = nrow(D), prediction = "none",
    rmse_sarar_trend_recal = nullr, rmse_ols = nullr,
    mean_lambda_train = NA_real_, stringsAsFactors = FALSE)
  # Plot area alone, same blocks: OLS of the log outcome on log plot area,
  # fitted on the training blocks (cv_benchmarks() in common.R).
  parea <- cv_benchmarks(D, blk)[["null_plot_area_only"]]
  cv_rows[[length(cv_rows) + 1]] <- data.frame(
    level = lv, source = src, w_spec = spec, measure = "null_plot_area_only",
    n_folds = NB, n_pred = nrow(D), prediction = "ols_plot_area",
    rmse_sarar_trend_recal = parea, rmse_ols = parea,
    mean_lambda_train = NA_real_, stringsAsFactors = FALSE)
  r <- do.call(rbind, cv_rows)
  r <- r[r$level == lv & r$source == src &
         !r$measure %in% c("null_training_block_mean", "null_plot_area_only"), ]
  log_msg(ctx, sprintf("CV %s / %-7s recal RMSE: %s   null %.4f   plot area only %.4f",
    lv, src,
    paste(sprintf("%s %.4f", r$measure, r$rmse_sarar_trend_recal), collapse = "  "),
    nullr, parea))
}
cvres <- do.call(rbind, cv_rows)

# =============================================================================
# 5. The comparison table: one row per (level, measure)
# =============================================================================

verdict <- function(b, p) sprintf("%s%s", if (b > 0) "+" else "-",
                                  if (p < ALPHA) " sig" else " n.s.")
cmp_rows <- list()
for (lv in names(DATA)) for (m in MEASURES) {
  a <- fits[fits$level == lv & fits$source == "frozen"  & fits$measure == m, ]
  b <- fits[fits$level == lv & fits$source == "rebuilt" & fits$measure == m, ]
  ca <- cvres[cvres$level == lv & cvres$source == "frozen"  & cvres$measure == m, ]
  cb <- cvres[cvres$level == lv & cvres$source == "rebuilt" & cvres$measure == m, ]
  if (!nrow(a) || !nrow(b)) next
  cmp_rows[[length(cmp_rows) + 1]] <- data.frame(
    level = lv, measure = m, w_spec = a$w_spec, k = a$k,
    n_frozen = a$n, n_rebuilt = b$n,
    beta_frozen = a$beta_measure, beta_rebuilt = b$beta_measure,
    d_beta = b$beta_measure - a$beta_measure,
    beta_std_frozen = a$beta_std, beta_std_rebuilt = b$beta_std,
    d_beta_std = b$beta_std - a$beta_std,
    rel_d_beta_std = (b$beta_std - a$beta_std) / abs(a$beta_std),
    p_frozen = a$p_measure, p_rebuilt = b$p_measure,
    verdict_frozen = verdict(a$beta_measure, a$p_measure),
    verdict_rebuilt = verdict(b$beta_measure, b$p_measure),
    sign_change = as.integer(sign(a$beta_measure) != sign(b$beta_measure)),
    sig_change = as.integer((a$p_measure < ALPHA) != (b$p_measure < ALPHA)),
    lambda_frozen = a$lambda_lag, lambda_rebuilt = b$lambda_lag,
    rho_frozen = a$rho_error, rho_rebuilt = b$rho_error,
    n_at_bound_frozen = a$error_at_bound,
    n_at_bound_rebuilt = b$error_at_bound,
    pseudo_r2_frozen = a$pseudo_r2, pseudo_r2_rebuilt = b$pseudo_r2,
    cv_rmse_frozen = if (nrow(ca)) ca$rmse_sarar_trend_recal else NA_real_,
    cv_rmse_rebuilt = if (nrow(cb)) cb$rmse_sarar_trend_recal else NA_real_,
    stringsAsFactors = FALSE)
}
cmp <- do.call(rbind, cmp_rows)
cmp$d_cv_rmse <- cmp$cv_rmse_rebuilt - cmp$cv_rmse_frozen
cmp$verdict_change <- ifelse(cmp$sign_change + cmp$sig_change > 0, "yes", "no")
# A sign flip on a coefficient indistinguishable from zero under both sources is
# not a change of verdict in any readable sense -- the point-level BC and FK
# coefficients are the usual case, and they flip sign under small
# perturbations. This column counts only flips on a measure that is significant
# somewhere, and it is the one the report leads with. Same distinction as
# `n_sign_changes_on_significant` in src/47_sample_sensitivity.R.
cmp$sig_either <- as.integer(cmp$p_frozen < ALPHA | cmp$p_rebuilt < ALPHA)
cmp$verdict_change_on_significant <- ifelse(
  cmp$sign_change * cmp$sig_either + cmp$sig_change > 0, "yes", "no")
# The rank of each measure on each criterion, per level and per source: a
# conclusion in this paper is a ranking, so the ranking is what has to be
# compared, not only the coefficients.
for (nm in c("rank_std_frozen", "rank_std_rebuilt", "rank_cv_frozen", "rank_cv_rebuilt"))
  cmp[[nm]] <- NA_integer_
for (lv in unique(cmp$level)) {
  i <- cmp$level == lv
  cmp$rank_std_frozen[i]  <- rank(-cmp$beta_std_frozen[i], ties.method = "min")
  cmp$rank_std_rebuilt[i] <- rank(-cmp$beta_std_rebuilt[i], ties.method = "min")
  cmp$rank_cv_frozen[i]   <- rank(cmp$cv_rmse_frozen[i], ties.method = "min")
  cmp$rank_cv_rebuilt[i]  <- rank(cmp$cv_rmse_rebuilt[i], ties.method = "min")
}
cmp$rank_change <- ifelse(cmp$rank_std_frozen != cmp$rank_std_rebuilt |
                          cmp$rank_cv_frozen != cmp$rank_cv_rebuilt, "yes", "no")

for (i in seq_len(nrow(cmp)))
  log_msg(ctx, sprintf("  %-13s %-4s std beta %+.4f -> %+.4f (%+.2f%%)  p %.4f -> %.4f  %-7s -> %-7s  CV %.4f -> %.4f  verdict change: %s",
    cmp$level[i], cmp$measure[i], cmp$beta_std_frozen[i], cmp$beta_std_rebuilt[i],
    100 * cmp$rel_d_beta_std[i], cmp$p_frozen[i], cmp$p_rebuilt[i],
    cmp$verdict_frozen[i], cmp$verdict_rebuilt[i],
    cmp$cv_rmse_frozen[i], cmp$cv_rmse_rebuilt[i],
    sprintf("%s (on a significant measure: %s)", cmp$verdict_change[i],
            cmp$verdict_change_on_significant[i])))

# =============================================================================
# 6. Write
# =============================================================================

w <- function(df, fn) {
  if (is.null(df)) return(invisible(NULL))
  p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p); p
}
w(funnel,   "stage4_rebuilt_dv_funnel.csv")
w(desc_cmp, "stage4_rebuilt_dv_table3.csv")
w(corr_cmp, "stage4_rebuilt_dv_table4.csv")
w(fits,     "stage4_rebuilt_dv_fits.csv")
w(cvres,    "stage4_rebuilt_dv_cv_rmse.csv")
w(cmp,      "stage4_rebuilt_dv_comparison.csv")

# ---- the report -------------------------------------------------------------
# Every sentence that states a result is built from this run's tables.
md_table <- function(df, digits = 4) {
  cells <- lapply(df, function(col) if (is.numeric(col))
    ifelse(is.na(col), "", formatC(col, format = "g", digits = digits)) else
    as.character(col))
  cells <- as.data.frame(cells, stringsAsFactors = FALSE)
  c(paste0("| ", paste(names(df), collapse = " | "), " |"),
    paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|"),
    apply(cells, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |")))
}
maxabs <- function(x) max(abs(x), na.rm = TRUE)
fmtn <- function(x) format(x, big.mark = ",")
# The number of addresses where the rebuild and the frozen column disagree,
# read off stage 1's own table.
rp <- file.path(REPO, ctx$cfg$paths$outputs, "tables", "stage1_residuals.csv")
N_RESID <- if (file.exists(rp)) nrow(read.csv(rp, stringsAsFactors = FALSE)) else NA_integer_
md <- character(0)
say <- function(...) md <<- c(md, ...)

d_dis <- cmp[cmp$level == "disaggregated", ]
d_agg <- cmp[cmp$level == "aggregated", ]
n_dis <- c(d_dis$n_frozen[1], d_dis$n_rebuilt[1])
n_agg <- c(d_agg$n_frozen[1], d_agg$n_rebuilt[1])
step_n <- function(lv, src, step) {
  x <- funnel[funnel$level == lv & funnel$source == src & funnel$step == step, ]
  if (nrow(x)) x$n[1] else NA_integer_
}
loaded_dis <- step_n("disaggregated", "frozen", "disaggregated: loaded")
pos_dis <- c(step_n("disaggregated", "frozen", "after positivity"),
             step_n("disaggregated", "rebuilt", "after positivity"))
n_verdict_sig <- sum(cmp$verdict_change_on_significant == "yes")
n_verdict_any <- sum(cmp$verdict_change == "yes")
rk <- cmp[cmp$rank_change == "yes", ]
rk_std <- cmp[cmp$rank_std_frozen != cmp$rank_std_rebuilt, ]
rk_cv  <- cmp[cmp$rank_cv_frozen != cmp$rank_cv_rebuilt, ]
flip  <- cmp[cmp$verdict_change == "yes" & cmp$verdict_change_on_significant == "no", ]
# The centrality rows only: the unit-value rows move with the variable itself.
desc_cen <- desc_cmp[desc_cmp$measure != "unit_value", ]
agg_desc_max <- maxabs(desc_cen$delta[desc_cen$level == "aggregated"])

say("# Stage 4 — the revision on the rebuilt dependent variable", "",
  sprintf("Generated by `src/48_rebuilt_dv.R` (`make rebuilt-dv`). Seed %s. W held fixed at the proposed matrices (`%s`, k = %s, aggregated; `%s`, k = %s, disaggregated) with the configured tie-break `%s`.",
    ctx$cfg$repro$seed,
    W_SPEC$aggregated, ctx$cfg$weights[[W_SPEC$aggregated]]$k,
    W_SPEC$disaggregated, ctx$cfg$weights[[W_SPEC$disaggregated]]$k,
    ctx$cfg$weights$identical_points$handling),
  "", "---", "",
  "## 0. The question, and the short answer", "",
  "The submitted tables were computed on the frozen point layer's `Preco` (total value on the plot) and `Puni` (`Preco` divided by the plot area), which already carry the unit multiplier and which no supplied script builds from the supplementary CSVs.",
  "",
  sprintf("Stage 1 rebuilds that variable from the raw records under one formula (`reports/stage1_dependent_variable.md` section 2). It agrees with the frozen column exactly on %.1f %% of the %s records and to within 1 %% on %.1f %%. This report measures what replacing the frozen column by the rebuilt one changes.",
    100 * AGREE$exact / AGREE$n, fmtn(AGREE$n), 100 * AGREE$within1 / AGREE$n),
  "",
  sprintf("- The sample moves by **%+d records** at the address level (%s → %s) and by **%+d** at the cell level (%s → %s).",
    n_dis[2] - n_dis[1], fmtn(n_dis[1]), fmtn(n_dis[2]),
    n_agg[2] - n_agg[1], fmtn(n_agg[1]), fmtn(n_agg[2])),
  sprintf("- Table 3's centrality descriptives move by at most **%.2f %%** and Table 4's correlations by at most **%.4f** in absolute Pearson r.",
    100 * max(abs(desc_cen$rel_delta), na.rm = TRUE), maxabs(corr_cmp$delta)),
  sprintf("- The largest move in any standardized coefficient is **%.4f** (%s %s, %+.1f %% of its own size).",
    maxabs(cmp$d_beta_std),
    cmp$level[which.max(abs(cmp$d_beta_std))], cmp$measure[which.max(abs(cmp$d_beta_std))],
    100 * cmp$rel_d_beta_std[which.max(abs(cmp$d_beta_std))]),
  sprintf("- **%d of the %d models** change their sign-and-significance verdict on a measure that is significant under either source; %d change the verdict at all.",
    n_verdict_sig, nrow(cmp), n_verdict_any),
  sprintf("- On the standardized coefficient, %d measure(s) change rank%s. On block-CV RMSE, %d measure(s) change rank%s.",
    nrow(rk_std),
    if (nrow(rk_std)) sprintf(" (%s)", paste(sprintf("%s %s", rk_std$level, rk_std$measure), collapse = ", ")) else "",
    nrow(rk_cv),
    if (nrow(rk_cv)) sprintf(" (%s; the largest frozen-source RMSE gap among them is %.4f in log units)",
      paste(sprintf("%s %s", rk_cv$level, rk_cv$measure), collapse = ", "),
      max(vapply(unique(rk_cv$level), function(lv)
        diff(range(rk_cv$cv_rmse_frozen[rk_cv$level == lv])), numeric(1)))) else ""),
  if (nrow(flip))
    sprintf("- Sign flips on coefficients insignificant under both sources: %s.",
      paste(sprintf("%s %s (%+.4f → %+.4f standardized; p = %.2f frozen, %.2f rebuilt)",
                    flip$level, flip$measure, flip$beta_std_frozen, flip$beta_std_rebuilt,
                    flip$p_frozen, flip$p_rebuilt), collapse = "; "))
  else character(0),
  if (length(repro_note) && nzchar(repro_note))
    c("", sprintf("Reproduction check of the `%s` rows against the W sweep: %s.",
                  DV$proposed_source, repro_note))
  else character(0),
  "", "---", "",
  "## 1. What changed in the data", "",
  "Only the dependent variable. The five centralities are frozen inputs and are attached, never recomputed; the plot area comes from the same layer in both runs (`plot_area_source: geolayer`); W, the specification, the estimator, the exclusion order and the tie-break are all held fixed. At the cell level the rebuilt layer is stage 2's re-aggregation of the rebuilt address values into the same hexagons under the same plain-cell-mean rule.",
  "", "### 1.1 The funnel", "",
  md_table(funnel[, c("level", "source", "step", "n", "delta")], 6), "",
  sprintf("At the address level the positivity step keeps %s records under the frozen source and %s under the rebuilt one (of %s loaded); the 1.5×IQR fence, which is computed on the land value itself, then leaves %s and %s. At the cell level the analysis count is %s under both sources%s.",
    fmtn(pos_dis[1]), fmtn(pos_dis[2]), fmtn(loaded_dis), fmtn(n_dis[1]), fmtn(n_dis[2]),
    if (n_agg[1] == n_agg[2]) fmtn(n_agg[1]) else sprintf("%s / %s", fmtn(n_agg[1]), fmtn(n_agg[2])),
    if (n_agg[1] == n_agg[2]) "" else " respectively"),
  "", "### 1.2 Table 3 — the descriptives", "",
  sprintf("Mean, median, relative range ((max − min) / mean), CV (100 · sd / mean) and the Pareto ratio (the sum of the top %.0f %% of values over the sum of the rest, ceil(%.1f n) top records), each on its source's analysis sample. The `unit_value` rows are the dependent variable in levels. The centrality rows move only because the analysis sample moves; at the cell level their largest absolute change is %s.",
    100 * TOP_SHARE, TOP_SHARE, formatC(agg_desc_max, format = "g", digits = 3)),
  "",
  md_table(desc_cmp[, c("level", "measure", "statistic", "submitted", "frozen",
                        "rebuilt", "delta", "rel_delta")], 6),
  "", "### 1.3 Table 4 — the correlations", "",
  md_table(corr_cmp[, c("level", "measure", "submitted", "frozen", "rebuilt", "delta",
                        "spearman_frozen", "spearman_rebuilt")], 6),
  "",
  "Pearson r is on logs; Spearman's rank correlation is the same on logs or levels.",
  "",
  sprintf("Largest movement: **%.4f** in Pearson r (%s %s, %.4f → %.4f). The frozen column reproduces the submitted correlations to within %.4f.",
    maxabs(corr_cmp$delta),
    corr_cmp$level[which.max(abs(corr_cmp$delta))],
    corr_cmp$measure[which.max(abs(corr_cmp$delta))],
    corr_cmp$frozen[which.max(abs(corr_cmp$delta))],
    corr_cmp$rebuilt[which.max(abs(corr_cmp$delta))],
    maxabs(corr_cmp$frozen - corr_cmp$submitted)),
  "", "---", "", "## 2. The models, frozen against rebuilt", "")

for (lv in c("disaggregated", "aggregated")) {
  s <- cmp[cmp$level == lv, ]
  sig_d <- ifelse(s$sig_either == 1L, s$d_beta_std, NA_real_)
  say(sprintf("### 2.%d %s level (%s, k = %s), N = %s → %s",
              if (lv == "disaggregated") 1 else 2, lv, s$w_spec[1], s$k[1],
              fmtn(s$n_frozen[1]), fmtn(s$n_rebuilt[1])),
      "",
      md_table(s[, c("measure", "beta_frozen", "beta_rebuilt", "d_beta",
                     "beta_std_frozen", "beta_std_rebuilt", "d_beta_std",
                     "p_frozen", "p_rebuilt", "verdict_frozen", "verdict_rebuilt",
                     "cv_rmse_frozen", "cv_rmse_rebuilt", "d_cv_rmse",
                     "verdict_change", "verdict_change_on_significant")], 5),
      "",
      sprintf("Largest move in a standardized coefficient: **%.4f** (%s), %.1f %% of its own size%s. Ranking on the standardized coefficient: frozen **%s**, rebuilt **%s**. Ranking on block-CV RMSE (lower is better): frozen **%s**, rebuilt **%s**. Spatial parameters: λ %.3f–%.3f frozen, %.3f–%.3f rebuilt; ρ %.3f–%.3f frozen, %.3f–%.3f rebuilt; fits with ρ on the estimator's bound: %d frozen, %d rebuilt.",
        maxabs(s$d_beta_std), s$measure[which.max(abs(s$d_beta_std))],
        100 * abs(s$rel_d_beta_std[which.max(abs(s$d_beta_std))]),
        if (any(!is.na(sig_d))) sprintf("; among the measures significant under either source, **%.4f** (%s), %.1f %%",
          maxabs(sig_d), s$measure[which.max(abs(sig_d))],
          100 * abs(s$rel_d_beta_std[which.max(abs(sig_d))])) else "",
        paste(s$measure[order(s$rank_std_frozen)], collapse = " > "),
        paste(s$measure[order(s$rank_std_rebuilt)], collapse = " > "),
        paste(s$measure[order(s$rank_cv_frozen)], collapse = " < "),
        paste(s$measure[order(s$rank_cv_rebuilt)], collapse = " < "),
        min(s$lambda_frozen), max(s$lambda_frozen),
        min(s$lambda_rebuilt), max(s$lambda_rebuilt),
        min(s$rho_frozen), max(s$rho_frozen),
        min(s$rho_rebuilt), max(s$rho_rebuilt),
        sum(s$n_at_bound_frozen), sum(s$n_at_bound_rebuilt)),
      "")
}

say("---", "", "## 3. Block-CV RMSE", "",
  sprintf("%d k-means blocks on the coordinates; W is rebuilt inside every training set so no held-out unit enters any neighbour list; prediction is trend-only with the intercept recalibrated on the training residual mean; RMSE is in log units.", NB),
  "",
  md_table(cvres[, c("level", "source", "measure", "n_folds", "n_pred",
                     "rmse_sarar_trend_recal", "rmse_ols", "mean_lambda_train")], 5),
  "", "---", "", "## 4. Summary", "",
  sprintf("- **Verdicts.** %d of %d models change sign or significance on a measure that is significant under either source.",
    n_verdict_sig, nrow(cmp)),
  sprintf("- **Rankings.** %d of %d (level × measure) rows change rank on the standardized coefficient and %d on block-CV RMSE.",
    nrow(rk_std), nrow(cmp), nrow(rk_cv)),
  sprintf("- **Sample.** The address-level sample changes by %+d records (%s against the submitted %s), because the 1.5×IQR fence is computed on the dependent variable itself.",
    n_dis[2] - n_dis[1], fmtn(n_dis[2]), fmtn(n_dis[1])),
  if (!is.na(N_RESID))
    sprintf("- **Residual.** The rebuild and the frozen column disagree on %s addresses (`outputs/tables/stage1_residuals.csv`); with the rebuilt variable the tables no longer depend on the frozen column.",
      fmtn(N_RESID)) else character(0),
  "", "---", "", "## 5. Outputs", "",
  "| file | what |", "|---|---|",
  "| `outputs/tables/stage4_rebuilt_dv_funnel.csv` | every filter, both sources, both levels |",
  "| `outputs/tables/stage4_rebuilt_dv_table3.csv` | Table 3 in full (unit value and the five measures: mean, median, relative range, CV, Pareto ratio), both sources, with deltas |",
  "| `outputs/tables/stage4_rebuilt_dv_table4.csv` | Table 4 correlations with deltas; Spearman beside Pearson (`spearman_frozen`, `spearman_rebuilt`) |",
  sprintf("| `outputs/tables/stage4_rebuilt_dv_fits.csv` | the %d SARAR fits, full detail |", nrow(fits)),
  "| `outputs/tables/stage4_rebuilt_dv_cv_rmse.csv` | block-CV RMSE per measure and source |",
  "| `outputs/tables/stage4_rebuilt_dv_comparison.csv` | the per-measure comparison, the table §2 prints |",
  "")

p <- out_report(ctx, "stage4_rebuilt_dv.md")
writeLines(md, p)
log_msg(ctx, "wrote ", p)

write_counts(ctx, "stage4_rebuilt_dv_counts.csv")
finish(ctx)
