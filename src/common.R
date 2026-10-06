# Shared plumbing for the reproduction pipeline (R side).
#
# Mirrors src/common.py: same config.yaml, same seed, same count-logging
# contract. Estimation lives in R; nothing here decides anything analytical --
# every convention is read from config.yaml.

suppressPackageStartupMessages({
  library(yaml)
})

# Defined here rather than taken from base R, which only ships `%||%` from
# R 4.4.0 on; the package supports R >= 4.3.
`%||%` <- function(a, b) if (is.null(a)) b else a

# Resolve the repository root whether a script is run via Rscript (the
# `--file=` argument names it) or common.R is sourced from the repository root.
.repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  fa <- grep("^--file=", args, value = TRUE)
  here <- if (length(fa)) dirname(sub("^--file=", "", fa[1])) else "src"
  normalizePath(file.path(here, ".."), mustWork = TRUE)
}

REPO <- .repo_root()

load_config <- function(path = file.path(REPO, "config.yaml")) {
  yaml::read_yaml(path)
}

# ---- run context ------------------------------------------------------------
#
# `stage` is a label for the log file name and the log header only; it does
# not control whether the script runs.
init <- function(stage, name) {
  cfg <- load_config()

  logdir <- file.path(REPO, cfg$paths$logs)
  dir.create(logdir, showWarnings = FALSE, recursive = TRUE)
  logfile <- file.path(logdir, sprintf("stage%02d_%s.log", stage, name))
  cat("", file = logfile)          # truncate

  set.seed(as.integer(cfg$repro$seed))

  ctx <- new.env(parent = emptyenv())
  ctx$cfg <- cfg
  ctx$stage <- stage
  ctx$name <- name
  ctx$logfile <- logfile
  ctx$started <- Sys.time()
  ctx$counts <- data.frame(step = character(), n = integer(),
                           delta = integer(), note = character(),
                           stringsAsFactors = FALSE)

  log_msg(ctx, sprintf("stage %d - %s", stage, name))
  log_msg(ctx, sprintf("seed=%d  crs=EPSG:%s", cfg$repro$seed, cfg$repro$crs_epsg))
  ctx
}

log_msg <- function(ctx, ...) {
  line <- sprintf("%s INFO    %s", format(Sys.time(), "%H:%M:%S"), paste0(...))
  cat(line, "\n", sep = "")
  cat(line, "\n", sep = "", file = ctx$logfile, append = TRUE)
  invisible(NULL)
}

# ---- paths ------------------------------------------------------------------

cfg_path <- function(ctx, ...) {
  node <- ctx$cfg$paths
  for (k in c(...)) node <- node[[k]]
  file.path(REPO, node)
}

out_table <- function(ctx, filename) {
  d <- file.path(REPO, ctx$cfg$paths$outputs, "tables")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  file.path(d, filename)
}

out_report <- function(ctx, filename) {
  d <- file.path(REPO, ctx$cfg$paths$reports)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  file.path(d, filename)
}

# ---- record counts in / out at every filter ---------------------------------

count_step <- function(ctx, step, n, note = "") {
  prev <- if (nrow(ctx$counts)) ctx$counts$n[nrow(ctx$counts)] else NA_integer_
  delta <- if (is.na(prev)) NA_integer_ else as.integer(n - prev)
  ctx$counts <- rbind(ctx$counts, data.frame(
    step = step, n = as.integer(n), delta = delta, note = note,
    stringsAsFactors = FALSE))
  arrow <- if (is.na(delta)) "" else sprintf("  (%+d)", delta)
  log_msg(ctx, sprintf("count  %-34s %8d%s %s", step, n, arrow, note))
  invisible(n)
}

write_counts <- function(ctx, filename) {
  p <- out_table(ctx, filename)
  write.csv(ctx$counts, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p)
  p
}

# ---- the dependent variable of the revised tables ----------------------------
#
# `dependent_variable.proposed_source` names where the revised tables take the
# dependent variable from. `frozen` returns the frozen layer as read. `rebuilt`
# returns, at the address level, the frozen layer with `Preco` / `Puni`
# replaced by stage 1's rebuild (joined by `id`, so the result does not depend
# on the order either file is stored in) and, at the cell level, stage 2's
# rebuilt 500 m layer, the one src/48_rebuilt_dv.R reads (same 821 cells, same
# frozen centralities, `Puni_mean` re-averaged from the rebuilt addresses; its
# `cell_id` is copied to `id`, the tie-break key at this level; its plot-area
# mean is re-averaged from the same frozen plot areas). The centralities are
# the frozen ones at both levels.
#
# `variant` selects one of stage 1's alternative parking conventions
# (`dependent_variable.variants`) at the address level; it needs the rebuilt
# source, because the alternatives exist only as rebuilds.
proposed_dv_layer <- function(ctx, level, variant = "headline",
                              source = ctx$cfg$dependent_variable$proposed_source) {
  DV <- ctx$cfg$dependent_variable
  if (!source %in% c("frozen", "rebuilt"))
    stop("dependent_variable.proposed_source must be frozen | rebuilt, not '",
         format(source), "'", call. = FALSE)
  if (!identical(variant, "headline") && !identical(source, "rebuilt"))
    stop("the parking variant '", variant, "' exists only as a rebuild", call. = FALSE)
  if (!variant %in% names(DV$variants))
    stop("unknown dependent-variable variant '", variant, "'", call. = FALSE)
  to_crs <- function(L) {
    if (is.na(sf::st_crs(L)$epsg) || sf::st_crs(L)$epsg != ctx$cfg$repro$crs_epsg)
      L <- sf::st_transform(L, ctx$cfg$repro$crs_epsg)
    if (isTRUE(sf::st_is_longlat(L)))
      stop("layer is in geographic coordinates", call. = FALSE)
    L
  }

  if (level == "disaggregated") {
    L <- to_crs(sf::st_read(cfg_path(ctx, "frozen", "cents_disaggregated"), quiet = TRUE))
    if (identical(source, "frozen")) return(L)
    rel <- if (identical(variant, "headline")) DV$rebuilt_path else
      file.path(ctx$cfg$paths$interim, sprintf("dependent_variable_%s.csv", variant))
    p <- file.path(REPO, rel)
    if (!file.exists(p))
      stop("stage-1 output missing: ", rel, " -- run `make dv` first", call. = FALSE)
    RB <- utils::read.csv(p, stringsAsFactors = FALSE)
    idx <- match(L$id, RB$id)
    if (nrow(RB) != nrow(L) || anyNA(idx) || anyDuplicated(RB$id))
      stop(sprintf("%s does not match the frozen layer one-to-one by id (%d vs %d rows)",
                   rel, nrow(RB), nrow(L)), call. = FALSE)
    if (max(abs(RB$Terreno[idx] - L$Terreno), na.rm = TRUE) > 0)
      stop(rel, ": plot area differs from the frozen layer's", call. = FALSE)
    L$Preco <- RB$Preco[idx]
    L$Puni  <- RB$Puni[idx]
    L$only_parking_address <- as.logical(RB$only_parking_address[idx])
    log_msg(ctx, sprintf("dependent variable (%s): rebuilt, variant '%s', from %s, joined by id",
                         level, variant, rel))
    return(L)
  }

  if (level != "aggregated")
    stop("proposed_dv_layer: level must be disaggregated | aggregated", call. = FALSE)
  if (!identical(variant, "headline"))
    stop("proposed_dv_layer: parking variants are address-level only", call. = FALSE)
  FZ <- to_crs(sf::st_read(cfg_path(ctx, "frozen", "cents_aggregated"), quiet = TRUE))
  if (identical(source, "frozen")) return(FZ)
  srcs <- ctx$cfg$revision$maup$sources
  keys <- vapply(srcs, function(s) as.character(s$key), character(1))
  if (!"500m_headline" %in% keys)
    stop("revision.maup.sources has no '500m_headline' record", call. = FALSE)
  rec <- srcs[[match("500m_headline", keys)]]
  p <- file.path(REPO, rec$path)
  if (!file.exists(p))
    stop("stage-2 output missing: ", rec$path, " -- run `make join` first", call. = FALSE)
  RB <- to_crs(sf::st_read(p, quiet = TRUE))
  if (nrow(FZ) != nrow(RB) || !all(FZ$id == RB$cell_id))
    stop("the rebuilt 500 m layer is not cell-aligned with the frozen one", call. = FALSE)
  RB$id <- RB$cell_id
  log_msg(ctx, sprintf("dependent variable (%s): rebuilt, from %s", level, rec$path))
  RB
}

# ---- the two cross-validation benchmarks -------------------------------------
#
# Both are scored on the same held-out blocks as the models, with the same RMSE
# (log units). `null_training_block_mean` predicts every held-out unit with the
# mean of the training blocks (intercept only). `null_plot_area_only` is the OLS
# of the log outcome on log plot area alone, fitted on the training blocks and
# predicted on the held-out block: the benchmark with the plot-area control and
# no centrality term, so that the difference between it and a model is the
# improvement attributable to the centrality measure.
cv_benchmarks <- function(D, blk, y = "ln_land_value", area = "ln_plot_area") {
  rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))
  p0 <- p1 <- rep(NA_real_, nrow(D))
  form <- stats::as.formula(sprintf("%s ~ %s", y, area))
  for (b in unique(blk)) {
    te <- which(blk == b)
    p0[te] <- mean(D[[y]][-te])
    o <- stats::lm(form, data = D[-te, , drop = FALSE])
    p1[te] <- as.numeric(stats::predict(o, newdata = D[te, , drop = FALSE]))
  }
  c(null_training_block_mean = rmse(D[[y]] - p0),
    null_plot_area_only      = rmse(D[[y]] - p1))
}

# ---- exclusions: the order is part of the specification ---------------------
#
# The steps run in the order `exclusions.order[<level>]` lists them. At the
# address level positivity before the IQR fence gives the submitted N = 7,767;
# the reverse order gives 7,714.
apply_exclusions <- function(ctx, df, level) {
  ex  <- ctx$cfg$exclusions
  cols <- ex$positivity[[level]]
  map <- ctx$cfg$measures$columns[[level]]

  resolve <- function(nm) if (!is.null(map[[nm]])) map[[nm]] else nm
  pos_cols <- vapply(cols, resolve, character(1))
  lv_col   <- resolve(ex$iqr_outliers$variable)

  count_step(ctx, sprintf("%s: loaded", level), nrow(df))

  order <- ex$order[[level]]
  if (is.null(order))
    stop("no exclusion order declared for level '", level, "' in config.yaml")
  log_msg(ctx, sprintf("exclusion order (%s): %s", level,
                       paste(order, collapse = " -> ")))

  for (step in order) {
    if (step == "positivity") {
      keep <- rep(TRUE, nrow(df))
      for (cc in pos_cols) {
        v <- df[[cc]]
        if (is.null(v)) stop("positivity column not found: ", cc)
        keep <- keep & !is.na(v) & v > 0
      }
      df <- df[keep, , drop = FALSE]
      count_step(ctx, "after positivity", nrow(df),
                 paste(pos_cols, collapse = ","))
    } else if (step == "iqr_outliers") {
      # The fence is computed on the rows that reach this step, i.e. after
      # every step listed before it.
      v  <- df[[lv_col]]
      qs <- stats::quantile(v, c(0.25, 0.75), na.rm = TRUE)
      m  <- ex$iqr_outliers$multiplier
      iqr <- qs[2] - qs[1]
      lo <- qs[1] - m * iqr
      hi <- qs[2] + m * iqr
      df <- df[!is.na(v) & v >= lo & v <= hi, , drop = FALSE]
      count_step(ctx, "after IQR outliers", nrow(df),
                 sprintf("%s in [%.2f, %.2f]", lv_col, lo, hi))
    } else {
      stop("unknown exclusion step: ", step)
    }
  }
  # Hand the frame back in weight-construction order, so that every caller's W
  # is decided by the rows' attributes and not by the file's order.
  df[order_for_weights(ctx, df, level), , drop = FALSE]
}

# ---- deterministic row order for weight construction ------------------------
#
# `knearneigh` breaks an exact distance tie by row order: whichever of two
# equidistant candidates comes first in the matrix wins the k-th slot. On the
# address layer 165 points share a coordinate with another point, and 135 of
# the 7,767 neighbour lists at k = 8 are exactly tied; on the 812-cell
# aggregated lattice 290 lists are tied at k = 6 without any duplicated
# centroid. With the file's own order W would therefore depend on how the
# GeoPackage happens to be stored -- reversing the file moves PC1's
# standardized beta by 6.3 %.
#
# Sorting the analysis frame on a key that is a pure function of the rows' own
# attributes, before any neighbour list is built, decides those ties
# reproducibly. It does not remove them.
#
# The sort is applied to the data frame, not inside build_weights(), because W
# and the frame are matched row for row; hence the call site at the end of
# apply_exclusions(), which every estimating script goes through.
# `method = "radix"` sorts characters in the C locale, so the order does not
# depend on LC_COLLATE either.
order_for_weights <- function(ctx, df, level) {
  ip <- ctx$cfg$weights$identical_points
  handling <- ip$handling
  if (identical(handling, "as_is")) return(seq_len(nrow(df)))
  if (!identical(handling, "sort_by_key"))
    stop("weights.identical_points.handling = '", format(handling),
         "' is not implemented (as_is | sort_by_key)", call. = FALSE)

  key <- ip$tie_break_key[[level]]
  if (is.null(key) || !length(key)) {
    log_msg(ctx, sprintf("order_for_weights: no tie_break_key for level '%s' -- row order kept",
                         level))
    return(seq_len(nrow(df)))
  }
  cols <- lapply(key, function(k) {
    v <- df[[k]]
    if (is.null(v)) stop("tie_break_key column not found at level '", level,
                         "': ", k, call. = FALSE)
    v
  })
  kk <- do.call(paste, c(lapply(cols, as.character), list(sep = "\r")))
  ndup <- sum(duplicated(kk))
  if (ndup > 0L)
    stop(sprintf("weights.identical_points.tie_break_key [%s] is not unique over the %s sample: %d duplicated key(s); the row order would still be arbitrary",
                 paste(key, collapse = ", "), level, ndup), call. = FALSE)
  ord <- do.call(order, c(cols, list(method = "radix", na.last = TRUE)))
  log_msg(ctx, sprintf("order_for_weights (%s): sorted by [%s]; %d of %d rows move (0 = the stored order already is key order)",
                       level, paste(key, collapse = ", "),
                       sum(ord != seq_along(ord)), length(ord)))
  ord
}

# ---- spatial weights --------------------------------------------------------

# `weights.active` is keyed by level. Pass either an explicit spec_name or the
# level. Coordinates must be projected (metres): kNN and distance bands are
# computed in the coordinates' own units.
build_weights <- function(ctx, coords, spec_name = NULL, level = NULL) {
  if (is.null(spec_name)) {
    act <- ctx$cfg$weights$active
    spec_name <- if (is.list(act)) {
      if (is.null(level)) stop("build_weights(): weights.active is per level; pass level=")
      act[[level]]
    } else act
  }
  spec <- ctx$cfg$weights[[spec_name]]
  if (is.null(spec)) stop("no weights spec named '", spec_name, "'")

  if (identical(spec$type, "knn")) {
    nb <- spdep::knn2nb(spdep::knearneigh(coords, k = as.integer(spec$k)))
    log_msg(ctx, sprintf("W = kNN k=%d style=%s (spec '%s')",
                         spec$k, spec$style, spec_name))
  } else if (identical(spec$type, "distance_band")) {
    nb <- spdep::dnearneigh(coords, d1 = 0, d2 = as.numeric(spec$d_max_m),
                            longlat = FALSE)
    log_msg(ctx, sprintf("W = distance band d<=%s m style=%s (spec '%s')",
                         spec$d_max_m, spec$style, spec_name))
  } else {
    stop("unknown weights type: ", spec$type)
  }

  if (isTRUE(spec$symmetrise)) {
    nb <- spdep::make.sym.nb(nb)          # explicit symmetrisation
    log_msg(ctx, "W symmetrised with make.sym.nb()")
  }

  card_n <- spdep::card(nb)
  log_msg(ctx, sprintf("neighbours: mean %.1f  min %d  max %d  islands %d",
                       mean(card_n), min(card_n), max(card_n), sum(card_n == 0)))
  lw <- spdep::nb2listw(nb, style = spec$style, zero.policy = FALSE)
  attr(lw, "spec_name") <- spec_name
  lw
}

# ---- island handling for distance bands -------------------------------------
#
# A distance band shorter than the largest first-nearest-neighbour distance
# leaves isolated units, and nb2listw(zero.policy = FALSE) refuses such a list.
# The isolated units are dropped for that spec only (`weights.band_islands:
# drop_isolated`) and N is recorded per spec. Removing a zero-degree node cannot
# isolate any other node, so one pass is enough.
#
# This is the only place a neighbour list is built outside build_weights(), and
# it lives here so that every analysis drops the same units for the same spec.
connected_idx <- function(ctx, coords, spec_name) {
  spec <- ctx$cfg$weights[[spec_name]]
  if (is.null(spec)) stop("no weights spec named '", spec_name, "'")
  if (!identical(spec$type, "distance_band")) return(seq_len(nrow(coords)))
  nb <- suppressWarnings(spdep::dnearneigh(coords, d1 = 0,
                            d2 = as.numeric(spec$d_max_m), longlat = FALSE))
  iso <- which(spdep::card(nb) == 0)
  if (!length(iso)) return(seq_len(nrow(coords)))
  handling <- ctx$cfg$weights$band_islands
  if (!identical(handling, "drop_isolated"))
    stop("weights.band_islands = '", format(handling), "' is not implemented; ",
         spec_name, " leaves ", length(iso), " island(s)", call. = FALSE)
  setdiff(seq_len(nrow(coords)), iso)
}

# ---- estimation helpers shared by every estimating script -------------------
#
# These live here, not in a stage script, for the same reason apply_exclusions()
# and build_weights() do: if one script computed the pseudo-R2, ran the LM
# suite or applied the decision rule its own way, its numbers would stop being
# comparable with the others'.

# coef() on a sphet fit can come back unnamed; take the names from summary().
# sphet names the spatial-lag parameter `lambda` and the error parameter `rho`.
sarar_row <- function(fit) {
  sm <- summary(fit)$Coef
  list(co = setNames(sm[, 1], rownames(sm)),
       se = setNames(sm[, 2], rownames(sm)),
       p  = setNames(sm[, 4], rownames(sm)))
}

# One entry point for every sphet fit. `het` comes from config; the remaining
# arguments are sphet 2.1.1's defaults, written out so that the estimator and
# its instrument set (WX and W^2X, q = 2, no lagged instruments for endogenous
# variables) do not change silently with a package update. Only the error
# parameter rho is bounded, by nlminb in (-0.9, 0.9); the lag parameter lambda
# is estimated by 2SLS and has no bound. A failure is logged and returns NULL.
fit_spreg <- function(ctx, form, data, listw, model) {
  het <- isTRUE(ctx$cfg$models$het)
  tryCatch(suppressWarnings(
    sphet::spreg(form, data = data, listw = listw, model = model, het = het,
                 lag.instr = FALSE, initial.value = 0.2, q = 2,
                 step1.c = FALSE, Durbin = FALSE)),
    error = function(e) { log_msg(ctx, "   ", toupper(model), " FAILED: ",
                                  conditionMessage(e)); NULL })
}

# Pseudo-R2 as used for the submitted tables: 1 - SSR / SST, with sphet's
# SARAR residuals y - X beta - lambda W y on the raw scale.
pseudo_r2 <- function(model, observed) {
  res <- residuals(model)
  1 - sum(res^2, na.rm = TRUE) /
      sum((observed - mean(observed, na.rm = TRUE))^2, na.rm = TRUE)
}

# spdep renamed lm.LMtests -> lm.RStests; support whichever is installed and
# return a named vector with canonical names (statistics and p-values -- the
# p-values are what lm_rule() keys on). NULL if the suite fails.
run_lm_tests <- function(form, data, listw) {
  ols <- lm(form, data = data)
  nms <- c("lm_err", "lm_lag", "rlm_err", "rlm_lag",
           "lm_err_p", "lm_lag_p", "rlm_err_p", "rlm_lag_p")
  pull <- function(out, keys) {
    out <- as.matrix(out)
    setNames(as.numeric(c(out[keys[1], 1], out[keys[2], 1],
                          out[keys[3], 1], out[keys[4], 1],
                          out[keys[1], 3], out[keys[2], 3],
                          out[keys[3], 3], out[keys[4], 3])), nms)
  }
  if ("lm.RStests" %in% getNamespaceExports("spdep")) {
    out <- tryCatch(summary(spdep::lm.RStests(ols, listw,
             test = c("RSerr", "RSlag", "adjRSerr", "adjRSlag")))$results,
             error = function(e) NULL)
    if (is.null(out)) return(NULL)
    pull(out, c("RSerr", "RSlag", "adjRSerr", "adjRSlag"))
  } else {
    out <- tryCatch(summary(spdep::lm.LMtests(ols, listw,
             test = c("LMerr", "LMlag", "RLMerr", "RLMlag")))$results,
             error = function(e) NULL)
    if (is.null(out)) return(NULL)
    pull(out, c("LMerr", "LMlag", "RLMerr", "RLMlag"))
  }
}

# Anselin's decision rules on the LM suite.
#
#   lt          the named vector from run_lm_tests(), or NULL
#   alpha       significance level (revision.lm_selection.alpha)
#   both_robust what to select when both robust tests are significant
#               (revision.lm_selection.both_robust_significant):
#                 "sarar"                   -> the SARAR model
#                 "larger_robust_statistic" -> "sem" if robust LM-error >=
#                                              robust LM-lag, else "sar"
#
# Neither plain LM test significant -> "ols"; exactly one -> that model ("sem"
# for error, "sar" for lag); both -> decided on the robust pair: only one robust
# test significant -> that model; neither -> the larger plain statistic; both
# -> `both_robust`.
#
# Returns a named character vector c(model = ..., basis = ...), model one of
# "ols", "sem", "sar", "sarar", or NA_character_ when the suite is missing or
# has a missing statistic (callers skip selection then).
lm_rule <- function(lt, alpha, both_robust) {
  if (!both_robust %in% c("sarar", "larger_robust_statistic"))
    stop("revision.lm_selection.both_robust_significant = '", format(both_robust),
         "' is not implemented (sarar | larger_robust_statistic)", call. = FALSE)
  need <- c("lm_err", "lm_lag", "rlm_err", "rlm_lag",
            "lm_err_p", "lm_lag_p", "rlm_err_p", "rlm_lag_p")
  if (is.null(lt) || !all(need %in% names(lt)) || anyNA(lt[need]))
    return(c(model = NA_character_, basis = "LM tests failed"))
  err  <- lt[["lm_err_p"]]  < alpha
  lag  <- lt[["lm_lag_p"]]  < alpha
  rerr <- lt[["rlm_err_p"]] < alpha
  rlag <- lt[["rlm_lag_p"]] < alpha
  if (!err && !lag) return(c(model = "ols",  basis = "neither LM significant"))
  if (err && !lag)  return(c(model = "sem",  basis = "only LM-error significant"))
  if (lag && !err)  return(c(model = "sar",  basis = "only LM-lag significant"))
  if (rerr && !rlag) return(c(model = "sem", basis = "both LM significant; only robust LM-error significant"))
  if (rlag && !rerr) return(c(model = "sar", basis = "both LM significant; only robust LM-lag significant"))
  if (!rerr && !rlag) {
    m <- if (lt[["lm_err"]] >= lt[["lm_lag"]]) "sem" else "sar"
    return(c(model = m, basis = "both LM significant, neither robust test is; larger plain LM statistic"))
  }
  if (identical(both_robust, "sarar"))
    return(c(model = "sarar", basis = "both robust tests significant"))
  m <- if (lt[["rlm_err"]] >= lt[["rlm_lag"]]) "sem" else "sar"
  c(model = m, basis = "both robust tests significant; larger robust statistic")
}

# ---- environment capture ----------------------------------------------------

# Writes logs/env/sessionInfo.txt (sessionInfo(), .libPaths() and the GDAL /
# GEOS / PROJ versions sf links against) and logs/env/r_packages.csv. `make env`
# copies logs/env/* into env/, which holds the record of the runs of record.
capture_env_r <- function(ctx) {
  d <- file.path(REPO, ctx$cfg$paths$logs, "env")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  p <- file.path(d, "sessionInfo.txt")
  sfv <- if (requireNamespace("sf", quietly = TRUE))
    capture.output(print(sf::sf_extSoftVersion())) else "sf not installed"
  writeLines(c(capture.output(sessionInfo()),
               "", ".libPaths():", paste0("  ", .libPaths()),
               "", "sf::sf_extSoftVersion():", sfv), p)
  pk <- as.data.frame(installed.packages()[, c("Package", "Version")],
                      stringsAsFactors = FALSE)
  write.csv(pk, file.path(d, "r_packages.csv"), row.names = FALSE)
  log_msg(ctx, "wrote ", p)
  p
}

finish <- function(ctx) {
  secs <- as.numeric(difftime(Sys.time(), ctx$started, units = "secs"))
  log_msg(ctx, sprintf("stage %d (%s) finished in %.1fs",
                       ctx$stage, ctx$name, secs))
}
