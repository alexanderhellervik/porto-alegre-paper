# =============================================================================
# Stage 4 -- the W-choice sweep: kNN at both levels, one grid, one code path.
#
# Not a standalone entry point. It is sourced by src/40_revision.R under
#     Rscript src/40_revision.R --mode w_sweep [aggregated|disaggregated|both]
#     Rscript src/40_revision.R --mode cv      [...]
#     Rscript src/40_revision.R --mode both    [...]        (sweep, then CV)
# and inherits `args`, `MODE`, `LEVEL` and everything in src/common.R from it.
#
# Dependent variable: `dependent_variable.proposed_source` (the rebuilt column,
# as in every table of the revised manuscript); at the address level the sample
# is then 7,775 records rather than the submitted 7,767.
#
# Purpose. The weight matrix is kNN at both levels: k = 6 at the cell level
# (hexagon adjacency) and, at the address level, the smallest k above which the
# estimates are interior and stable. That rule needs a sweep, and this file
# runs it and tabulates the rule. It reads `weights.*` and never modifies it.
#
# What it produces, per level and per k in `weights.sensitivity.w_choice`:
#   1. Neighbourhood geometry  -- distance to the k-th neighbour (mean / median /
#      max, metres), mean distance over all k neighbours, k as a share of the
#      sample, and the tie exposure: how many neighbour lists have the k-th and
#      (k+1)-th neighbours equidistant, so that the tie-break decides them.
#   2. SARAR by GMM (het per config) for all five measures: beta with SE and p,
#      standardized beta by re-estimation on z-scored covariates, lambda and rho
#      with SEs, the flag for rho on sphet's nlminb bound, pseudo-R2.
#   3. The LM suite and Anselin's decision rule (lm_rule() in common.R).
#   4. Spatial block CV: 10 seeded k-means blocks, W rebuilt inside every
#      training set, trend-only prediction with a training-side intercept
#      recalibration, RMSE in log units, OLS and two benchmarks (the
#      training-block mean, and OLS on log plot area alone), random k-fold
#      as the comparison line.
#   5. The choice rule (`weights.sensitivity.w_choice.rule`), applied and
#      tabulated.
#
# Every W comes from build_weights(), every filter from apply_exclusions(), the
# pseudo-R2, the LM suite and the decision rule from common.R, so this sweep
# computes exactly what the stage-3 gate and `--mode full` compute.
#
# Compute. Fits run in compute.max_workers forked workers (parallel::mclapply),
# capped at compute.host_ceiling. With a single-threaded (reference) BLAS one
# worker is one core; with a multithreaded BLAS set OMP_NUM_THREADS /
# OPENBLAS_NUM_THREADS to 1 so the thread pools do not nest.
# =============================================================================

suppressPackageStartupMessages({
  library(parallel)
})

LEVELS   <- if (identical(LEVEL, "both")) c("aggregated", "disaggregated") else LEVEL
if (!all(LEVELS %in% c("aggregated", "disaggregated")))
  stop("--mode w_sweep/cv accepts levels: aggregated | disaggregated | both",
       call. = FALSE)

ctx <- init(4, "w_choice")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
WC       <- ctx$cfg$weights$sensitivity$w_choice
if (is.null(WC))
  stop("config.yaml: weights.sensitivity.w_choice is missing", call. = FALSE)
RULE     <- WC$rule
BOUND    <- as.numeric(RULE$bound)
BEPS     <- as.numeric(RULE$bound_eps)
LAM_MAX  <- as.numeric(RULE$lambda_max)
ALPHA    <- as.numeric(RULE$alpha)
BOTH_ROB <- ctx$cfg$revision$lm_selection$both_robust_significant
NWORK    <- max(1L, min(as.integer(ctx$cfg$compute$max_workers),
                        as.integer(ctx$cfg$compute$host_ceiling)))
CTRL     <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
frm      <- function(m) as.formula(sprintf("ln_land_value ~ %sln_%s", CTRL, m))
DV_SOURCE <- ctx$cfg$dependent_variable$proposed_source
DO_SWEEP <- MODE %in% c("w_sweep", "both")
DO_CV    <- MODE %in% c("cv", "both")

log_msg(ctx, sprintf("mode=%s  levels=%s  workers=%d  dependent variable: %s",
                     MODE, paste(LEVELS, collapse = "+"), NWORK, DV_SOURCE))
log_msg(ctx, sprintf("choice rule: |rho| < %.6f and lambda < %.2f for all 5 fits, ",
                     BOUND - BEPS, LAM_MAX),
        sprintf("and no measure significant at either k (alpha = %.2f) changes sign or significance against the next larger k", ALPHA))

# ---- load one level ---------------------------------------------------------
#
# The layer on the dependent variable of the revised tables and the exclusion
# order for that level. apply_exclusions()
# returns the rows in tie-break-key order (weights.identical_points), so W does
# not depend on the order the file is stored in; nothing below re-sorts them.
# 165 address points share a coordinate with another point, which is why that
# order matters: `knearneigh` breaks exact distance ties by row order.
load_level <- function(level) {
  # The dependent variable of the revised tables
  # (`dependent_variable.proposed_source`, see proposed_dv_layer() in common.R);
  # the centralities are the frozen ones.
  Land <- proposed_dv_layer(ctx, level)
  crs_expected <- ctx$cfg$repro$crs_epsg
  if (is.na(st_crs(Land)$epsg) || st_crs(Land)$epsg != crs_expected) {
    log_msg(ctx, sprintf("!! %s CRS is %s, expected EPSG:%s -- transforming",
                         level, st_crs(Land)$epsg, crs_expected))
    Land <- st_transform(Land, crs_expected)
  }
  if (isTRUE(st_is_longlat(Land)))
    stop("layer is in geographic coordinates; kNN would be wrong", call. = FALSE)
  log_msg(ctx, sprintf("%s: CRS EPSG:%s (%s), %d rows as read",
                       level, st_crs(Land)$epsg, st_crs(Land)$Name, nrow(Land)))

  map <- ctx$cfg$measures$columns[[level]]
  for (canon in names(map)) {
    srcn <- map[[canon]]
    if (!is.null(Land[[srcn]]) && srcn != canon) Land[[canon]] <- Land[[srcn]]
  }
  Land <- apply_exclusions(ctx, Land, level)
  tgt  <- ctx$cfg$exclusions$targets[[paste0(level, "_n")]]
  if (identical(DV_SOURCE, "frozen")) {
    # On the frozen column the sample must be the submitted one.
    log_msg(ctx, sprintf("%s: N = %d (submitted %d) %s", level, nrow(Land), tgt,
                         if (nrow(Land) == tgt) "MATCH" else "*** MISMATCH ***"))
    if (nrow(Land) != tgt)
      stop(sprintf("%s N = %d does not match the submitted %d; stopping",
                   level, nrow(Land), tgt), call. = FALSE)
  } else {
    log_msg(ctx, sprintf("%s: N = %d on the rebuilt dependent variable (submitted sample %d)",
                         level, nrow(Land), tgt))
  }

  zh <- ctx$cfg$transforms$zero_handling
  lg <- function(x) if (zh == "log_plus_c")
    log(x + ctx$cfg$transforms$log_plus_c) else log(x)
  for (v in c("land_value", MEASURES, "plot_area"))
    Land[[paste0("ln_", v)]] <- lg(Land[[v]])

  coords <- if (level == "aggregated") {
    suppressWarnings(st_coordinates(st_point_on_surface(st_geometry(Land)))[, 1:2])
  } else {
    st_coordinates(Land)[, 1:2]
  }
  if (level == "aggregated") ADJ <<- adjacency_audit(Land, coords)
  list(D = st_drop_geometry(Land), coords = coords, level = level)
}

# ---- adjacency of the analysis cells ----------------------------------------
#
# Why kNN k = 6 rather than contiguity at the cell level. Rook contiguity on the
# hexagon polygons (shared edge, spdep::poly2nb(queen = FALSE)) leaves the
# cells without any adjacent analysis cell as islands; and on a complete
# lattice the six nearest cells sit at the lattice spacing, so the share of
# cells whose sixth-nearest cell is within 1 % of `grid.spacing_m` says how
# often k = 6 is exactly the first-order neighbourhood.
ADJ <- NULL
adjacency_audit <- function(Land, coords, k = 6L, tol = 0.01) {
  sp  <- as.numeric(ctx$cfg$grid$spacing_m)
  nb  <- suppressWarnings(spdep::poly2nb(Land, queen = FALSE))
  cd  <- spdep::card(nb)
  kn  <- suppressWarnings(spdep::knearneigh(coords, k = k))
  dk  <- sqrt(rowSums((coords - coords[kn$nn[, k], , drop = FALSE])^2))
  at  <- abs(dk - sp) <= tol * sp
  out <- data.frame(
    level = "aggregated", n_cells = nrow(coords), spacing_m = sp,
    contiguity = "rook (shared edge), poly2nb(queen = FALSE)",
    n_cells_no_adjacent = sum(cd == 0L),
    adjacent_mean = mean(cd), adjacent_median = median(cd),
    n_cells_6_adjacent = sum(cd == 6L),
    k = k, rel_tol = tol,
    n_cells_kth_nearest_at_spacing = sum(at),
    share_kth_nearest_at_spacing = mean(at),
    d_kth_nearest_mean_m = mean(dk), stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("adjacency (aggregated, %d cells): %d with no rook-adjacent analysis cell; the %d-th nearest cell within %.0f%% of %g m for %d (%.1f%%)",
                       nrow(coords), out$n_cells_no_adjacent, k, 100 * tol, sp,
                       out$n_cells_kth_nearest_at_spacing,
                       100 * out$share_kth_nearest_at_spacing))
  out
}

# ---- neighbourhood geometry, per spec ---------------------------------------
#
# Reported for the cross-level comparability point: a k on 7,767 points and a k
# on 812 hexagons are the same NUMBER of neighbours and nothing else, so the
# table gives the physical radius in metres and the share of the sample too.
#
# Tie exposure. `knearneigh` warns "identical points found / kd_tree not
# available"; the WARNING is recorded verbatim below. What it costs is countable:
# a list is order-dependent exactly when the k-th and (k+1)-th nearest points
# are equidistant, so the tie could have fallen either way. That count is
# `n_order_dependent_lists`. `n_lists_with_duplicate_member` is the looser
# exposure -- lists that contain, or belong to, a duplicated coordinate at all.
nb_stats <- function(coords, spec_name) {
  spec <- ctx$cfg$weights[[spec_name]]
  n    <- nrow(coords)
  warn <- character(0)
  cap  <- function(expr) withCallingHandlers(expr, warning = function(w) {
    warn <<- unique(c(warn, conditionMessage(w))); invokeRestart("muffleWarning") })

  key  <- paste(coords[, 1], coords[, 2], sep = "|")
  dupk <- key %in% key[duplicated(key)]

  out <- data.frame(
    w_spec = spec_name, w_type = spec$type,
    k = if (identical(spec$type, "knn")) as.integer(spec$k) else NA_integer_,
    d_max_m = if (identical(spec$type, "distance_band")) as.numeric(spec$d_max_m) else NA_real_,
    n = n, stringsAsFactors = FALSE)

  if (identical(spec$type, "knn")) {
    k   <- as.integer(spec$k)
    kn  <- cap(spdep::knearneigh(coords, k = k))
    dist_to <- function(idx) sqrt(rowSums((coords - coords[idx, , drop = FALSE])^2))
    dk  <- dist_to(kn$nn[, k])
    dall <- mean(vapply(seq_len(k), function(j) mean(dist_to(kn$nn[, j])), numeric(1)))
    # (k+1)-th neighbour: the tie test. spdep refuses k >= 500, so a grid point
    # at k = 499 could not be probed this way; the grid tops out at k = 300.
    ties <- NA_integer_
    if (k + 1L < 500L) {
      kn1  <- cap(spdep::knearneigh(coords, k = k + 1L))
      dk1  <- dist_to(kn1$nn[, k + 1L])
      ties <- sum(dk1 - dk <= 1e-9 * pmax(1, dk))
    }
    dupmem <- sum(dupk | apply(kn$nn, 1, function(r) any(dupk[r])))
    out$nbrs_mean <- k; out$nbrs_min <- k; out$nbrs_max <- k
    out$share_of_sample_pct       <- 100 * k / n
    d1  <- dist_to(kn$nn[, 1])
    out$d_first_mean_m            <- mean(d1)
    out$d_first_median_m          <- median(d1)
    out$d_first_max_m             <- max(d1)
    out$d_kth_mean_m              <- mean(dk)
    out$d_kth_median_m            <- median(dk)
    out$d_kth_max_m               <- max(dk)
    out$d_kth_min_m               <- min(dk)
    out$d_all_nbrs_mean_m         <- dall
    out$n_duplicate_coord_points  <- sum(dupk)
    out$n_order_dependent_lists   <- ties
    out$n_lists_with_duplicate_member <- dupmem
  } else {
    nb <- cap(spdep::dnearneigh(coords, d1 = 0, d2 = as.numeric(spec$d_max_m),
                                longlat = FALSE))
    cd <- spdep::card(nb)
    out$nbrs_mean <- mean(cd); out$nbrs_min <- min(cd); out$nbrs_max <- max(cd)
    out$share_of_sample_pct <- 100 * mean(cd) / n
    for (v in c("d_first_mean_m", "d_first_median_m", "d_first_max_m",
                "d_kth_mean_m", "d_kth_median_m", "d_kth_max_m", "d_kth_min_m",
                "d_all_nbrs_mean_m")) out[[v]] <- NA_real_
    out$n_duplicate_coord_points <- sum(dupk)
    out$n_order_dependent_lists  <- NA_integer_
    out$n_lists_with_duplicate_member <- NA_integer_
  }
  out$knearneigh_warnings <- if (length(warn)) paste(warn, collapse = " | ") else ""
  out
}

# ---- one (spec, measure) fit ------------------------------------------------
one_fit <- function(dat, W, m, level, spec_name) {
  form <- frm(m)
  lt   <- run_lm_tests(form, dat, W)
  sel  <- lm_rule(lt, ALPHA, BOTH_ROB)
  if (is.null(lt))                     # keep the frame's shape when the suite fails
    lt <- setNames(rep(NA_real_, 8), c("lm_err", "lm_lag", "rlm_err", "rlm_lag",
                                       "lm_err_p", "lm_lag_p", "rlm_err_p", "rlm_lag_p"))
  fit  <- fit_spreg(ctx, form, dat, W, "sarar")
  if (is.null(fit)) return(NULL)
  z  <- sarar_row(fit)
  bn <- paste0("ln_", m)
  r2 <- pseudo_r2(fit, dat$ln_land_value)
  ab <- abs(abs(z$co[["rho"]]) - BOUND) < BEPS      # only rho is bounded

  # standardized beta by re-estimation on z-scored covariates; the analytic
  # beta_raw * sd is carried alongside as the cross-check.
  sd_m <- sd(dat[[bn]]); sd_a <- sd(dat$ln_plot_area)
  dz <- dat
  dz$ln_plot_area <- as.numeric(scale(dat$ln_plot_area))
  dz[[bn]]        <- as.numeric(scale(dat[[bn]]))
  fz <- fit_spreg(ctx, form, dz, W, "sarar")
  bstd <- sstd <- NA_real_
  if (!is.null(fz)) { zz <- sarar_row(fz); bstd <- zz$co[[bn]]; sstd <- zz$se[[bn]] }

  list(
    sweep = data.frame(
      level = level, w_spec = spec_name,
      k = ctx$cfg$weights[[spec_name]]$k %||% NA_integer_,
      w_type = ctx$cfg$weights[[spec_name]]$type,
      n = nrow(dat), measure = m,
      beta_area = z$co[["ln_plot_area"]], se_area = z$se[["ln_plot_area"]],
      p_area = z$p[["ln_plot_area"]],
      beta_measure = z$co[[bn]], se_measure = z$se[[bn]], p_measure = z$p[[bn]],
      beta_std = bstd, se_std = sstd,
      beta_std_analytic = z$co[[bn]] * sd_m,
      sd_ln_measure = sd_m, sd_ln_plot_area = sd_a,
      lambda_lag = z$co[["lambda"]], se_lambda = z$se[["lambda"]],
      rho_error = z$co[["rho"]], se_rho = z$se[["rho"]],
      error_at_bound = as.integer(ab),
      lambda_nonstationary = as.integer(z$co[["lambda"]] >= LAM_MAX),
      pseudo_r2 = r2, stringsAsFactors = FALSE),
    msel = data.frame(
      level = level, w_spec = spec_name, measure = m, n = nrow(dat),
      lm_err = lt[["lm_err"]], lm_err_p = lt[["lm_err_p"]],
      lm_lag = lt[["lm_lag"]], lm_lag_p = lt[["lm_lag_p"]],
      rlm_err = lt[["rlm_err"]], rlm_err_p = lt[["rlm_err_p"]],
      rlm_lag = lt[["rlm_lag"]], rlm_lag_p = lt[["rlm_lag_p"]],
      selected = sel[["model"]], basis = sel[["basis"]],
      stringsAsFactors = FALSE))
}

# =============================================================================
# 1. The sweep
# =============================================================================

DATA  <- list()
sweep_rows <- msel_rows <- nb_rows <- list()

for (lv in LEVELS) {
  log_msg(ctx, strrep("=", 74))
  log_msg(ctx, "LEVEL: ", lv)
  DATA[[lv]] <- load_level(lv)
}

if (DO_SWEEP) for (lv in LEVELS) {
  d      <- DATA[[lv]]
  specs  <- c(as.character(unlist(WC$specs[[lv]])),
              as.character(unlist(WC$reference_specs[[lv]])))
  log_msg(ctx, strrep("-", 74))
  log_msg(ctx, sprintf("%s sweep: %s", lv, paste(specs, collapse = ", ")))

  for (ws in specs) {
    nbs <- nb_stats(d$coords, ws)
    nbs$level <- lv
    nb_rows[[length(nb_rows) + 1]] <- nbs
    if (nzchar(nbs$knearneigh_warnings))
      log_msg(ctx, sprintf("   knearneigh warning(s) at %s/%s: %s",
                           lv, ws, nbs$knearneigh_warnings))
    if (identical(nbs$w_type, "knn"))
      log_msg(ctx, sprintf("   %-24s k=%-4d %.2f%% of N   d(k-th nbr) mean %.0f m, median %.0f m, max %.0f m   order-dependent lists %s of %d",
                           ws, nbs$k, nbs$share_of_sample_pct, nbs$d_kth_mean_m,
                           nbs$d_kth_median_m, nbs$d_kth_max_m,
                           format(nbs$n_order_dependent_lists), nbs$n))
  }

  # W is built once per spec in the parent; forked workers share it copy-on-write.
  Ws   <- lapply(specs, function(ws) suppressWarnings(
            build_weights(ctx, d$coords, spec_name = ws)))
  names(Ws) <- specs

  jobs <- expand.grid(ws = specs, m = MEASURES, stringsAsFactors = FALSE)
  t0   <- Sys.time()
  res  <- mclapply(seq_len(nrow(jobs)), function(i) {
    j  <- jobs[i, ]
    tt <- Sys.time()
    o  <- one_fit(d$D, Ws[[j$ws]], j$m, lv, j$ws)
    if (!is.null(o)) o$secs <- as.numeric(difftime(Sys.time(), tt, units = "secs"))
    o
  }, mc.cores = min(NWORK, nrow(jobs)), mc.preschedule = FALSE)

  bad <- vapply(res, function(r) !is.list(r) || inherits(r, "try-error"), logical(1))
  if (any(bad))
    log_msg(ctx, sprintf("!! %d of %d %s sweep fits returned nothing",
                         sum(bad), length(res), lv))
  for (r in res[!bad]) {
    sweep_rows[[length(sweep_rows) + 1]] <- r$sweep
    msel_rows[[length(msel_rows) + 1]]   <- r$msel
  }
  log_msg(ctx, sprintf("%s sweep: %d fits in %.0f s wall on %d workers", lv,
                       sum(!bad), as.numeric(difftime(Sys.time(), t0, units = "secs")),
                       min(NWORK, nrow(jobs))))

  for (ws in specs) {
    sub <- do.call(rbind, lapply(res[!bad], `[[`, "sweep"))
    sub <- sub[sub$w_spec == ws, ]
    if (!nrow(sub)) next
    log_msg(ctx, sprintf("  %-24s rho at bound %d/5  lambda>=1 %d/5  rho [%+.3f,%+.3f]  lambda [%+.3f,%+.3f]  R2 [%.3f,%.3f]",
      ws, sum(sub$error_at_bound), sum(sub$lambda_nonstationary),
      min(sub$rho_error), max(sub$rho_error),
      min(sub$lambda_lag), max(sub$lambda_lag),
      min(sub$pseudo_r2), max(sub$pseudo_r2)))
  }
}

sweep <- if (length(sweep_rows)) do.call(rbind, sweep_rows) else NULL
msel  <- if (length(msel_rows))  do.call(rbind, msel_rows)  else NULL
nbtab <- if (length(nb_rows))    do.call(rbind, nb_rows)    else NULL

# =============================================================================
# 2. The choice rule, applied
# =============================================================================

rule_rows <- list()
if (!is.null(sweep)) for (lv in LEVELS) {
  grid  <- as.integer(unlist(WC$k_grid[[lv]]))
  sub   <- sweep[sweep$level == lv & sweep$w_type == "knn", ]
  verdict <- function(k) {
    s <- sub[!is.na(sub$k) & sub$k == k, ]
    if (nrow(s) != length(MEASURES)) return(NULL)
    s[match(MEASURES, s$measure), ]
  }
  for (i in seq_along(grid)) {
    k <- grid[i]; s <- verdict(k)
    if (is.null(s)) next
    interior <- all(abs(s$rho_error) < BOUND - BEPS) && all(s$lambda_lag < LAM_MAX)
    nxt <- if (i < length(grid)) verdict(grid[i + 1L]) else NULL
    # Stability against the next larger k. The rule counts a change only on a
    # measure that is significant (p < alpha) at either k: its sign or its
    # significance must not change. A sign flip of a coefficient insignificant
    # at both k is not a change of verdict (the convention of
    # `verdict_change_on_significant` in src/47 and src/49). The stricter
    # all-signs reading is kept beside it (`*_all` columns).
    if (is.null(nxt)) {
      stable_all <- stable <- NA; sign_agree <- NA_integer_; sig_agree <- NA_integer_
      n_flip_sig <- NA_integer_
      note <- "largest k in the grid: no successor, stability undefined"
    } else {
      sg <- sign(s$beta_measure) == sign(nxt$beta_measure)
      sv <- (s$p_measure < ALPHA) == (nxt$p_measure < ALPHA)
      sig_either <- (s$p_measure < ALPHA) | (nxt$p_measure < ALPHA)
      sign_agree <- sum(sg); sig_agree <- sum(sv)
      n_flip_sig <- sum(!sg & sig_either)
      stable_all <- all(sg) && all(sv)
      stable     <- all(sv) && n_flip_sig == 0L
      bad  <- unique(c(MEASURES[!sg & sig_either], MEASURES[!sv]))
      badx <- setdiff(MEASURES[!sg], bad)
      note <- paste(c(
        if (length(bad)) paste0("changes against k=", grid[i + 1L], " on a significant measure: ",
                                paste(bad, collapse = ",")),
        if (length(badx)) paste0("sign flip of an insignificant coefficient (not counted): ",
                                 paste(badx, collapse = ","))), collapse = "; ")
    }
    rule_rows[[length(rule_rows) + 1]] <- data.frame(
      level = lv, k = k, n = s$n[1],
      n_rho_at_bound = sum(abs(abs(s$rho_error) - BOUND) < BEPS),
      n_rho_ge_bound = sum(abs(s$rho_error) >= BOUND - BEPS),
      n_lambda_ge_1 = sum(s$lambda_lag >= LAM_MAX),
      rho_min = min(s$rho_error), rho_max = max(s$rho_error),
      lambda_min = min(s$lambda_lag), lambda_max = max(s$lambda_lag),
      interior = as.integer(interior),
      next_k = if (i < length(grid)) grid[i + 1L] else NA_integer_,
      sign_agree_all = sign_agree, sig_agree_all = sig_agree,
      stable_all_signs = if (is.na(stable_all)) NA_integer_ else as.integer(stable_all),
      rule_satisfied_all_signs = if (is.na(stable_all)) NA_integer_ else
        as.integer(interior && stable_all),
      n_sign_flips_on_significant = n_flip_sig,
      stable_on_significant = if (is.na(stable)) NA_integer_ else as.integer(stable),
      rule_satisfied = if (is.na(stable)) NA_integer_ else as.integer(interior && stable),
      note = note, stringsAsFactors = FALSE)
  }
}
rule <- if (length(rule_rows)) do.call(rbind, rule_rows) else NULL

# Two "above which" readings of the rule, reported beside the literal pairwise
# one so all three are checkable from the printed table:
#   interior_at_k_and_above -- k and every larger k in the grid are INTERIOR
#   rule_holds_from_k       -- every larger k with a DEFINED verdict satisfies
#                              the rule (the largest k's verdict is undefined by
#                              construction, so it is skipped, not counted false)
# `rule_satisfied` alone can be TRUE at a k that a larger k then contradicts,
# hence the extra columns.
if (!is.null(rule)) {
  rule$interior_at_k_and_above <- NA_integer_
  rule$rule_holds_from_k       <- NA_integer_
  for (lv in unique(rule$level)) {
    ii <- which(rule$level == lv); ii <- ii[order(rule$k[ii])]
    for (j in seq_along(ii)) {
      tail_i <- ii[j:length(ii)]
      rule$interior_at_k_and_above[ii[j]] <- as.integer(all(rule$interior[tail_i] == 1L))
      v <- rule$rule_satisfied[tail_i]; v <- v[!is.na(v)]
      rule$rule_holds_from_k[ii[j]] <- as.integer(length(v) > 0 && all(v == 1L))
    }
  }
  for (i in seq_len(nrow(rule)))
    log_msg(ctx, sprintf("RULE %-14s k=%-4d interior=%s (rho [%+.3f,%+.3f], lambda max %+.3f)  stable_vs_k=%s: %s  -> %s %s",
      rule$level[i], rule$k[i], ifelse(rule$interior[i] == 1L, "YES", "no "),
      rule$rho_min[i], rule$rho_max[i], rule$lambda_max[i],
      format(rule$next_k[i]),
      ifelse(is.na(rule$stable_on_significant[i]), "n/a",
             ifelse(rule$stable_on_significant[i] == 1L, "YES", "no")),
      ifelse(is.na(rule$rule_satisfied[i]), "UNDEFINED",
             ifelse(rule$rule_satisfied[i] == 1L, "SATISFIED", "fails")),
      rule$note[i]))
  for (lv in unique(rule$level)) {
    s  <- rule[rule$level == lv & !is.na(rule$rule_satisfied) & rule$rule_satisfied == 1L, ]
    h  <- rule[rule$level == lv & rule$rule_holds_from_k == 1L, ]
    log_msg(ctx, sprintf("choice rule at %s: %s", lv,
      if (nrow(s)) sprintf("satisfied at k = %s; SMALLEST = %d",
                           paste(s$k, collapse = ", "), min(s$k))
      else "NO k in the grid satisfies the rule"))
    log_msg(ctx, sprintf("choice rule at %s: 'and above' reading -- %s", lv,
      if (nrow(h)) sprintf("holds from k = %d upward", min(h$k))
      else "NO k from which the rule holds for every larger k in the grid"))
  }
}

# =============================================================================
# 3. Spatial block CV (revision.spatial_cv)
# =============================================================================

cvc   <- ctx$cfg$revision$spatial_cv
NB    <- as.integer(cvc$n_blocks)
NRF   <- as.integer(cvc$n_folds_random)
RECAL <- isTRUE(cvc$intercept_recalibration)
if (!identical(cvc$prediction, "trend_only"))
  stop("only revision.spatial_cv.prediction = 'trend_only' is implemented",
       call. = FALSE)
rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

cv_rows <- list()
if (DO_CV) for (lv in LEVELS) {
  d      <- DATA[[lv]]
  D      <- d$D; coords <- d$coords
  specs  <- as.character(unlist(WC$cv_specs[[lv]]))
  if (!length(specs)) next

  set.seed(as.integer(ctx$cfg$repro$seed))          # blocks must be reproducible
  km  <- kmeans(coords, centers = NB, nstart = 25, iter.max = 100)
  blk <- km$cluster
  set.seed(as.integer(ctx$cfg$repro$seed) + 1L)
  rfold <- sample(rep_len(seq_len(NRF), nrow(D)))
  log_msg(ctx, strrep("-", 74))
  log_msg(ctx, sprintf("%s block CV: %d k-means blocks, sizes %s", lv, NB,
                       paste(as.integer(table(blk)), collapse = " ")))
  log_msg(ctx, sprintf("%s CV specs: %s", lv, paste(specs, collapse = ", ")))

  schemes <- c("spatial_block",
               if (isTRUE(cvc$random_kfold_comparison)) "random_kfold")
  jobs <- do.call(rbind, lapply(specs, function(ws)
            do.call(rbind, lapply(schemes, function(sc)
              data.frame(ws = ws, scheme = sc, fold = seq_len(if (sc == "spatial_block") NB else NRF),
                         stringsAsFactors = FALSE)))))

  t0 <- Sys.time()
  fold_res <- mclapply(seq_len(nrow(jobs)), function(i) {
    j  <- jobs[i, ]
    te <- if (j$scheme == "spatial_block") which(blk == j$fold) else which(rfold == j$fold)
    tr <- setdiff(seq_len(nrow(D)), te)
    # W rebuilt on the TRAINING coordinates alone -- no held-out unit enters any
    # neighbour list, so nothing leaks through W.
    keep <- tr[connected_idx(ctx, coords[tr, , drop = FALSE], j$ws)]
    dtr  <- D[keep, , drop = FALSE]
    Wtr  <- suppressWarnings(build_weights(ctx, coords[keep, , drop = FALSE],
                                           spec_name = j$ws))
    out <- list(idx = te, y = D$ln_land_value[te], job = j)
    for (m in MEASURES) {
      form <- frm(m)
      Xte  <- model.matrix(form, data = D[te, , drop = FALSE])
      Xtr  <- model.matrix(form, data = dtr)
      f    <- fit_spreg(ctx, form, dtr, Wtr, "sarar")
      o    <- lm(form, data = dtr)
      out[[m]] <- if (is.null(f)) {
        list(sarar = rep(NA_real_, length(te)), recal = rep(NA_real_, length(te)),
             ols = as.numeric(predict(o, newdata = D[te, , drop = FALSE])),
             lambda = NA_real_)
      } else {
        b <- sarar_row(f)$co
        b <- b[colnames(Xte)]                        # trend coefficients only
        p <- as.numeric(Xte %*% b)
        list(sarar = p,
             recal = p + (mean(dtr$ln_land_value) - mean(as.numeric(Xtr %*% b))),
             ols = as.numeric(predict(o, newdata = D[te, , drop = FALSE])),
             lambda = sarar_row(f)$co[["lambda"]])
      }
    }
    out
  }, mc.cores = min(NWORK, nrow(jobs)), mc.preschedule = FALSE)

  ok <- !vapply(fold_res, function(r) !is.list(r) || inherits(r, "try-error"), logical(1))
  if (any(!ok)) log_msg(ctx, sprintf("!! %d of %d %s CV folds failed",
                                     sum(!ok), length(fold_res), lv))
  log_msg(ctx, sprintf("%s CV: %d folds in %.0f s wall on %d workers", lv, sum(ok),
                       as.numeric(difftime(Sys.time(), t0, units = "secs")),
                       min(NWORK, nrow(jobs))))

  for (ws in specs) for (sc in schemes) {
    sel <- which(ok & vapply(fold_res, function(r)
      is.list(r) && !inherits(r, "try-error") &&
        r$job$ws == ws && r$job$scheme == sc, logical(1)))
    if (!length(sel)) next
    for (m in MEASURES) {
      y  <- unlist(lapply(fold_res[sel], `[[`, "y"))
      ps <- unlist(lapply(fold_res[sel], function(r) r[[m]]$sarar))
      pr <- unlist(lapply(fold_res[sel], function(r) r[[m]]$recal))
      po <- unlist(lapply(fold_res[sel], function(r) r[[m]]$ols))
      lm_ <- vapply(fold_res[sel], function(r) r[[m]]$lambda, numeric(1))
      cv_rows[[length(cv_rows) + 1]] <- data.frame(
        level = lv, scheme = sc, w_spec = ws,
        k = ctx$cfg$weights[[ws]]$k %||% NA_integer_,
        measure = m, n_folds = length(sel), n_pred = length(y),
        prediction = "trend_only",
        rmse_sarar_trend = rmse(y - ps),
        rmse_sarar_trend_recal = rmse(y - pr),
        rmse_ols = rmse(y - po),
        mean_lambda_train = mean(lm_, na.rm = TRUE), stringsAsFactors = FALSE)
    }
    r <- do.call(rbind, cv_rows)
    r <- r[r$level == lv & r$scheme == sc & r$w_spec == ws, ]
    log_msg(ctx, sprintf("CV %-13s %-24s recal RMSE: %s", sc, ws,
      paste(sprintf("%s %.4f", r$measure, r$rmse_sarar_trend_recal), collapse = "  ")))
  }

  # Null model on the SAME blocks: predict the training-block mean.
  nullr <- local({
    p <- numeric(nrow(D))
    for (b in unique(blk)) { te <- which(blk == b); p[te] <- mean(D$ln_land_value[-te]) }
    rmse(D$ln_land_value - p)
  })
  cv_rows[[length(cv_rows) + 1]] <- data.frame(
    level = lv, scheme = "spatial_block", w_spec = "-", k = NA_integer_,
    measure = "null_training_block_mean", n_folds = NB, n_pred = nrow(D),
    prediction = "none", rmse_sarar_trend = NA_real_,
    rmse_sarar_trend_recal = nullr, rmse_ols = nullr,
    mean_lambda_train = NA_real_, stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("%s CV null (training-block mean) RMSE = %.4f", lv, nullr))
  # Plot area alone, same blocks: OLS of the log outcome on log plot area,
  # fitted on the training blocks (cv_benchmarks() in common.R).
  parea <- cv_benchmarks(D, blk)[["null_plot_area_only"]]
  cv_rows[[length(cv_rows) + 1]] <- data.frame(
    level = lv, scheme = "spatial_block", w_spec = "-", k = NA_integer_,
    measure = "null_plot_area_only", n_folds = NB, n_pred = nrow(D),
    prediction = "ols_plot_area", rmse_sarar_trend = NA_real_,
    rmse_sarar_trend_recal = parea, rmse_ols = parea,
    mean_lambda_train = NA_real_, stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("%s CV plot area only (OLS on the training blocks) RMSE = %.4f",
                       lv, parea))
}
cvres <- if (length(cv_rows)) do.call(rbind, cv_rows) else NULL

# =============================================================================
# 4. Write
# =============================================================================

w <- function(df, fn) {
  if (is.null(df)) return(invisible(NULL))
  p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p); p
}
w(nbtab, "stage4_w_choice_neighbours.csv")
w(ADJ,   "stage4_w_choice_adjacency.csv")
w(sweep, "stage4_w_choice_sweep.csv")
w(msel,  "stage4_w_choice_model_selection.csv")
w(rule,  "stage4_w_choice_rule.csv")
w(cvres, "stage4_w_choice_cv_rmse.csv")
write_counts(ctx, "stage4_w_choice_counts.csv")

finish(ctx)
