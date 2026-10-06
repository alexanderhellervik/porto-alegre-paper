#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — MAUP: the aggregated analysis at 250 m, 500 m and 1,000 m.
#
# A MAUP run is the aggregated analysis at another cell size, on the same
# frozen centralities and the same estimator. The frozen 500 m centrality
# columns are aggregates of street segments, not of the 9,030 address points,
# so re-aggregating the point layer at another cell size would answer the MAUP
# question with a different quantity. Stage 2 therefore assigns every segment
# to each cell it intersects (the rule that reproduces the per-cell segment
# counts of the frozen 500 m layer) and aggregates the segment centralities onto
# the 250 m and 1,000 m grids; this script models them.
#
# What it computes, per cell size and per measure
#   ln_land_value ~ ln_plot_area + ln_<measure>, SARAR by GMM (sphet, het = true)
#   1. N and the zero-handling loss per measure (transforms.zero_handling)
#   2. Pearson / Spearman correlations with land value
#   3. the weight specifications of revision.maup.w_specs: the proposed
#      aggregated kNN k = 6 (role `revision_headline`, the matrix Table 6 is
#      quoted from), and for comparability the submitted 6,100 m band, a
#      constant neighbour count (k = 242), a constant neighbour share
#      (k = 0.298 N) and a local k = 10 -- the error-bound flag on every fit
#   4. raw and standardized beta with SEs, lambda, rho, pseudo-R2, the LM suite
#      and the model the LM rules select, estimated as well as SARAR
#   5. a sign table across point level -> 250 m -> 500 m -> 1,000 m
#   6. spatial block CV, with OLS and null baselines
#   7. the two parking alternates, SARAR coefficients only
#
# The 500 m row is run twice -- once on the frozen submitted layer and once on
# stage 2's rebuild of it -- so the 250 m and 1,000 m results are anchored to
# the submitted ones rather than merely adjacent to them.
#
# Every weight matrix comes from build_weights(), every filter from
# apply_exclusions(), the pseudo-R2, LM suite and rule, island rule and sphet
# wrapper from common.R. Nothing analytical is decided here; it is all
# config.yaml (`revision.maup`, `weights.*`, `measures.columns.maup_*`,
# `exclusions.*`).
#
# Run:  Rscript src/45_maup.R          (make maup)   ~40 min, 250 m dominates
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "revision_maup")
capture_env_r(ctx)

MP       <- ctx$cfg$revision$maup
MEASURES <- c(unlist(ctx$cfg$measures$accessibility),
              unlist(ctx$cfg$measures$intermediation))
BOUND    <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
ALPHA    <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
SIZES    <- as.integer(unlist(MP$hex_sizes_m))
FULLV    <- MP$full_analysis_variant
CV_SPEC  <- MP$cv_w_spec
CV_SIZES <- as.integer(unlist(MP$cv_sizes_m))
ZH       <- ctx$cfg$transforms$zero_handling
CTRL     <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""

W_BY_SIZE <- setNames(
  lapply(MP$w_specs, function(x) as.character(unlist(x$specs))),
  vapply(MP$w_specs, function(x) as.character(x$size_m), character(1)))
W_ROLE <- function(ws) {
  r <- MP$w_roles[[ws]]
  if (is.null(r)) stop("revision.maup.w_roles has no entry for '", ws, "'", call. = FALSE)
  as.character(r)
}

# `revision.maup.w_roles` says what each matrix is for, and one role --
# `revision_headline`, the proposed aggregated W (kNN k = 6) -- names the matrix
# Table 6 of the manuscript is quoted from. The submitted 6,100 m band keeps
# `constant_metric_band` and stays in the sweep as a comparability column.
# Exactly one spec per size may carry the headline role, so the manuscript
# table has one unambiguous cell per (size, measure).
HEADLINE_ROLE <- as.character(MP$headline_w_role)
headline_spec <- function(size_m) {
  specs <- W_BY_SIZE[[as.character(size_m)]]
  if (is.null(specs)) stop("no revision.maup.w_specs entry for size ", size_m)
  h <- specs[vapply(specs, function(s) identical(W_ROLE(s), HEADLINE_ROLE),
                    logical(1))]
  if (length(h) != 1L)
    stop(sprintf("exactly one w_spec at %d m must carry w_role '%s'; found %d",
                 size_m, HEADLINE_ROLE, length(h)), call. = FALSE)
  h[[1]]
}
HEADLINE_SPECS <- setNames(vapply(SIZES, headline_spec, character(1)),
                           as.character(SIZES))

log_msg(ctx, "MAUP cell sizes: ", paste(SIZES, collapse = " / "), " m")
log_msg(ctx, sprintf("manuscript-table W: role '%s' -> %s",
                     HEADLINE_ROLE,
                     paste(sprintf("%dm: %s", SIZES, HEADLINE_SPECS),
                           collapse = ", ")))
if (!all(HEADLINE_SPECS == CV_SPEC))
  log_msg(ctx, sprintf("!! block CV runs on '%s', which is NOT the headline matrix at every size; the manuscript quotes both on one matrix",
                       CV_SPEC))
log_msg(ctx, "measures: ", paste(MEASURES, collapse = ", "),
        "   zero handling = ", ZH)

# ---- tidy long accumulator --------------------------------------------------

results <- list()
add <- function(size_m, variant, w_spec, measure, model, stat, value) {
  results[[length(results) + 1]] <<- data.frame(
    size_m = size_m, variant = variant, w_spec = w_spec, measure = measure,
    model = model, statistic = stat, value = as.numeric(value),
    stringsAsFactors = FALSE)
  invisible(NULL)
}

lg   <- function(x) if (ZH == "log_plus_c")
  log(x + ctx$cfg$transforms$log_plus_c) else log(x)
frm  <- function(m) as.formula(sprintf("ln_land_value ~ %sln_%s", CTRL, m))
rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))
pval <- function(b, se) 2 * pnorm(-abs(b / se))

# Anselin's decision rules on the LM suite: lm_rule() in common.R, with the
# alpha and the both-robust rule from revision.lm_selection.
BOTH_ROBUST <- ctx$cfg$revision$lm_selection$both_robust_significant

# ---- input integrity --------------------------------------------------------
# The stage-2 GPKGs are byte-deterministic, so their checksums are a
# real test that the layer being modelled is the layer stage 2 reported on.

man_path <- file.path(REPO, ctx$cfg$paths$interim, MP$manifest)
manifest <- if (file.exists(man_path)) jsonlite::fromJSON(man_path) else NULL
verify_source <- function(src) {
  key <- sub("\\.gpkg$", "", basename(src$path))
  if (is.null(manifest) || is.null(manifest[[key]])) {
    log_msg(ctx, sprintf("   checksum: %s not in %s (frozen input, checked at stage 0a)",
                         key, MP$manifest)); return(NA)
  }
  got <- digest::digest(file.path(REPO, src$path), algo = "sha256", file = TRUE)
  ok  <- identical(got, manifest[[key]]$sha256)
  log_msg(ctx, sprintf("   checksum: %s %s", key,
                       if (ok) "MATCHES stage2_manifest" else "*** MISMATCH ***"))
  if (!ok) stop("stage-2 checksum mismatch for ", key, call. = FALSE)
  ok
}

# ---- load one source layer and put it on the logical column names -----------

load_source <- function(src) {
  path <- file.path(REPO, src$path)
  if (!file.exists(path)) stop("source layer missing: ", path, call. = FALSE)
  verify_source(src)
  g <- if (is.null(src$layer)) st_read(path, quiet = TRUE) else
                               st_read(path, layer = src$layer, quiet = TRUE)

  crs_expected <- ctx$cfg$repro$crs_epsg
  if (is.na(st_crs(g)$epsg) || st_crs(g)$epsg != crs_expected) {
    log_msg(ctx, sprintf("!! CRS is %s, expected EPSG:%s -- transforming",
                         st_crs(g)$epsg, crs_expected))
    g <- st_transform(g, crs_expected)
  }
  if (isTRUE(st_is_longlat(g)))
    stop("layer is in geographic coordinates; distance weights would be wrong",
         call. = FALSE)

  map <- ctx$cfg$measures$columns[[src$columns]]
  if (is.null(map)) stop("no measures.columns map named '", src$columns, "'")
  for (canon in names(map)) {
    s <- map[[canon]]
    if (is.null(g[[s]]))
      stop(sprintf("column '%s' (logical '%s') not in %s", s, canon, src$path),
           call. = FALSE)
    if (s != canon) g[[canon]] <- g[[s]]
  }

  # Put the layer's own cell identifier on the canonical name `cell_id`,
  # so `weights.identical_points.tie_break_key` can declare one key per maup
  # level whatever the source layer calls its identifier (the frozen 500 m
  # layer says `id`, the stage-2 layers say `cell_id`; at 500 m they are the
  # same values). order_for_weights() then enforces uniqueness over the
  # post-exclusion sample and stops if the key does not separate every row.
  idc <- as.character(src$cell_id_column %||% MP$cell_id_column)
  if (is.null(g[[idc]]))
    stop(sprintf("cell id column '%s' not found in %s (revision.maup.cell_id_column)",
                 idc, src$path), call. = FALSE)
  if (!identical(idc, "cell_id")) g$cell_id <- g[[idc]]

  log_msg(ctx, sprintf("loaded %-38s %5d cells  CRS EPSG:%s  map '%s'",
                       src$key, nrow(g), st_crs(g)$epsg, src$columns))
  g
}

# ---- what the zero/NA handling costs, per measure, before the joint filter --

zero_rows <- list()
zero_audit <- function(src, g) {
  d  <- st_drop_geometry(g)
  lv <- d$land_value
  pop <- !is.na(lv) & lv > 0                       # cells with any address data
  for (m in c(MEASURES, "land_value", "plot_area")) {
    v <- d[[m]]
    zero_rows[[length(zero_rows) + 1]] <<- data.frame(
      key = src$key, size_m = src$size_m, variant = src$variant,
      provenance = src$provenance, measure = m,
      n_grid_cells = nrow(d),
      n_populated  = sum(pop),
      n_na         = sum(is.na(v)),
      n_na_populated   = sum(pop & is.na(v)),
      n_zero_populated = sum(pop & !is.na(v) & v == 0),
      n_neg_populated  = sum(pop & !is.na(v) & v < 0),
      n_positive_populated = sum(pop & !is.na(v) & v > 0),
      zero_handling = ZH, stringsAsFactors = FALSE)
  }
  invisible(NULL)
}

# ---- prepare a modelling frame ---------------------------------------------

prepare <- function(src) {
  g <- load_source(src)
  zero_audit(src, g)
  lvl <- sprintf("maup_%dm", src$size_m)
  count_step(ctx, sprintf("%s: grid cells", src$key), nrow(g))
  g <- apply_exclusions(ctx, g, lvl)
  for (v in c("land_value", MEASURES, "plot_area"))
    g[[paste0("ln_", v)]] <- lg(g[[v]])
  pts <- suppressWarnings(st_point_on_surface(g))
  list(src = src, D = st_drop_geometry(g),
       coords = st_coordinates(pts)[, 1:2, drop = FALSE])
}

# =============================================================================
# The per-source analysis
# =============================================================================

wsens <- list(); msel <- list(); stdz <- list(); corr <- list()
cv_rows <- list(); sign_rows <- list(); var_rows <- list(); wsum <- list()
dropped <- list()

analyse <- function(P, full) {
  src <- P$src; D <- P$D; coords <- P$coords
  sz  <- src$size_m; key <- src$key
  specs <- W_BY_SIZE[[as.character(sz)]]
  if (is.null(specs)) stop("no revision.maup.w_specs entry for size ", sz)

  # ---- 2. correlations ------------------------------------------------------
  if (full) for (m in MEASURES) {
    pear <- cor(D$ln_land_value, D[[paste0("ln_", m)]], method = "pearson")
    spea <- cor(D$land_value,    D[[m]],                method = "spearman")
    corr[[length(corr) + 1]] <<- data.frame(
      key = key, size_m = sz, variant = src$variant,
      provenance = src$provenance, measure = m, n = nrow(D),
      pearson_ln = pear, spearman_raw = spea, stringsAsFactors = FALSE)
    add(sz, src$variant, "-", m, "correlation", "pearson_ln", pear)
    add(sz, src$variant, "-", m, "correlation", "spearman_raw", spea)
  }

  for (ws in specs) {
    spec <- ctx$cfg$weights[[ws]]
    keep <- connected_idx(ctx, coords, ws)
    if (length(keep) < nrow(coords))
      log_msg(ctx, sprintf("!! %s / %s isolates %d unit(s); dropped (weights.band_islands)",
                           key, ws, nrow(coords) - length(keep)))
    dat <- D[keep, , drop = FALSE]
    # A W that cannot be constructed is recorded and skipped, never replaced by
    # a smaller one -- spdep::knearneigh() refuses k >= 500 outright, and
    # silently capping k would put a differently-specified row in a table whose
    # whole point is that the rows are comparable.
    W <- tryCatch(build_weights(ctx, coords[keep, , drop = FALSE], spec_name = ws),
                  error = function(e) { log_msg(ctx, sprintf(
                    "!! W spec '%s' cannot be built at %d m: %s -- dropped, not capped",
                    ws, sz, conditionMessage(e))); NULL })
    if (is.null(W)) {
      dropped[[length(dropped) + 1]] <<- data.frame(
        key = key, size_m = sz, w_spec = ws, w_role = W_ROLE(ws),
        reason = "build_weights() failed (see log); dropped, not capped",
        stringsAsFactors = FALSE)
      next
    }
    card_n <- spdep::card(W$neighbours)
    count_step(ctx, sprintf("%s / %s", key, ws), nrow(dat),
               sprintf("%s, style=%s, mean nb %.1f",
                       if (identical(spec$type, "knn")) paste0("kNN k=", spec$k)
                       else paste0("band d<=", spec$d_max_m, "m"),
                       spec$style, mean(card_n)))
    if (full) {
      # The neighbourhood's scale in metres, on the same coordinates and
      # the same neighbour list the model uses. For a kNN spec each unit's
      # largest neighbour distance is its k-th-nearest-neighbour distance, so
      # the median of that column is "the median k-th-neighbour distance" and is
      # what makes the k = 6 rows comparable across cell sizes.
      nbd <- spdep::nbdists(W$neighbours, coords[keep, , drop = FALSE])
      kth <- vapply(nbd, function(d) if (length(d)) max(d) else NA_real_,
                    numeric(1))
      wsum[[length(wsum) + 1]] <<- data.frame(
        key = key, size_m = sz, w_spec = ws, w_role = W_ROLE(ws),
        w_type = spec$type,
        w_param = if (identical(spec$type, "knn")) paste0("k=", spec$k)
                  else paste0("d<=", spec$d_max_m, "m"),
        n = nrow(dat), n_dropped_islands = nrow(D) - nrow(dat),
        mean_neighbours = mean(card_n), min_neighbours = min(card_n),
        max_neighbours = max(card_n),
        neighbour_share = mean(card_n) / nrow(dat),
        median_kth_nb_dist_m = stats::median(kth, na.rm = TRUE),
        mean_kth_nb_dist_m   = mean(kth, na.rm = TRUE),
        min_kth_nb_dist_m    = min(kth, na.rm = TRUE),
        max_kth_nb_dist_m    = max(kth, na.rm = TRUE),
        stringsAsFactors = FALSE)
    }
    add(sz, src$variant, ws, "-", "W", "n", nrow(dat))
    add(sz, src$variant, ws, "-", "W", "mean_neighbours", mean(card_n))

    for (m in MEASURES) {
      form <- frm(m)
      t0 <- Sys.time()

      # ---- LM suite (depends on W, not on the GMM fit) ---------------------
      lt <- NULL; sel <- c(model = NA_character_, basis = "not run")
      if (full) {
        lt <- run_lm_tests(form, dat, W)
        if (!is.null(lt)) for (nm in names(lt))
          add(sz, src$variant, ws, m, "LM", nm, lt[[nm]])
        sel <- lm_rule(lt, ALPHA, BOTH_ROBUST)
      }

      # ---- SARAR, the estimator of the submitted tables ---------------------
      fit <- fit_spreg(ctx, form, dat, W, "sarar")
      if (is.null(fit)) next
      z  <- sarar_row(fit)
      bn <- paste0("ln_", m)
      for (nm in names(z$co)) {
        add(sz, src$variant, ws, m, "SARAR", paste0("coef_", nm), z$co[[nm]])
        add(sz, src$variant, ws, m, "SARAR", paste0("se_",   nm), z$se[[nm]])
        add(sz, src$variant, ws, m, "SARAR", paste0("p_",    nm), z$p[[nm]])
      }
      # sphet bounds only the error parameter rho
      at_bound <- abs(abs(z$co[["rho"]]) - BOUND) < 1e-6
      add(sz, src$variant, ws, m, "SARAR", "error_at_bound", as.numeric(at_bound))
      r2 <- pseudo_r2(fit, dat$ln_land_value)
      add(sz, src$variant, ws, m, "SARAR", "pseudo_r2", r2)

      log_msg(ctx, sprintf("   %-5dm %-24s %-3s  beta=%+.4f (se %.4f, p %.4f)  Area=%+.4f  lam=%+.4f  rho=%+.4f%s  R2=%.4f  [%.0fs]",
        sz, ws, m, z$co[[bn]], z$se[[bn]], z$p[[bn]], z$co[["ln_plot_area"]],
        z$co[["lambda"]], z$co[["rho"]],
        if (at_bound) " *BOUND*" else "", r2,
        as.numeric(difftime(Sys.time(), t0, units = "secs"))))

      row <- data.frame(
        key = key, size_m = sz, variant = src$variant,
        provenance = src$provenance, w_spec = ws, w_role = W_ROLE(ws),
        n = nrow(dat), measure = m,
        beta_area = z$co[["ln_plot_area"]], se_area = z$se[["ln_plot_area"]],
        beta_measure = z$co[[bn]], se_measure = z$se[[bn]],
        p_measure = z$p[[bn]],
        sign = if (z$co[[bn]] >= 0) "+" else "-",
        significant_05 = as.integer(z$p[[bn]] < 0.05),
        lambda_lag = z$co[["lambda"]], se_lambda = z$se[["lambda"]],
        rho_error = z$co[["rho"]], se_rho = z$se[["rho"]],
        error_at_bound = as.integer(at_bound),
        pseudo_r2 = r2, stringsAsFactors = FALSE)

      if (full) {
        wsens[[length(wsens) + 1]] <<- row
        sign_rows[[length(sign_rows) + 1]] <<- row[, c(
          "size_m", "variant", "provenance", "w_spec", "w_role", "measure", "n",
          "beta_measure", "se_measure", "p_measure", "sign", "significant_05")]
      } else {
        var_rows[[length(var_rows) + 1]] <<- row
      }
      if (!full) next

      # ---- the model the LM rules actually select --------------------------
      selm <- sel[["model"]]
      sel_r2 <- sel_beta <- sel_se <- sel_lambda <- sel_rho <- NA_real_
      if (identical(selm, "sarar")) {
        sel_r2 <- r2; sel_beta <- z$co[[bn]]; sel_se <- z$se[[bn]]
        sel_lambda <- z$co[["lambda"]]; sel_rho <- z$co[["rho"]]
      } else if (identical(selm, "ols")) {
        o <- lm(form, data = dat); so <- summary(o)$coefficients
        sel_r2 <- pseudo_r2(o, dat$ln_land_value)
        sel_beta <- so[bn, 1]; sel_se <- so[bn, 2]
        for (nm in rownames(so)) {
          add(sz, src$variant, ws, m, "OLS", paste0("coef_", nm), so[nm, 1])
          add(sz, src$variant, ws, m, "OLS", paste0("se_", nm),   so[nm, 2])
        }
        add(sz, src$variant, ws, m, "OLS", "pseudo_r2", sel_r2)
      } else if (!is.na(selm)) {
        smod <- if (identical(selm, "sem")) "error" else "lag"
        sf2  <- fit_spreg(ctx, form, dat, W, smod)
        if (!is.null(sf2)) {
          z2 <- sarar_row(sf2); lbl <- toupper(selm)
          for (nm in names(z2$co)) {
            add(sz, src$variant, ws, m, lbl, paste0("coef_", nm), z2$co[[nm]])
            add(sz, src$variant, ws, m, lbl, paste0("se_",   nm), z2$se[[nm]])
          }
          sel_r2   <- pseudo_r2(sf2, dat$ln_land_value)
          sel_beta <- z2$co[[bn]]; sel_se <- z2$se[[bn]]
          add(sz, src$variant, ws, m, lbl, "pseudo_r2", sel_r2)
          # sphet names the error parameter `rho`, the lag parameter `lambda`.
          sp <- if (identical(selm, "sem")) z2$co[["rho"]] else z2$co[["lambda"]]
          if (identical(selm, "sem")) sel_rho <- sp else sel_lambda <- sp
          add(sz, src$variant, ws, m, lbl, "spatial_at_bound",
              as.numeric(abs(abs(sp) - BOUND) < 1e-6))
        }
      }
      ltv <- function(k) if (is.null(lt)) NA_real_ else as.numeric(lt[[k]])
      msel[[length(msel) + 1]] <<- data.frame(
        key = key, size_m = sz, w_spec = ws, w_role = W_ROLE(ws),
        measure = m, n = nrow(dat),
        lm_err = ltv("lm_err"), lm_err_p = ltv("lm_err_p"),
        lm_lag = ltv("lm_lag"), lm_lag_p = ltv("lm_lag_p"),
        rlm_err = ltv("rlm_err"), rlm_err_p = ltv("rlm_err_p"),
        rlm_lag = ltv("rlm_lag"), rlm_lag_p = ltv("rlm_lag_p"),
        selected = selm, basis = sel[["basis"]],
        selected_beta = sel_beta, selected_se = sel_se,
        selected_lambda = sel_lambda, selected_rho = sel_rho,
        selected_pseudo_r2 = sel_r2,
        sarar_beta = z$co[[bn]], sarar_pseudo_r2 = r2,
        stringsAsFactors = FALSE)
      log_msg(ctx, sprintf("         LM rules select %-5s (%s)",
                           toupper(selm), sel[["basis"]]))

      # ---- standardized coefficients (covariates z-scored, refitted) --------
      if (isTRUE(ctx$cfg$revision$standardize_coefficients)) {
        dz <- dat
        sd_area <- sd(dat$ln_plot_area); sd_m <- sd(dat[[bn]])
        dz$ln_plot_area <- as.numeric(scale(dat$ln_plot_area))
        dz[[bn]] <- as.numeric(scale(dat[[bn]]))
        fz <- fit_spreg(ctx, form, dz, W, "sarar")
        if (!is.null(fz)) {
          zz <- sarar_row(fz)
          for (nm in names(zz$co)) {
            add(sz, src$variant, ws, m, "SARAR_std", paste0("coef_", nm), zz$co[[nm]])
            add(sz, src$variant, ws, m, "SARAR_std", paste0("se_",   nm), zz$se[[nm]])
          }
          add(sz, src$variant, ws, m, "SARAR_std", "pseudo_r2",
              pseudo_r2(fz, dz$ln_land_value))
          stdz[[length(stdz) + 1]] <<- data.frame(
            key = key, size_m = sz, w_spec = ws, w_role = W_ROLE(ws),
            measure = m, n = nrow(dat), sd_ln_measure = sd_m,
            beta_raw = z$co[[bn]], se_raw = z$se[[bn]],
            beta_std = zz$co[[bn]], se_std = zz$se[[bn]],
            beta_std_analytic = z$co[[bn]] * sd_m,
            area_beta_raw = z$co[["ln_plot_area"]],
            area_beta_std = zz$co[["ln_plot_area"]],
            sd_ln_plot_area = sd_area, stringsAsFactors = FALSE)
        }
      }
    }
  }
  invisible(NULL)
}

# =============================================================================
# Spatial block CV (revision.spatial_cv): k-means blocks on the coordinates,
# W rebuilt inside every training set, trend-only prediction
# =============================================================================

run_cv <- function(P) {
  src <- P$src; D <- P$D; coords <- P$coords; sz <- src$size_m
  cvc <- ctx$cfg$revision$spatial_cv
  NB  <- as.integer(cvc$n_blocks)
  RECAL <- isTRUE(cvc$intercept_recalibration)
  if (!identical(cvc$prediction, "trend_only"))
    stop("only revision.spatial_cv.prediction = 'trend_only' is implemented",
         call. = FALSE)

  log_msg(ctx, strrep("-", 74))
  log_msg(ctx, sprintf("Block CV %s: %d blocks by %s, W = '%s' rebuilt inside every training set, prediction = %s%s",
                       src$key, NB, cvc$method, CV_SPEC, cvc$prediction,
                       if (RECAL) " + training-side intercept recalibration" else ""))

  set.seed(as.integer(ctx$cfg$repro$seed))       # blocks must be reproducible
  km  <- kmeans(coords, centers = NB, nstart = 25, iter.max = 100)
  blk <- km$cluster
  log_msg(ctx, "block sizes: ", paste(as.integer(table(blk)), collapse = " "))
  for (b in seq_len(NB))
    add(sz, src$variant, CV_SPEC, "-", "CV_blocks", sprintf("block_%02d_n", b),
        sum(blk == b))

  cv_fold <- function(test_idx, m) {
    tr <- setdiff(seq_len(nrow(D)), test_idx)
    # W is rebuilt on the training coordinates only -- no test unit ever enters
    # any neighbour list, so nothing leaks through W.
    keep_tr <- tr[connected_idx(ctx, coords[tr, , drop = FALSE], CV_SPEC)]
    dtr  <- D[keep_tr, , drop = FALSE]
    Wtr  <- build_weights(ctx, coords[keep_tr, , drop = FALSE], spec_name = CV_SPEC)
    form <- frm(m)
    Xte  <- model.matrix(form, data = D[test_idx, , drop = FALSE])
    Xtr  <- model.matrix(form, data = dtr)
    out  <- list(y = D$ln_land_value[test_idx])
    f <- fit_spreg(ctx, form, dtr, Wtr, "sarar")
    if (is.null(f)) {
      out$sarar <- rep(NA_real_, length(out$y))
      out$sarar_recal <- out$sarar
    } else {
      b <- sarar_row(f)$co[colnames(Xte)]          # trend coefficients only
      out$sarar <- as.numeric(Xte %*% b)
      out$sarar_recal <- out$sarar +
        (mean(dtr$ln_land_value) - mean(as.numeric(Xtr %*% b)))
      out$lambda <- sarar_row(f)$co[["lambda"]]
    }
    o <- lm(form, data = dtr)
    out$ols <- as.numeric(predict(o, newdata = D[test_idx, , drop = FALSE]))
    out
  }

  folds <- split(seq_len(nrow(D)), blk)
  for (m in MEASURES) {
    p_s <- p_r <- p_o <- y_all <- numeric(0); lams <- numeric(0)
    for (f in folds) {
      o <- cv_fold(f, m)
      y_all <- c(y_all, o$y); p_s <- c(p_s, o$sarar)
      p_r <- c(p_r, o$sarar_recal); p_o <- c(p_o, o$ols)
      if (!is.null(o$lambda)) lams <- c(lams, o$lambda)
    }
    r_s <- rmse(y_all - p_s); r_r <- rmse(y_all - p_r); r_o <- rmse(y_all - p_o)
    cv_rows[[length(cv_rows) + 1]] <<- data.frame(
      key = src$key, size_m = sz, variant = src$variant,
      provenance = src$provenance, scheme = "spatial_block", measure = m,
      n_folds = length(folds), n_pred = length(y_all), w_spec = CV_SPEC,
      prediction = "trend_only",
      rmse_sarar_trend = r_s, rmse_sarar_trend_recal = r_r, rmse_ols = r_o,
      mean_lambda_train = if (length(lams)) mean(lams) else NA_real_,
      stringsAsFactors = FALSE)
    add(sz, src$variant, CV_SPEC, m, "CV_spatial_block", "rmse_sarar_trend", r_s)
    add(sz, src$variant, CV_SPEC, m, "CV_spatial_block", "rmse_sarar_trend_recal", r_r)
    add(sz, src$variant, CV_SPEC, m, "CV_spatial_block", "rmse_ols", r_o)
    log_msg(ctx, sprintf("   CV %-5dm %-3s  SARAR(trend)=%.4f  recal=%.4f  OLS=%.4f",
                         sz, m, r_s, r_r, r_o))
  }
  # Null model: predict the training-block mean.
  p <- numeric(nrow(D))
  for (b in unique(blk)) { te <- which(blk == b); p[te] <- mean(D$ln_land_value[-te]) }
  nr <- rmse(D$ln_land_value - p)
  cv_rows[[length(cv_rows) + 1]] <<- data.frame(
    key = src$key, size_m = sz, variant = src$variant,
    provenance = src$provenance, scheme = "spatial_block", measure = "null_mean",
    n_folds = NB, n_pred = nrow(D), w_spec = CV_SPEC, prediction = "training_mean",
    rmse_sarar_trend = NA_real_, rmse_sarar_trend_recal = nr, rmse_ols = nr,
    mean_lambda_train = NA_real_, stringsAsFactors = FALSE)
  add(sz, src$variant, CV_SPEC, "-", "CV_spatial_block", "rmse_null_mean", nr)
  log_msg(ctx, sprintf("   CV %-5dm null (training-block mean) RMSE = %.4f", sz, nr))
  # Plot area alone, same blocks: OLS of the log outcome on log plot area,
  # fitted on the training blocks (cv_benchmarks() in common.R).
  pa <- cv_benchmarks(D, blk)[["null_plot_area_only"]]
  cv_rows[[length(cv_rows) + 1]] <<- data.frame(
    key = src$key, size_m = sz, variant = src$variant,
    provenance = src$provenance, scheme = "spatial_block",
    measure = "null_plot_area_only",
    n_folds = NB, n_pred = nrow(D), w_spec = CV_SPEC, prediction = "ols_plot_area",
    rmse_sarar_trend = NA_real_, rmse_sarar_trend_recal = pa, rmse_ols = pa,
    mean_lambda_train = NA_real_, stringsAsFactors = FALSE)
  add(sz, src$variant, CV_SPEC, "-", "CV_spatial_block", "rmse_null_plot_area_only", pa)
  log_msg(ctx, sprintf("   CV %-5dm plot area only (OLS) RMSE = %.4f", sz, pa))
  invisible(NULL)
}

# =============================================================================
# Run every source
# =============================================================================

sources <- MP$sources
prepared <- list()
for (src in sources) {
  log_msg(ctx, strrep("=", 74))
  full <- identical(src$variant, FULLV) || identical(src$provenance, "frozen")
  log_msg(ctx, sprintf("SOURCE %s  (%d m, variant '%s', %s) -- %s",
                       src$key, src$size_m, src$variant, src$provenance,
                       if (full) "full analysis" else "SARAR coefficients only"))
  P <- prepare(src)
  prepared[[src$key]] <- list(n = nrow(P$D), size_m = src$size_m,
                              variant = src$variant, provenance = src$provenance)
  analyse(P, full)
  if (full && src$size_m %in% CV_SIZES) run_cv(P)
  rm(P); invisible(gc(verbose = FALSE))
}

# =============================================================================
# The point-level anchor for the sign table (stage-3 gate, point_level_w_spec)
# =============================================================================

pl_path <- file.path(REPO, MP$point_level_reference)
pl <- read.csv(pl_path, stringsAsFactors = FALSE)
# the gate writes N on a row of its own (w_spec and model "-"), so read it
# before filtering to the SARAR rows of the anchor W
nn <- pl$value[pl$statistic == "N"][1]
if (is.na(nn)) stop("no N row in ", basename(pl_path), call. = FALSE)
pl <- pl[pl$w_spec == MP$point_level_w_spec & pl$model == "SARAR", ]
pt_rows <- list()
for (m in MEASURES) {
  b  <- pl$value[pl$measure == m & pl$statistic == "coef_measure"]
  se <- pl$value[pl$measure == m & pl$statistic == paste0("se_ln_", m)]
  if (!length(b) || !length(se)) next
  pt_rows[[length(pt_rows) + 1]] <- data.frame(
    size_m = 0L, variant = "point_level", provenance = "frozen",
    w_spec = MP$point_level_w_spec, w_role = "point_level_anchor",
    measure = m, n = as.integer(nn),
    beta_measure = b[1], se_measure = se[1], p_measure = pval(b[1], se[1]),
    sign = if (b[1] >= 0) "+" else "-",
    significant_05 = as.integer(pval(b[1], se[1]) < 0.05),
    stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("point level %-3s beta=%+.5f (se %.5f, p %.3f) -- from %s",
                       m, b[1], se[1], pval(b[1], se[1]), basename(pl_path)))
}

# =============================================================================
# Write everything
# =============================================================================

bind <- function(l) if (length(l)) do.call(rbind, l) else NULL
res       <- bind(results)
zero_tab  <- bind(zero_rows)
wsens_tab <- bind(wsens)
msel_tab  <- bind(msel)
stdz_tab  <- bind(stdz)
corr_tab  <- bind(corr)
cv_tab    <- bind(cv_rows)
var_tab   <- bind(var_rows)
wsum_tab  <- bind(wsum)
sign_tab  <- rbind(bind(pt_rows), bind(sign_rows))
sign_tab  <- sign_tab[order(sign_tab$measure, sign_tab$size_m, sign_tab$w_spec), ]

# n_cells: one row per (source, measure) with the zero-handling loss, plus the
# joint post-positivity N the models actually run on.
n_by_key <- vapply(prepared, function(x) x$n, integer(1))
zero_tab$n_analysis <- as.integer(n_by_key[zero_tab$key])
zero_tab$n_lost_to_zero_handling <-
  zero_tab$n_populated - zero_tab$n_positive_populated

w <- function(df, fn) {
  p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p); p
}
w(res,       "stage4_maup_models.csv")
w(zero_tab,  "stage4_maup_n_cells.csv")
w(corr_tab,  "stage4_maup_correlations.csv")
w(sign_tab,  "stage4_maup_sign_table.csv")
w(cv_tab,    "stage4_maup_cv_rmse.csv")
w(var_tab,   "stage4_maup_parking_variants.csv")
w(wsens_tab, "stage4_maup_w_sensitivity.csv")
w(wsum_tab,  "stage4_maup_w_summary.csv")
w(msel_tab,  "stage4_maup_model_selection.csv")
drop_tab <- bind(dropped)
if (is.null(drop_tab))
  drop_tab <- data.frame(key = character(), size_m = integer(),
                         w_spec = character(), w_role = character(),
                         reason = character(), stringsAsFactors = FALSE)
w(drop_tab,  "stage4_maup_dropped_specs.csv")
w(stdz_tab,  "stage4_maup_standardized.csv")

# ---- the manuscript table, assembled once ------------------------------------
# One row per (source, measure) under the headline matrix, carrying everything
# the manuscript's Table 6 and its surrounding sentences quote: raw and standardized
# beta, lambda/rho with the bound flag, pseudo-R2 and the block-CV RMSE. The
# 500 m row flagged `manuscript_cell = 1` is the rebuilt-DV row, i.e. the same
# cell as the revised Table 5; the frozen 500 m row rides along as the anchor.
hl_tab <- wsens_tab[wsens_tab$w_role == HEADLINE_ROLE, , drop = FALSE]
if (nrow(hl_tab)) {
  hl_tab <- merge(hl_tab,
    stdz_tab[, c("key", "w_spec", "measure", "sd_ln_measure",
                 "beta_std", "se_std")],
    by = c("key", "w_spec", "measure"), all.x = TRUE)
  if (!is.null(cv_tab)) {
    hl_tab <- merge(hl_tab, cv_tab[!cv_tab$measure %in% c("null_mean", "null_plot_area_only"),
      c("key", "measure", "rmse_sarar_trend", "rmse_sarar_trend_recal",
        "rmse_ols", "mean_lambda_train")],
      by = c("key", "measure"), all.x = TRUE)
    nullr <- cv_tab[cv_tab$measure == "null_mean",
                    c("key", "rmse_sarar_trend_recal")]
    names(nullr)[2] <- "rmse_null_mean"
    hl_tab <- merge(hl_tab, nullr, by = "key", all.x = TRUE)
    parea <- cv_tab[cv_tab$measure == "null_plot_area_only",
                    c("key", "rmse_sarar_trend_recal")]
    names(parea)[2] <- "rmse_null_plot_area_only"
    hl_tab <- merge(hl_tab, parea, by = "key", all.x = TRUE)
  }
  hl_tab$manuscript_cell <- as.integer(hl_tab$variant == FULLV)
  hl_tab <- hl_tab[order(hl_tab$size_m, hl_tab$variant, hl_tab$measure), ]
  w(hl_tab, "stage4_maup_headline.csv")
}
write_counts(ctx, "stage4_counts_maup.csv")

# ---- 500 m anchor: frozen submitted layer vs stage-2 rebuild ----------------
# Same centralities, same grid; the land value differs by the stage-1 residual.
anch <- merge(
  wsens_tab[wsens_tab$key == "500m_frozen",
            c("w_spec", "measure", "n", "beta_measure", "se_measure",
              "p_measure", "lambda_lag", "rho_error", "pseudo_r2")],
  wsens_tab[wsens_tab$key == "500m_headline",
            c("w_spec", "measure", "n", "beta_measure", "se_measure",
              "p_measure", "lambda_lag", "rho_error", "pseudo_r2")],
  by = c("w_spec", "measure"), suffixes = c("_frozen", "_rebuilt"))
if (nrow(anch)) {
  anch$d_beta  <- anch$beta_measure_rebuilt - anch$beta_measure_frozen
  anch$d_r2    <- anch$pseudo_r2_rebuilt - anch$pseudo_r2_frozen
  anch$same_sign <- as.integer(sign(anch$beta_measure_rebuilt) ==
                               sign(anch$beta_measure_frozen))
  w(anch, "stage4_maup_anchor_500m.csv")
  log_msg(ctx, sprintf("500 m anchor: max |d beta| = %.4f, max |d R2| = %.4f, signs agree in %d of %d",
                       max(abs(anch$d_beta)), max(abs(anch$d_r2)),
                       sum(anch$same_sign), nrow(anch)))
}

# ---- console digest ---------------------------------------------------------
log_msg(ctx, strrep("=", 74))
# one line per source layer, so the two 500 m layers are not summed together
b <- aggregate(error_at_bound ~ size_m + key + w_spec, data = wsens_tab,
               FUN = function(x) c(hits = sum(x), fits = length(x)))
b <- b[order(b$size_m, b$key, b$w_spec), ]
for (i in seq_len(nrow(b)))
  log_msg(ctx, sprintf("bound hits  %5d m  %-18s %-26s error %d/%d",
                       b$size_m[i], b$key[i], b$w_spec[i],
                       b$error_at_bound[i, "hits"], b$error_at_bound[i, "fits"]))
for (m in c("BC", "FK")) {
  st <- sign_tab[sign_tab$measure == m & sign_tab$variant %in%
                   c("point_level", FULLV, "frozen_submitted"), ]
  log_msg(ctx, sprintf("%s sign by size: %s", m,
    paste(sprintf("%dm/%s:%s%s", st$size_m, st$w_spec, st$sign,
                  ifelse(st$significant_05 == 1, "*", "")), collapse = "  ")))
  # The "N of N positive" count over the aggregated fits, and its
  # headline-matrix subset. The frozen 500 m layer is the anchor for the same
  # cell as the rebuilt 500 m layer, so it is left out of the count.
  agg <- st[st$size_m > 0 & st$variant == FULLV, ]
  hl  <- agg[agg$w_role == HEADLINE_ROLE, ]
  log_msg(ctx, sprintf("%s aggregated fits: %d of %d positive (%d of them significant at 5%%); headline matrix only: %d of %d positive, %d significant",
    m, sum(agg$sign == "+"), nrow(agg),
    sum(agg$sign == "+" & agg$significant_05 == 1),
    sum(hl$sign == "+"), nrow(hl),
    sum(hl$sign == "+" & hl$significant_05 == 1)))
}
finish(ctx)
