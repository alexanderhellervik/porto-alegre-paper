#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 -- revision analyses at the aggregated (500 m cell) level.
#
# `--mode full` (the default) runs the aggregated analyses below on the frozen
# 500 m hexagon layer. The address-level analyses have their own scripts: the
# W-choice sweep and block CV (`--mode w_sweep|cv|both`, src/40_w_choice.R),
# the sample sensitivity (src/47), the dependent-variable comparison (src/48),
# the extra controls and property-type split (src/49) and Table 5's LM rows
# (src/52).
#
# What `--mode full` computes:
#   1. W sensitivity   -- the submitted 6,100 m band, bands of 4,500 / 3,000 m
#                         and kNN k in {10, 50, 100, 300}: LM suite, SARAR,
#                         betas with SEs, lambda/rho, pseudo-R2, and a flag for
#                         an error parameter rho on sphet's nlminb bound.
#   2. Model selection -- Anselin's LM decision rules per (W, measure), via
#                         lm_rule() in common.R; the selected model is
#                         estimated as well as SARAR.
#   3. Standardized coefficients -- covariates z-scored, response in logs;
#                         raw and standardized betas side by side.
#   4. Spatial block CV -- 10 seeded k-means blocks on hexagon centroids, W
#                         rebuilt inside each training set, trend-only
#                         prediction; plus an OLS baseline, a training-mean
#                         null model, an OLS-on-log-plot-area-only benchmark
#                         and a random k-fold comparison.
#   5. PC density-penalty (Dp) variants -- the frozen layer's cd*_mean columns,
#                         labelled by the (gamma, beta, Dp) parameters recorded
#                         in `measures.pc_variant_parameters`.
#
# MAUP is src/45_maup.R.
#
# Every weight matrix comes from build_weights(), every filter from
# apply_exclusions(), the pseudo-R2, LM suite and decision rule from common.R,
# so these numbers are computed exactly as the stage-3 gate computes them.
#
# Run:  Rscript src/40_revision.R [aggregated]
#       Rscript src/40_revision.R --mode w_sweep [aggregated|disaggregated|both]
#       Rscript src/40_revision.R --mode cv      [aggregated|disaggregated|both]
#       Rscript src/40_revision.R --mode both    [aggregated|disaggregated|both]
#
# The last three hand off to src/40_w_choice.R, which writes its own files
# (`outputs/tables/stage4_w_choice_*.csv`) and none of this script's.
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

args  <- commandArgs(trailingOnly = TRUE)

# `--mode <m>` and `--mode=<m>` both parse. Positional argument = the level.
.mode_arg <- function(a) {
  i <- which(a == "--mode")
  if (length(i) && length(a) > i[1]) return(a[i[1] + 1L])
  eq <- grep("^--mode=", a, value = TRUE)
  if (length(eq)) return(sub("^--mode=", "", eq[1]))
  "full"
}
MODE  <- .mode_arg(args)
.pos  <- args[!grepl("^--", args)]
.pos  <- setdiff(.pos, MODE)          # drop the value of a space-separated --mode
LEVEL <- if (length(.pos)) .pos[1] else if (MODE == "full") "aggregated" else "both"

# ---- the W-choice sweep ------------------------------------------------------
if (MODE %in% c("w_sweep", "cv", "both")) {
  source(file.path(dirname(sub("^--file=", "",
    grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "40_w_choice.R"))
  quit(save = "no", status = 0)
}
if (MODE != "full")
  stop(sprintf("unknown --mode '%s' (full | w_sweep | cv | both)", MODE),
       call. = FALSE)
if (LEVEL != "aggregated")
  stop(sprintf(paste0(
    "--mode full covers the aggregated level only (got '%s'). The address-level ",
    "analyses are `make w-choice`, `make sample-sens`, `make rebuilt-dv`, ",
    "`make controls-types` and `make table5-lm`."), LEVEL), call. = FALSE)

ctx <- init(4, "revision_aggregated")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
BOUND    <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
ALPHA    <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
BOTH_ROB <- ctx$cfg$revision$lm_selection$both_robust_significant

results <- list()
add <- function(w_spec, measure, model, stat, value) {
  results[[length(results) + 1]] <<- data.frame(
    level = LEVEL, w_spec = w_spec, measure = measure, model = model,
    statistic = stat, value = as.numeric(value), stringsAsFactors = FALSE)
  invisible(NULL)
}

# ---- load, map, exclude -----------------------------------------------------

path <- cfg_path(ctx, "frozen", "cents_aggregated")
Land <- st_read(path, quiet = TRUE)

crs_expected <- ctx$cfg$repro$crs_epsg
if (is.na(st_crs(Land)$epsg) || st_crs(Land)$epsg != crs_expected) {
  log_msg(ctx, sprintf("!! CRS is %s, expected EPSG:%s -- transforming",
                       st_crs(Land)$epsg, crs_expected))
  Land <- st_transform(Land, crs_expected)
}
if (isTRUE(st_is_longlat(Land)))
  stop("layer is in geographic coordinates; distance weights would be wrong",
       call. = FALSE)
log_msg(ctx, sprintf("CRS confirmed EPSG:%s (%s)", st_crs(Land)$epsg,
                     st_crs(Land)$Name))

map <- ctx$cfg$measures$columns[[LEVEL]]
for (canon in names(map)) {
  src <- map[[canon]]
  if (!is.null(Land[[src]]) && src != canon) Land[[canon]] <- Land[[src]]
}
# The Dp variants ride along on the same sample; their parameters come from
# measures.pc_variant_parameters, keyed by the column stem.
PCV <- unlist(ctx$cfg$measures$pc_variants_aggregated)

Land <- apply_exclusions(ctx, Land, LEVEL)
target_n <- ctx$cfg$exclusions$targets[[paste0(LEVEL, "_n")]]
log_msg(ctx, sprintf("N = %d (submitted aggregated N = %d) %s", nrow(Land),
                     target_n, if (nrow(Land) == target_n) "MATCH" else "*** MISMATCH ***"))
if (nrow(Land) != target_n)
  stop("aggregated N does not match the submitted sample; stopping", call. = FALSE)

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c")
  log(x + ctx$cfg$transforms$log_plus_c) else log(x)
for (v in c("land_value", MEASURES, "plot_area"))
  Land[[paste0("ln_", v)]] <- lg(Land[[v]])
log_msg(ctx, sprintf("zero handling = %s (submitted analysis: exclude)", zh))

pts    <- suppressWarnings(st_point_on_surface(Land))
coords <- st_coordinates(pts)[, 1:2]
D      <- st_drop_geometry(Land)          # modelling frame; geometry not needed
CTRL   <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
frm <- function(m) as.formula(sprintf("ln_land_value ~ %sln_%s", CTRL, m))

# =============================================================================
# 1-3. W sensitivity, model selection by LM rules, standardized coefficients
# =============================================================================

SPECS <- unlist(ctx$cfg$weights$sensitivity$specs[[LEVEL]])
log_msg(ctx, "W specs swept: ", paste(SPECS, collapse = ", "))

wsens <- list(); msel <- list(); stdz <- list()

for (ws in SPECS) {
  log_msg(ctx, strrep("=", 74))
  spec <- ctx$cfg$weights[[ws]]
  keep <- connected_idx(ctx, coords, ws)
  if (length(keep) < nrow(coords))
    log_msg(ctx, sprintf("!! spec '%s' isolates %d unit(s); dropped (weights.band_islands)",
                         ws, nrow(coords) - length(keep)))
  dat <- D[keep, , drop = FALSE]
  W   <- build_weights(ctx, coords[keep, , drop = FALSE], spec_name = ws)
  count_step(ctx, paste0("W ", ws), nrow(dat),
             sprintf("%s, style=%s", if (identical(spec$type, "knn"))
               paste0("kNN k=", spec$k) else paste0("band d<=", spec$d_max_m, "m"),
               spec$style))
  add(ws, "-", "-", "N", nrow(dat))

  for (m in MEASURES) {
    form <- frm(m)
    log_msg(ctx, sprintf("-- %-3s  %s", m, deparse(form)))

    # --- LM suite (depends on W, not on the GMM fit) -------------------------
    lt <- run_lm_tests(form, dat, W)
    if (!is.null(lt)) for (nm in names(lt)) add(ws, m, "LM", nm, lt[[nm]])

    sel <- lm_rule(lt, ALPHA, BOTH_ROB)
    if (is.null(lt))                     # keep the table's shape when the suite fails
      lt <- setNames(rep(NA_real_, 8), c("lm_err", "lm_lag", "rlm_err", "rlm_lag",
                                         "lm_err_p", "lm_lag_p", "rlm_err_p", "rlm_lag_p"))
    add(ws, m, "LM", "selected_is_sarar", as.numeric(identical(sel[["model"]], "sarar")))

    # --- SARAR, the estimator of the submitted tables -------------------------
    fit <- fit_spreg(ctx, form, dat, W, "sarar")
    if (is.null(fit)) next
    z <- sarar_row(fit)
    for (nm in names(z$co)) {
      add(ws, m, "SARAR", paste0("coef_", nm), z$co[[nm]])
      add(ws, m, "SARAR", paste0("se_",   nm), z$se[[nm]])
      add(ws, m, "SARAR", paste0("p_",    nm), z$p[[nm]])
    }
    # Only the error parameter rho is bounded (nlminb, +-BOUND).
    at_bound <- abs(abs(z$co[["rho"]]) - BOUND) < 1e-6
    add(ws, m, "SARAR", "error_at_bound", as.numeric(at_bound))
    r2 <- pseudo_r2(fit, dat$ln_land_value)
    add(ws, m, "SARAR", "pseudo_r2", r2)

    log_msg(ctx, sprintf("   Area=%+.4f (se %.4f)  beta=%+.4f (se %.4f)  lambda=%+.4f  rho=%+.4f%s  R2=%.4f",
      z$co[["ln_plot_area"]], z$se[["ln_plot_area"]],
      z$co[[paste0("ln_", m)]], z$se[[paste0("ln_", m)]],
      z$co[["lambda"]], z$co[["rho"]],
      if (at_bound) "  *** rho AT nlminb BOUND ***" else "", r2))

    wsens[[length(wsens) + 1]] <- data.frame(
      w_spec = ws, w_type = spec$type,
      w_param = if (identical(spec$type, "knn")) paste0("k=", spec$k)
                else paste0("d<=", spec$d_max_m, "m"),
      n = nrow(dat), measure = m,
      beta_area = z$co[["ln_plot_area"]], se_area = z$se[["ln_plot_area"]],
      beta_measure = z$co[[paste0("ln_", m)]],
      se_measure = z$se[[paste0("ln_", m)]],
      p_measure = z$p[[paste0("ln_", m)]],
      lambda_lag = z$co[["lambda"]], se_lambda = z$se[["lambda"]],
      rho_error = z$co[["rho"]], se_rho = z$se[["rho"]],
      error_at_bound = as.integer(at_bound),
      pseudo_r2 = r2, stringsAsFactors = FALSE)

    # --- the model the LM rules actually select ------------------------------
    selm <- sel[["model"]]
    sel_r2 <- NA_real_; sel_beta <- NA_real_; sel_se <- NA_real_
    sel_lambda <- NA_real_; sel_rho <- NA_real_
    if (identical(selm, "sarar")) {
      sel_r2 <- r2; sel_beta <- z$co[[paste0("ln_", m)]]
      sel_se <- z$se[[paste0("ln_", m)]]
      sel_lambda <- z$co[["lambda"]]; sel_rho <- z$co[["rho"]]
    } else if (identical(selm, "ols")) {
      o  <- lm(form, data = dat); so <- summary(o)$coefficients
      sel_r2 <- pseudo_r2(o, dat$ln_land_value)
      sel_beta <- so[paste0("ln_", m), 1]; sel_se <- so[paste0("ln_", m), 2]
      for (nm in rownames(so)) {
        add(ws, m, "OLS", paste0("coef_", nm), so[nm, 1])
        add(ws, m, "OLS", paste0("se_", nm),   so[nm, 2])
      }
      add(ws, m, "OLS", "pseudo_r2", sel_r2)
    } else if (!is.na(selm)) {
      smod <- if (identical(selm, "sem")) "error" else "lag"
      sf2  <- fit_spreg(ctx, form, dat, W, smod)
      if (!is.null(sf2)) {
        z2 <- sarar_row(sf2)
        lbl <- toupper(selm)
        for (nm in names(z2$co)) {
          add(ws, m, lbl, paste0("coef_", nm), z2$co[[nm]])
          add(ws, m, lbl, paste0("se_",   nm), z2$se[[nm]])
        }
        sel_r2 <- pseudo_r2(sf2, dat$ln_land_value)
        add(ws, m, lbl, "pseudo_r2", sel_r2)
        sel_beta <- z2$co[[paste0("ln_", m)]]
        sel_se   <- z2$se[[paste0("ln_", m)]]
        # sphet names the error parameter `rho` and the lag parameter `lambda`;
        # only rho is bounded, so only the error model gets a bound flag.
        if (identical(selm, "sem")) {
          sel_rho <- z2$co[["rho"]]
          add(ws, m, lbl, "error_at_bound",
              as.numeric(abs(abs(sel_rho) - BOUND) < 1e-6))
        } else sel_lambda <- z2$co[["lambda"]]
      }
    }

    msel[[length(msel) + 1]] <- data.frame(
      w_spec = ws, measure = m, n = nrow(dat),
      lm_err = lt[["lm_err"]], lm_err_p = lt[["lm_err_p"]],
      lm_lag = lt[["lm_lag"]], lm_lag_p = lt[["lm_lag_p"]],
      rlm_err = lt[["rlm_err"]], rlm_err_p = lt[["rlm_err_p"]],
      rlm_lag = lt[["rlm_lag"]], rlm_lag_p = lt[["rlm_lag_p"]],
      selected = selm, basis = sel[["basis"]],
      selected_beta = sel_beta, selected_se = sel_se,
      selected_lambda = sel_lambda, selected_rho = sel_rho,
      selected_pseudo_r2 = sel_r2,
      sarar_beta = z$co[[paste0("ln_", m)]], sarar_pseudo_r2 = r2,
      stringsAsFactors = FALSE)
    log_msg(ctx, sprintf("   LM rules select %-5s (%s)", toupper(format(selm)), sel[["basis"]]))

    # --- standardized coefficients (covariates z-scored, response in logs) ---
    if (isTRUE(ctx$cfg$revision$standardize_coefficients)) {
      dz <- dat
      sd_area <- sd(dat$ln_plot_area); sd_m <- sd(dat[[paste0("ln_", m)]])
      dz$ln_plot_area <- as.numeric(scale(dat$ln_plot_area))
      dz[[paste0("ln_", m)]] <- as.numeric(scale(dat[[paste0("ln_", m)]]))
      fz <- fit_spreg(ctx, form, dz, W, "sarar")
      if (!is.null(fz)) {
        zz <- sarar_row(fz)
        for (nm in names(zz$co)) {
          add(ws, m, "SARAR_std", paste0("coef_", nm), zz$co[[nm]])
          add(ws, m, "SARAR_std", paste0("se_",   nm), zz$se[[nm]])
        }
        add(ws, m, "SARAR_std", "pseudo_r2", pseudo_r2(fz, dz$ln_land_value))
        stdz[[length(stdz) + 1]] <- data.frame(
          w_spec = ws, measure = m, n = nrow(dat),
          sd_ln_measure = sd_m,
          beta_raw = z$co[[paste0("ln_", m)]], se_raw = z$se[[paste0("ln_", m)]],
          beta_std = zz$co[[paste0("ln_", m)]], se_std = zz$se[[paste0("ln_", m)]],
          beta_std_analytic = z$co[[paste0("ln_", m)]] * sd_m,
          area_beta_raw = z$co[["ln_plot_area"]],
          area_beta_std = zz$co[["ln_plot_area"]],
          sd_ln_plot_area = sd_area,
          stringsAsFactors = FALSE)
      }
    }
  }
}

wsens <- do.call(rbind, wsens)
msel  <- do.call(rbind, msel)
stdz  <- do.call(rbind, stdz)

# =============================================================================
# 4. Spatial block k-fold CV (+ OLS baseline, + random k-fold comparison)
# =============================================================================

cvc      <- ctx$cfg$revision$spatial_cv
CV_SPEC  <- cvc$w_spec[[LEVEL]]
NB       <- as.integer(cvc$n_blocks)
NRF      <- as.integer(cvc$n_folds_random)
RECAL    <- isTRUE(cvc$intercept_recalibration)

log_msg(ctx, strrep("=", 74))
log_msg(ctx, sprintf("Spatial block CV: %d k-means blocks on the coordinates, W = '%s' rebuilt inside every training set",
                     NB, CV_SPEC))
log_msg(ctx, sprintf("prediction = %s  (X_test %%*%% beta_train%s)", cvc$prediction,
                     if (RECAL) "; intercept recalibrated on the training residual mean" else ""))
if (!identical(cvc$prediction, "trend_only"))
  stop("only revision.spatial_cv.prediction = 'trend_only' is implemented",
       call. = FALSE)

set.seed(as.integer(ctx$cfg$repro$seed))       # blocks must be reproducible
km <- kmeans(coords, centers = NB, nstart = 25, iter.max = 100)
blk <- km$cluster
log_msg(ctx, "block sizes: ", paste(as.integer(table(blk)), collapse = " "))
for (b in seq_len(NB)) add("-", "-", "CV_blocks", sprintf("block_%02d_n", b), sum(blk == b))

set.seed(as.integer(ctx$cfg$repro$seed) + 1L)
rfold <- sample(rep_len(seq_len(NRF), nrow(D)))

# One fold of one scheme: returns predictions for the held-out rows.
cv_fold <- function(test_idx, m, kind) {
  tr <- setdiff(seq_len(nrow(D)), test_idx)
  # W is rebuilt on the TRAINING coordinates only -- no test unit ever enters
  # any neighbour list, so nothing leaks through W.
  keep_tr <- tr[connected_idx(ctx, coords[tr, , drop = FALSE], CV_SPEC)]
  dtr <- D[keep_tr, , drop = FALSE]
  Wtr <- build_weights(ctx, coords[keep_tr, , drop = FALSE], spec_name = CV_SPEC)
  form <- frm(m)
  Xte  <- model.matrix(form, data = D[test_idx, , drop = FALSE])
  yte  <- D$ln_land_value[test_idx]
  Xtr  <- model.matrix(form, data = dtr)

  out <- list(y = yte)
  if (kind %in% c("sarar", "both")) {
    f <- fit_spreg(ctx, form, dtr, Wtr, "sarar")
    if (is.null(f)) { out$sarar <- rep(NA_real_, length(yte))
    } else {
      b <- sarar_row(f)$co
      b <- b[colnames(Xte)]                       # trend coefficients only
      out$sarar <- as.numeric(Xte %*% b)
      out$sarar_recal <- out$sarar +
        (mean(dtr$ln_land_value) - mean(as.numeric(Xtr %*% b)))
      out$lambda <- sarar_row(f)$co[["lambda"]]
    }
  }
  if (kind %in% c("ols", "both")) {
    o <- lm(form, data = dtr)
    out$ols <- as.numeric(predict(o, newdata = D[test_idx, , drop = FALSE]))
  }
  out
}

rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

cv_rows <- list()
for (scheme in c("spatial_block", if (isTRUE(cvc$random_kfold_comparison)) "random_kfold")) {
  folds <- if (scheme == "spatial_block") split(seq_len(nrow(D)), blk) else
                                          split(seq_len(nrow(D)), rfold)
  for (m in MEASURES) {
    p_sarar <- p_recal <- p_ols <- y_all <- numeric(0)
    lams <- numeric(0)
    for (f in folds) {
      o <- cv_fold(f, m, "both")
      y_all   <- c(y_all, o$y)
      p_sarar <- c(p_sarar, o$sarar)
      p_recal <- c(p_recal, o$sarar_recal %||% rep(NA_real_, length(o$y)))
      p_ols   <- c(p_ols, o$ols)
      if (!is.null(o$lambda)) lams <- c(lams, o$lambda)
    }
    r_s <- rmse(y_all - p_sarar); r_r <- rmse(y_all - p_recal)
    r_o <- rmse(y_all - p_ols)
    cv_rows[[length(cv_rows) + 1]] <- data.frame(
      scheme = scheme, measure = m, n_folds = length(folds), n_pred = length(y_all),
      w_spec = CV_SPEC, prediction = "trend_only",
      rmse_sarar_trend = r_s, rmse_sarar_trend_recal = r_r, rmse_ols = r_o,
      mean_lambda_train = if (length(lams)) mean(lams) else NA_real_,
      stringsAsFactors = FALSE)
    add(CV_SPEC, m, paste0("CV_", scheme), "rmse_sarar_trend", r_s)
    add(CV_SPEC, m, paste0("CV_", scheme), "rmse_sarar_trend_recal", r_r)
    add(CV_SPEC, m, paste0("CV_", scheme), "rmse_ols", r_o)
    log_msg(ctx, sprintf("CV %-13s %-3s  RMSE SARAR(trend)=%.4f  recal=%.4f  OLS=%.4f",
                         scheme, m, r_s, r_r, r_o))
  }
}
cvres <- do.call(rbind, cv_rows)
# Null model: predict the training mean. Anchors "is any of this useful?".
null_rmse <- local({
  p <- numeric(nrow(D)); for (b in unique(blk)) {
    te <- which(blk == b); p[te] <- mean(D$ln_land_value[-te]) }
  rmse(D$ln_land_value - p)
})
add(CV_SPEC, "-", "CV_spatial_block", "rmse_null_mean", null_rmse)
log_msg(ctx, sprintf("CV spatial_block null (training mean) RMSE = %.4f", null_rmse))
# Plot area alone on the same blocks: OLS of the log outcome on log plot area,
# fitted on the training blocks (cv_benchmarks() in common.R).
parea_rmse <- cv_benchmarks(D, blk)[["null_plot_area_only"]]
add(CV_SPEC, "-", "CV_spatial_block", "rmse_null_plot_area_only", parea_rmse)
log_msg(ctx, sprintf("CV spatial_block plot area only (OLS) RMSE = %.4f", parea_rmse))

# =============================================================================
# 5. PC density-penalty (Dp) variants, aggregated
# =============================================================================

log_msg(ctx, strrep("=", 74))
log_msg(ctx, "PC Dp variants -- labelled by the parameters in measures.pc_variant_parameters")

vcols <- PCV[vapply(PCV, function(cn) !is.null(Land[[cn]]), logical(1))]
# (gamma, beta, Dp) of a variant column, from the parameter table keyed by the
# segment-level column stem (the cell layer adds an aggregation suffix).
PARAMS <- ctx$cfg$measures$pc_variant_parameters
param_of <- function(col, field) {
  p <- PARAMS[[sub("_(mean|sum)$", "", col)]]
  if (is.null(p) || is.null(p[[field]])) NA_real_ else as.numeric(p[[field]])
}
for (vn in names(vcols))
  log_msg(ctx, sprintf("   %-4s %-18s gamma=%g beta=%g Dp=%g", vn, vcols[[vn]],
                       param_of(vcols[[vn]], "gamma"), param_of(vcols[[vn]], "beta"),
                       param_of(vcols[[vn]], "dp")))
pcv_rows <- list()
keep_h <- connected_idx(ctx, coords, CV_SPEC)
Wh <- build_weights(ctx, coords[keep_h, , drop = FALSE], spec_name = CV_SPEC)
Dh <- D[keep_h, , drop = FALSE]

ranks <- list()
for (vn in names(vcols)) {
  cn <- vcols[[vn]]
  v  <- D[[cn]]
  ok <- !is.na(v) & v > 0
  count_step(ctx, paste0("PC variant ", cn), sum(ok),
             sprintf("positive on the %d-cell sample", nrow(D)))
  pear <- cor(D$ln_land_value[ok], lg(v[ok]), method = "pearson")
  spea <- cor(D$land_value[ok], v[ok], method = "spearman")
  ranks[[cn]] <- ifelse(ok, rank(ifelse(ok, v, NA), na.last = "keep"), NA)

  dh <- Dh; dh$ln_variant <- lg(dh[[cn]])
  okh <- is.finite(dh$ln_variant)
  fitv <- fit_spreg(ctx, as.formula(sprintf("ln_land_value ~ %sln_variant", CTRL)),
                    dh[okh, , drop = FALSE], Wh, "sarar")
  bb <- se <- lam <- rr <- r2 <- NA_real_; ab <- NA_integer_
  if (!is.null(fitv) && sum(okh) == nrow(dh)) {
    zz <- sarar_row(fitv)
    bb <- zz$co[["ln_variant"]]; se <- zz$se[["ln_variant"]]
    lam <- zz$co[["lambda"]]; rr <- zz$co[["rho"]]
    r2 <- pseudo_r2(fitv, dh$ln_land_value[okh])
    ab <- as.integer(abs(abs(rr) - BOUND) < 1e-6)
    add(CV_SPEC, cn, "SARAR_pc_variant", "coef_measure", bb)
    add(CV_SPEC, cn, "SARAR_pc_variant", "se_measure", se)
    add(CV_SPEC, cn, "SARAR_pc_variant", "coef_lambda", lam)
    add(CV_SPEC, cn, "SARAR_pc_variant", "coef_rho", rr)
    add(CV_SPEC, cn, "SARAR_pc_variant", "pseudo_r2", r2)
    add(CV_SPEC, cn, "SARAR_pc_variant", "error_at_bound", ab)
  }
  add("-", cn, "PC_variant", "pearson_ln", pear)
  add("-", cn, "PC_variant", "spearman_raw", spea)
  pcv_rows[[length(pcv_rows) + 1]] <- data.frame(
    variant_key = vn, column = cn,
    gamma = param_of(cn, "gamma"), beta = param_of(cn, "beta"), dp = param_of(cn, "dp"),
    n_positive = sum(ok), pearson_ln_landvalue = pear,
    spearman_landvalue = spea, sd_ln = sd(lg(v[ok])),
    sarar_beta = bb, sarar_se = se, sarar_beta_std = bb * sd(lg(v[ok])),
    lambda_lag = lam, rho_error = rr, error_at_bound = ab, pseudo_r2 = r2,
    stringsAsFactors = FALSE)
}
pcv <- do.call(rbind, pcv_rows)

# Spearman rank stability between variants, pairwise, on the common sample.
rk <- as.data.frame(ranks)
rank_stab <- suppressWarnings(cor(rk, use = "pairwise.complete.obs",
                                  method = "spearman"))
rs_long <- data.frame(
  a = rep(rownames(rank_stab), times = ncol(rank_stab)),
  b = rep(colnames(rank_stab), each = nrow(rank_stab)),
  spearman = as.numeric(rank_stab), stringsAsFactors = FALSE)
rs_long <- rs_long[rs_long$a < rs_long$b, ]
for (i in seq_len(nrow(rs_long)))
  add("-", paste(rs_long$a[i], rs_long$b[i], sep = "|"), "PC_variant",
      "spearman_rank_stability", rs_long$spearman[i])
for (i in seq_len(nrow(rs_long)))
  log_msg(ctx, sprintf("   rank stability %-16s vs %-16s rho = %.4f",
                       rs_long$a[i], rs_long$b[i], rs_long$spearman[i]))

# =============================================================================
# Write everything
# =============================================================================

res <- do.call(rbind, results)
w <- function(df, fn) { p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
                        log_msg(ctx, "wrote ", p); p }

w(res,       "stage4_aggregated_models.csv")
w(wsens,     "stage4_aggregated_w_sensitivity.csv")
w(msel,      "stage4_aggregated_model_selection.csv")
w(stdz,      "stage4_aggregated_standardized.csv")
w(cvres,     "stage4_aggregated_cv_rmse.csv")
w(pcv,       "stage4_aggregated_pc_variants.csv")
w(rs_long,   "stage4_aggregated_pc_rank_stability.csv")
write_counts(ctx, "stage4_counts_aggregated.csv")

# ---- console digest ---------------------------------------------------------
log_msg(ctx, strrep("=", 74))
bnd <- aggregate(error_at_bound ~ w_spec, data = wsens, FUN = sum)
for (i in seq_len(nrow(bnd)))
  log_msg(ctx, sprintf("rho at bound  %-26s %d/%d", bnd$w_spec[i],
                       bnd$error_at_bound[i], length(MEASURES)))
log_msg(ctx, "MAUP (250 m / 1,000 m hexagons) is a separate run: ",
        "Rscript src/45_maup.R (make maup) -- outputs/tables/stage4_maup_*.csv")
finish(ctx)
