#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — floor price vs development density: the channel from centrality to
# the dependent variable, mapped and quantified by property type.
#
# Why this run exists. src/49_controls_types.R shows that with ln built area in
# the model the dependent variable behaves like building intensity times a
# price per m2 of floor space. This run turns that into an exact
# decomposition and asks whether there is a spatial pattern in apartment floor
# prices at all.
#
# The identity. Stage 1 builds Puni = price_stat * U / A. Define
#
#   FP(a)  = price_stat(a) / (B(a) / U(a))     floor price, BRL per m2 of floor
#   DEN(a) = B(a) / A(a)                       built floor area per m2 of plot
#
# with B(a) the registry built area summed over the counted units (as in
# src/49_built_area.py), U(a) stage 1's unit multiplier and A(a) the plot area.
# Then
#
#   ln Puni = ln FP + ln DEN                   exactly, by construction
#
# and the run measures the residual rather than asserting it. Every regression
# of a component on ln PC1 is therefore a decomposition of the same gross
# effect: under OLS on a common right-hand side the two component slopes add to
# the slope on ln Puni to machine precision, so the split into "through density"
# and "through floor price" is exact, not approximate. Under SARAR they only
# add approximately, because each equation gets its own lambda and rho; both
# are reported and the gap is stated.
#
# Sample. FP and DEN need a determinate built area, so this runs on the
# determinate subsample of src/49_controls_types.R. Vacant plots have no
# building and effectively leave, so a group is estimated here when it has
# n >= revision.property_type_split.min_n_per_group determinate records, a
# stricter floor than the property-type split's.
#
# Everything else is held at the specification of src/48_rebuilt_dv.R and
# src/49_controls_types.R:
#   - dependent variable: the rebuilt headline column;
#   - exclusions: the submitted order, positivity -> 1.5xIQR;
#   - W: the proposed point-level W (kNN k = 8, row-standardised, configured
#     tie-break), and for a group the list is rebuilt on the group's own points;
#   - estimator: sphet::spreg(model = "sarar", het per config);
#   - standardized beta by re-estimation on z-scored covariates.
#
# COMPUTE. `compute.max_workers`, further capped by PIPELINE_MAX_WORKERS.
#
# Run:  Rscript src/50_floor_price.R            (or `make floor-price`)
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet); library(parallel)
  library(mgcv); library(ggplot2); library(cowplot)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "floor_price")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
DV       <- ctx$cfg$dependent_variable
CCF      <- ctx$cfg$revision$controls
PTS      <- ctx$cfg$revision$property_type_split
W_SPEC   <- ctx$cfg$weights$proposed$disaggregated
K        <- as.integer(ctx$cfg$weights[[W_SPEC]]$k)
ALPHA    <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
MIN_N    <- as.integer(PTS$min_n_per_group)
BOUND    <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
BEPS     <- 1e-6

# The smooth-surface basis. One number, held constant across every group so the
# R2 values are comparable; stated in the report rather than tuned per group.
TPS_K    <- 100L

NWORK <- max(1L, min(as.integer(ctx$cfg$compute$max_workers),
                     as.integer(ctx$cfg$compute$host_ceiling)))
ovr <- Sys.getenv("PIPELINE_MAX_WORKERS", "")
if (nzchar(ovr) && !is.na(suppressWarnings(as.integer(ovr)))) {
  NWORK <- max(1L, min(NWORK, as.integer(ovr)))
  log_msg(ctx, sprintf("workers capped to %d by PIPELINE_MAX_WORKERS",
                       NWORK))
}

log_msg(ctx, sprintf("dependent variable: rebuilt; W '%s' (kNN k = %d); tie-break '%s'; het = %s; alpha = %.2f; TPS basis k = %d",
                     W_SPEC, K,
                     ctx$cfg$weights$identical_points$handling,
                     isTRUE(ctx$cfg$models$het), ALPHA, TPS_K))
log_msg(ctx, sprintf("workers = %d", NWORK))

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c") log(x + ctx$cfg$transforms$log_plus_c) else log(x)

# =============================================================================
# 1. The analysis frame — identical assembly to src/49_controls_types.R
# =============================================================================

L <- st_read(cfg_path(ctx, "frozen", "cents_disaggregated"), quiet = TRUE)
if (is.na(st_crs(L)$epsg) || st_crs(L)$epsg != ctx$cfg$repro$crs_epsg)
  L <- st_transform(L, ctx$cfg$repro$crs_epsg)
if (isTRUE(st_is_longlat(L))) stop("layer is in geographic coordinates", call. = FALSE)
log_msg(ctx, sprintf("frozen point layer: %d records, EPSG:%s",
                     nrow(L), st_crs(L)$epsg))

p_dv <- file.path(REPO, DV$rebuilt_path)
if (!file.exists(p_dv))
  stop("stage-1 output missing: ", p_dv, " -- run `make dv` first", call. = FALSE)
RB <- read.csv(p_dv, stringsAsFactors = FALSE)
if (nrow(RB) != nrow(L) || !all(RB$id == L$id))
  stop("the rebuilt DV file is not id-aligned with the frozen layer", call. = FALSE)
if (max(abs(RB$Preco_frozen - L$Preco), na.rm = TRUE) > 1e-6)
  stop("the stage-1 file's `Preco_frozen` does not match the layer's `Preco`",
       call. = FALSE)

p_ct <- file.path(REPO, CCF$path)
if (!file.exists(p_ct))
  stop("controls file missing: ", p_ct,
       " -- run `python src/49_built_area.py` first", call. = FALSE)
CTL <- read.csv(p_ct, stringsAsFactors = FALSE)
if (nrow(CTL) != nrow(L) || !all(CTL$id == L$id))
  stop("the controls file is not id-aligned with the frozen layer", call. = FALSE)
if (max(abs(CTL$n_units - RB$n_units), na.rm = TRUE) > 0)
  stop("the controls file's `n_units` is not stage 1's", call. = FALSE)

L$Preco <- RB$Preco
L$Puni  <- RB$Puni
L$price_stat             <- RB$price_stat
L$unit_count             <- CTL$n_units
L$built_area_units       <- CTL$built_area_units
L$built_area_sold_mean   <- CTL$built_area_sold_mean
L$built_area_determinate <- CTL$built_area_determinate

MAP <- ctx$cfg$measures$columns$disaggregated
for (cn in names(MAP)) {
  srcn <- MAP[[cn]]
  if (!is.null(L[[srcn]]) && srcn != cn) L[[cn]] <- L[[srcn]]
}
L$built_area <- NULL   # the layer's per-record Area_Const is not our B(a)

E <- apply_exclusions(ctx, L, "disaggregated")
log_msg(ctx, sprintf("analysis sample (rebuilt DV, submitted exclusion order): N = %d",
                     nrow(E)))

D  <- st_drop_geometry(E)
XY <- st_coordinates(E)[, 1:2]
D$X <- XY[, 1]; D$Y <- XY[, 2]
for (v in c("land_value", MEASURES, "plot_area")) D[[paste0("ln_", v)]] <- lg(D[[v]])

# ---- the stage-1 identity, re-measured here so this report can quote it -----
ident_dv <- max(abs(D$ln_land_value -
                    (log(D$price_stat) + log(D$unit_count) - log(D$plot_area))),
                na.rm = TRUE)
log_msg(ctx, sprintf("stage-1 identity: max |ln Puni - (ln price_stat + ln U - ln A)| = %.3g over %d records",
                     ident_dv, nrow(D)))

# =============================================================================
# 2. FP, DEN and the decomposition identity
# =============================================================================

det <- D$built_area_determinate == 1 & is.finite(D$built_area_units) &
       D$built_area_units > 0
DD  <- D[det, , drop = FALSE]
count_step(ctx, "floor price: determinate built area", nrow(DD),
           sprintf("%d of %d records leave; rule '%s'", sum(!det), nrow(D),
                   CCF$built_area$determinacy))

DD$B   <- DD$built_area_units
DD$U   <- DD$unit_count
DD$A   <- DD$plot_area
DD$BPU <- DD$B / DD$U                      # mean built area of a counted unit
DD$FP  <- DD$price_stat / DD$BPU           # BRL per m2 of floor
DD$DEN <- DD$B / DD$A                      # m2 of floor per m2 of plot
DD$ln_FP  <- log(DD$FP)
DD$ln_DEN <- log(DD$DEN)
DD$ln_BPU <- log(DD$BPU)

# the alternative floor-price denominator: the built area of the units that
# actually sold (the ITBI mean), not of every counted unit at the address.
DD$FP_sold    <- ifelse(is.finite(DD$built_area_sold_mean) &
                        DD$built_area_sold_mean > 0,
                        DD$price_stat / DD$built_area_sold_mean, NA_real_)
DD$ln_FP_sold <- log(DD$FP_sold)

ident_fp <- max(abs(DD$ln_land_value - (DD$ln_FP + DD$ln_DEN)), na.rm = TRUE)
log_msg(ctx, sprintf("DECOMPOSITION IDENTITY: max |ln Puni - (ln FP + ln DEN)| = %.3g over %d records (mean %.3g)",
                     ident_fp, nrow(DD),
                     mean(abs(DD$ln_land_value - (DD$ln_FP + DD$ln_DEN)))))
if (ident_fp > 1e-9)
  stop(sprintf("the decomposition identity does not hold to 1e-9 (max residual %.3g) -- FP or DEN is mis-specified",
               ident_fp), call. = FALSE)

XYD <- as.matrix(DD[, c("X", "Y")])

# =============================================================================
# 3. Groups
# =============================================================================

grp <- do.call(rbind, lapply(sort(unique(D$Type)), function(g) {
  i <- D$Type == g; j <- DD$Type == g
  data.frame(group = g, n_analysis = sum(i), n_determinate = sum(j),
             pct_determinate = 100 * sum(j) / sum(i),
             estimated = sum(j) >= MIN_N,
             median_FP = if (sum(j)) median(DD$FP[j]) else NA_real_,
             median_DEN = if (sum(j)) median(DD$DEN[j]) else NA_real_,
             median_BPU = if (sum(j)) median(DD$BPU[j]) else NA_real_,
             var_ln_FP = if (sum(j) > 1) var(DD$ln_FP[j]) else NA_real_,
             var_ln_DEN = if (sum(j) > 1) var(DD$ln_DEN[j]) else NA_real_,
             var_ln_Puni = if (sum(j) > 1) var(DD$ln_land_value[j]) else NA_real_,
             stringsAsFactors = FALSE)
}))
grp <- grp[order(-grp$n_determinate), ]
GROUPS <- grp$group[grp$estimated]
log_msg(ctx, sprintf("groups with >= %d DETERMINATE records: %s; not estimated: %s",
                     MIN_N, paste(GROUPS, collapse = "; "),
                     paste(grp$group[!grp$estimated], collapse = "; ")))
for (i in seq_len(nrow(grp)))
  log_msg(ctx, sprintf("   %-40s analysis %5d  determinate %5d (%.1f %%)  var(ln DEN) %s",
                       grp$group[i], grp$n_analysis[i], grp$n_determinate[i],
                       grp$pct_determinate[i],
                       if (is.na(grp$var_ln_DEN[i])) "  --" else
                         sprintf("%.4f", grp$var_ln_DEN[i])))

# The analysis units: pooled (every determinate record) plus each group.
UNITS <- c(list("(pooled)" = rep(TRUE, nrow(DD))),
           setNames(lapply(GROUPS, function(g) DD$Type == g), GROUPS))

# W per unit: the k = 8 list rebuilt on the unit's own points.
# `dd` is already in tie-break key order, so every subset inherits it.
WL <- lapply(names(UNITS), function(u) {
  i <- UNITS[[u]]
  suppressWarnings(build_weights(ctx, XYD[i, , drop = FALSE], spec_name = W_SPEC))
})
names(WL) <- names(UNITS)

# =============================================================================
# 4. "Is there even a spatial pattern?" — Moran's I, a smooth surface, and the
#    correlation with centrality, per unit and per variable
# =============================================================================

VARS <- c(ln_FP = "ln FP (floor price)", ln_DEN = "ln DEN (density)",
          ln_land_value = "ln Puni (the dependent variable)")

# 500 m hex-cell membership, for the fixed-effects cross-check of the smooth
# surface. The frozen 821-cell lattice is the study area (grid.maup.study_area).
HEX <- st_read(cfg_path(ctx, "frozen", "cents_aggregated"), quiet = TRUE)
if (is.na(st_crs(HEX)$epsg) || st_crs(HEX)$epsg != ctx$cfg$repro$crs_epsg)
  HEX <- st_transform(HEX, ctx$cfg$repro$crs_epsg)
PTS_SF <- st_as_sf(DD[, c("X", "Y")], coords = c("X", "Y"),
                   crs = ctx$cfg$repro$crs_epsg)
.wi <- suppressMessages(st_within(PTS_SF, HEX, sparse = TRUE))
DD$hex_id <- HEX$id[vapply(.wi, function(z) if (length(z)) z[1] else NA_integer_,
                           integer(1))]
log_msg(ctx, sprintf("500 m hex membership: %d of %d determinate points fall in one of the %d frozen cells (%d outside)",
                     sum(!is.na(DD$hex_id)), nrow(DD), nrow(HEX),
                     sum(is.na(DD$hex_id))))

spat_row <- function(u, vname) {
  i <- UNITS[[u]]; d <- DD[i, , drop = FALSE]; y <- d[[vname]]
  W <- WL[[u]]
  mt <- tryCatch(spdep::moran.test(y, W, randomisation = TRUE,
                                   alternative = "two.sided", zero.policy = FALSE),
                 error = function(e) NULL)
  # thin-plate spline on the projected coordinates (in km, for conditioning)
  d$xk <- d$X / 1000; d$yk <- d$Y / 1000
  g <- tryCatch(mgcv::gam(as.formula(paste(vname, "~ s(xk, yk, bs = 'tp', k =", TPS_K, ")")),
                          data = d, method = "REML"),
                error = function(e) NULL)
  # 500 m hex fixed effects, on the points that fall in a cell
  hs <- !is.na(d$hex_id)
  hm <- if (sum(hs) > 10 && length(unique(d$hex_id[hs])) > 1)
    summary(lm(d[[vname]][hs] ~ factor(d$hex_id[hs]))) else NULL
  data.frame(
    unit = u, variable = vname, n = nrow(d),
    sd = sd(y), var = var(y),
    moran_I = if (is.null(mt)) NA_real_ else unname(mt$estimate[["Moran I statistic"]]),
    moran_E = if (is.null(mt)) NA_real_ else unname(mt$estimate[["Expectation"]]),
    moran_sd = if (is.null(mt)) NA_real_ else sqrt(unname(mt$estimate[["Variance"]])),
    moran_z = if (is.null(mt)) NA_real_ else unname(mt$statistic),
    moran_p = if (is.null(mt)) NA_real_ else mt$p.value,
    tps_r2_adj = if (is.null(g)) NA_real_ else summary(g)$r.sq,
    tps_dev_expl = if (is.null(g)) NA_real_ else summary(g)$dev.expl,
    tps_edf = if (is.null(g)) NA_real_ else sum(summary(g)$edf),
    # the size of the spatial signal in the variable's own log units, not just
    # its share: R2 x var. A variable can be 40 % "explained by location" and
    # still carry almost no absolute spatial variation if it barely varies.
    spatial_var = if (is.null(g)) NA_real_ else summary(g)$r.sq * var(y),
    hex_n = sum(hs),
    hex_cells = length(unique(d$hex_id[hs])),
    hex_r2 = if (is.null(hm)) NA_real_ else hm$r.squared,
    hex_r2_adj = if (is.null(hm)) NA_real_ else hm$adj.r.squared,
    r_lnPC1 = cor(y, d$ln_PC1),
    rho_lnPC1 = suppressWarnings(cor(y, d$ln_PC1, method = "spearman")),
    p_r_lnPC1 = cor.test(y, d$ln_PC1)$p.value,
    stringsAsFactors = FALSE)
}

t0 <- Sys.time()
sjobs <- expand.grid(u = names(UNITS), v = names(VARS), stringsAsFactors = FALSE)
sres <- mclapply(seq_len(nrow(sjobs)),
                 function(i) spat_row(sjobs$u[i], sjobs$v[i]),
                 mc.cores = min(NWORK, nrow(sjobs)), mc.preschedule = FALSE)
sbad <- vapply(sres, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(sbad)) log_msg(ctx, sprintf("!! %d of %d spatial-pattern rows failed",
                                    sum(sbad), length(sres)))
SP <- do.call(rbind, sres[!sbad])
SP <- SP[order(match(SP$variable, names(VARS)), match(SP$unit, names(UNITS))), ]
log_msg(ctx, sprintf("spatial-pattern block: %d rows in %.0f s wall on %d workers",
                     nrow(SP), as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     min(NWORK, nrow(sjobs))))
for (i in seq_len(nrow(SP)))
  log_msg(ctx, sprintf("  %-14s %-14s N %5d  Moran I %+.4f (z %+8.2f, p %.3g)  TPS R2 %.4f (edf %.0f)  hex R2 %.4f (adj %.4f)  r(ln PC1) %+.4f",
    SP$unit[i], SP$variable[i], SP$n[i], SP$moran_I[i], SP$moran_z[i],
    SP$moran_p[i], SP$tps_r2_adj[i], SP$tps_edf[i], SP$hex_r2[i],
    SP$hex_r2_adj[i], SP$r_lnPC1[i]))

# =============================================================================
# 5. The channel decomposition
# =============================================================================
#
# Same right-hand side for all three outcomes, so the OLS slopes add exactly:
#     ln_<outcome> ~ ln_plot_area + ln_<measure>
# Standardized beta = the coefficient from the same fit re-estimated on
# z-scored covariates, which is beta * sd(x) since the covariates and
# not the outcome are standardized (as everywhere in the revision).
# That scaling is linear in beta, so the standardized slopes add too.

OUTCOMES <- c(ln_land_value = "ln Puni", ln_FP = "ln FP", ln_DEN = "ln DEN")

chan_fit <- function(u, m, outcome, ycol = outcome) {
  i <- UNITS[[u]]; d <- DD[i, , drop = FALSE]; W <- WL[[u]]
  bn <- paste0("ln_", m)
  form <- as.formula(paste(ycol, "~ ln_plot_area +", bn))
  o <- lm(form, data = d)
  dz <- d; for (v in c("ln_plot_area", bn)) dz[[v]] <- as.numeric(scale(d[[v]]))
  oz <- lm(form, data = dz)
  f  <- fit_spreg(ctx, form, d, W, "sarar")
  fz <- fit_spreg(ctx, form, dz, W, "sarar")
  z  <- if (is.null(f))  NULL else sarar_row(f)
  zz <- if (is.null(fz)) NULL else sarar_row(fz)
  data.frame(
    unit = u, measure = m, outcome = outcome, y_column = ycol, n = nrow(d),
    ols_beta = unname(coef(o)[bn]),
    ols_se = unname(summary(o)$coefficients[bn, 2]),
    ols_p = unname(summary(o)$coefficients[bn, 4]),
    ols_beta_std = unname(coef(oz)[bn]),
    ols_r2 = summary(o)$r.squared,
    sarar_beta = if (is.null(z)) NA_real_ else unname(z$co[[bn]]),
    sarar_se = if (is.null(z)) NA_real_ else unname(z$se[[bn]]),
    sarar_p = if (is.null(z)) NA_real_ else unname(z$p[[bn]]),
    sarar_beta_std = if (is.null(zz)) NA_real_ else unname(zz$co[[bn]]),
    sarar_se_std = if (is.null(zz)) NA_real_ else unname(zz$se[[bn]]),
    sarar_beta_area = if (is.null(z)) NA_real_ else unname(z$co[["ln_plot_area"]]),
    lambda_lag = if (is.null(z)) NA_real_ else unname(z$co[["lambda"]]),
    rho_error = if (is.null(z)) NA_real_ else unname(z$co[["rho"]]),
    error_at_bound = if (is.null(z)) NA_integer_ else
      as.integer(abs(abs(unname(z$co[["rho"]])) - BOUND) < BEPS),
    lambda_nonstationary = if (is.null(z)) NA_integer_ else
      as.integer(unname(z$co[["lambda"]]) >= 1),
    pseudo_r2 = if (is.null(f)) NA_real_ else pseudo_r2(f, d[[ycol]]),
    stringsAsFactors = FALSE)
}

# headline: PC1 on every unit and every outcome; the other four measures pooled
# and within the apartment and single-family-open groups, so the channel split
# can be shown not to be a PC1 artefact.
cjobs <- rbind(
  expand.grid(u = names(UNITS), m = "PC1", o = names(OUTCOMES),
              stringsAsFactors = FALSE),
  expand.grid(u = intersect(c("(pooled)", "Apartment",
                              "Single-family house in open community"),
                            names(UNITS)),
              m = setdiff(MEASURES, "PC1"), o = names(OUTCOMES),
              stringsAsFactors = FALSE))
t0 <- Sys.time()
cres <- mclapply(seq_len(nrow(cjobs)),
                 function(i) chan_fit(cjobs$u[i], cjobs$m[i], cjobs$o[i]),
                 mc.cores = min(NWORK, nrow(cjobs)), mc.preschedule = FALSE)
cbad <- vapply(cres, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(cbad)) log_msg(ctx, sprintf("!! %d of %d channel fits returned nothing",
                                    sum(cbad), length(cres)))
CH <- do.call(rbind, cres[!cbad])
log_msg(ctx, sprintf("channel block: %d of %d fits in %.0f s wall on %d workers",
                     nrow(CH), nrow(cjobs),
                     as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     min(NWORK, nrow(cjobs))))

verdict <- function(b, p) sprintf("%s%s", if (b > 0) "+" else "-",
                                  if (p < ALPHA) " sig" else " n.s.")
CH$verdict_ols   <- mapply(verdict, CH$ols_beta, CH$ols_p)
CH$verdict_sarar <- mapply(verdict, CH$sarar_beta, CH$sarar_p)

# ---- the split table: one row per unit x measure -----------------------------
pick <- function(u, m, o, col) {
  v <- CH[[col]][CH$unit == u & CH$measure == m & CH$outcome == o]
  if (!length(v)) NA_real_ else v[1]
}
pickc <- function(u, m, o, col) {
  v <- CH[[col]][CH$unit == u & CH$measure == m & CH$outcome == o]
  if (!length(v)) NA_character_ else as.character(v[1])
}
split_keys <- unique(CH[, c("unit", "measure")])
SPLIT <- do.call(rbind, lapply(seq_len(nrow(split_keys)), function(i) {
  u <- split_keys$unit[i]; m <- split_keys$measure[i]
  tot_o <- pick(u, m, "ln_land_value", "ols_beta_std")
  fp_o  <- pick(u, m, "ln_FP",  "ols_beta_std")
  dn_o  <- pick(u, m, "ln_DEN", "ols_beta_std")
  tot_s <- pick(u, m, "ln_land_value", "sarar_beta_std")
  fp_s  <- pick(u, m, "ln_FP",  "sarar_beta_std")
  dn_s  <- pick(u, m, "ln_DEN", "sarar_beta_std")
  data.frame(
    unit = u, measure = m, n = pick(u, m, "ln_land_value", "n"),
    ols_std_total = tot_o, ols_std_FP = fp_o, ols_std_DEN = dn_o,
    ols_additivity_residual = tot_o - (fp_o + dn_o),
    ols_share_DEN = dn_o / tot_o, ols_share_FP = fp_o / tot_o,
    sarar_std_total = tot_s, sarar_std_FP = fp_s, sarar_std_DEN = dn_s,
    sarar_sum_components = fp_s + dn_s,
    sarar_additivity_residual = tot_s - (fp_s + dn_s),
    # a share is only interpretable when the two components pull the same way;
    # when they have opposite signs the ratio explodes and is left blank.
    sarar_share_DEN = if (isTRUE(sign(fp_s) == sign(dn_s))) dn_s / (fp_s + dn_s) else NA_real_,
    sarar_share_FP  = if (isTRUE(sign(fp_s) == sign(dn_s))) fp_s / (fp_s + dn_s) else NA_real_,
    sarar_flags = sum(pick(u, m, "ln_land_value", "error_at_bound"),
                      pick(u, m, "ln_land_value", "lambda_nonstationary"),
                      pick(u, m, "ln_FP", "error_at_bound"),
                      pick(u, m, "ln_FP", "lambda_nonstationary"),
                      pick(u, m, "ln_DEN", "error_at_bound"),
                      pick(u, m, "ln_DEN", "lambda_nonstationary"), na.rm = TRUE),
    verdict_total = pickc(u, m, "ln_land_value", "verdict_sarar"),
    verdict_FP    = pickc(u, m, "ln_FP",  "verdict_sarar"),
    verdict_DEN   = pickc(u, m, "ln_DEN", "verdict_sarar"),
    var_ln_DEN = var(DD$ln_DEN[UNITS[[u]]]),
    var_ln_FP  = var(DD$ln_FP[UNITS[[u]]]),
    stringsAsFactors = FALSE)
}))
SPLIT <- SPLIT[order(match(SPLIT$measure, MEASURES),
                     match(SPLIT$unit, names(UNITS))), ]
for (i in seq_len(nrow(SPLIT)))
  log_msg(ctx, sprintf("  SPLIT %-14s %-4s N %5d  total %+.4f = DEN %+.4f + FP %+.4f (OLS residual %.2g)  share DEN %5.1f %%  |  SARAR total %+.4f, DEN %+.4f, FP %+.4f (gap %+.4f)",
    SPLIT$unit[i], SPLIT$measure[i], SPLIT$n[i], SPLIT$ols_std_total[i],
    SPLIT$ols_std_DEN[i], SPLIT$ols_std_FP[i], SPLIT$ols_additivity_residual[i],
    100 * SPLIT$ols_share_DEN[i], SPLIT$sarar_std_total[i],
    SPLIT$sarar_std_DEN[i], SPLIT$sarar_std_FP[i],
    SPLIT$sarar_additivity_residual[i]))

OLS_ADD_MAX <- max(abs(SPLIT$ols_additivity_residual), na.rm = TRUE)
log_msg(ctx, sprintf("OLS additivity: max |std beta(ln Puni) - std beta(ln FP) - std beta(ln DEN)| = %.3g over %d unit x measure cells",
                     OLS_ADD_MAX, nrow(SPLIT)))

# ---- the sold-unit built-area sensitivity (a caveat with a number) ----------
sold_ok <- is.finite(DD$ln_FP_sold)
log_msg(ctx, sprintf("sold-unit denominator available on %d of %d determinate records (%.1f %%)",
                     sum(sold_ok), nrow(DD), 100 * sum(sold_ok) / nrow(DD)))
alt_units <- intersect(c("(pooled)", "Apartment",
                         "Single-family house in open community"), names(UNITS))
ALT <- do.call(rbind, lapply(alt_units, function(u) {
  i <- UNITS[[u]] & sold_ok
  if (sum(i) < MIN_N) return(NULL)
  d <- DD[i, , drop = FALSE]
  W <- suppressWarnings(build_weights(ctx, XYD[i, , drop = FALSE], spec_name = W_SPEC))
  mt <- tryCatch(spdep::moran.test(d$ln_FP_sold, W, randomisation = TRUE,
                                   alternative = "two.sided"), error = function(e) NULL)
  o  <- lm(ln_FP_sold ~ ln_plot_area + ln_PC1, data = d)
  dz <- d; for (v in c("ln_plot_area", "ln_PC1")) dz[[v]] <- as.numeric(scale(d[[v]]))
  oz <- lm(ln_FP_sold ~ ln_plot_area + ln_PC1, data = dz)
  data.frame(unit = u, n = nrow(d),
             r_headline_vs_sold = cor(d$ln_FP, d$ln_FP_sold),
             moran_I_FP_sold = if (is.null(mt)) NA_real_ else
               unname(mt$estimate[["Moran I statistic"]]),
             moran_p_FP_sold = if (is.null(mt)) NA_real_ else mt$p.value,
             ols_std_FP_sold = unname(coef(oz)["ln_PC1"]),
             ols_p_FP_sold = unname(summary(o)$coefficients["ln_PC1", 4]),
             ols_std_FP_headline = pick(u, "PC1", "ln_FP", "ols_beta_std"),
             stringsAsFactors = FALSE)
}))
if (!is.null(ALT)) for (i in seq_len(nrow(ALT)))
  log_msg(ctx, sprintf("  ALT-FP %-14s N %5d  r(headline, sold) %.4f  Moran I %+.4f (p %.3g)  std beta on ln PC1 %+.4f (headline %+.4f)",
    ALT$unit[i], ALT$n[i], ALT$r_headline_vs_sold[i], ALT$moran_I_FP_sold[i],
    ALT$moran_p_FP_sold[i], ALT$ols_std_FP_sold[i], ALT$ols_std_FP_headline[i]))

# =============================================================================
# 6. The maps
# =============================================================================
#
# Figure 5 is a hexagon choropleth on the frozen 500 m lattice (the cells of the
# aggregated models): each cell is filled with the median of the panel's
# records in it, in classed steps labelled in natural units. A cell with fewer
# than MAP_MIN_N records is drawn grey -- the records exist, but their median
# is not shown. The classes are shared within a row so the panels compare.
#
# Floor price is drawn in US$, the paper's currency, at the one fixed rate of
# the submitted data: the frozen 500 m layer's `US$` column is Puni divided by
# it, and the rate is recovered from that column and required to be the same
# in every cell rather than typed in. A constant factor moves every ln FP by
# the same amount, so Moran's I and the smooth-surface R2 in the subtitles,
# both computed on the address points, are unchanged by it.
#
# Publication-lean: drawn at journal full width (174 mm, at most 234 mm tall)
# with real 6.5-10 pt text; light surface; one-hue ramps (blue for floor price,
# orange for density); the IBGE municipal outline for land context; legends in
# the free column beside the tall maps; a scale bar, no north arrow
# (EPSG:31982 is north up).

FIGDIR <- file.path(REPO, ctx$cfg$paths$outputs, "figures", "floor_price")
dir.create(FIGDIR, showWarnings = FALSE, recursive = TRUE)

MAP_MIN_N <- 3L
# class edges, in the units the legend prints (US$ per m2 of floor; m2 of floor
# per m2 of plot); the classes below the first and above the last edge are open
MAP_STEPS <- list(fp  = c(400, 500, 600, 700, 800, 1000),
                  den = c(0.1, 0.25, 0.5, 1, 1.5, 2, 3))

BLUES   <- c("#e3eefc", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab",
             "#104281", "#0a2a55")
ORANGES <- c("#fde8db", "#f9c6a6", "#f3a06f", "#e9783e", "#d85517", "#a83e13",
             "#7a2b0c", "#4d1a06")
INK  <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"
SURFACE <- "#fcfcfb"; LAND <- "#f3f2ee"; COAST <- "#c9c7bf"; FEW <- "#dddbd3"

# ---- the US$ rate of the submitted data --------------------------------------
usd_col <- intersect(c("US$", "US."), names(HEX))[1]
if (is.na(usd_col)) stop("the frozen 500 m layer has no `US$` column", call. = FALSE)
.ok <- is.finite(HEX[[usd_col]]) & HEX[[usd_col]] > 0
.rt <- HEX$Puni_mean[.ok] / HEX[[usd_col]][.ok]
if (!length(.rt) || diff(range(.rt)) > 1e-9)
  stop("the frozen `US$` column is not Puni at one fixed rate", call. = FALSE)
BRL_PER_USD <- median(.rt)
log_msg(ctx, sprintf("US$ rate of the submitted data: %.6f BRL per US$ (identical in all %d frozen cells with a value)",
                     BRL_PER_USD, sum(.ok)))

# ---- land context: the municipality (IBGE Malha Municipal Digital 2025) ------
MB  <- ctx$cfg$revision$boundary_sensitivity$municipal$boundary
POA <- st_read(file.path(REPO, MB$path), quiet = TRUE)
POA <- POA[as.character(POA[[MB$id_column]]) == as.character(MB$cd_mun), ]
if (nrow(POA) != 1) stop("municipal boundary: CD_MUN ", MB$cd_mun, " not found once",
                         call. = FALSE)
POA <- st_transform(POA, ctx$cfg$repro$crs_epsg)

BB   <- st_bbox(HEX)
PAD  <- 400
XLIM <- c(BB[["xmin"]] - PAD, BB[["xmax"]] + PAD)
YLIM <- c(BB[["ymin"]] - PAD, BB[["ymax"]] + PAD)
ASPECT <- diff(YLIM) / diff(XLIM)           # 1.34: the study area is tall

SB_M  <- 5000                               # 5 km scale bar
sb_x0 <- XLIM[1] + 0.06 * diff(XLIM)
sb_y0 <- YLIM[1] + 0.04 * diff(YLIM)

map_theme <- theme_void(base_size = 8) +
  theme(plot.background = element_rect(fill = SURFACE, colour = NA),
        panel.background = element_rect(fill = SURFACE, colour = NA),
        plot.title = element_text(colour = INK, face = "bold", size = 9,
                                  hjust = 0, margin = margin(b = 1.5)),
        plot.subtitle = element_text(colour = INK2, size = 6.5, hjust = 0,
                                     margin = margin(b = 2)),
        plot.caption = element_text(colour = MUTED, size = 6, hjust = 0,
                                    lineheight = 1.2, margin = margin(t = 4)),
        legend.position = "right", legend.justification = "center",
        legend.title = element_text(colour = INK2, size = 7, lineheight = 1.1),
        legend.text = element_text(colour = INK2, size = 6.5),
        legend.key.height = unit(8, "mm"), legend.key.width = unit(3, "mm"),
        plot.margin = margin(2, 2, 2, 2))

# per-cell median of a panel's variable and its record count; DD$hex_id is the
# 500 m membership of section 4
hex_cells <- function(dat, vcol) {
  i <- !is.na(dat$hex_id)
  a <- aggregate(dat[[vcol]][i], list(id = dat$hex_id[i]),
                 function(z) c(med = median(z), n = length(z)))
  merge(HEX[, "id"], data.frame(id = a$id, med = a$x[, "med"], n = a$x[, "n"]),
        by = "id")
}

panel <- function(dat, var, title, subtitle) {
  vcol <- c(fp = "ln_FP", den = "ln_DEN")[[var]]
  h <- hex_cells(dat, vcol)
  if (var == "fp") h$med <- h$med - log(BRL_PER_USD)             # BRL -> US$
  br   <- log(MAP_STEPS[[var]])
  # one colour per class (length(br) + 1 classes), the darkest end of the ramp
  pal  <- if (var == "fp") BLUES else ORANGES
  ramp <- pal[(length(pal) - length(br)):length(pal)]
  lab  <- if (var == "fp") function(x) format(round(exp(x)), big.mark = ",", trim = TRUE)
          else function(x) formatC(exp(x), format = "fg", digits = 2)
  ggplot() +
    geom_sf(data = POA, fill = LAND, colour = COAST, linewidth = 0.25) +
    geom_sf(data = h[h$n < MAP_MIN_N, ], fill = FEW, colour = SURFACE,
            linewidth = 0.15) +
    geom_sf(data = h[h$n >= MAP_MIN_N, ], aes(fill = med), colour = SURFACE,
            linewidth = 0.15) +
    scale_fill_stepsn(colours = ramp, breaks = br, labels = lab,
                      limits = range(br) + c(-1, 1) * 0.5 * mean(diff(br)),
                      oob = scales::squish,
                      name = c(fp = "US$ per m²\nof floor",
                               den = "m² of floor\nper m² of plot")[[var]],
                      guide = guide_coloursteps(even.steps = TRUE)) +
    annotate("segment", x = sb_x0, xend = sb_x0 + SB_M, y = sb_y0, yend = sb_y0,
             colour = INK2, linewidth = 0.5) +
    annotate("segment", x = c(sb_x0, sb_x0 + SB_M), xend = c(sb_x0, sb_x0 + SB_M),
             y = sb_y0 - 150, yend = sb_y0 + 150, colour = INK2, linewidth = 0.5) +
    annotate("text", x = sb_x0 + SB_M / 2, y = sb_y0 + 700, label = "5 km",
             colour = INK2, size = 2.2) +
    coord_sf(xlim = XLIM, ylim = YLIM, expand = FALSE,
             crs = st_crs(ctx$cfg$repro$crs_epsg), datum = NA) +
    labs(title = title, subtitle = subtitle) +
    map_theme
}

APT <- DD[DD$Type == "Apartment", , drop = FALSE]
SFH <- DD[DD$Type == "Single-family house in open community", , drop = FALSE]

sub_of <- function(d, v) {
  r <- SP[SP$unit == v[1] & SP$variable == v[2], ]
  sprintf("n = %s  ·  Moran's I = %s  ·  smooth-surface R² = %s",
          format(nrow(d), big.mark = ","),
          if (!nrow(r)) "--" else sprintf("%.2f", r$moran_I[1]),
          if (!nrow(r)) "--" else sprintf("%.0f %%", 100 * r$tps_r2_adj[1]))
}

P <- list(
  fp_apartment  = panel(APT, "fp",  "(i)  Apartments",
                        sub_of(APT, c("Apartment", "ln_FP"))),
  fp_sfh_open   = panel(SFH, "fp",  "(ii)  Single-family houses, open community",
                        sub_of(SFH, c("Single-family house in open community", "ln_FP"))),
  den_apartment = panel(APT, "den", "(iii)  Apartments",
                        sub_of(APT, c("Apartment", "ln_DEN"))),
  den_pooled    = panel(DD,  "den", "(iv)  All property types",
                        sub_of(DD, c("(pooled)", "ln_DEN"))))
ROW_OF <- c(fp_apartment = "Floor price", fp_sfh_open = "Floor price",
            den_apartment = "Development density", den_pooled = "Development density")

for (nm in names(P)) {
  h <- hex_cells(if (nm == "den_pooled") DD else if (nm == "fp_sfh_open") SFH else APT,
                 if (startsWith(nm, "fp")) "ln_FP" else "ln_DEN")
  log_msg(ctx, sprintf("map %-14s %d cells with records, %d with >= %d (coloured)",
                       nm, nrow(h), sum(h$n >= MAP_MIN_N), MAP_MIN_N))
}

STAMP <- sprintf(
  "Porto Alegre address records with a determinate registry built area (N = %s of %s analysis records); vacant plots have no building and are absent by construction. EPSG:31982, north up. Each 500 m hexagon is coloured by the median of its records; hexagons with fewer than %d records are shown in grey. Classes shared within a row. Floor price converted at %.4f BRL per US$, the rate of the submitted data. Moran's I and R² are computed on the address records.",
  format(nrow(DD), big.mark = ","), format(nrow(D), big.mark = ","), MAP_MIN_N,
  BRL_PER_USD)

# a vertical legend column: the colour steps plus the grey "few records" key
legend_col <- function(p) {
  g <- get_plot_component(p, "guide-box-right", return_all = TRUE)
  if (is.list(g) && !inherits(g, "grob")) g <- g[[1]]
  hx <- data.frame(x = cos(seq(30, 330, by = 60) * pi / 180),
                   y = sin(seq(30, 330, by = 60) * pi / 180))
  key <- ggplot() + geom_polygon(data = hx, aes(x, y), fill = FEW) +
    annotate("text", x = 1.8, y = 0, hjust = 0, size = 6.5 / .pt, colour = INK2,
             lineheight = 1.05, label = sprintf("fewer than\n%d records", MAP_MIN_N)) +
    coord_fixed(xlim = c(-1.2, 12), ylim = c(-2.5, 2.5), clip = "off") + theme_void()
  plot_grid(NULL, ggdraw() + draw_grob(g, x = 0.1, width = 0.9, hjust = 0),
            plot_grid(NULL, key, nrow = 1, rel_widths = c(0.1, 0.9)), NULL,
            ncol = 1, rel_heights = c(0.12, 0.56, 0.08, 0.24))
}

# ---- the single panels -------------------------------------------------------
for (nm in names(P)) {
  p <- P[[nm]] +
    labs(title = paste0(ROW_OF[[nm]], " — ",
                        sub("^(.)", "\\L\\1", sub("^\\(\\w+\\)  ", "", P[[nm]]$labels$title),
                            perl = TRUE)),
         caption = paste(strwrap(STAMP, 95), collapse = "\n"))
  fn <- file.path(FIGDIR, paste0(nm, ".png"))
  ggsave(fn, p, width = 4.6, height = 3.5 * ASPECT + 1.3, dpi = 300, units = "in",
         bg = SURFACE)
  log_msg(ctx, "wrote ", fn)
}

# ---- the combined 2 x 2 ------------------------------------------------------
# Absolute geometry in inches. Each row = a heading (the variable), the two
# group panels and one legend column to their right: the maps are tall, so the
# width beside them is free, and a legend under each row would cost height.
FIG_W <- 6.85; FIG_H_MAX <- 9.2                 # 174 mm wide, <= 234 mm tall
LEG_W <- 1.05; HEAD_H <- 0.24; TITLE_H <- 0.36; GAP_H <- 0.10; MARG <- 0.06
# The combined figure carries no caption text (the manuscript caption has it).
# The panel geometry still reserves the caption's height, so the maps keep the
# size they had with it; the figure is just that much shorter.
cap_h <- 0.06 + 0.112 * length(strwrap(STAMP, width = 128))
map_h <- (FIG_H_MAX - 2 * MARG - cap_h - GAP_H - 2 * (HEAD_H + TITLE_H)) / 2
map_w <- min(map_h / ASPECT, (FIG_W - 2 * MARG - LEG_W) / 2 - 0.04)
map_h <- map_w * ASPECT
cell_w <- map_w + 0.04; panel_h <- map_h + TITLE_H; row_h <- HEAD_H + panel_h
nolg <- function(p) p + theme(legend.position = "none")
row_of <- function(a, b, heading) plot_grid(
  ggdraw() + draw_label(heading, fontface = "bold", size = 10, colour = INK,
                        hjust = 0, x = 0.004),
  plot_grid(nolg(a), nolg(b), legend_col(a), nrow = 1,
            rel_widths = c(cell_w, cell_w, FIG_W - 2 * MARG - 2 * cell_w)),
  ncol = 1, rel_heights = c(HEAD_H, panel_h))
combined <- plot_grid(row_of(P$fp_apartment, P$fp_sfh_open, "Floor price"), NULL,
                      row_of(P$den_apartment, P$den_pooled, "Development density"),
                      ncol = 1, rel_heights = c(row_h, GAP_H, row_h)) +
  theme(plot.background = element_rect(fill = SURFACE, colour = NA),
        plot.margin = margin(MARG * 72, MARG * 72, MARG * 72, MARG * 72))
FIG_H <- 2 * row_h + GAP_H + 2 * MARG
fn <- file.path(FIGDIR, "floor_price_density_panels.png")
ggsave(fn, combined, width = FIG_W, height = FIG_H, dpi = 300, units = "in",
       bg = SURFACE)
log_msg(ctx, sprintf("wrote %s (%.2f x %.2f in)", fn, FIG_W, FIG_H))
fn <- file.path(FIGDIR, "floor_price_density_panels.pdf")
ggsave(fn, combined, width = FIG_W, height = FIG_H, units = "in", bg = SURFACE,
       device = cairo_pdf)
log_msg(ctx, "wrote ", fn)

# =============================================================================
# 7. Tables
# =============================================================================

w <- function(df, fn) {
  if (is.null(df) || !nrow(df)) return(invisible(NULL))
  p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p); p
}
w(grp,   "floor_price_groups.csv")
w(SP,    "floor_price_spatial_pattern.csv")
w(CH,    "floor_price_channel_fits.csv")
w(SPLIT, "floor_price_channel_split.csv")
w(ALT,   "floor_price_sold_unit_sensitivity.csv")

# =============================================================================
# 8. The report — every sentence that states a result is built from the tables
# =============================================================================

md_table <- function(df, digits = 4) {
  cells <- lapply(df, function(col) if (is.numeric(col))
    ifelse(is.na(col), "", formatC(col, format = "g", digits = digits)) else
    as.character(col))
  cells <- as.data.frame(cells, stringsAsFactors = FALSE)
  c(paste0("| ", paste(names(df), collapse = " | "), " |"),
    paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|"),
    apply(cells, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |")))
}
fmtn <- function(x) format(x, big.mark = ",")
md <- character(0); say <- function(...) md <<- c(md, ...)

sp_val <- function(u, v, col) {
  x <- SP[[col]][SP$unit == u & SP$variable == v]
  if (!length(x)) NA_real_ else x[1]
}
sp_p <- function(u, v) {
  p <- sp_val(u, v, "moran_p")
  if (is.na(p)) "--" else if (p < 1e-4) "< 0.0001" else sprintf("= %.4f", p)
}
APT_G <- "Apartment"; SFH_G <- "Single-family house in open community"
sh <- function(u, m, col) SPLIT[[col]][SPLIT$unit == u & SPLIT$measure == m][1]
grp_var <- function(g) grp$var_ln_DEN[grp$group == g][1]

# ---- the corollary, evaluated across every estimated group ------------------
# The corollary under test: the floor-price channel should be relatively
# stronger where density has less room to vary. `var(ln DEN)` is the room
# measure; the test is whether it ranks against the floor-price share
# negatively. With few estimated groups this is a rank correlation over a
# handful of points and is reported as such, not as a test.
COR <- SPLIT[SPLIT$measure == "PC1" & SPLIT$unit != "(pooled)", ]
COR <- COR[order(COR$var_ln_DEN), ]
cor_rho <- if (nrow(COR) >= 3)
  suppressWarnings(cor(COR$var_ln_DEN, COR$ols_share_FP, method = "spearman")) else NA_real_
corr_holds <- is.finite(cor_rho) && cor_rho < 0
# the pairwise comparison the corollary is aimed at: apartments vs single-family
# houses in open community. Less room for density should mean a larger
# floor-price share.
.av <- grp_var(APT_G); .sv <- grp_var(SFH_G)
.af <- sh(APT_G, "PC1", "ols_share_FP"); .sf <- sh(SFH_G, "PC1", "ols_share_FP")
pair_holds <- isTRUE((.av < .sv) == (.af > .sf))
log_msg(ctx, sprintf("corollary, pairwise (apartments vs single-family open): var(ln DEN) %.3f vs %.3f, floor-price share %.1f %% vs %.1f %% -- %s",
                     .av, .sv, 100 * .af, 100 * .sf,
                     if (pair_holds) "the predicted direction" else "the opposite direction"))
log_msg(ctx, sprintf("corollary across %d estimated groups: Spearman(var ln DEN, OLS floor-price share) = %s -- the corollary predicts negative",
                     nrow(COR), if (is.na(cor_rho)) "NA" else sprintf("%+.3f", cor_rho)))

# ---- the SARAR health check on the component equations ----------------------
FLAGS <- CH[CH$error_at_bound == 1 | CH$lambda_nonstationary == 1, ]
N_FLAG <- table(factor(FLAGS$outcome, levels = names(OUTCOMES)))
log_msg(ctx, sprintf("SARAR flags: %d of %d fits have rho on the +-%.1f bound or lambda >= 1 -- %s",
                     nrow(FLAGS), nrow(CH), BOUND,
                     paste(sprintf("%d on %s", as.integer(N_FLAG), names(N_FLAG)), collapse = ", ")))
NEG_PR2 <- sum(CH$pseudo_r2 < 0, na.rm = TRUE)

# spatial amplitude, the number that separates "a pattern exists" from "the
# pattern is big": R2 x variance, in the variable's own log units.
amp <- function(u, v) sp_val(u, v, "spatial_var")
AMP_RATIO_APT <- amp(APT_G, "ln_DEN") / amp(APT_G, "ln_FP")
AMP_RATIO_SFH <- amp(SFH_G, "ln_DEN") / amp(SFH_G, "ln_FP")

pool_sh   <- sh("(pooled)", "PC1", "ols_share_DEN")
apt_sh    <- sh(APT_G, "PC1", "ols_share_DEN")
sfh_sh    <- sh(SFH_G, "PC1", "ols_share_DEN")
cb <- CH[CH$unit == "(pooled)" & CH$outcome == "ln_land_value" & CH$measure == "PC1", ]
PAPT_SHARE <- 100 * sum(DD$Type == APT_G) / nrow(DD)
flag_desc <- if (nrow(FLAGS)) paste(sprintf("%s / %s / %s (λ = %.3f, ρ = %+.3f)",
  FLAGS$unit, FLAGS$measure, FLAGS$outcome, FLAGS$lambda_lag, FLAGS$rho_error),
  collapse = "; ") else "none"

say("# Stage 4 — floor price, density and the channel from centrality to land value", "",
  sprintf("Generated by `src/50_floor_price.R` (`make floor-price`). Seed %s. Dependent variable: the rebuilt headline column. W: `%s`, kNN k = %d row-standardised, tie-break `%s`, rebuilt inside each group. Estimator: `sphet::spreg(model = \"sarar\", het = %s)`, with OLS reported beside it because only OLS makes the decomposition exact. Standardized β by re-estimation on z-scored covariates. α = %.2f.",
    ctx$cfg$repro$seed, W_SPEC, K, ctx$cfg$weights$identical_points$handling,
    isTRUE(ctx$cfg$models$het), ALPHA),
  "", "---", "",
  "## 0. The question, and the short answer", "",
  "The value per m² of plot factorises exactly into a price per m² of floor space and the floor space built per m² of plot. This run measures that decomposition, maps the two halves, and asks: **is there a spatial pattern in apartment floor prices?**",
  "",
  sprintf("**Spatial pattern.** Apartment ln floor price has Moran's I = **%.3f** (p %s) on the group's own k = %d neighbours, and a smooth surface in the coordinates alone carries **%.0f %%** of its variance.",
    sp_val(APT_G, "ln_FP", "moran_I"), sp_p(APT_G, "ln_FP"), K,
    100 * sp_val(APT_G, "ln_FP", "tps_r2_adj")),
  "",
  sprintf("**Size of the pattern.** Within apartments var(ln FP) = **%.3f** against var(ln DEN) = **%.3f**. In absolute log units the smooth spatial signal is **%.3f** in floor price against **%.3f** in density, a ratio of **%.1f**.",
    sp_val(APT_G, "ln_FP", "var"), sp_val(APT_G, "ln_DEN", "var"),
    amp(APT_G, "ln_FP"), amp(APT_G, "ln_DEN"), AMP_RATIO_APT),
  "",
  sprintf("**The channel, pooled.** Of PC1's gross standardized effect on ln Puni (**%.4f**, OLS), **%.4f** runs through density and **%.4f** through floor price — **%.0f %% via density**. The components add to the total to within %.2g over all %d unit × measure cells.",
    sh("(pooled)", "PC1", "ols_std_total"), sh("(pooled)", "PC1", "ols_std_DEN"),
    sh("(pooled)", "PC1", "ols_std_FP"), 100 * pool_sh,
    OLS_ADD_MAX, nrow(SPLIT)),
  "",
  sprintf("**By property type.** The density share of PC1's effect is %.0f %% among apartments (N = %s) and %.0f %% among single-family houses in open community (N = %s), against %.0f %% pooled.",
    100 * apt_sh, fmtn(sh(APT_G, "PC1", "n")), 100 * sfh_sh, fmtn(sh(SFH_G, "PC1", "n")),
    100 * pool_sh),
  "", "---", "",
  "## 1. Definitions, and the identity that makes this a decomposition", "",
  "Per address `a`, all built from the rows stage 1's unit multiplier already counts:",
  "",
  "- **B(a)** — the registry `Built Area` summed over the counted units; **U(a)** — stage 1's unit multiplier; **A(a)** — the plot area; **price_stat(a)** — the mean non-parking sale price.",
  "- **Floor price** `FP(a) = price_stat(a) ÷ (B(a)/U(a))` — the mean sale price divided by the mean built area of the counted units, i.e. **BRL per m² of floor space**.",
  "- **Density** `DEN(a) = B(a)/A(a)` — **m² of floor per m² of plot** (a floor-area ratio).",
  "",
  "Stage 1 builds `Puni = price_stat · U / A`, so",
  "",
  "```",
  "ln Puni  =  ln FP  +  ln DEN",
  "```",
  "",
  sprintf("holds exactly. This run measures it: the largest residual over the %s records with a determinate built area is **%.3g** (mean %.3g), and the script stops above 1e-9. The stage-1 identity behind it, `ln Puni = ln price_stat + ln U − ln A`, has residual **%.3g** over all %s analysis records.",
    fmtn(nrow(DD)), ident_fp,
    mean(abs(DD$ln_land_value - (DD$ln_FP + DD$ln_DEN))), ident_dv, fmtn(nrow(D))),
  "",
  sprintf("**The sample.** FP and DEN both need B(a), so this runs on the determinate subsample: **%s of %s** analysis records (%.2f %%). A group is estimated here only if it keeps **%d** determinate records; estimated: %s; not estimated: %s.",
    fmtn(nrow(DD)), fmtn(nrow(D)), 100 * nrow(DD) / nrow(D), MIN_N,
    paste(GROUPS, collapse = ", "), paste(grp$group[!grp$estimated], collapse = ", ")),
  "", md_table(grp), "",
  sprintf("`var_ln_DEN` measures how much a group's floor-area ratio varies — the room density has to respond. By group: %s. `var_ln_FP` is its counterpart for floor price, and it is smaller than `var_ln_DEN` in %d of the %d groups with a value.",
    paste(sprintf("%s %.3f", grp$group[!is.na(grp$var_ln_DEN)],
                  grp$var_ln_DEN[!is.na(grp$var_ln_DEN)]), collapse = "; "),
    sum(grp$var_ln_FP < grp$var_ln_DEN, na.rm = TRUE),
    sum(!is.na(grp$var_ln_FP) & !is.na(grp$var_ln_DEN))),
  "", "---", "",
  "## 2. The maps", "",
  sprintf("One PNG per panel plus the combined figure (300 dpi PNG, and a vector PDF of the combined figure), generated by this script. Each panel is a choropleth on the frozen 500 m hexagon lattice: a cell is filled with the median of the panel's records in it, in classed steps shared within a row so the panels compare; a cell with fewer than %d records is grey. Floor price on a single-hue blue scale in US$ per m² of floor, converted at %.4f BRL per US$ — the one fixed rate of the submitted data, recovered from the frozen 500 m layer's `US$` column; density on a single-hue orange scale in m² of floor per m² of plot. A constant rate shifts every ln FP equally, so Moran's I and the smooth-surface R² (computed on the address records, Section 3) are unaffected. EPSG:31982, north up; land context is the IBGE 2025 municipal outline.", MAP_MIN_N, BRL_PER_USD),
  "",
  "![Floor price and density by property type](../outputs/figures/floor_price/floor_price_density_panels.png)",
  "",
  "| panel | file |",
  "|---|---|",
  "| (i) ln FP, apartments | [`floor_price/fp_apartment.png`](../outputs/figures/floor_price/fp_apartment.png) |",
  "| (ii) ln FP, single-family open | [`floor_price/fp_sfh_open.png`](../outputs/figures/floor_price/fp_sfh_open.png) |",
  "| (iii) ln DEN, apartments | [`floor_price/den_apartment.png`](../outputs/figures/floor_price/den_apartment.png) |",
  "| (iv) ln DEN, all types pooled | [`floor_price/den_pooled.png`](../outputs/figures/floor_price/den_pooled.png) |",
  "| combined 2 × 2 | [`floor_price/floor_price_density_panels.png`](../outputs/figures/floor_price/floor_price_density_panels.png) |",
  "",
  "---", "",
  "## 3. Is there a spatial pattern? Three numbers per group and variable", "",
  sprintf("Moran's I is computed on the group's own k = %d list, under randomisation, two-sided. The smooth surface is a thin-plate spline in the projected coordinates, `mgcv::gam(y ~ s(x, y, bs = \"tp\", k = %d))` fitted by REML, and the reported R² is the adjusted R² of that fit. The 500 m hex fixed-effects R² is a coarser cross-check (%s of %s points fall in one of the %d frozen cells); it spends one parameter per cell, so the adjusted figure is the one to read.",
    K, TPS_K, fmtn(sum(!is.na(DD$hex_id))), fmtn(nrow(DD)), nrow(HEX)),
  "",
  "`spatial_var` is `tps_r2_adj × var`: the size of the smooth spatial signal in the variable's own log units, so a share is not mistaken for an amount.",
  "",
  md_table(SP[, c("unit", "variable", "n", "sd", "var", "moran_I", "moran_z",
                  "moran_p", "tps_r2_adj", "spatial_var", "tps_edf", "hex_cells",
                  "hex_r2", "hex_r2_adj", "r_lnPC1", "rho_lnPC1")]), "",
  sprintf("**Apartments.** ln FP: I = %.3f (p %s), smooth-surface share %.0f %%, spatial signal %.3f. ln DEN: spatial signal %.3f. Correlation with ln PC1: %+.3f for floor price, %+.3f for density.",
    sp_val(APT_G, "ln_FP", "moran_I"), sp_p(APT_G, "ln_FP"),
    100 * sp_val(APT_G, "ln_FP", "tps_r2_adj"), amp(APT_G, "ln_FP"), amp(APT_G, "ln_DEN"),
    sp_val(APT_G, "ln_FP", "r_lnPC1"), sp_val(APT_G, "ln_DEN", "r_lnPC1")),
  "",
  sprintf("**Single-family houses in open community.** ln FP: smooth-surface share %.0f %% (apartments %.0f %%) over a variance of %.3f (apartments %.3f), so the spatial signal in floor price is %.3f — %.1f times the apartments'. The density-to-price amplitude ratio is %.1f among houses against %.1f among apartments.",
    100 * sp_val(SFH_G, "ln_FP", "tps_r2_adj"), 100 * sp_val(APT_G, "ln_FP", "tps_r2_adj"),
    sp_val(SFH_G, "ln_FP", "var"), sp_val(APT_G, "ln_FP", "var"),
    amp(SFH_G, "ln_FP"), amp(SFH_G, "ln_FP") / amp(APT_G, "ln_FP"),
    AMP_RATIO_SFH, AMP_RATIO_APT),
  "", "---", "",
  "## 4. The channel decomposition", "",
  "Every outcome is regressed on the same right-hand side, `ln_plot_area + ln_<measure>`, so the component slopes add to the total slope. Under **OLS** the addition is exact and the shares below are exact; under **SARAR** each equation gets its own λ and ρ, so the components only approximately add and the gap is reported. The standardized β is the coefficient re-estimated on z-scored covariates, a linear rescaling that preserves the addition.",
  "",
  sprintf("**SARAR health.** %d of the %d SARAR fits have ρ on the ±%.1f bound or λ ≥ 1 (%s): %s.%s Where the SARAR fits are flagged, read the OLS columns as the decomposition; every flag is in `outputs/tables/floor_price_channel_fits.csv`. A blank `sarar_share_DEN` means the two SARAR components came back with opposite signs, which makes a share meaningless; the OLS share is always defined.",
    nrow(FLAGS), nrow(CH), BOUND,
    paste(sprintf("%d on %s", as.integer(N_FLAG), names(N_FLAG)), collapse = ", "),
    flag_desc,
    if (NEG_PR2 > 0) sprintf(" %d fits return a negative pseudo-R².", NEG_PR2) else ""),
  "",
  md_table(SPLIT[, c("unit", "measure", "n", "ols_std_total", "ols_std_DEN",
                     "ols_std_FP", "ols_additivity_residual", "ols_share_DEN",
                     "sarar_std_total", "sarar_std_DEN", "sarar_std_FP",
                     "sarar_additivity_residual", "sarar_share_DEN",
                     "verdict_total", "verdict_DEN", "verdict_FP",
                     "sarar_flags", "var_ln_DEN")]), "",
  sprintf("**Pooled.** PC1's gross standardized effect on ln Puni is %.4f (OLS), of which %.4f (**%.0f %%**) is density and %.4f (**%.0f %%**) floor price. Under SARAR: density %+.4f (%s), floor price %+.4f (%s); the gap between the total-equation coefficient and the sum of the components is %+.4f.",
    sh("(pooled)", "PC1", "ols_std_total"), sh("(pooled)", "PC1", "ols_std_DEN"),
    100 * pool_sh, sh("(pooled)", "PC1", "ols_std_FP"),
    100 * sh("(pooled)", "PC1", "ols_share_FP"),
    sh("(pooled)", "PC1", "sarar_std_DEN"), sh("(pooled)", "PC1", "verdict_DEN"),
    sh("(pooled)", "PC1", "sarar_std_FP"), sh("(pooled)", "PC1", "verdict_FP"),
    sh("(pooled)", "PC1", "sarar_additivity_residual")),
  "",
  sprintf("**By type.** Density share of PC1's effect (OLS): apartments %.0f %%, single-family open %.0f %%, pooled %.0f %%. Within apartments (%.0f %% of the determinate sample) the total is %.4f (OLS) / %.4f (SARAR, %s); the SARAR floor-price coefficient is %+.4f (%s) and the density coefficient %+.4f (%s).",
    100 * apt_sh, 100 * sfh_sh, 100 * pool_sh, PAPT_SHARE,
    sh(APT_G, "PC1", "ols_std_total"), sh(APT_G, "PC1", "sarar_std_total"),
    sh(APT_G, "PC1", "verdict_total"),
    sh(APT_G, "PC1", "sarar_std_FP"), sh(APT_G, "PC1", "verdict_FP"),
    sh(APT_G, "PC1", "sarar_std_DEN"), sh(APT_G, "PC1", "verdict_DEN")),
  "",
  "### 4.1 The corollary: where density cannot vary, is the effect in floor price?", "",
  sprintf("If the floor-price channel is stronger where density has less room to vary, the rank correlation between var(ln DEN) and the floor-price share should be negative. Across the %d estimated groups it is **%s** — %s. With this few groups it is a description of the table, not a test.",
    nrow(COR), if (is.na(cor_rho)) "not computable" else sprintf("%+.2f", cor_rho),
    if (corr_holds) "the predicted sign" else "not the predicted sign"),
  "", md_table(COR[, c("unit", "n", "var_ln_DEN", "var_ln_FP", "ols_std_total",
                       "ols_share_DEN", "ols_share_FP")]), "",
  sprintf("The pairwise comparison the corollary is aimed at — apartments against single-family houses in open community — goes **%s**: var(ln DEN) is %.3f among apartments and %.3f among houses, and the floor-price share of PC1's effect is %.0f %% and %.0f %%.",
    if (pair_holds) "the predicted way" else "the opposite way", .av, .sv, 100 * .af, 100 * .sf),
  "", "---", "",
  "## 5. Caveats", "")

alt_txt <- if (is.null(ALT)) "not computable" else sprintf(
  "on the pooled determinate sample the two floor prices correlate r = %.3f, and PC1's standardized OLS coefficient on ln FP moves %+.4f → %+.4f when the denominator is the sold units' built area",
  ALT$r_headline_vs_sold[ALT$unit == "(pooled)"][1],
  ALT$ols_std_FP_headline[ALT$unit == "(pooled)"][1],
  ALT$ols_std_FP_sold[ALT$unit == "(pooled)"][1])
left <- grp[grp$n_analysis > grp$n_determinate, ]
left <- left[order(-(left$n_analysis - left$n_determinate)), ]
nei_med <- vapply(names(UNITS)[-1], function(u) {
  i <- UNITS[[u]]; xy <- XYD[i, , drop = FALSE]
  kn <- suppressWarnings(spdep::knearneigh(xy, k = K))
  median(sqrt(rowSums((xy - xy[kn$nn[, K], , drop = FALSE])^2)))
}, numeric(1))

say(sprintf("1. **FP inherits the registry's `%s` gaps.** B(a) is determinate on %s of the %s analysis records; the %d that leave are, by group: %s. Every number in this report is therefore about the built stock.",
  CCF$built_area$indeterminate_code, fmtn(nrow(DD)), fmtn(nrow(D)), nrow(D) - nrow(DD),
  paste(sprintf("%s %d", left$group, left$n_analysis - left$n_determinate), collapse = ", ")),
  sprintf("2. **The denominator is all counted units, not the units that sold.** `FP` divides by the mean built area of every counted unit at the address, while the numerator is the mean price of the units that transacted. If sold units are systematically larger or smaller than the building's average, FP is biased by that ratio. Measured: %s. The alternative denominator is available on %s of %s determinate records (%.1f %%) and is reported in `outputs/tables/floor_price_sold_unit_sensitivity.csv`; it is a sensitivity line, not the headline, because the headline definition is the one that makes the identity exact.",
    alt_txt, fmtn(sum(sold_ok)), fmtn(nrow(DD)), 100 * sum(sold_ok) / nrow(DD)),
  "3. **The decomposition is arithmetic, not causal.** `ln Puni = ln FP + ln DEN` is an identity, so the split of a coefficient across the two components is exact as a description of the estimated association. It does not identify a mechanism, and nothing here rules out that density and floor price are jointly determined by the same unobserved local demand.",
  sprintf("4. **The spatial-pattern statistics depend on their tuning.** Moran's I uses each group's own k = %d list, so it is measured over different physical neighbourhoods in different groups (median %d-th-neighbour distance: %s). The thin-plate spline uses a fixed basis of k = %d in every group, chosen once and not tuned per group; a larger basis raises every R², and the comparison between groups is what carries the argument.",
    K, K, paste(sprintf("%s %.0f m", names(nei_med), nei_med), collapse = "; "), TPS_K),
  sprintf("5. **W and the dependent variable are held fixed.** k = %d is the proposed point-level W and the dependent variable is the rebuilt column; every number here is a difference between runs of one specification.", K),
  sprintf("6. **%d addresses have their multiplier collapsed to one by stage 1's parent-only rule while B(a) still sums every counted row.** They are flagged in `%s`; FP divides by B/U, so this is the one variable where that flag touches the numerator and the denominator differently.",
    sum(CTL$parent_only_collapse), CCF$path),
  "", "---", "",
  "## 6. Files", "",
  "- `outputs/figures/floor_price/` — the four panels and the combined figure, 300 dpi PNG.",
  "- `outputs/tables/floor_price_groups.csv` — group counts, medians and the variances.",
  "- `outputs/tables/floor_price_spatial_pattern.csv` — Moran's I, TPS and hex R², correlations.",
  "- `outputs/tables/floor_price_channel_fits.csv` — every OLS and SARAR fit behind the split.",
  "- `outputs/tables/floor_price_channel_split.csv` — the channel table.",
  "- `outputs/tables/floor_price_sold_unit_sensitivity.csv` — the alternative FP denominator.",
  "- `src/50_floor_price.R`, `make floor-price`.",
  "")

pmd <- out_report(ctx, "floor_price_channel.md")
writeLines(md, pmd)
log_msg(ctx, "wrote ", pmd)

write_counts(ctx, "floor_price_counts.csv")
finish(ctx)
