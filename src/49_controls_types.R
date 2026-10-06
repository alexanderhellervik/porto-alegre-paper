#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — the additional controls and the property-type split.
#
# Two referee requests:
#
#   Reviewer 4: "...either add built area / unit count as controls or rename the
#   variable (eg improved-property value per m2 of land) and pull back the
#   'land value' claims accordingly."
#   Reviewer 3: "...conduct sensitivity analyses excluding parking transactions,
#   separating major property types, and controlling for built area or
#   development intensity."
#
# Everything except the right-hand side (part 1) or the subsample (part 2) is
# held fixed:
#   - dependent variable: the rebuilt headline column
#     (`dependent_variable.rebuilt_path`), joined to the frozen point layer by
#     `id` exactly as src/48_rebuilt_dv.R does;
#   - exclusions: the submitted order, positivity -> 1.5xIQR;
#   - W: the proposed point-level W (`weights.proposed.disaggregated`, kNN
#     k = 8 row-standardised), with the configured tie-break sort;
#   - estimator: sphet::spreg(model = "sarar", het per config);
#   - standardized beta by re-estimation on z-scored covariates;
#   - the error-bound flag, pseudo-R2, spatial block CV (revision.spatial_cv).
#
# PART 1 — CONTROLS. Per measure, four specifications:
#   (a) ln_land_value ~ ln_plot_area + ln_<measure>                  [headline]
#   (b)                 + ln_built_area
#   (c)                 + ln_unit_count
#   (d)                 + ln_built_area + ln_unit_count
# Built area is only defined where the registry gives one (`INDETERMINADO`
# otherwise), so (a)-(d) all run on the determinate subsample and (a) is also
# run on the full sample, so that the comparison is like-for-like and the cost
# of the subsample is separated from the cost of the controls.
#
# `unit_count` is U(a), the dependent variable's own multiplier:
# ln Puni = ln price_stat + ln U - ln Terreno is an identity, checked
# numerically below. Specifications (c) and (d) therefore do not "control for"
# development intensity in the usual sense -- they move U from the left-hand
# side to the right, i.e. they change the question to "mean unit price per m2
# of land". That is the alternative Reviewer 4 offers to renaming the variable,
# so it is reported, but it is not an unbiased robustness check of the headline.
#
# PART 2 — PROPERTY TYPE. The frozen layer's `Type` column, groups with
# n >= revision.property_type_split.min_n_per_group in the analysis sample.
# Two weight matrices per group:
#   within_group  kNN k = 8 rebuilt on the group's own points   [headline]
#   restricted    the pooled k = 8 list with the out-of-group columns deleted
#                 (units left without an in-group neighbour are dropped)
# `within_group` is the headline because it keeps the specification (every unit
# has exactly 8 neighbours) while the group changes; `restricted` keeps the
# pooled geometry but the neighbour counts collapse with the group's density,
# so it is reported as the robustness line. Neighbour distances are reported for
# both so the reader can see how far "the eight nearest" reaches in each group.
#
# COMPUTE. `compute.max_workers`, further capped by the environment variable
# PIPELINE_MAX_WORKERS when set.
#
# Run:  Rscript src/49_controls_types.R          (or `make controls-types`)
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet); library(parallel)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "controls_types")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
DV       <- ctx$cfg$dependent_variable
CC       <- ctx$cfg$revision$controls
PTS      <- ctx$cfg$revision$property_type_split
W_SPEC   <- ctx$cfg$weights$proposed$disaggregated
K        <- as.integer(ctx$cfg$weights[[W_SPEC]]$k)
BOUND    <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
BEPS     <- 1e-6
LAM_MAX  <- 1.0
ALPHA    <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
MIN_N    <- as.integer(PTS$min_n_per_group)
CTRL     <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""

NWORK <- max(1L, min(as.integer(ctx$cfg$compute$max_workers),
                     as.integer(ctx$cfg$compute$host_ceiling)))
ovr <- Sys.getenv("PIPELINE_MAX_WORKERS", "")
if (nzchar(ovr) && !is.na(suppressWarnings(as.integer(ovr)))) {
  NWORK <- max(1L, min(NWORK, as.integer(ovr)))
  log_msg(ctx, sprintf("workers capped to %d by PIPELINE_MAX_WORKERS",
                       NWORK))
}

if (!isTRUE(PTS$enabled))
  stop("revision.property_type_split.enabled is not true", call. = FALSE)

log_msg(ctx, sprintf("dependent variable: rebuilt; W '%s' (kNN k = %d); tie-break '%s'; het = %s; alpha = %.2f",
                     W_SPEC, K,
                     ctx$cfg$weights$identical_points$handling,
                     isTRUE(ctx$cfg$models$het), ALPHA))
log_msg(ctx, sprintf("workers = %d", NWORK))

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c") log(x + ctx$cfg$transforms$log_plus_c) else log(x)

# =============================================================================
# 1. The analysis frame: rebuilt DV + the two extra controls
# =============================================================================

L <- st_read(cfg_path(ctx, "frozen", "cents_disaggregated"), quiet = TRUE)
if (is.na(st_crs(L)$epsg) || st_crs(L)$epsg != ctx$cfg$repro$crs_epsg)
  L <- st_transform(L, ctx$cfg$repro$crs_epsg)
if (isTRUE(st_is_longlat(L))) stop("layer is in geographic coordinates", call. = FALSE)

p_dv <- file.path(REPO, DV$rebuilt_path)
if (!file.exists(p_dv))
  stop("stage-1 output missing: ", p_dv, " -- run `make dv` first", call. = FALSE)
RB <- read.csv(p_dv, stringsAsFactors = FALSE)
if (nrow(RB) != nrow(L) || !all(RB$id == L$id))
  stop(sprintf("%s is not id-aligned with the frozen layer (%d vs %d rows)",
               DV$rebuilt_path, nrow(RB), nrow(L)), call. = FALSE)
if (max(abs(RB$Preco_frozen - L$Preco), na.rm = TRUE) > 1e-6)
  stop("the stage-1 file's `Preco_frozen` does not match the layer's `Preco`",
       call. = FALSE)

p_ct <- file.path(REPO, CC$path)
if (!file.exists(p_ct))
  stop("controls file missing: ", p_ct, " -- run `python src/49_built_area.py` first",
       call. = FALSE)
CTL <- read.csv(p_ct, stringsAsFactors = FALSE)
if (nrow(CTL) != nrow(L) || !all(CTL$id == L$id))
  stop(sprintf("%s is not id-aligned with the frozen layer (%d vs %d rows)",
               CC$path, nrow(CTL), nrow(L)), call. = FALSE)
if (max(abs(CTL$n_units - RB$n_units), na.rm = TRUE) > 0)
  stop("the controls file's `n_units` is not stage 1's", call. = FALSE)
log_msg(ctx, sprintf("controls file: %s, %d records id-aligned; determinate built area on %d of them (rule '%s'; the lenient rule would keep %d)",
                     CC$path, nrow(CTL), sum(CTL$built_area_determinate == 1),
                     CC$built_area$determinacy,
                     sum(CTL$built_area_determinate_lenient == 1)))

# the rebuilt dependent variable, and the controls, onto the layer
L$Preco <- RB$Preco
L$Puni  <- RB$Puni
L$price_stat            <- RB$price_stat
L$unit_count            <- CTL$n_units
L$built_area_units      <- CTL$built_area_units
L$built_area_all_rows   <- CTL$built_area_all_rows
L$built_area_sold_mean  <- CTL$built_area_sold_mean
L$built_area_per_unit   <- CTL$built_area_per_unit
L$area_const            <- CTL$area_const
L$built_area_determinate <- CTL$built_area_determinate
L$built_area_partial     <- CTL$built_area_partial
L$parent_only_collapse   <- CTL$parent_only_collapse

MAP <- ctx$cfg$measures$columns$disaggregated
for (cn in names(MAP)) {
  srcn <- MAP[[cn]]
  if (!is.null(L[[srcn]]) && srcn != cn) L[[cn]] <- L[[srcn]]
}
# `built_area` in measures.columns.disaggregated is the layer's per-record
# Area_Const, which is not the address-level built area this analysis needs; the
# canonicalisation above would shadow it, so name ours explicitly and never use
# the canonical name for it.
L$built_area <- NULL

E <- apply_exclusions(ctx, L, "disaggregated")
log_msg(ctx, sprintf("analysis sample (rebuilt DV, submitted exclusion order): N = %d",
                     nrow(E)))

D <- st_drop_geometry(E)
XY <- st_coordinates(E)[, 1:2]
for (v in c("land_value", MEASURES, "plot_area")) D[[paste0("ln_", v)]] <- lg(D[[v]])
D$ln_built_area <- lg(D$built_area_units)
D$ln_unit_count <- lg(D$unit_count)

# ---- the identity that makes `unit_count` a special kind of control ---------
# ln Puni == ln price_stat + ln U - ln Terreno, by construction (stage 1's
# `Preco = price_stat * n_units`, `Puni = Preco / Terreno`). Measured, not
# asserted, so the report can quote the residual.
ident <- max(abs(D$ln_land_value -
                 (log(D$price_stat) + log(D$unit_count) - log(D$plot_area))),
             na.rm = TRUE)
log_msg(ctx, sprintf("identity check: max |ln(land value) - (ln price_stat + ln U - ln plot area)| = %.3g -- the unit count IS the dependent variable's numerator",
                     ident))

# =============================================================================
# 2. The determinate subsample
# =============================================================================

det <- D$built_area_determinate == 1 & is.finite(D$ln_built_area)
DD  <- D[det, , drop = FALSE]
XYD <- XY[det, , drop = FALSE]
count_step(ctx, "controls: determinate built area", nrow(DD),
           sprintf("%d of %d records leave; rule '%s'", sum(!det), nrow(D),
                   CC$built_area$determinacy))
log_msg(ctx, sprintf("determinate subsample: N = %d of %d (%.2f %%); dropped %d",
                     nrow(DD), nrow(D), 100 * nrow(DD) / nrow(D), sum(!det)))

# who leaves, by property type
by_type <- do.call(rbind, lapply(sort(unique(D$Type)), function(g) {
  i <- D$Type == g
  data.frame(Type = g, n_analysis = sum(i), n_determinate = sum(i & det),
             n_dropped = sum(i & !det),
             pct_determinate = 100 * sum(i & det) / sum(i),
             stringsAsFactors = FALSE)
}))
by_type <- by_type[order(-by_type$n_analysis), ]
for (i in seq_len(nrow(by_type)))
  log_msg(ctx, sprintf("   %-40s analysis %5d  determinate %5d (%.1f %%)",
                       by_type$Type[i], by_type$n_analysis[i],
                       by_type$n_determinate[i], by_type$pct_determinate[i]))

# =============================================================================
# 3. One fit
# =============================================================================

SPECS <- list(
  a_full = list(extra = character(0), sample = "full",
                label = "(a) headline, full sample"),
  a      = list(extra = character(0), sample = "determinate",
                label = "(a) headline, determinate subsample"),
  b      = list(extra = "ln_built_area", sample = "determinate",
                label = "(b) + ln built area"),
  c      = list(extra = "ln_unit_count", sample = "determinate",
                label = "(c) + ln unit count"),
  d      = list(extra = c("ln_built_area", "ln_unit_count"), sample = "determinate",
                label = "(d) + both")
)

frm <- function(m, extra = character(0)) {
  rhs <- c(if (nzchar(CTRL)) "ln_plot_area" else character(0), paste0("ln_", m), extra)
  as.formula(paste("ln_land_value ~", paste(rhs, collapse = " + ")))
}

# Variance inflation factors from the OLS design matrix: VIF_j = 1/(1 - R2_j),
# R2_j from regressing covariate j on the other covariates. Reported because
# built area and the unit count are strongly related to each other and to the
# plot area, and the reader is entitled to the number rather than the adjective.
vifs <- function(dat, form) {
  X <- model.matrix(form, data = dat)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  if (ncol(X) < 2) return(setNames(rep(1, ncol(X)), colnames(X)))
  setNames(vapply(seq_len(ncol(X)), function(j) {
    r2 <- summary(lm(X[, j] ~ X[, -j, drop = FALSE]))$r.squared
    1 / (1 - r2)
  }, numeric(1)), colnames(X))
}

one_fit <- function(dat, W, m, extra, tag, extra_cols = list()) {
  form <- frm(m, extra); bn <- paste0("ln_", m)
  fit  <- fit_spreg(ctx, form, dat, W, "sarar")
  if (is.null(fit)) return(NULL)
  z  <- sarar_row(fit)
  ab <- abs(abs(z$co[["rho"]]) - BOUND) < BEPS    # sphet bounds only rho
  covs <- c(if (nzchar(CTRL)) "ln_plot_area" else character(0), bn, extra)
  dz <- dat
  for (v in covs) dz[[v]] <- as.numeric(scale(dat[[v]]))
  fz <- fit_spreg(ctx, form, dz, W, "sarar")
  bstd <- sstd <- NA_real_
  if (!is.null(fz)) { zz <- sarar_row(fz); bstd <- zz$co[[bn]]; sstd <- zz$se[[bn]] }
  vv <- tryCatch(vifs(dat, form), error = function(e) NULL)
  getco <- function(nm, what) if (nm %in% names(z$co)) z[[what]][[nm]] else NA_real_
  out <- data.frame(
    spec = tag, n = nrow(dat), measure = m,
    formula = paste(deparse(form), collapse = ""),
    beta_area = z$co[["ln_plot_area"]], se_area = z$se[["ln_plot_area"]],
    p_area = z$p[["ln_plot_area"]],
    beta_measure = z$co[[bn]], se_measure = z$se[[bn]], p_measure = z$p[[bn]],
    beta_std = bstd, se_std = sstd, beta_std_analytic = z$co[[bn]] * sd(dat[[bn]]),
    beta_built_area = getco("ln_built_area", "co"),
    se_built_area = getco("ln_built_area", "se"),
    p_built_area = getco("ln_built_area", "p"),
    beta_unit_count = getco("ln_unit_count", "co"),
    se_unit_count = getco("ln_unit_count", "se"),
    p_unit_count = getco("ln_unit_count", "p"),
    lambda_lag = z$co[["lambda"]], se_lambda = z$se[["lambda"]],
    rho_error = z$co[["rho"]], se_rho = z$se[["rho"]],
    error_at_bound = as.integer(ab),
    lambda_nonstationary = as.integer(z$co[["lambda"]] >= LAM_MAX),
    pseudo_r2 = pseudo_r2(fit, dat$ln_land_value),
    vif_max = if (is.null(vv)) NA_real_ else max(vv),
    vif_measure = if (is.null(vv)) NA_real_ else unname(vv[bn]),
    vif_plot_area = if (is.null(vv)) NA_real_ else unname(vv["ln_plot_area"]),
    vif_built_area = if (is.null(vv) || !("ln_built_area" %in% names(vv))) NA_real_
                     else unname(vv["ln_built_area"]),
    vif_unit_count = if (is.null(vv) || !("ln_unit_count" %in% names(vv))) NA_real_
                     else unname(vv["ln_unit_count"]),
    stringsAsFactors = FALSE)
  for (nm in names(extra_cols)) out[[nm]] <- extra_cols[[nm]]
  out
}

verdict <- function(b, p) sprintf("%s%s", if (b > 0) "+" else "-",
                                  if (p < ALPHA) " sig" else " n.s.")

# =============================================================================
# 4. PART 1 — the controls
# =============================================================================

W_full <- suppressWarnings(build_weights(ctx, XY,  spec_name = W_SPEC))
W_det  <- suppressWarnings(build_weights(ctx, XYD, spec_name = W_SPEC))

jobs <- expand.grid(s = names(SPECS), m = MEASURES, stringsAsFactors = FALSE)
t0 <- Sys.time()
res <- mclapply(seq_len(nrow(jobs)), function(i) {
  j <- jobs[i, ]; sp <- SPECS[[j$s]]
  if (sp$sample == "full") one_fit(D, W_full, j$m, sp$extra, j$s)
  else one_fit(DD, W_det, j$m, sp$extra, j$s)
}, mc.cores = min(NWORK, nrow(jobs)), mc.preschedule = FALSE)
bad <- vapply(res, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(bad)) log_msg(ctx, sprintf("!! %d of %d control fits returned nothing",
                                   sum(bad), length(res)))
cf <- do.call(rbind, res[!bad])
log_msg(ctx, sprintf("%d of %d control fits in %.0f s wall on %d workers",
                     sum(!bad), nrow(jobs),
                     as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     min(NWORK, nrow(jobs))))
cf$spec <- factor(cf$spec, levels = names(SPECS))
cf <- cf[order(match(cf$measure, MEASURES), cf$spec), ]
cf$spec <- as.character(cf$spec)
cf$spec_label <- vapply(cf$spec, function(s) SPECS[[s]]$label, character(1))
cf$sample <- vapply(cf$spec, function(s) SPECS[[s]]$sample, character(1))
cf$verdict <- mapply(verdict, cf$beta_measure, cf$p_measure)

# deltas against (a) on the same (determinate) subsample -- the like-for-like
# comparison -- and, separately, the cost of the subsample itself.
ref  <- cf[cf$spec == "a", ]; rownames(ref) <- ref$measure
full <- cf[cf$spec == "a_full", ]; rownames(full) <- full$measure
cf$d_beta_std_vs_a      <- cf$beta_std - ref[cf$measure, "beta_std"]
cf$rel_d_beta_std_vs_a  <- cf$d_beta_std_vs_a / abs(ref[cf$measure, "beta_std"])
cf$d_beta_std_vs_full   <- cf$beta_std - full[cf$measure, "beta_std"]
cf$sign_change_vs_a     <- as.integer(sign(cf$beta_measure) != sign(ref[cf$measure, "beta_measure"]))
cf$sig_change_vs_a      <- as.integer((cf$p_measure < ALPHA) != (ref[cf$measure, "p_measure"] < ALPHA))
cf$sig_either_vs_a      <- as.integer(cf$p_measure < ALPHA | ref[cf$measure, "p_measure"] < ALPHA)
cf$verdict_change_vs_a  <- ifelse(cf$sign_change_vs_a + cf$sig_change_vs_a > 0, "yes", "no")
cf$verdict_change_on_significant_vs_a <- ifelse(
  cf$sign_change_vs_a * cf$sig_either_vs_a + cf$sig_change_vs_a > 0, "yes", "no")

for (i in seq_len(nrow(cf)))
  log_msg(ctx, sprintf("  %-4s %-6s N %5d  beta %+8.4f (p %.3g)  std beta %+.4f (%+.1f %% vs (a))  lambda %+.3f rho %+.3f  R2 %.4f  %-7s  verdict change vs (a): %s",
    cf$measure[i], cf$spec[i], cf$n[i], cf$beta_measure[i], cf$p_measure[i],
    cf$beta_std[i], 100 * cf$rel_d_beta_std_vs_a[i], cf$lambda_lag[i],
    cf$rho_error[i], cf$pseudo_r2[i], cf$verdict[i],
    cf$verdict_change_on_significant_vs_a[i]))

# ---- alternative built-area definitions (report the alternative if it matters)
ALTS <- list(
  registry_all_rows   = list(col = "built_area_all_rows",
    label = "registry Built Area over ALL rows at the address (parking units in)"),
  itbi_sold_mean      = list(col = "built_area_sold_mean",
    label = "mean Built Area of the ITBI transactions that set the price"),
  geolayer_area_const = list(col = "area_const",
    label = "the frozen layer's own per-RECORD Area_Const (one unit, not the plot)")
)
alt_rows <- list()
for (an in names(ALTS)) {
  col <- ALTS[[an]]$col
  keep <- is.finite(DD[[col]]) & DD[[col]] > 0
  dat  <- DD[keep, , drop = FALSE]
  dat$ln_built_area <- lg(dat[[col]])
  Wa   <- suppressWarnings(build_weights(ctx, XYD[keep, , drop = FALSE],
                                         spec_name = W_SPEC))
  rr <- mclapply(MEASURES, function(m)
    one_fit(dat, Wa, m, c("ln_built_area", "ln_unit_count"), paste0("d_", an),
            extra_cols = list(built_area_definition = an,
                              definition_label = ALTS[[an]]$label)),
    mc.cores = min(NWORK, length(MEASURES)), mc.preschedule = FALSE)
  rr <- rr[!vapply(rr, function(r) is.null(r) || inherits(r, "try-error"), logical(1))]
  if (length(rr)) alt_rows[[an]] <- do.call(rbind, rr)
  log_msg(ctx, sprintf("alternative built-area definition '%s': N = %d (%s)",
                       an, nrow(dat), ALTS[[an]]$label))
}
alts <- if (length(alt_rows)) do.call(rbind, alt_rows) else NULL
if (!is.null(alts)) {
  dref <- cf[cf$spec == "d", ]; rownames(dref) <- dref$measure
  alts$beta_std_headline_definition <- dref[alts$measure, "beta_std"]
  alts$d_beta_std <- alts$beta_std - alts$beta_std_headline_definition
  alts$verdict <- mapply(verdict, alts$beta_measure, alts$p_measure)
  alts$verdict_headline_definition <- dref[alts$measure, "verdict"]
  alts$verdict_change <- ifelse(alts$verdict != alts$verdict_headline_definition,
                                "yes", "no")
}

# ---- collinearity, in correlations as well as VIFs ---------------------------
cor_vars <- c("ln_plot_area", "ln_built_area", "ln_unit_count",
              paste0("ln_", MEASURES))
CM <- cor(DD[, cor_vars], use = "complete.obs")
coll <- do.call(rbind, lapply(seq_along(cor_vars), function(i)
  do.call(rbind, lapply(seq_along(cor_vars), function(j)
    data.frame(var1 = cor_vars[i], var2 = cor_vars[j], pearson_r = CM[i, j],
               stringsAsFactors = FALSE)))))
coll <- coll[coll$var1 < coll$var2, ]
log_msg(ctx, sprintf("collinearity on the determinate subsample: r(ln built area, ln unit count) = %.4f, r(ln built area, ln plot area) = %.4f, r(ln unit count, ln plot area) = %.4f",
                     CM["ln_built_area", "ln_unit_count"],
                     CM["ln_built_area", "ln_plot_area"],
                     CM["ln_unit_count", "ln_plot_area"]))

# ---- spatial block CV (revision.spatial_cv), per specification --------------
cvc <- ctx$cfg$revision$spatial_cv
NB  <- as.integer(cvc$n_blocks)
if (!identical(cvc$prediction, "trend_only"))
  stop("only revision.spatial_cv.prediction = 'trend_only' is implemented", call. = FALSE)
rmse <- function(e) sqrt(mean(e^2, na.rm = TRUE))

cv_block <- function(dat, coords, specs, tag) {
  set.seed(as.integer(ctx$cfg$repro$seed))
  blk <- kmeans(coords, centers = NB, nstart = 25, iter.max = 100)$cluster
  log_msg(ctx, sprintf("%s block CV: %d k-means blocks, sizes %s", tag, NB,
                       paste(as.integer(table(blk)), collapse = " ")))
  t0 <- Sys.time()
  fr <- mclapply(seq_len(NB), function(b) {
    te <- which(blk == b); tr <- setdiff(seq_len(nrow(dat)), te)
    dtr <- dat[tr, , drop = FALSE]
    Wtr <- suppressWarnings(build_weights(ctx, coords[tr, , drop = FALSE],
                                          spec_name = W_SPEC))
    out <- list(y = dat$ln_land_value[te])
    for (s in names(specs)) for (m in MEASURES) {
      form <- frm(m, specs[[s]])
      Xte  <- model.matrix(form, data = dat[te, , drop = FALSE])
      Xtr  <- model.matrix(form, data = dtr)
      f    <- fit_spreg(ctx, form, dtr, Wtr, "sarar")
      key  <- paste(s, m)
      out[[key]] <- if (is.null(f)) rep(NA_real_, length(te)) else {
        bb <- sarar_row(f)$co[colnames(Xte)]
        as.numeric(Xte %*% bb) +
          (mean(dtr$ln_land_value) - mean(as.numeric(Xtr %*% bb)))
      }
    }
    out
  }, mc.cores = min(NWORK, NB), mc.preschedule = FALSE)
  ok <- !vapply(fr, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
  if (any(!ok)) log_msg(ctx, sprintf("!! %d of %d %s CV folds failed", sum(!ok), NB, tag))
  log_msg(ctx, sprintf("%s CV: %d folds in %.0f s wall", tag, sum(ok),
                       as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  y <- unlist(lapply(fr[ok], `[[`, "y"))
  rows <- list()
  for (s in names(specs)) for (m in MEASURES) {
    key <- paste(s, m)
    p <- unlist(lapply(fr[ok], function(r) r[[key]]))
    rows[[length(rows) + 1]] <- data.frame(
      spec = s, measure = m, sample = tag, n = nrow(dat), n_folds = sum(ok),
      n_pred = length(y), rmse_sarar_trend_recal = rmse(y - p),
      stringsAsFactors = FALSE)
  }
  nullr <- local({
    p <- numeric(nrow(dat))
    for (b in unique(blk)) { te <- which(blk == b); p[te] <- mean(dat$ln_land_value[-te]) }
    rmse(dat$ln_land_value - p)
  })
  rows[[length(rows) + 1]] <- data.frame(
    spec = "null", measure = "null_training_block_mean", sample = tag,
    n = nrow(dat), n_folds = NB, n_pred = nrow(dat),
    rmse_sarar_trend_recal = nullr, stringsAsFactors = FALSE)
  # Plot area alone, same blocks: OLS of the log outcome on log plot area,
  # fitted on the training blocks (cv_benchmarks() in common.R).
  rows[[length(rows) + 1]] <- data.frame(
    spec = "null", measure = "null_plot_area_only", sample = tag,
    n = nrow(dat), n_folds = NB, n_pred = nrow(dat),
    rmse_sarar_trend_recal = cv_benchmarks(dat, blk)[["null_plot_area_only"]],
    stringsAsFactors = FALSE)
  do.call(rbind, rows)
}

cv <- rbind(
  cv_block(D,  XY,  list(a_full = character(0)), "full"),
  cv_block(DD, XYD, lapply(SPECS[c("a", "b", "c", "d")], `[[`, "extra"), "determinate"))
for (i in seq_len(nrow(cv)))
  log_msg(ctx, sprintf("  CV %-11s %-6s %-4s RMSE %.4f", cv$sample[i], cv$spec[i],
                       cv$measure[i], cv$rmse_sarar_trend_recal[i]))
cvk <- cv[cv$spec != "null", ]
cvk$key <- paste(cvk$spec, cvk$measure)
cf$cv_rmse <- cvk$rmse_sarar_trend_recal[match(paste(cf$spec, cf$measure), cvk$key)]

# ---- the ranking, which is what the paper's claim actually is ---------------
# The performance criterion ranks the five measures on standardized beta and on
# block-CV RMSE. A control that halves every coefficient but leaves the order
# alone costs the paper its effect size, not its comparison, and the report has
# to be able to say which of the two happened.
cf$rank_std <- NA_integer_; cf$rank_cv <- NA_integer_
for (s in unique(cf$spec)) {
  i <- cf$spec == s
  cf$rank_std[i] <- rank(-cf$beta_std[i], ties.method = "min")
  cf$rank_cv[i]  <- rank(cf$cv_rmse[i], ties.method = "min")
}
ranks <- do.call(rbind, lapply(names(SPECS), function(s) {
  x <- cf[cf$spec == s, ]
  x <- x[match(MEASURES, x$measure), ]
  data.frame(spec = s, label = SPECS[[s]]$label, n = x$n[1],
             order_on_std_beta = paste(x$measure[order(-x$beta_std)], collapse = " > "),
             order_on_cv_rmse = paste(x$measure[order(x$cv_rmse)], collapse = " < "),
             stringsAsFactors = FALSE)
}))
for (i in seq_len(nrow(ranks)))
  log_msg(ctx, sprintf("  rank %-6s std beta: %-28s   CV RMSE: %s", ranks$spec[i],
                       ranks$order_on_std_beta[i], ranks$order_on_cv_rmse[i]))

# =============================================================================
# 5. PART 2 — the property-type split
# =============================================================================

grp_all <- do.call(rbind, lapply(sort(unique(D$Type)), function(g) {
  i <- D$Type == g
  data.frame(group = g, n_analysis = sum(i),
             n_layer = sum(L$Type == g, na.rm = TRUE),
             estimated = sum(i) >= MIN_N,
             median_land_value = median(D$land_value[i]),
             median_plot_area = median(D$plot_area[i]),
             stringsAsFactors = FALSE)
}))
grp_all <- grp_all[order(-grp_all$n_analysis), ]
GROUPS <- grp_all$group[grp_all$estimated]
log_msg(ctx, sprintf("property-type split: %d of %d groups reach min_n_per_group = %d (%s); the rest are reported as counts only (%s)",
                     length(GROUPS), nrow(grp_all), MIN_N,
                     paste(GROUPS, collapse = "; "),
                     paste(grp_all$group[!grp_all$estimated], collapse = "; ")))

# ---- the two weight matrices per group --------------------------------------
# `restricted` deletes the out-of-group columns of the pooled k = 8 list. That
# leaves group members with no in-group neighbour at all; they are dropped for
# that variant only, iterating because dropping one node can isolate another --
# the same `drop_isolated` handling that distance-band islands get.
NB_FULL <- suppressWarnings(spdep::knn2nb(spdep::knearneigh(XY, k = K)))

restrict_nb <- function(keep) {
  idx <- which(keep)
  repeat {
    sub <- suppressWarnings(spdep::subset.nb(NB_FULL, seq_len(nrow(XY)) %in% idx))
    iso <- which(spdep::card(sub) == 0)
    if (!length(iso)) return(list(nb = sub, idx = idx))
    idx <- idx[-iso]
    if (length(idx) < 2) stop("restricted W left fewer than two connected units",
                              call. = FALSE)
  }
}

knn_dists <- function(coords, k) {
  kn <- suppressWarnings(spdep::knearneigh(coords, k = k))
  d  <- sqrt(rowSums((coords - coords[kn$nn[, k], , drop = FALSE])^2))
  d1 <- sqrt(rowSums((coords - coords[kn$nn[, 1], , drop = FALSE])^2))
  c(mean_d1 = mean(d1), median_d1 = median(d1),
    mean_dk = mean(d), median_dk = median(d), max_dk = max(d))
}

GD <- list(); NEI <- list()
dfull <- knn_dists(XY, K)
NEI[[length(NEI) + 1]] <- data.frame(
  group = "(pooled)", w = "within_group", n = nrow(D), n_used = nrow(D),
  mean_neighbours = mean(spdep::card(NB_FULL)),
  mean_d1_m = dfull[["mean_d1"]], median_d1_m = dfull[["median_d1"]],
  mean_dk_m = dfull[["mean_dk"]], median_dk_m = dfull[["median_dk"]],
  max_dk_m = dfull[["max_dk"]], n_islands_dropped = 0L, stringsAsFactors = FALSE)

for (g in GROUPS) {
  keep <- D$Type == g
  dg   <- D[keep, , drop = FALSE]; xg <- XY[keep, , drop = FALSE]
  Wg   <- suppressWarnings(build_weights(ctx, xg, spec_name = W_SPEC))
  dk   <- knn_dists(xg, K)
  GD[[paste(g, "within_group")]] <- list(D = dg, W = Wg)
  NEI[[length(NEI) + 1]] <- data.frame(
    group = g, w = "within_group", n = nrow(dg), n_used = nrow(dg),
    mean_neighbours = K,
    mean_d1_m = dk[["mean_d1"]], median_d1_m = dk[["median_d1"]],
    mean_dk_m = dk[["mean_dk"]], median_dk_m = dk[["median_dk"]],
    max_dk_m = dk[["max_dk"]], n_islands_dropped = 0L, stringsAsFactors = FALSE)

  rr  <- restrict_nb(keep)
  dr  <- D[rr$idx, , drop = FALSE]
  Wr  <- spdep::nb2listw(rr$nb, style = ctx$cfg$weights[[W_SPEC]]$style,
                         zero.policy = FALSE)
  attr(Wr, "spec_name") <- paste0(W_SPEC, "_restricted")
  GD[[paste(g, "restricted")]] <- list(D = dr, W = Wr)
  NEI[[length(NEI) + 1]] <- data.frame(
    group = g, w = "restricted", n = nrow(dg), n_used = nrow(dr),
    mean_neighbours = mean(spdep::card(rr$nb)),
    mean_d1_m = NA_real_, median_d1_m = NA_real_,
    mean_dk_m = NA_real_, median_dk_m = NA_real_, max_dk_m = NA_real_,
    n_islands_dropped = nrow(dg) - nrow(dr), stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("group '%s': N = %d; within-group k = %d reaches a median of %.0f m (pooled %.0f m); restricted W keeps %d units, mean %.2f neighbours",
                       g, nrow(dg), K, dk[["median_dk"]], dfull[["median_dk"]],
                       nrow(dr), mean(spdep::card(rr$nb))))
}
nei <- do.call(rbind, NEI)

tjobs <- expand.grid(key = names(GD), m = MEASURES, stringsAsFactors = FALSE)
t0 <- Sys.time()
tres <- mclapply(seq_len(nrow(tjobs)), function(i) {
  j <- tjobs[i, ]
  parts <- strsplit(j$key, " (?=[^ ]+$)", perl = TRUE)[[1]]
  o <- one_fit(GD[[j$key]]$D, GD[[j$key]]$W, j$m, character(0), "type",
               extra_cols = list(group = parts[1], w = parts[2]))
  o
}, mc.cores = min(NWORK, nrow(tjobs)), mc.preschedule = FALSE)
tbad <- vapply(tres, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(tbad)) log_msg(ctx, sprintf("!! %d of %d property-type fits returned nothing",
                                    sum(tbad), length(tres)))
tf <- do.call(rbind, tres[!tbad])
log_msg(ctx, sprintf("%d of %d property-type fits in %.0f s wall on %d workers",
                     sum(!tbad), nrow(tjobs),
                     as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     min(NWORK, nrow(tjobs))))

pool <- cf[cf$spec == "a_full", ]; rownames(pool) <- pool$measure
tf$verdict <- mapply(verdict, tf$beta_measure, tf$p_measure)
tf$beta_std_pooled <- pool[tf$measure, "beta_std"]
tf$verdict_pooled  <- pool[tf$measure, "verdict"]
tf$d_beta_std      <- tf$beta_std - tf$beta_std_pooled
tf$sign_change     <- as.integer(sign(tf$beta_measure) != sign(pool[tf$measure, "beta_measure"]))
tf$sig_change      <- as.integer((tf$p_measure < ALPHA) != (pool[tf$measure, "p_measure"] < ALPHA))
tf$sig_either      <- as.integer(tf$p_measure < ALPHA | pool[tf$measure, "p_measure"] < ALPHA)
tf$verdict_change  <- ifelse(tf$sign_change + tf$sig_change > 0, "yes", "no")
tf$verdict_change_on_significant <- ifelse(
  tf$sign_change * tf$sig_either + tf$sig_change > 0, "yes", "no")
tf <- tf[order(match(tf$w, c("within_group", "restricted")),
               match(tf$group, GROUPS), match(tf$measure, MEASURES)), ]

for (i in seq_len(nrow(tf)))
  log_msg(ctx, sprintf("  %-14s %-40s %-4s N %5d  beta %+9.4f  std beta %+.4f (pooled %+.4f)  p %.3g  %-7s (pooled %-7s)  change: %s",
    tf$w[i], tf$group[i], tf$measure[i], tf$n[i], tf$beta_measure[i],
    tf$beta_std[i], tf$beta_std_pooled[i], tf$p_measure[i], tf$verdict[i],
    tf$verdict_pooled[i], tf$verdict_change_on_significant[i]))

# =============================================================================
# 6. Write the tables
# =============================================================================

w <- function(df, fn) {
  if (is.null(df) || !nrow(df)) return(invisible(NULL))
  p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p); p
}

controls_wide <- do.call(rbind, lapply(MEASURES, function(m) {
  s <- cf[cf$measure == m, ]; rownames(s) <- s$spec
  row <- data.frame(measure = m,
                    n_full = s["a_full", "n"], n_determinate = s["a", "n"],
                    stringsAsFactors = FALSE)
  for (sp in names(SPECS)) {
    row[[paste0("beta_std_", sp)]] <- s[sp, "beta_std"]
    row[[paste0("p_", sp)]]        <- s[sp, "p_measure"]
    row[[paste0("verdict_", sp)]]  <- s[sp, "verdict"]
  }
  row$d_std_subsample <- s["a", "beta_std"] - s["a_full", "beta_std"]
  row$d_std_built     <- s["b", "beta_std"] - s["a", "beta_std"]
  row$d_std_units     <- s["c", "beta_std"] - s["a", "beta_std"]
  row$d_std_both      <- s["d", "beta_std"] - s["a", "beta_std"]
  row$rel_d_std_both  <- row$d_std_both / abs(s["a", "beta_std"])
  row$verdict_survives_controls <-
    if (all(s[c("a", "b", "c", "d"), "verdict"] == s["a", "verdict"])) "yes" else "no"
  row$beta_built_area_d <- s["d", "beta_built_area"]
  row$p_built_area_d    <- s["d", "p_built_area"]
  row$beta_unit_count_d <- s["d", "beta_unit_count"]
  row$p_unit_count_d    <- s["d", "p_unit_count"]
  row$vif_max_d         <- s["d", "vif_max"]
  row
}))

types_wide <- do.call(rbind, lapply(c("within_group", "restricted"), function(wv)
  do.call(rbind, lapply(GROUPS, function(g) {
    s <- tf[tf$group == g & tf$w == wv, ]
    if (!nrow(s)) return(NULL)
    s <- s[match(MEASURES, s$measure), ]
    row <- data.frame(w = wv, group = g, n = s$n[1], stringsAsFactors = FALSE)
    for (i in seq_along(MEASURES)) {
      row[[paste0("beta_std_", MEASURES[i])]] <- s$beta_std[i]
      row[[paste0("p_", MEASURES[i])]]        <- s$p_measure[i]
      row[[paste0("verdict_", MEASURES[i])]]  <- s$verdict[i]
    }
    row$n_sign_changes <- sum(s$sign_change, na.rm = TRUE)
    row$n_sig_changes  <- sum(s$sig_change, na.rm = TRUE)
    row$n_changes_on_significant <- sum(s$verdict_change_on_significant == "yes",
                                        na.rm = TRUE)
    row$max_abs_d_beta_std <- max(abs(s$d_beta_std), na.rm = TRUE)
    row$n_at_bound <- sum(s$error_at_bound, na.rm = TRUE)
    row
  }))))

w(cf,            "stage4_controls_fits.csv")
w(ranks,         "stage4_controls_rank_stability.csv")
w(controls_wide, "stage4_controls_summary.csv")
w(by_type,       "stage4_controls_determinacy_by_type.csv")
w(coll,          "stage4_controls_collinearity.csv")
w(alts,          "stage4_controls_alternate_definitions.csv")
w(cv,            "stage4_controls_cv_rmse.csv")
w(tf,            "stage4_types_fits.csv")
w(types_wide,    "stage4_types_summary.csv")
w(grp_all,       "stage4_types_groups.csv")
w(nei,           "stage4_types_neighbours.csv")

# =============================================================================
# 7. The report — every sentence that states a result is built from the tables
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
listx <- function(x) if (length(x)) paste(x, collapse = ", ") else "none"
md <- character(0); say <- function(...) md <<- c(md, ...)

val <- function(m, sp, col) {
  v <- cf[[col]][cf$measure == m & cf$spec == sp]
  if (!length(v)) NA_real_ else v[1]
}
grp_val <- function(g, m, col, wv = "within_group") {
  v <- tf[[col]][tf$group == g & tf$measure == m & tf$w == wv]
  if (!length(v)) NA_real_ else v[1]
}
ACCESS <- unlist(ctx$cfg$measures$accessibility)
INTERM <- unlist(ctx$cfg$measures$intermediation)
sig_in <- function(sp, ms) ms[vapply(ms, function(m) isTRUE(val(m, sp, "p_measure") < ALPHA), logical(1))]
grp_sig <- function(g, ms, wv = "within_group")
  ms[vapply(ms, function(m) isTRUE(grp_val(g, m, "p_measure", wv) < ALPHA), logical(1))]

N_FULL  <- controls_wide$n_full[1]
N_DET   <- controls_wide$n_determinate[1]
iv      <- match("Vacant plot", by_type$Type)
VAC_N   <- if (is.na(iv)) NA_integer_ else by_type$n_analysis[iv]
VAC_DET <- if (is.na(iv)) NA_integer_ else by_type$n_determinate[iv]
twf     <- tf[tf$w == "within_group", ]
N_CHG   <- sum(twf$verdict_change_on_significant == "yes")
BIG     <- GROUPS[1]                                   # the largest group
BIG_SHARE <- 100 * grp_all$n_analysis[grp_all$group == BIG] / nrow(D)
FIRST_STD <- vapply(names(SPECS), function(s) {
  x <- cf[cf$spec == s, ]; x$measure[which.max(x$beta_std)] }, character(1))
FIRST_CV  <- vapply(names(SPECS), function(s) {
  x <- cf[cf$spec == s, ]; x$measure[which.min(x$cv_rmse)] }, character(1))
STD_ORDERS <- unique(ranks$order_on_std_beta)
CV_SPREAD  <- max(vapply(c("b", "c", "d"),
                         function(s) diff(range(cf$cv_rmse[cf$spec == s])), numeric(1)))
lost_b <- setdiff(sig_in("a", ACCESS), sig_in("b", ACCESS))
group_line <- function(g, wv) sprintf("%s (N = %s): %s", g,
  fmtn(grp_val(g, "PC1", "n", wv)), listx(grp_sig(g, ACCESS, wv)))

say("# Stage 4 — additional controls and the property-type split", "",
  sprintf("Generated by `src/49_controls_types.R` (`make controls-types`); `src/49_built_area.py` builds the two controls. Seed %s. Dependent variable: the rebuilt headline column. W: `%s`, kNN k = %d row-standardised, tie-break `%s`. Estimator: `sphet::spreg(model = \"sarar\", het = %s)`. Standardized β by re-estimation on z-scored covariates. α = %.2f. Every number here is a difference between two runs of one specification.",
    ctx$cfg$repro$seed, W_SPEC, K, ctx$cfg$weights$identical_points$handling,
    isTRUE(ctx$cfg$models$het), ALPHA),
  "", "---", "",
  "## 0. What was asked, and the short answer", "",
  "Reviewer 4: *\"…either add built area / unit count as controls or rename the variable (eg improved-property value per m² of land) and pull back the 'land value' claims accordingly.\"*",
  "",
  "Reviewer 3: *\"…conduct sensitivity analyses excluding parking transactions, **separating major property types**, and **controlling for built area or development intensity**.\"*",
  "",
  sprintf("1. **Built area.** Adding log built area to the headline model takes pseudo-R² from **%.3f to %.3f** and block-CV RMSE from **%.3f to %.3f**, with a coefficient of **%.3f** on log built area against **%.3f** on log plot area. PC1's standardized β goes **%.4f → %.4f** (%+.0f %%); with both controls in, its p-value is %.3f. Accessibility measures significant in (a) but not in (b): %s.",
    val("PC1", "a", "pseudo_r2"), val("PC1", "b", "pseudo_r2"),
    val("PC1", "a", "cv_rmse"), val("PC1", "b", "cv_rmse"),
    val("PC1", "b", "beta_built_area"), val("PC1", "b", "beta_area"),
    val("PC1", "a", "beta_std"), val("PC1", "b", "beta_std"),
    100 * (val("PC1", "b", "beta_std") / val("PC1", "a", "beta_std") - 1),
    val("PC1", "d", "p_measure"), listx(lost_b)),
  "",
  sprintf("2. **The comparison between measures.** First on standardized β across the five specifications: %s. First on block-CV RMSE: %s. The standardized ordering is **%s** in %d of the five%s. In the specifications with a control, the five block-CV RMSEs lie within **%.3f** of each other.",
    paste(sprintf("%s %s", names(FIRST_STD), FIRST_STD), collapse = ", "),
    paste(sprintf("%s %s", names(FIRST_CV), FIRST_CV), collapse = ", "),
    STD_ORDERS[1], sum(ranks$order_on_std_beta == STD_ORDERS[1]),
    if (length(STD_ORDERS) > 1) sprintf("; the other order(s): %s", paste(STD_ORDERS[-1], collapse = "; ")) else "",
    CV_SPREAD),
  "",
  sprintf("3. **Property-type split** (within-group W). Accessibility measures significant at α = %.2f, by group: %s. %d of the %d estimated group × measure cells differ from the pooled model in sign or significance on a measure that is significant somewhere.",
    ALPHA, paste(vapply(GROUPS, group_line, character(1), wv = "within_group"), collapse = "; "),
    N_CHG, nrow(twf)),
  if (!is.na(VAC_N) && "Vacant plot" %in% GROUPS)
    c("", sprintf("4. **Vacant plots**, where value per m² of plot has no building in it: on %s observations the standardized β of PC1 / PC2 / CC is %.3f / %.3f / %.3f (p = %.3f / %.3f / %.3f).",
      fmtn(VAC_N), grp_val("Vacant plot", "PC1", "beta_std"),
      grp_val("Vacant plot", "PC2", "beta_std"), grp_val("Vacant plot", "CC", "beta_std"),
      grp_val("Vacant plot", "PC1", "p_measure"), grp_val("Vacant plot", "PC2", "p_measure"),
      grp_val("Vacant plot", "CC", "p_measure"))) else character(0),
  "", "---", "",
  "## 1. How the two controls are defined", "",
  "Both come from the same rows the dependent variable's own unit multiplier counts — the deduplicated registry, `multiplier_units: non_parking`, parent rows included, with the only-parking fallback. `src/49_built_area.py` sums the built area over exactly those rows, and this script stops unless the controls file's `n_units` equals stage 1's on every record.",
  "",
  "- **Unit count U(a)** = stage 1's `n_units`, i.e. the dependent variable's multiplier itself.",
  "- **Built area B(a)** = the registry's `Built Area` summed over those same rows.",
  sprintf("- **Determinate** = none of the counted rows carries the literal `%s` and B(a) > 0. **%s** of the %s layer records qualify before the exclusions and **%s** of the **%s** in the analysis sample; the lenient rule (any counted row numeric) would keep %s of the %s.",
    CC$built_area$indeterminate_code,
    fmtn(sum(CTL$built_area_determinate == 1)), fmtn(nrow(CTL)), fmtn(N_DET), fmtn(N_FULL),
    fmtn(sum(CTL$built_area_determinate_lenient == 1)), fmtn(nrow(CTL))),
  "",
  "**The unit count is the dependent variable's numerator.** By construction",
  "",
  "```",
  "ln(land value) = ln(mean unit price) + ln U(a) − ln(plot area)",
  "```",
  "",
  sprintf("and this run measures the residual of that identity as **%.3g** over all %s records. Putting `ln U(a)` on the right-hand side moves U from one side of the equation to the other, so specifications (c) and (d) answer a different question: how mean unit price per m² of land relates to centrality. They are reported because that is the alternative Reviewer 4 offers to renaming the variable, not as a robustness check of the headline.",
    ident, fmtn(nrow(D))),
  "", "### 1.1 Which records lose their built area", "",
  md_table(by_type), "",
  if (!is.na(VAC_N)) sprintf("%d of the %d vacant records in the analysis sample carry a positive determinate built area, so vacant plots effectively leave the sample when `ln built area` enters; evidence on them comes from the property-type split in §3. Overall %s of %s records leave (%.2f %%).",
    VAC_DET, VAC_N, fmtn(N_FULL - N_DET), fmtn(N_FULL),
    100 * (N_FULL - N_DET) / N_FULL) else
    sprintf("Overall %s of %s records leave (%.2f %%).", fmtn(N_FULL - N_DET), fmtn(N_FULL),
            100 * (N_FULL - N_DET) / N_FULL),
  "", "---", "",
  "## 2. The controls", "",
  "Four specifications per measure, all with `ln_plot_area`:",
  "",
  "| spec | model |",
  "|---|---|",
  "| (a) | `ln_land_value ~ ln_plot_area + ln_<measure>` — the headline |",
  "| (b) | (a) + `ln_built_area` |",
  "| (c) | (a) + `ln_unit_count` |",
  "| (d) | (a) + `ln_built_area` + `ln_unit_count` |",
  "",
  "(a) is run **twice** — once on the full sample and once on the determinate subsample — so the cost of the *subsample* is separated from the cost of the *controls*. W is rebuilt on whichever points are in the sample.",
  "", md_table(controls_wide[, c("measure", "n_full", "n_determinate",
    "beta_std_a_full", "beta_std_a", "beta_std_b", "beta_std_c", "beta_std_d",
    "d_std_subsample", "d_std_both", "verdict_survives_controls")]), "",
  sprintf("Moving from the full sample to the determinate subsample, before any control is added, changes PC1's standardized β by %+.4f (%.4f → %.4f, %+.0f %%).",
    controls_wide$d_std_subsample[controls_wide$measure == "PC1"],
    val("PC1", "a_full", "beta_std"), val("PC1", "a", "beta_std"),
    100 * (val("PC1", "a", "beta_std") / val("PC1", "a_full", "beta_std") - 1)),
  "", "### 2.1 Every fit", "",
  md_table(cf[, c("measure", "spec", "n", "beta_measure", "se_measure", "p_measure",
                  "beta_std", "lambda_lag", "rho_error",
                  "error_at_bound", "pseudo_r2", "cv_rmse", "verdict",
                  "verdict_change_on_significant_vs_a")]), "",
  sprintf("ρ sits on the estimator's ±%.1f bound in %d of these %d fits. For PC1, λ moves from %.2f in (a) to %.2f in (b) and ρ from %+.2f to %+.2f.",
    BOUND, sum(cf$error_at_bound, na.rm = TRUE), nrow(cf),
    val("PC1", "a", "lambda_lag"), val("PC1", "b", "lambda_lag"),
    val("PC1", "a", "rho_error"), val("PC1", "b", "rho_error")),
  "", "### 2.2 The control coefficients themselves", "",
  md_table(cf[cf$spec %in% c("b", "c", "d"),
              c("measure", "spec", "beta_built_area", "p_built_area",
                "beta_unit_count", "p_unit_count", "beta_area", "p_area",
                "vif_built_area", "vif_unit_count", "vif_max")]), "",
  sprintf("In (b), for PC1, the coefficients are **%.3f** on ln built area and **%.3f** on ln plot area; they sum to %.2f. A sum near zero means the dependent variable moves with `built area / plot area`, i.e. with building intensity.",
    val("PC1", "b", "beta_built_area"), val("PC1", "b", "beta_area"),
    val("PC1", "b", "beta_built_area") + val("PC1", "b", "beta_area")),
  "", "### 2.3 Collinearity", "",
  md_table(coll), "",
  sprintf("On the determinate subsample r(ln B, ln U) = **%.3f**, r(ln B, ln plot area) = %.3f and r(ln U, ln plot area) = %.3f. The largest VIF in specification (d) is **%.2f**; the largest VIF on any `ln <measure>` anywhere in the table is %.2f.",
    CM["ln_built_area", "ln_unit_count"], CM["ln_built_area", "ln_plot_area"],
    CM["ln_unit_count", "ln_plot_area"], max(cf$vif_max[cf$spec == "d"], na.rm = TRUE),
    max(cf$vif_measure, na.rm = TRUE)),
  "", "### 2.4 The ranking of the measures", "",
  md_table(ranks), "",
  sprintf("Distinct orders on the standardized coefficient across the five specifications: %d. Distinct orders on block-CV RMSE: %d; with a control in the model the five RMSEs lie within %.3f of one another.",
          length(STD_ORDERS), length(unique(ranks$order_on_cv_rmse)), CV_SPREAD),
  "", "### 2.5 Alternative built-area definitions", "")
if (!is.null(alts)) say(
  md_table(alts[, c("built_area_definition", "measure", "n", "beta_measure",
                    "p_measure", "beta_std", "beta_std_headline_definition",
                    "d_beta_std", "verdict", "verdict_change")]), "",
  sprintf("The headline definition sums the registry over the counted units. The alternates are reported because the choice is a convention: summing over *all* registry rows adds the parking boxes back; the ITBI mean is the built area of the units that actually sold; `Area_Const` is the frozen layer's own per-record figure, a single unit's floor area. %d of the %d alternate fits change the verdict relative to the headline definition.",
    sum(alts$verdict_change == "yes"), nrow(alts)), "")

say("---", "", "## 3. The property-type split", "",
  md_table(grp_all), "",
  sprintf("Groups below n = %d are reported as counts and not estimated. The estimated groups get two weight matrices: `within_group`, the k = %d list rebuilt on the group's own points, which is the headline because it keeps the specification fixed while the group changes; and `restricted`, the pooled %s-point list with the out-of-group columns deleted, which keeps the pooled geometry but lets the neighbour count collapse with the group's density.",
    MIN_N, K, fmtn(nrow(D))),
  "", sprintf("### 3.1 What k = %d means in each group", K), "",
  md_table(nei), "",
  sprintf("Median distance to the %d-th nearest neighbour within each group: %s. The within-group models are all k = %d models, but they are not models of the same physical neighbourhood.",
    K, paste(sprintf("%s %.0f m", nei$group[nei$w == "within_group"],
                     nei$median_dk_m[nei$w == "within_group"]), collapse = "; "), K),
  "", "### 3.2 Results per group and measure (headline W = within_group)", "",
  md_table(twf[, c("group", "measure", "n", "beta_measure", "se_measure",
                   "p_measure", "beta_std", "beta_std_pooled", "lambda_lag",
                   "rho_error", "pseudo_r2", "verdict", "verdict_pooled",
                   "verdict_change_on_significant")]), "",
  "### 3.3 Where a sign or a significance differs from the pooled model", "")

chg <- twf[twf$verdict_change == "yes",
           c("group", "measure", "n", "beta_measure", "p_measure", "beta_std",
             "verdict", "verdict_pooled", "sign_change", "sig_change",
             "verdict_change_on_significant")]
if (nrow(chg)) say(md_table(chg), "") else
  say("No group × measure cell changes sign or significance relative to the pooled model.", "")
say(sprintf("**%d** of these involve a measure that is significant in the pooled model or in the group; the rest are sign flips on coefficients indistinguishable from zero on both sides.",
            N_CHG),
  "",
  sprintf("The largest group, **%s**, holds %.0f %% of the sample (N = %s). Its PC1 standardized β is **%.4f** (p = %.3f) and its CC standardized β is %.4f (p = %.3f).",
    BIG, BIG_SHARE, fmtn(grp_val(BIG, "PC1", "n")), grp_val(BIG, "PC1", "beta_std"),
    grp_val(BIG, "PC1", "p_measure"), grp_val(BIG, "CC", "beta_std"),
    grp_val(BIG, "CC", "p_measure")),
  "", "### 3.4 The robustness line: the pooled W restricted to the group", "",
  md_table(types_wide[types_wide$w == "restricted",
                      c("group", "n", "n_sign_changes", "n_sig_changes",
                        "n_changes_on_significant", "max_abs_d_beta_std")]), "",
  sprintf("Restricting the pooled list drops **%d** units across the %d groups (members left with no in-group neighbour) and leaves mean neighbour counts between **%.2f** and **%.2f** instead of %d. Accessibility measures significant under the restricted W, by group: %s.",
    sum(nei$n_islands_dropped), length(GROUPS),
    min(nei$mean_neighbours[nei$w == "restricted"]),
    max(nei$mean_neighbours[nei$w == "restricted"]), K,
    paste(vapply(GROUPS, group_line, character(1), wv = "restricted"), collapse = "; ")),
  "", "---", "",
  "## 4. Summary", "",
  sprintf("- **Sign of the accessibility coefficients.** Positive in %d of the %d control-specification fits, and PC1 positive in %d of the %d estimated property-type groups.",
    sum(cf$beta_measure[cf$measure %in% ACCESS] > 0), sum(cf$measure %in% ACCESS),
    sum(vapply(GROUPS, function(g) isTRUE(grp_val(g, "PC1", "beta_measure") > 0),
               logical(1))), length(GROUPS)),
  sprintf("- **Magnitude.** With built area in the model PC1's standardized β is %.0f %% of its value in (a); with both controls its p-value is %.3f.",
    100 * val("PC1", "b", "beta_std") / val("PC1", "a", "beta_std"), val("PC1", "d", "p_measure")),
  sprintf("- **Intermediation measures significant at α = %.2f.** Control specifications: %s. Property-type groups (within-group W): %s.",
    ALPHA,
    listx(with(cf[cf$measure %in% INTERM & cf$p_measure < ALPHA, ], paste(measure, spec))),
    listx(with(twf[twf$measure %in% INTERM & twf$p_measure < ALPHA, ], paste(measure, group, sep = " in ")))),
  sprintf("- **Within the largest group** (%s, %.0f %% of the sample), significant accessibility measures: %s.",
    BIG, BIG_SHARE, listx(grp_sig(BIG, ACCESS))),
  "", "---", "",
  "## 5. Scope and limits", "",
  sprintf("- The dependent variable is the rebuilt column and W is the proposed k = %d matrix. Both are held fixed across every line here, so every number is a difference between two runs of one specification.", K),
  "- Block CV is reported for the controls (k-means blocks, W rebuilt inside each training set, trend-only prediction with the intercept recalibrated on the training residual mean). It is not reported per property-type group: a group of a few hundred addresses scattered over the whole municipality does not decompose into spatial blocks in any meaningful sense, and RMSE in log units is not comparable across groups whose dependent variables have different dispersions.",
  sprintf("- %d addresses have their multiplier collapsed to one by stage 1's parent-only rule while B(a) still sums every counted row; they are flagged in `%s` (`parent_only_collapse`).",
    sum(CTL$parent_only_collapse), CC$path),
  "- The exclusion order, the estimator, the standardization, the bound flag and the CV protocol are the same code paths as `src/47_sample_sensitivity.R` and `src/48_rebuilt_dv.R`, so these rows are comparable cell for cell with `reports/stage4_rebuilt_dv.md`.",
  "- Nothing here re-runs a centrality: PC1, PC2, CC, BC and FK are frozen inputs throughout.",
  "")

pmd <- out_report(ctx, "stage4_controls_and_types.md")
writeLines(md, pmd)
log_msg(ctx, "wrote ", pmd)

write_counts(ctx, "stage4_controls_types_counts.csv")
finish(ctx)
