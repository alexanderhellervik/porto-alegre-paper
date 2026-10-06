#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 -- preferential-centrality (gamma, beta) sensitivity, cell level.
#
# Does the analysis depend on the preferential-centrality model's parameters?
# `revision.pc_sensitivity` defines a (gamma, beta) grid -- beta in
# `beta_grid`, gamma in `gamma_grid` -- with every other setting in
# `inputs.pc_solver.submitted_run` held fixed. The runs are produced by
# src/11_pc_runs.py and put on the 500 m cells and the 9,030 address points by
# src/21_pc_beta_layers.py.
#
# Measure slots. `PC1` is the gamma = 1 run, `PC2` the gamma = 0 run
# (preferential centrality without agglomeration) -- the paper's two measures --
# and `PC3` the optional gamma = 0.5 run at the same beta. The slots live in
# `revision.pc_sensitivity` alone; `measures.accessibility` is untouched.
#
# What is here:
#   1. Correlations with land value per (gamma, beta) -- Pearson on logs,
#      Spearman on levels, at the cell level and at the address level.
#   2. SARAR per (gamma, beta) at the cell level, under both weight
#      specifications: the W of the submitted tables (`weights.active`, the
#      6,100 m band) and the revision's kNN W (`weights.proposed`, k = 6). Same
#      formula as every other cell-level model, raw and standardized
#      coefficients, and the flag for rho on sphet's nlminb bound.
#   3. Spearman rank stability -- between variants and against the submitted
#      columns, at both levels.
#   4. The (gamma, beta) convergence map: the solver's own final step norm at
#      each grid point, and which of them are converged at the iteration budget.
#
# At the address level this script reports correlations only; no model is
# estimated there.
#
# Dependent variable: `dependent_variable.proposed_source` (the rebuilt column,
# as in every table of the revised manuscript), at both levels.
#
# Run:  make pc-beta      (Rscript src/41_pc_beta_sensitivity.R)
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

LEVEL <- "aggregated"
ctx <- init(4, "pc_beta_sensitivity")
capture_env_r(ctx)

# Cap the BLAS underneath sphet at compute.max_workers. Set here as well as in
# the Makefile so that a bare `Rscript` call is capped too; a BLAS that sizes
# its thread pool from `nproc` sees the host, not a container's quota.
MAXW <- as.integer(ctx$cfg$compute$max_workers)
Sys.setenv(OMP_NUM_THREADS = MAXW, OPENBLAS_NUM_THREADS = MAXW,
           MKL_NUM_THREADS = MAXW)
log_msg(ctx, sprintf("thread cap = %d (compute.max_workers); OMP_NUM_THREADS=%s",
                     MAXW, Sys.getenv("OMP_NUM_THREADS")))

W_SUB  <- ctx$cfg$weights$active[[LEVEL]]          # the submitted tables' W
W_PROP <- ctx$cfg$weights$proposed[[LEVEL]]        # the revision's W
W_SPECS <- c(submitted = W_SUB, proposed = W_PROP)
describe_w <- function(sp) {
  s <- ctx$cfg$weights[[sp]]
  if (identical(s$type, "knn")) sprintf("kNN k = %s", s$k)
  else sprintf("%s m distance band", format(as.numeric(s$d_max_m), big.mark = ","))
}
# Converged, for the purposes of this report: the last iterate moved the state
# by less than this in the solver's own `max_step_rel_mean` criterion.
CONV_TOL <- 1e-8
BOUND  <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
ALPHA  <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
CTRL   <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
SENS   <- ctx$cfg$revision$pc_sensitivity
STD    <- isTRUE(ctx$cfg$revision$standardize_coefficients)
RUN0   <- ctx$cfg$inputs$pc_solver$submitted_run

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c")
  log(x + ctx$cfg$transforms$log_plus_c) else log(x)

# --- the run grid, labelled by parameters, not by column name ----------------
runs <- SENS$input_files
if (!length(runs)) stop("revision.pc_sensitivity.input_files is empty", call. = FALSE)
PARAMS <- ctx$cfg$measures$pc_variant_parameters
MEAS_SLOTS <- c("PC1", "PC2", "PC3")            # gamma 1 / gamma 0 / gamma 0.5
slots_of <- function(r)
  MEAS_SLOTS[vapply(MEAS_SLOTS, function(m) !is.null(r[[m]]), logical(1))]
# The parameter table, not the column name, is the authority on (gamma, beta,
# Dp): `b225` reads as 22.5 to a name-only parser.
param_of <- function(col, field) {
  p <- PARAMS[[sub("_(mean|sum)$", "", col)]]
  if (is.null(p) || is.null(p[[field]])) NA_real_ else as.numeric(p[[field]])
}
lab <- function(col, meas) {
  # the cell layer's columns carry the aggregation suffix; the parameter table
  # is keyed by the segment-level stem
  stem <- sub("_(mean|sum)$", "", col)
  p <- PARAMS[[stem]]
  if (is.null(p)) return(sprintf("%s (`%s`)", meas, col))
  sprintf("%s  gamma=%g beta=%g Dp=%g", meas, as.numeric(p$gamma),
          as.numeric(p$beta), as.numeric(p$dp))
}

# =============================================================================
# 1. The cell-level sample -- exactly as every other cell-level analysis
# =============================================================================

# The dependent variable of the revised tables
# (`dependent_variable.proposed_source`; proposed_dv_layer() in common.R).
DV_SOURCE <- ctx$cfg$dependent_variable$proposed_source
Land <- proposed_dv_layer(ctx, LEVEL)
crs_expected <- ctx$cfg$repro$crs_epsg
if (is.na(st_crs(Land)$epsg) || st_crs(Land)$epsg != crs_expected)
  Land <- st_transform(Land, crs_expected)
if (isTRUE(st_is_longlat(Land)))
  stop("layer is in geographic coordinates; distance weights would be wrong",
       call. = FALSE)
log_msg(ctx, sprintf("CRS confirmed EPSG:%s", st_crs(Land)$epsg))

id_col <- ctx$cfg$grid$frozen_500m$id_column
Land$cell_id <- Land[[id_col]]

# the beta layer, joined on the frozen cell id
bpath <- file.path(REPO, SENS$aggregated_layer)
if (!file.exists(bpath))
  stop("missing ", SENS$aggregated_layer, " -- run `make pc-beta-layers` first",
       call. = FALSE)
B <- st_drop_geometry(st_read(bpath, quiet = TRUE))
log_msg(ctx, sprintf("beta layer: %d cells, %d columns", nrow(B), ncol(B)))
bcols <- setdiff(names(B), c("cell_id", "row_index", "col_index"))
Land <- merge(Land, B[, c("cell_id", bcols)], by = "cell_id", all.x = TRUE,
              sort = FALSE)

map <- ctx$cfg$measures$columns[[LEVEL]]
for (canon in names(map)) {
  src <- map[[canon]]
  if (!is.null(Land[[src]]) && src != canon) Land[[canon]] <- Land[[src]]
}
Land <- apply_exclusions(ctx, Land, LEVEL)
target_n <- ctx$cfg$exclusions$targets[[paste0(LEVEL, "_n")]]
log_msg(ctx, sprintf("N = %d on the %s dependent variable (submitted aggregated N = %d)%s",
                     nrow(Land), DV_SOURCE, target_n,
                     if (nrow(Land) == target_n) " MATCH" else " -- differs"))
if (identical(DV_SOURCE, "frozen") && nrow(Land) != target_n)
  stop("aggregated N does not match the submitted sample; stopping", call. = FALSE)

for (v in c("land_value", "PC1", "PC2", "plot_area"))
  Land[[paste0("ln_", v)]] <- lg(Land[[v]])

pts    <- suppressWarnings(st_point_on_surface(Land))
coords <- st_coordinates(pts)[, 1:2]
D      <- st_drop_geometry(Land)

# Both weight specifications, each with its own connected subset (a distance
# band can leave islands; a kNN cannot).
WS <- list()
for (nm in names(W_SPECS)) {
  sp   <- W_SPECS[[nm]]
  keep <- connected_idx(ctx, coords, sp)
  WS[[nm]] <- list(spec = sp, keep = keep,
                   W = build_weights(ctx, coords[keep, , drop = FALSE], spec_name = sp),
                   D = D[keep, , drop = FALSE])
  count_step(ctx, sprintf("cells entering the SARAR (%s W)", nm), length(keep),
             sprintf("W = %s", sp))
}

# =============================================================================
# 2. Per-(gamma, beta) correlations and SARAR, cell level
# =============================================================================

# The submitted columns are the beta = 2 anchor: the frozen `cd1g1b2k0_mean` /
# `cd4g0b2k0_mean` of the submitted analysis. Every variant is reported beside
# them.
series <- list()
series[["submitted|PC1"]] <- list(
  label = "submitted", beta = 2.0, gamma = 1, measure = "PC1", role = "submitted",
  column = map$PC1, cell_col = map$PC1, values = D[[map$PC1]])
series[["submitted|PC2"]] <- list(
  label = "submitted", beta = 2.0, gamma = 0, measure = "PC2", role = "submitted",
  column = map$PC2, cell_col = map$PC2, values = D[[map$PC2]])
for (r in runs) {
  for (meas in slots_of(r)) {
    cn <- sprintf("%s_%s_mean", r$label, meas)
    if (is.null(D[[cn]])) { log_msg(ctx, "!! missing column ", cn); next }
    series[[paste(r$label, meas, sep = "|")]] <- list(
      label = r$label, beta = as.numeric(r$beta),
      gamma = param_of(r[[meas]], "gamma"), measure = meas, role = r$role,
      column = r[[meas]], cell_col = cn, values = D[[cn]])
  }
}
log_msg(ctx, sprintf("series to score: %s", paste(names(series), collapse = ", ")))

rows <- list()
for (nm in names(series)) {
  s  <- series[[nm]]
  v  <- s$values
  ok <- !is.na(v) & v > 0
  count_step(ctx, paste0("PC series ", nm), sum(ok),
             sprintf("positive on the %d-cell sample", nrow(D)))
  pear <- cor(D$ln_land_value[ok], lg(v[ok]), method = "pearson")
  spea <- cor(D$land_value[ok], v[ok], method = "spearman")
  sdln <- sd(lg(v[ok]))

  for (wnm in names(WS)) {
    dh <- WS[[wnm]]$D
    dh$ln_variant <- lg(dh[[s$cell_col]])
    okh <- is.finite(dh$ln_variant)
    fit <- fit_spreg(ctx, as.formula(sprintf("ln_land_value ~ %sln_variant", CTRL)),
                     dh[okh, , drop = FALSE], WS[[wnm]]$W, "sarar")
    bb <- se <- pv <- lam <- rr <- r2 <- NA_real_; ab <- NA_integer_
    if (!is.null(fit) && sum(okh) == nrow(dh)) {
      zz  <- sarar_row(fit)
      bb  <- zz$co[["ln_variant"]]; se <- zz$se[["ln_variant"]]
      pv  <- zz$p[["ln_variant"]]
      lam <- zz$co[["lambda"]];     rr <- zz$co[["rho"]]
      r2  <- pseudo_r2(fit, dh$ln_land_value[okh])
      ab  <- as.integer(abs(abs(rr)  - BOUND) < 1e-6)   # only rho is bounded
    }
    rows[[length(rows) + 1]] <- data.frame(
      series = nm, label = s$label, role = s$role, measure = s$measure,
      gamma = s$gamma, beta = s$beta, column = s$column,
      parameter_label = lab(s$column, s$measure),
      n_positive = sum(ok), pearson_ln_landvalue = pear, spearman_landvalue = spea,
      sd_ln = sdln, w_role = wnm, w_spec = WS[[wnm]]$spec, n_sarar = sum(okh),
      sarar_beta = bb, sarar_se = se, sarar_p = pv,
      sarar_beta_std = if (STD) bb * sdln else NA_real_,
      lambda_lag = lam, rho_error = rr,
      error_at_bound = ab, pseudo_r2 = r2,
      stringsAsFactors = FALSE)
    log_msg(ctx, sprintf(
      "%-14s gamma=%s beta=%.2f %s [%s W]: r=%.4f rho_s=%.4f  SARAR b=%+.4f (se %.4f) std=%+.4f  lambda=%+.3f rho=%+.3f%s  R2=%.4f",
      s$label, format(s$gamma), s$beta, s$measure, wnm, pear, spea, bb, se,
      bb * sdln, lam, rr, if (isTRUE(ab == 1L)) " AT BOUND" else "", r2))
  }
}
agg_tab <- do.call(rbind, rows)

# =============================================================================
# 3. Rank stability, cell level -- pairwise Spearman
# =============================================================================

AGG_LEVEL <- sprintf("aggregated (%d cells)", nrow(D))
rk <- as.data.frame(lapply(series, function(s) rank(s$values, na.last = "keep")))
names(rk) <- names(series)
stab <- suppressWarnings(cor(rk, use = "pairwise.complete.obs", method = "spearman"))
rs <- data.frame(
  a = rep(rownames(stab), times = ncol(stab)),
  b = rep(colnames(stab), each = nrow(stab)),
  spearman = as.numeric(stab), stringsAsFactors = FALSE)
rs <- rs[rs$a < rs$b, ]
rs$level <- AGG_LEVEL
rs$same_measure <- sub(".*\\|", "", rs$a) == sub(".*\\|", "", rs$b)
for (i in seq_len(nrow(rs)))
  log_msg(ctx, sprintf("   rank stability %-16s vs %-16s rho = %.6f",
                       rs$a[i], rs$b[i], rs$spearman[i]))

# =============================================================================
# 4. Address-level correlations (no model is estimated at this level here)
# =============================================================================

dis_tab <- NULL; rs_dis <- NULL
ppath <- file.path(REPO, SENS$points_table)
if (file.exists(ppath)) {
  P  <- read.csv(ppath)
  # proposed_dv_layer() keeps the frozen layer's own row order
  Pt <- st_drop_geometry(proposed_dv_layer(ctx, "disaggregated"))
  if (nrow(P) != nrow(Pt))
    stop("point table has ", nrow(P), " rows, frozen layer ", nrow(Pt), call. = FALSE)
  # `point_index` is the frozen layer's own row order, which is how
  # src/21_pc_beta_layers.py wrote it.
  P <- P[order(P$point_index), ]
  dmap <- ctx$cfg$measures$columns$disaggregated
  for (r in runs) for (meas in slots_of(r))
    Pt[[paste(r$label, meas, sep = "_")]] <- P[[paste(r$label, meas, sep = "_")]]
  Pt <- apply_exclusions(ctx, Pt, "disaggregated")
  tgt <- ctx$cfg$exclusions$targets$disaggregated_n
  log_msg(ctx, sprintf("disaggregated N = %d on the %s dependent variable (submitted %d)%s",
                       nrow(Pt), DV_SOURCE, tgt,
                       if (nrow(Pt) == tgt) " MATCH" else " -- differs"))
  if (identical(DV_SOURCE, "frozen") && nrow(Pt) != tgt)
    stop("disaggregated N does not match the submitted sample; stopping", call. = FALSE)

  dser <- list()
  dser[["submitted|PC1"]] <- list(label = "submitted", beta = 2.0, gamma = 1,
                                  measure = "PC1", role = "submitted",
                                  column = dmap$PC1, values = Pt[[dmap$PC1]])
  dser[["submitted|PC2"]] <- list(label = "submitted", beta = 2.0, gamma = 0,
                                  measure = "PC2", role = "submitted",
                                  column = dmap$PC2, values = Pt[[dmap$PC2]])
  for (r in runs) for (meas in slots_of(r)) {
    cn <- paste(r$label, meas, sep = "_")
    if (is.null(Pt[[cn]])) next
    dser[[paste(r$label, meas, sep = "|")]] <- list(
      label = r$label, beta = as.numeric(r$beta),
      gamma = param_of(r[[meas]], "gamma"), measure = meas, role = r$role,
      column = r[[meas]], values = Pt[[cn]])
  }
  lv <- Pt[[dmap$land_value]]
  drows <- list()
  for (nm in names(dser)) {
    s <- dser[[nm]]; v <- s$values
    ok <- !is.na(v) & v > 0 & !is.na(lv) & lv > 0
    drows[[length(drows) + 1]] <- data.frame(
      series = nm, label = s$label, role = s$role, measure = s$measure,
      gamma = s$gamma, beta = s$beta, column = s$column, n_positive = sum(ok),
      pearson_ln_landvalue = cor(log(lv[ok]), lg(v[ok]), method = "pearson"),
      spearman_landvalue = cor(lv[ok], v[ok], method = "spearman"),
      note = "correlation only; no model estimated at the address level",
      stringsAsFactors = FALSE)
  }
  dis_tab <- do.call(rbind, drows)
  for (i in seq_len(nrow(dis_tab)))
    log_msg(ctx, sprintf("disaggregated %-14s %s: r=%.4f rho_s=%.4f (n=%d)",
                         dis_tab$label[i], dis_tab$measure[i],
                         dis_tab$pearson_ln_landvalue[i],
                         dis_tab$spearman_landvalue[i], dis_tab$n_positive[i]))

  rkd <- as.data.frame(lapply(dser, function(s) rank(s$values, na.last = "keep")))
  names(rkd) <- names(dser)
  sd_ <- suppressWarnings(cor(rkd, use = "pairwise.complete.obs", method = "spearman"))
  rs_dis <- data.frame(
    a = rep(rownames(sd_), times = ncol(sd_)),
    b = rep(colnames(sd_), each = nrow(sd_)),
    spearman = as.numeric(sd_), stringsAsFactors = FALSE)
  rs_dis <- rs_dis[rs_dis$a < rs_dis$b, ]
  rs_dis$level <- sprintf("disaggregated (%d points)", nrow(Pt))
  rs_dis$same_measure <- sub(".*\\|", "", rs_dis$a) == sub(".*\\|", "", rs_dis$b)
} else {
  log_msg(ctx, "!! no point table at ", SENS$points_table, " -- skipping the ",
          "address-level correlations")
}

rank_tab <- if (is.null(rs_dis)) rs else rbind(rs, rs_dis)

# ---- summary quantities for the report --------------------------------------
AGG1 <- agg_tab[agg_tab$w_role == "submitted", ]    # one row per series
anchor <- AGG1[AGG1$role == "submitted", ]
cross <- rank_tab[rank_tab$same_measure & rank_tab$level == AGG_LEVEL, ]
is_sub <- function(df) grepl("^submitted\\|", df$a) | grepl("^submitted\\|", df$b)
lab_of <- function(x) sub("\\|.*$", "", x)
is_repro <- function(df) lab_of(df$a) == "b2" | lab_of(df$b) == "b2"
vs_sub_agg <- cross[is_sub(cross) & !is_repro(cross), ]
crossd <- rank_tab[rank_tab$same_measure & rank_tab$level != AGG_LEVEL, ]
vs_sub_dis <- crossd[is_sub(crossd) & !is_repro(crossd), ]
min_or_na <- function(x) if (length(x) && any(!is.na(x))) min(x, na.rm = TRUE) else NA_real_
min_rho_vs_sub_agg <- min_or_na(vs_sub_agg$spearman)
min_rho_vs_sub_dis <- min_or_na(vs_sub_dis$spearman)
min_rho_any <- min_or_na(cross$spearman)
worst_pair <- cross[which.min(cross$spearman), ]

# =============================================================================
# 5. The (gamma, beta) convergence map -- the solver's own record
# =============================================================================
# `final_step_norm` is what the last iteration was still moving by, in the
# solver's `max_step_rel_mean` criterion. Read back from the run table written
# by src/11_pc_runs.py rather than restated, so the report cannot drift from
# the runs.
rt_path <- out_table(ctx, "stage0_pc_runs.csv")
if (!file.exists(rt_path))
  stop("missing ", rt_path, " -- run `make pc-runs` first", call. = FALSE)
runtab <- read.csv(rt_path)
runtab$converged <- runtab$final_step_norm < CONV_TOL
runtab <- runtab[order(-runtab$gamma, runtab$beta), ]
betas  <- sort(unique(runtab$beta))
gammas <- sort(unique(runtab$gamma), decreasing = TRUE)
cell_of <- function(g, b) {
  r <- runtab[runtab$gamma == g & runtab$beta == b, ]
  if (!nrow(r)) return("—")
  sprintf("%s %s", formatC(r$final_step_norm[1], format = "e", digits = 2),
          if (r$converged[1]) "✓" else "**✗**")
}
conv_map <- data.frame(gamma = formatC(gammas, format = "g"),
                       check.names = FALSE, stringsAsFactors = FALSE)
for (b in betas)
  conv_map[[sprintf("beta = %s", formatC(b, format = "g"))]] <-
    vapply(gammas, cell_of, character(1), b = b)

conv_tab <- data.frame(
  run = runtab$label, gamma = formatC(runtab$gamma, format = "g"),
  beta = formatC(runtab$beta, format = "g"),
  column = paste0("`", runtab$density_column, "`"),
  status = runtab$status, iterations = runtab$iterations,
  final_step_norm = formatC(runtab$final_step_norm, format = "e", digits = 3),
  converged = ifelse(runtab$converged, "yes", "**no**"),
  max_density = formatC(runtab$max_density, format = "g", digits = 6),
  solve_s = formatC(runtab$solve_wall_seconds, format = "f", digits = 1),
  check.names = FALSE, stringsAsFactors = FALSE)

n_conv  <- sum(runtab$converged)
not_conv <- runtab[!runtab$converged, ]
step_of <- function(g, b) {
  x <- runtab$final_step_norm[runtab$gamma == g & runtab$beta == b]
  if (length(x)) x[1] else NA_real_
}
sub_step <- step_of(1, as.numeric(RUN0$beta))     # the submitted gamma = 1 run
b25_g1  <- runtab[runtab$gamma == 1   & runtab$beta == 2.5, ]
b25_g05 <- runtab[runtab$gamma == 0.5 & runtab$beta == 2.5, ]
b25_rescued <- nrow(b25_g05) > 0 && isTRUE(b25_g05$converged[1])
e3 <- function(x) ifelse(is.na(x), "n/a", formatC(x, format = "e", digits = 3))
b25_answer <- if (!nrow(b25_g05) || !nrow(b25_g1)) "Not run" else if (b25_rescued)
  sprintf("Yes — at γ = 0.5 the β = 2.5 run converges (final step norm %s, against %s at γ = 1); the largest segment density is %s, against %s at γ = 1",
          e3(b25_g05$final_step_norm[1]), e3(b25_g1$final_step_norm[1]),
          formatC(b25_g05$max_density[1], format = "g", digits = 6),
          formatC(b25_g1$max_density[1], format = "g", digits = 6)) else
  sprintf("No — at γ = 0.5 the β = 2.5 run is still moving at the last iteration (final step norm %s, against %s at γ = 1)",
          e3(b25_g05$final_step_norm[1]), e3(b25_g1$final_step_norm[1]))

# =============================================================================
# Write
# =============================================================================

w <- function(df, fn) { p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
                        log_msg(ctx, "wrote ", p); p }
w(agg_tab,  "stage4_pc_beta_aggregated.csv")
w(rank_tab, "stage4_pc_beta_rank_stability.csv")
w(runtab,   "stage4_pc_beta_convergence.csv")
if (!is.null(dis_tab)) w(dis_tab, "stage4_pc_beta_disaggregated_correlations.csv")

# ---- report ------------------------------------------------------------------
fmt <- function(x, d = 4) ifelse(is.na(x), "n/a", formatC(x, format = "f", digits = d))
gf  <- function(x) ifelse(is.na(x), "n/a", formatC(as.numeric(x), format = "g"))
glist <- function(x) if (length(x)) paste(gf(x), collapse = ", ") else "none"
md_table <- function(df) {
  # a series name is `label|measure`, and a bare pipe would end the cell
  df[] <- lapply(df, function(x) gsub("|", " \\| ", as.character(x), fixed = TRUE))
  hdr <- paste0("| ", paste(names(df), collapse = " | "), " |")
  sep <- paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|")
  body <- apply(df, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  c(hdr, sep, body)
}

L <- c(
  "# Stage 4 — PC parameter (γ, β) sensitivity",
  "",
  sprintf(paste0("*Generated by `src/41_pc_beta_sensitivity.R` (`make pc-beta`). ",
                 "Seed %s. Cell level, N = %d. Two weight specifications: ",
                 "`%s` (%s, the W of the submitted tables) and `%s` (%s, the ",
                 "revision's W). Runs produced by `src/11_pc_runs.py` with `prefcent` ",
                 "%s @ `%s`.*"),
          ctx$cfg$repro$seed, nrow(D), W_SUB, describe_w(W_SUB),
          W_PROP, describe_w(W_PROP),
          ctx$cfg$inputs$pc_solver$version,
          substr(ctx$cfg$inputs$pc_solver$upstream_commit, 1, 7)),
  "",
  "## What was varied, and what was not",
  "",
  sprintf(paste0("β is the decay exponent of the interaction kernel, f(c) = (c + d₀)^(−β); ",
                 "γ is the weight on the agglomerative (preferential) term. ",
                 "The submitted measures use β = %s at γ = 1 (PC1) and γ = 0 (PC2). ",
                 "These runs move β over {%s} and γ over {%s} and hold every other ",
                 "setting of `inputs.pc_solver.submitted_run` fixed: the same network ",
                 "and capacities, d₀ = %s, Dp = %s, a₀ = %s, ω = %s, the `%s` ",
                 "renormalisation and the same %s-iteration budget. The β = %s runs at ",
                 "γ ∈ {1, 0} recompute the submitted runs; `make pc-accept` scores ",
                 "them against the submitted columns."),
          gf(RUN0$beta), glist(unlist(SENS$beta_grid)), glist(unlist(SENS$gamma_grid)),
          format(as.numeric(RUN0$d0), big.mark = ","), gf(RUN0$kappa), RUN0$a0,
          gf(RUN0$omega), RUN0$normalization, RUN0$max_iter, gf(RUN0$beta)),
  "",
  paste0("**Slots.** `PC1` = γ 1, `PC2` = γ 0 (preferential centrality without ",
         "agglomeration), `PC3` = γ 0.5, the intermediate strength of the ",
         "agglomerative term."),
  "",
  sprintf("## 1. Correlation with land value, cell level (N = %d cells)", nrow(D)),
  "")
tab1 <- data.frame(
  run = AGG1$label, gamma = gf(AGG1$gamma), beta = gf(AGG1$beta),
  slot = AGG1$measure, column = paste0("`", AGG1$column, "`"),
  n = AGG1$n_positive,
  `Pearson ln` = fmt(AGG1$pearson_ln_landvalue),
  `Spearman` = fmt(AGG1$spearman_landvalue),
  check.names = FALSE, stringsAsFactors = FALSE)
L <- c(L, md_table(tab1), "",
  "## 2. SARAR per (γ, β), under both weight specifications",
  "",
  paste0("`ln_land_value ~ ", CTRL, "ln_<PC>`, `sphet::spreg(model = \"sarar\", ",
         "het = ", isTRUE(ctx$cfg$models$het), ")`. `W = submitted` is `", W_SUB,
         "` (", describe_w(W_SUB), "); `W = proposed` is `", W_PROP, "` (",
         describe_w(W_PROP), "). `at_bound` = 1 marks a GMM error parameter ",
         "sitting on sphet's nlminb bound of ±", BOUND, ", i.e. not an interior ",
         "estimate."),
  "")
tab2 <- data.frame(
  run = agg_tab$label, gamma = gf(agg_tab$gamma), beta = gf(agg_tab$beta),
  slot = agg_tab$measure, W = agg_tab$w_role, N = agg_tab$n_sarar,
  beta_hat = fmt(agg_tab$sarar_beta), se = fmt(agg_tab$sarar_se),
  p = fmt(agg_tab$sarar_p, 4),
  std = fmt(agg_tab$sarar_beta_std),
  lambda = fmt(agg_tab$lambda_lag, 3), rho = fmt(agg_tab$rho_error, 3),
  at_bound = agg_tab$error_at_bound, pseudo_R2 = fmt(agg_tab$pseudo_r2),
  check.names = FALSE, stringsAsFactors = FALSE)
tab2 <- tab2[order(tab2$slot, tab2$beta, tab2$W), ]
L <- c(L, md_table(tab2), "",
  "## 3. Spearman rank stability",
  "",
  paste0("Pairwise Spearman between the same slot under different (γ, β), and ",
         "against the submitted column. Rows where the two series are different ",
         "slots are in the CSV but not printed here."),
  "")
tab3 <- data.frame(
  a = cross$a, b = cross$b, level = cross$level,
  spearman = fmt(cross$spearman, 6),
  check.names = FALSE, stringsAsFactors = FALSE)
L <- c(L, md_table(tab3), "")
if (!is.null(rs_dis)) {
  crossd2 <- rs_dis[rs_dis$same_measure, ]
  L <- c(L, "Same, at the address level (correlations only):", "",
         md_table(data.frame(a = crossd2$a, b = crossd2$b, level = crossd2$level,
                             spearman = fmt(crossd2$spearman, 6),
                             check.names = FALSE, stringsAsFactors = FALSE)), "")
}
if (!is.null(dis_tab)) {
  L <- c(L, "## 4. Correlation with land value, address level", "",
         paste0("A correlation needs no weight matrix; this script estimates no ",
                "model at the address level."),
         "",
         md_table(data.frame(
           run = dis_tab$label, gamma = gf(dis_tab$gamma), beta = gf(dis_tab$beta),
           slot = dis_tab$measure, n = dis_tab$n_positive,
           `Pearson ln` = fmt(dis_tab$pearson_ln_landvalue),
           Spearman = fmt(dis_tab$spearman_landvalue),
           check.names = FALSE, stringsAsFactors = FALSE)), "")
}

# the convergence summary, per gamma, from the run table
conv_by_gamma <- vapply(gammas, function(g) {
  r <- runtab[runtab$gamma == g, ]
  nc <- r[!r$converged, ]
  if (!nrow(nc)) sprintf("γ = %s: converged at every β run (%s)", gf(g), glist(r$beta))
  else sprintf("γ = %s: not converged at β = %s", gf(g),
               paste(sprintf("%s (%s)", gf(nc$beta),
                             formatC(nc$final_step_norm, format = "e", digits = 2)),
                     collapse = ", "))
}, character(1))

L <- c(L,
  "## 5. The (γ, β) convergence map",
  "",
  sprintf(paste0("The %s-iteration budget is part of the submitted specification ",
                 "(`stop: %s`), so every run stops there whatever it is doing. ",
                 "The table gives the solver's own `final_step_norm` — how far the last ",
                 "iterate moved — at each grid point. **Converged** here means a final ",
                 "step norm below %s; the submitted γ = 1, β = %s run's own value is %s. ",
                 "✓ = converged, **✗** = not."),
          RUN0$max_iter, RUN0$stop, formatC(CONV_TOL, format = "e", digits = 0),
          gf(RUN0$beta), e3(sub_step)),
  "",
  md_table(conv_map),
  "",
  md_table(conv_tab),
  "",
  sprintf("**%d of the %d grid points are converged at the iteration budget.**",
          n_conv, nrow(runtab)),
  "",
  paste0("- ", conv_by_gamma),
  "",
  "### Does β = 2.5 converge at γ = 0.5?",
  "",
  sprintf("**%s.**", b25_answer),
  "",
  paste0("A non-converged run is not dropped: it is the declared model iterated ",
         "for the same budget the submitted run used, and it is reported as such."),
  "",
  "## Summary",
  "")

sgn_anchor <- sign(anchor$sarar_beta[anchor$measure == "PC1"][1])
all_b <- agg_tab$sarar_beta[!is.na(agg_tab$sarar_beta)]
sign_stable <- length(all_b) > 0 && !is.na(sgn_anchor) && all(sign(all_b) == sgn_anchor)
sig_sentence <- function(meas, rows) {
  r <- rows[rows$measure == meas & !is.na(rows$sarar_p), ]
  if (!nrow(r)) return(NULL)
  sprintf("%s: p < %.2f at β = %s; not at β = %s", meas, ALPHA,
          glist(r$beta[r$sarar_p < ALPHA & r$role != "submitted"]),
          glist(r$beta[r$sarar_p >= ALPHA & r$role != "submitted"]))
}
sig_by_w <- unlist(lapply(names(W_SPECS), function(wn) {
  r <- agg_tab[agg_tab$w_role == wn, ]
  s <- unlist(lapply(MEAS_SLOTS, sig_sentence, rows = r))
  if (length(s)) sprintf("  - `%s` W — %s.", wn, paste(s, collapse = "; ")) else NULL
}))
b25_within1_cells <- {
  a <- D[["b25_PC1_mean"]]; b <- D[[map$PC1]]
  if (is.null(a)) NA_integer_ else {
    ok <- is.finite(a) & is.finite(b) & b != 0
    sum(abs(a[ok] - b[ok]) / abs(b[ok]) < 0.01)
  }
}
rho_sub_b25_pc1 <- cross$spearman[
  (cross$a == "b25|PC1" & cross$b == "submitted|PC1") |
  (cross$b == "b25|PC1" & cross$a == "submitted|PC1")][1]
std_rng <- function(x) if (any(!is.na(x))) sprintf("%s to %s", fmt(min(x, na.rm = TRUE)),
                                                  fmt(max(x, na.rm = TRUE))) else "n/a"

L <- c(L,
  sprintf("- **Sign.** %d SARAR coefficients across the grid and both weight specifications; %s",
          length(all_b),
          if (sign_stable) sprintf("all have the sign of the submitted PC1 coefficient (%s).",
                                   if (sgn_anchor > 0) "positive" else "negative")
          else "they are **not** all of one sign; see §2."),
  sprintf("- **Significance** (variants only, α = %.2f; the submitted series: PC1 p = %s, PC2 p = %s under the submitted W):",
          ALPHA, fmt(anchor$sarar_p[anchor$measure == "PC1"][1]),
          fmt(anchor$sarar_p[anchor$measure == "PC2"][1])),
  sig_by_w,
  sprintf("- **Ranking.** Against the submitted column, the same-slot rank correlation of a variant is at least ρ = %s at the cell level (%s at the address level); the lowest same-slot value anywhere in the grid is ρ = %s (%s against %s).",
          fmt(min_rho_vs_sub_agg), fmt(min_rho_vs_sub_dis), fmt(min_rho_any),
          if (nrow(worst_pair)) worst_pair$a[1] else "n/a",
          if (nrow(worst_pair)) worst_pair$b[1] else "n/a"),
  if (!is.na(b25_within1_cells))
    sprintf("- **Levels vs ranks at β = 2.5, γ = 1.** %d of the %d cells are within 1 %% of the submitted PC1 cell value, while the rank correlation with it is ρ = %s.",
            b25_within1_cells, nrow(D), fmt(rho_sub_b25_pc1)) else NULL,
  sprintf("- **Magnitude.** The standardized coefficients span %s under the submitted W and %s under the proposed W (submitted series under the submitted W: %s for PC1, %s for PC2).",
          std_rng(AGG1$sarar_beta_std),
          std_rng(agg_tab$sarar_beta_std[agg_tab$w_role == "proposed"]),
          fmt(anchor$sarar_beta_std[anchor$measure == "PC1"][1]),
          fmt(anchor$sarar_beta_std[anchor$measure == "PC2"][1])),
  sprintf("- **Convergence.** %d of the %d (γ, β) grid points do not reach the convergence threshold within the budget%s.",
          nrow(not_conv), nrow(runtab),
          if (nrow(not_conv)) paste0(": ", paste(sprintf("γ = %s, β = %s", gf(not_conv$gamma),
                                                       gf(not_conv$beta)), collapse = "; "))
          else ""),
  sprintf("- **Error parameter on the bound.** ρ sits on the nlminb bound of ±%s in %d of the %d fits under the submitted W and in %d of %d under the proposed W.",
          fmt(BOUND, 1),
          sum(AGG1$error_at_bound, na.rm = TRUE), sum(!is.na(AGG1$error_at_bound)),
          sum(agg_tab$error_at_bound[agg_tab$w_role == "proposed"], na.rm = TRUE),
          sum(!is.na(agg_tab$error_at_bound[agg_tab$w_role == "proposed"]))),
  "",
  "## Tables",
  "",
  "- `outputs/tables/stage4_pc_beta_aggregated.csv` — one row per (series, W)",
  "- `outputs/tables/stage4_pc_beta_rank_stability.csv`",
  "- `outputs/tables/stage4_pc_beta_convergence.csv` — the (γ, β) convergence map",
  if (!is.null(dis_tab)) "- `outputs/tables/stage4_pc_beta_disaggregated_correlations.csv`" else NULL,
  "- `outputs/tables/stage0_pc_runs.csv` — the runs themselves, with timings and fingerprints",
  "")

rp <- out_report(ctx, "stage4_pc_beta_sensitivity.md")
writeLines(L, rp)
log_msg(ctx, "wrote ", rp)
write_counts(ctx, "stage4_pc_beta_counts.csv")
finish(ctx)
