#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — disaggregated analysis-sample sensitivity.
#
# One W (the proposed point-level W, `weights.proposed.disaggregated`: kNN
# k = 8, row-standardised), one dependent variable (the one of the revised
# tables, `dependent_variable.proposed_source`: the rebuilt column), one
# specification (ln_land_value ~ ln_plot_area + ln_<measure>, SARAR by GMM with
# het per config). The only thing that moves is which records are in the
# sample, or, in the two parking-convention rows, how the dependent variable is
# constructed.
#
# Variants (all five measures each):
#   baseline            the analysis sample, submitted exclusion order, rows
#                       in the order the layer stores them
#   baseline_reversed   identical sample, row order reversed -> measures how
#                       much W depends on how knearneigh breaks distance ties
#   iqr_before_positivity
#                       the IQR fence applied before the positivity filter
#   drop_dup_records    the repeated address records dropped first
#   coords_jitter       the duplicate-coordinate points jittered <= 0.5 m
#   coords_dedupe       each duplicate-coordinate cluster collapsed to one record
#   drop_imovel_especial  the IMOVEL ESPECIAL records dropped first
#   cap_plot_area_100k    plots above 100,000 m2 dropped before the exclusions
#   cap_plot_area_100k_in_sample
#                       plots above 100,000 m2 dropped from the analysis sample
#                       (IQR fence unchanged)
#   sort_by_key         the configured tie-break: the frame is sorted by
#                       weights.identical_points.tie_break_key before W is built
#   sort_by_key_reversed_input
#                       the same rule applied to the frozen layer read in
#                       reversed row order -- the test that the rule removes
#                       the file-order dependence (must be bit-identical to
#                       `sort_by_key`)
#   parking_strict      the addresses whose only transactions are parking
#                       (stage 1's `only_parking_address`, where the headline
#                       construction falls back to the parking prices) dropped
#                       first; positivity then the IQR fence, recomputed on the
#                       remaining records. The row also reports the N obtained
#                       by removing those addresses from the baseline sample
#                       with the baseline fence held (`n_fence_held`).
#   parking_in_multiplier_only, parking_in_both
#                       the two alternative dependent-variable constructions
#                       stage 1 writes (`dependent_variable.variants`), through
#                       the same exclusion order; they need the rebuilt source
#
# All variants except the two sort_by_key ones are pinned to `handling: as_is`
# inside this script whatever config.yaml says, so they measure the stored row
# order and the two sort rows read as deltas against them.
#
# A second, smaller block runs the same rule at the aggregated level (the 500 m
# lattice has no duplicated centroid, but whole rings of cells sit at identical
# distances, so many k = 6 lists are still order-dependent), under the proposed
# aggregated W.
#
# The alternatives are applied to a copy of the config inside this script only;
# config.yaml is read, never written. Every filter still goes through
# apply_exclusions(), every W through build_weights(), and the standardized
# beta is the stage-4 one, so these rows are comparable, cell for cell, with
# the k = 8 row of outputs/tables/stage4_w_choice_sweep.csv.
#
# Run:  Rscript src/47_sample_sensitivity.R
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet); library(parallel)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "sample_sensitivity")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
W_SPEC   <- ctx$cfg$weights$proposed$disaggregated
K        <- as.integer(ctx$cfg$weights[[W_SPEC]]$k)
BOUND    <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
BEPS     <- 1e-6
LAM_MAX  <- 1.0
ALPHA    <- as.numeric(ctx$cfg$weights$sensitivity$w_choice$rule$alpha)
NWORK    <- max(1L, min(as.integer(ctx$cfg$compute$max_workers),
                        as.integer(ctx$cfg$compute$host_ceiling)))
CTRL     <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
frm      <- function(m) as.formula(sprintf("ln_land_value ~ %sln_%s", CTRL, m))
JITTER_M <- 0.5

log_msg(ctx, sprintf("W = '%s' (kNN k = %d, proposed); het = %s; workers = %d",
                     W_SPEC, K, isTRUE(ctx$cfg$models$het), NWORK))
log_msg(ctx, "weights.active, exclusions.order and dependent_variable.* are READ-ONLY here")

# ---- load the layer once; keep its own row order -----------------------------
# The frozen point layer, with the dependent variable of the revised tables
# (proposed_dv_layer() in common.R joins the rebuilt column by id and keeps the
# layer's row order).
DV_SOURCE <- ctx$cfg$dependent_variable$proposed_source
LAND0 <- proposed_dv_layer(ctx, "disaggregated")
if (isTRUE(st_is_longlat(LAND0))) stop("layer is in geographic coordinates", call. = FALSE)
log_msg(ctx, sprintf("disaggregated layer: %d rows, CRS EPSG:%s (row order kept), dependent variable: %s",
                     nrow(LAND0), st_crs(LAND0)$epsg, DV_SOURCE))

MAP <- ctx$cfg$measures$columns$disaggregated
canon <- function(L) {
  for (cn in names(MAP)) {
    srcn <- MAP[[cn]]
    if (!is.null(L[[srcn]]) && srcn != cn) L[[cn]] <- L[[srcn]]
  }
  L
}
LAND0 <- canon(LAND0)

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c") log(x + ctx$cfg$transforms$log_plus_c) else log(x)
add_logs <- function(D) {
  for (v in c("land_value", MEASURES, "plot_area")) D[[paste0("ln_", v)]] <- lg(D[[v]])
  D
}

# apply_exclusions() reads the order from ctx$cfg; run an alternative order on a
# shallow copy of the context so the real config is never mutated.
# `handling` is pinned per variant rather than inherited from config.yaml, so
# that switching weights.identical_points.handling cannot silently restate the
# as_is rows.
excl <- function(L, order = NULL, handling = "as_is", level = "disaggregated") {
  c2 <- new.env(parent = emptyenv())
  for (nm in ls(ctx)) assign(nm, get(nm, envir = ctx), envir = c2)
  if (!is.null(order)) c2$cfg$exclusions$order[[level]] <- order
  c2$cfg$weights$identical_points$handling <- handling
  out <- apply_exclusions(c2, L, level)
  ctx$counts <<- c2$counts
  out
}

# ---- tie exposure at k, for one coordinate matrix ---------------------------
# A neighbour list is order-dependent exactly when the k-th and (k+1)-th nearest
# points are equidistant: the tie could have fallen either way. Same definition
# and the same code as nb_stats() in src/40_w_choice.R.
tie_stats <- function(coords, k = K) {
  key  <- paste(coords[, 1], coords[, 2], sep = "|")
  dupk <- key %in% key[duplicated(key)]
  q <- function(e) suppressWarnings(e)
  kn  <- q(spdep::knearneigh(coords, k = k))
  kn1 <- q(spdep::knearneigh(coords, k = k + 1L))
  dist_to <- function(nn) sqrt(rowSums((coords - coords[nn, , drop = FALSE])^2))
  dk  <- dist_to(kn$nn[, k]); dk1 <- dist_to(kn1$nn[, k + 1L])
  list(n_dup_coord_points = sum(dupk),
       n_dup_coord_clusters = length(unique(key[dupk])),
       n_order_dependent_lists = sum(dk1 - dk <= 1e-9 * pmax(1, dk)))
}

# =============================================================================
# The variants. Each returns the analysis frame, its coordinates and a note.
# Record-level sample choices are applied before the exclusions (they are part
# of defining the record universe); the two coordinate handlings are applied
# after, because the duplicate-coordinate clusters are defined on the analysis
# sample itself.
# =============================================================================

SORT_VARIANTS <- c("sort_by_key", "sort_by_key_reversed_input")
DV_VARIANTS   <- c("parking_in_multiplier_only", "parking_in_both")
PARKING_VARIANTS <- c("parking_strict", DV_VARIANTS)

build_variant <- function(name) {
  # The two alternative constructions replace the dependent variable and
  # nothing else; every other variant starts from the headline column.
  L <- if (name %in% DV_VARIANTS)
    canon(proposed_dv_layer(ctx, "disaggregated", variant = name)) else LAND0
  note <- ""
  n_fence_held <- NA_integer_
  handling <- if (name %in% SORT_VARIANTS) "sort_by_key" else "as_is"
  if (name == "sort_by_key_reversed_input") {
    # The rule's own test. Hand apply_exclusions() the frozen layer in the
    # opposite row order; if the sort is doing its job, everything downstream --
    # W, the fits, every digit -- must come back identical to `sort_by_key`.
    L <- L[rev(seq_len(nrow(L))), ]
  }
  if (name == "drop_dup_records") {
    # The layer's `n` is a duplicate counter (1 on the first occurrence of an
    # address record, 2, 3, ... on its repeats). Keep the first occurrence.
    drop <- L$n > 1
    L <- L[!drop, ]
    note <- sprintf("dropped %d repeated address records before exclusions", sum(drop))
  } else if (name == "drop_imovel_especial") {
    drop <- !is.na(L$Finalidade) & L$Finalidade == "IMOVEL ESPECIAL"
    L <- L[!drop, ]
    note <- sprintf("dropped %d IMOVEL ESPECIAL records before exclusions", sum(drop))
  } else if (name == "cap_plot_area_100k") {
    drop <- !is.na(L$plot_area) & L$plot_area > 100000
    L <- L[!drop, ]
    note <- sprintf("dropped %d plots > 100,000 m2 before exclusions", sum(drop))
  } else if (name == "parking_strict") {
    if (is.null(L$only_parking_address))
      stop("parking_strict needs stage 1's `only_parking_address` (rebuilt source)",
           call. = FALSE)
    drop <- !is.na(L$only_parking_address) & L$only_parking_address
    L <- L[!drop, ]
    # The same addresses taken out of the baseline analysis sample instead,
    # with the baseline fence held: the second N the row reports.
    b <- V[["baseline"]]$D
    n_fence_held <- sum(!(!is.na(b$only_parking_address) & b$only_parking_address))
    count_step(ctx, "parking_strict: only-parking addresses dropped", nrow(L),
               sprintf("%d addresses", sum(drop)))
  } else if (name %in% DV_VARIANTS) {
    note <- sprintf("dependent variable built under the `%s` convention (dependent_variable.variants), same exclusion order",
                    name)
  }

  order <- if (name == "iqr_before_positivity") c("iqr_outliers", "positivity") else NULL
  if (name == "iqr_before_positivity") note <- "IQR fence applied before the positivity filter"
  E <- excl(L, order = order, handling = handling)
  if (name == "sort_by_key")
    note <- sprintf("same %s records, sorted by weights.identical_points.tie_break_key$disaggregated = [%s] before W",
                    format(nrow(E), big.mark = ","),
                    paste(ctx$cfg$weights$identical_points$tie_break_key$disaggregated,
                          collapse = ", "))
  if (name == "sort_by_key_reversed_input")
    note <- "frozen layer read in reversed row order, then the tie-break sort -- must equal `sort_by_key` exactly"

  D  <- add_logs(st_drop_geometry(E))
  xy <- st_coordinates(E)[, 1:2]
  if (name == "parking_strict")
    note <- sprintf("dropped the %d only-parking addresses before the exclusions; positivity, then the IQR fence recomputed: N = %s (removing the %d of them in the baseline sample with its fence held: N = %s)",
                    sum(!is.na(LAND0$only_parking_address) & LAND0$only_parking_address),
                    format(nrow(D), big.mark = ","),
                    nrow(V[["baseline"]]$D) - n_fence_held,
                    format(n_fence_held, big.mark = ","))

  if (name == "cap_plot_area_100k_in_sample") {
    # The cap applied to the sample rather than to the record universe: it drops
    # the large plots that survive the exclusions, with the IQR fence left where
    # it was. Reported beside `cap_plot_area_100k` so the effect of those plots
    # is separated from the effect of the fence moving when they leave.
    keep <- !(D$plot_area > 100000)
    note <- sprintf("dropped %d plots > 100,000 m2 from the analysis sample (IQR fence unchanged)",
                    sum(!keep))
    D <- D[keep, , drop = FALSE]; xy <- xy[keep, , drop = FALSE]
    count_step(ctx, "cap in sample", nrow(D), note)
  } else if (name == "baseline_reversed") {
    ord <- rev(seq_len(nrow(D)))
    D <- D[ord, , drop = FALSE]; xy <- xy[ord, , drop = FALSE]
    note <- sprintf("same %s records, row order reversed (order-dependence probe)",
                    format(nrow(D), big.mark = ","))
  } else if (name == "coords_jitter") {
    key  <- paste(xy[, 1], xy[, 2], sep = "|")
    dupk <- which(key %in% key[duplicated(key)])
    set.seed(as.integer(ctx$cfg$repro$seed))     # seeded; coordinates only
    r  <- JITTER_M * sqrt(runif(length(dupk)))
    th <- runif(length(dupk), 0, 2 * pi)
    xy[dupk, 1] <- xy[dupk, 1] + r * cos(th)
    xy[dupk, 2] <- xy[dupk, 2] + r * sin(th)
    note <- sprintf("%d duplicate-coordinate points jittered uniformly in a %.1f m disc (seed %d); values unchanged",
                    length(dupk), JITTER_M, ctx$cfg$repro$seed)
    count_step(ctx, "coords_jitter: jittered", nrow(D),
               sprintf("%d points moved <= %.1f m", length(dupk), JITTER_M))
  } else if (name == "coords_dedupe") {
    key <- paste(xy[, 1], xy[, 2], sep = "|")
    grp <- match(key, unique(key))
    lnv <- paste0("ln_", c("land_value", MEASURES, "plot_area"))
    keep_first <- !duplicated(grp)
    Dn <- D[keep_first, , drop = FALSE]
    gk <- grp[keep_first]
    for (v in lnv) {
      mu <- tapply(D[[v]], grp, mean)
      Dn[[v]] <- as.numeric(mu)[match(gk, as.integer(names(mu)))]
    }
    xy <- xy[keep_first, , drop = FALSE]
    ncl <- sum(table(grp) > 1); npt <- sum(table(grp)[table(grp) > 1])
    D <- Dn
    note <- sprintf("%d duplicate-coordinate clusters (%d points) collapsed to one record each, cluster means in logs",
                    ncl, npt)
    count_step(ctx, "coords_dedupe: collapsed", nrow(D), note)
  }

  list(name = name, D = D, coords = xy, note = note, n_fence_held = n_fence_held)
}

VARIANTS <- c("baseline", "baseline_reversed", "iqr_before_positivity", "drop_dup_records",
              "coords_jitter", "coords_dedupe", "drop_imovel_especial",
              "cap_plot_area_100k", "cap_plot_area_100k_in_sample",
              SORT_VARIANTS)
# The parking rows need stage 1's rebuild: its flag and its alternative
# constructions exist only there.
if (identical(DV_SOURCE, "rebuilt")) {
  VARIANTS <- c(VARIANTS, PARKING_VARIANTS)
} else {
  log_msg(ctx, "dependent variable is frozen: the parking variants (",
          paste(PARKING_VARIANTS, collapse = ", "), ") need the rebuilt source and are not run")
}

V <- list(); TIES <- list()
for (v in VARIANTS) {
  log_msg(ctx, strrep("-", 74))
  log_msg(ctx, "VARIANT: ", v)
  V[[v]] <- build_variant(v)
  if (nzchar(V[[v]]$note)) log_msg(ctx, "   ", V[[v]]$note)
  ts <- tie_stats(V[[v]]$coords)
  TIES[[v]] <- data.frame(variant = v, n = nrow(V[[v]]$D),
                          n_dup_coord_points = ts$n_dup_coord_points,
                          n_dup_coord_clusters = ts$n_dup_coord_clusters,
                          n_order_dependent_lists = ts$n_order_dependent_lists,
                          stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("   N = %d   duplicate-coordinate points %d in %d clusters   order-dependent k=%d lists %d",
                       nrow(V[[v]]$D), ts$n_dup_coord_points, ts$n_dup_coord_clusters,
                       K, ts$n_order_dependent_lists))
}
ties <- do.call(rbind, TIES)

# ---- one (variant, measure) fit ---------------------------------------------
# Same body as one_fit() in src/40_w_choice.R minus the LM block: standardized
# beta by re-estimation on z-scored covariates, and the error-bound flag.
one_fit <- function(dat, W, m, vname) {
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
    variant = vname, w_spec = W_SPEC, k = K, n = nrow(dat), measure = m,
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

Ws <- lapply(V, function(v) suppressWarnings(build_weights(ctx, v$coords, spec_name = W_SPEC)))
names(Ws) <- VARIANTS

jobs <- expand.grid(v = VARIANTS, m = MEASURES, stringsAsFactors = FALSE)
t0 <- Sys.time()
res <- mclapply(seq_len(nrow(jobs)), function(i) {
  j <- jobs[i, ]
  one_fit(V[[j$v]]$D, Ws[[j$v]], j$m, j$v)
}, mc.cores = min(NWORK, nrow(jobs)), mc.preschedule = FALSE)
bad <- vapply(res, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(bad)) log_msg(ctx, sprintf("!! %d of %d fits returned nothing", sum(bad), length(res)))
fits <- do.call(rbind, res[!bad])
log_msg(ctx, sprintf("%d of %d fits in %.0f s wall on %d workers", sum(!bad), nrow(jobs),
                     as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     min(NWORK, nrow(jobs))))

fits$variant <- factor(fits$variant, levels = VARIANTS)
fits <- fits[order(fits$variant, match(fits$measure, MEASURES)), ]
fits$variant <- as.character(fits$variant)

# ---- the reproduction check against the k = 8 sweep row ---------------------
sw <- file.path(REPO, ctx$cfg$paths$outputs, "tables", "stage4_w_choice_sweep.csv")
if (file.exists(sw)) {
  S <- read.csv(sw, stringsAsFactors = FALSE)
  S <- S[S$level == "disaggregated" & S$w_spec == "knn_k8", ]
  B <- fits[fits$variant == "baseline", ]
  if (nrow(S) == length(MEASURES)) {
    S <- S[match(B$measure, S$measure), ]
    d <- c(beta = max(abs(B$beta_measure - S$beta_measure)),
           std  = max(abs(B$beta_std - S$beta_std)),
           rho  = max(abs(B$rho_error - S$rho_error)),
           lam  = max(abs(B$lambda_lag - S$lambda_lag)),
           r2   = max(abs(B$pseudo_r2 - S$pseudo_r2)))
    log_msg(ctx, sprintf("baseline vs stage4_w_choice knn_k8: max |delta| beta %.3g, std beta %.3g, rho %.3g, lambda %.3g, R2 %.3g%s",
                         d["beta"], d["std"], d["rho"], d["lam"], d["r2"],
                         if (max(d) == 0) "  -- BIT-IDENTICAL" else ""))
  }
}

# ---- deltas vs baseline, and the verdict test -------------------------------
base <- fits[fits$variant == "baseline", ]
rownames(base) <- base$measure
fits$d_beta_std   <- fits$beta_std - base[fits$measure, "beta_std"]
fits$d_beta       <- fits$beta_measure - base[fits$measure, "beta_measure"]
fits$sign_change  <- as.integer(sign(fits$beta_measure) != sign(base[fits$measure, "beta_measure"]))
fits$sig_change   <- as.integer((fits$p_measure < ALPHA) !=
                                (base[fits$measure, "p_measure"] < ALPHA))
fits$verdict_change <- as.integer(fits$sign_change == 1L | fits$sig_change == 1L)
fits$level <- "disaggregated"

# =============================================================================
# The aggregated level: the same tie-break rule on the 500 m hexagon lattice.
#
# The lattice has no duplicated centroid, and still many of the k = 6
# neighbour lists are order-dependent, because whole rings of cells sit at
# identical distances on a regular grid. Three rows: `as_is`, the tie-break
# sort, and the sort applied to a reversed reading of the layer (which must
# reproduce the sort row exactly).
# =============================================================================

W_AGG <- ctx$cfg$weights$proposed$aggregated
K_AGG <- as.integer(ctx$cfg$weights[[W_AGG]]$k)
MAP_A <- ctx$cfg$measures$columns$aggregated
AGG0  <- proposed_dv_layer(ctx, "aggregated")
for (cn in names(MAP_A)) {
  srcn <- MAP_A[[cn]]
  if (!is.null(AGG0[[srcn]]) && srcn != cn) AGG0[[cn]] <- AGG0[[srcn]]
}
log_msg(ctx, sprintf("aggregated layer: %d rows, CRS EPSG:%s; W = '%s' (kNN k = %d)",
                     nrow(AGG0), st_crs(AGG0)$epsg, W_AGG, K_AGG))

build_agg_variant <- function(name) {
  A <- AGG0
  handling <- if (name %in% SORT_VARIANTS) "sort_by_key" else "as_is"
  note <- switch(name,
    baseline = "analysis cells in the layer's stored order (as_is)",
    sort_by_key = sprintf("analysis cells sorted by tie_break_key$aggregated = [%s] before W",
                          paste(ctx$cfg$weights$identical_points$tie_break_key$aggregated,
                                collapse = ", ")),
    sort_by_key_reversed_input =
      "layer read in reversed row order, then the tie-break sort -- must equal `sort_by_key` exactly")
  if (name == "sort_by_key_reversed_input") A <- A[rev(seq_len(nrow(A))), ]
  E  <- excl(A, handling = handling, level = "aggregated")
  D  <- add_logs(st_drop_geometry(E))
  xy <- suppressWarnings(st_coordinates(st_point_on_surface(st_geometry(E)))[, 1:2])
  list(name = name, D = D, coords = xy, note = note)
}

AGG_VARIANTS <- c("baseline", SORT_VARIANTS)
VA <- list(); TIES_A <- list()
for (v in AGG_VARIANTS) {
  log_msg(ctx, strrep("-", 74))
  log_msg(ctx, "AGGREGATED VARIANT: ", v)
  VA[[v]] <- build_agg_variant(v)
  log_msg(ctx, "   ", VA[[v]]$note)
  ts <- tie_stats(VA[[v]]$coords, k = K_AGG)
  TIES_A[[v]] <- data.frame(variant = v, n = nrow(VA[[v]]$D),
                            n_dup_coord_points = ts$n_dup_coord_points,
                            n_dup_coord_clusters = ts$n_dup_coord_clusters,
                            n_order_dependent_lists = ts$n_order_dependent_lists,
                            stringsAsFactors = FALSE)
  log_msg(ctx, sprintf("   N = %d   duplicate-coordinate cells %d   order-dependent k=%d lists %d",
                       nrow(VA[[v]]$D), ts$n_dup_coord_points, K_AGG,
                       ts$n_order_dependent_lists))
}
ties_a <- do.call(rbind, TIES_A)

Wa <- lapply(VA, function(v) suppressWarnings(build_weights(ctx, v$coords, spec_name = W_AGG)))
names(Wa) <- AGG_VARIANTS
ajobs <- expand.grid(v = AGG_VARIANTS, m = MEASURES, stringsAsFactors = FALSE)
ares <- mclapply(seq_len(nrow(ajobs)), function(i) {
  j <- ajobs[i, ]
  o <- one_fit(VA[[j$v]]$D, Wa[[j$v]], j$m, j$v)
  if (!is.null(o)) { o$w_spec <- W_AGG; o$k <- K_AGG }
  o
}, mc.cores = min(NWORK, nrow(ajobs)), mc.preschedule = FALSE)
abad <- vapply(ares, function(r) is.null(r) || inherits(r, "try-error"), logical(1))
if (any(abad)) log_msg(ctx, sprintf("!! %d of %d aggregated fits returned nothing",
                                    sum(abad), length(ares)))
afits <- do.call(rbind, ares[!abad])
afits$variant <- factor(afits$variant, levels = AGG_VARIANTS)
afits <- afits[order(afits$variant, match(afits$measure, MEASURES)), ]
afits$variant <- as.character(afits$variant)

abase <- afits[afits$variant == "baseline", ]
rownames(abase) <- abase$measure
afits$d_beta_std  <- afits$beta_std - abase[afits$measure, "beta_std"]
afits$d_beta      <- afits$beta_measure - abase[afits$measure, "beta_measure"]
afits$sign_change <- as.integer(sign(afits$beta_measure) != sign(abase[afits$measure, "beta_measure"]))
afits$sig_change  <- as.integer((afits$p_measure < ALPHA) !=
                                (abase[afits$measure, "p_measure"] < ALPHA))
afits$verdict_change <- as.integer(afits$sign_change == 1L | afits$sig_change == 1L)
afits$level <- "aggregated"

# ---- the two identity tests --------------------------------------------------
# Reversing the file order before the tie-break sort must change nothing.
id_test <- function(F, lab) {
  A <- F[F$variant == "sort_by_key", ]
  B <- F[F$variant == "sort_by_key_reversed_input", ]
  if (!nrow(A) || !nrow(B)) return(invisible(NULL))
  B <- B[match(A$measure, B$measure), ]
  num <- c("beta_measure", "se_measure", "p_measure", "beta_std", "se_std",
           "lambda_lag", "rho_error", "pseudo_r2")
  d <- max(abs(as.matrix(A[, num]) - as.matrix(B[, num])))
  log_msg(ctx, sprintf("tie-break identity test (%s): reversed input vs sorted, max |delta| over %s = %.3g%s",
                       lab, paste(num, collapse = "/"), d,
                       if (d == 0) "   -- BIT-IDENTICAL" else "   *** NOT IDENTICAL ***"))
  d
}
id_test(fits,  sprintf("disaggregated, k = %d", K))
id_test(afits, sprintf("aggregated, k = %d", K_AGG))

detail <- rbind(fits, afits)[, c("level", "variant", "w_spec", "k", "n", "measure",
                   "beta_area", "se_area", "p_area",
                   "beta_measure", "se_measure", "p_measure", "d_beta",
                   "beta_std", "se_std", "d_beta_std", "beta_std_analytic",
                   "lambda_lag", "se_lambda", "rho_error", "se_rho",
                   "error_at_bound", "lambda_nonstationary",
                   "pseudo_r2", "sign_change", "sig_change", "verdict_change")]
p1 <- out_table(ctx, "stage4_sample_sensitivity_detail.csv")
write.csv(detail, p1, row.names = FALSE)
log_msg(ctx, "wrote ", p1)

# ---- the one wide table -----------------------------------------------------
# Both levels share it; `level` is the first column.
make_wide <- function(F, variants, TIESdf, notes, base_df, lvl, nfh = list()) do.call(rbind, lapply(variants, function(v) {
  s <- F[F$variant == v, ]; s <- s[match(MEASURES, s$measure), ]
  base <- base_df
  tv <- TIESdf[TIESdf$variant == v, ]
  row <- data.frame(level = lvl, variant = v, n = s$n[1],
                    n_fence_held = if (is.null(nfh[[v]])) NA_integer_ else nfh[[v]],
                    n_dup_coord_points = tv$n_dup_coord_points,
                    n_order_dependent_lists = tv$n_order_dependent_lists,
                    stringsAsFactors = FALSE)
  for (i in seq_along(MEASURES)) {
    m <- MEASURES[i]
    row[[paste0("beta_", m)]]     <- s$beta_measure[i]
    row[[paste0("p_", m)]]        <- s$p_measure[i]
    row[[paste0("beta_std_", m)]] <- s$beta_std[i]
  }
  row$max_abs_d_beta_std <- max(abs(s$d_beta_std))
  row$max_abs_d_beta     <- max(abs(s$d_beta))
  row$n_sign_changes     <- sum(s$sign_change)
  row$n_sig_changes      <- sum(s$sig_change)
  # A sign flip on a coefficient that is indistinguishable from zero in both the
  # baseline and the variant is not a change of verdict in any readable sense
  # (BC and FK at the point level are the usual case). This column counts only
  # flips on a measure that is significant at alpha somewhere.
  sig_either <- (s$p_measure < ALPHA) | (base[s$measure, "p_measure"] < ALPHA)
  row$n_sign_changes_on_significant <- sum(s$sign_change & sig_either)
  row$any_verdict_change <- if (isTRUE(sum(s$verdict_change) > 0)) "yes" else
                            if (anyNA(s$verdict_change)) NA_character_ else "no"
  row$any_verdict_change_on_significant <-
    if (isTRUE(row$n_sign_changes_on_significant + row$n_sig_changes > 0)) "yes" else "no"
  row$lambda_min <- min(s$lambda_lag); row$lambda_max <- max(s$lambda_lag)
  row$rho_min <- min(s$rho_error);     row$rho_max <- max(s$rho_error)
  row$n_at_bound <- sum(s$error_at_bound)
  row$pseudo_r2_min <- min(s$pseudo_r2); row$pseudo_r2_max <- max(s$pseudo_r2)
  row$note <- notes[[v]]
  row
}))

wide <- rbind(
  make_wide(fits,  VARIANTS,     ties,   lapply(V,  `[[`, "note"), base,  "disaggregated",
            lapply(V, `[[`, "n_fence_held")),
  make_wide(afits, AGG_VARIANTS, ties_a, lapply(VA, `[[`, "note"), abase, "aggregated"))
p2 <- out_table(ctx, "stage4_sample_sensitivity.csv")
write.csv(wide, p2, row.names = FALSE)
log_msg(ctx, "wrote ", p2)

for (i in seq_len(nrow(wide)))
  log_msg(ctx, sprintf("  %-13s %-28s N %5d  max|d std beta| %.4f  sign chg %d (on a significant measure %d)  sig chg %d  verdict change: %s / on significant: %s",
                       wide$level[i], wide$variant[i], wide$n[i], wide$max_abs_d_beta_std[i],
                       wide$n_sign_changes[i], wide$n_sign_changes_on_significant[i],
                       wide$n_sig_changes[i], wide$any_verdict_change[i],
                       wide$any_verdict_change_on_significant[i]))

write_counts(ctx, "stage4_sample_sensitivity_counts.csv")
finish(ctx)
