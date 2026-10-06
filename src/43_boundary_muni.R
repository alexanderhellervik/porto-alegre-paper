#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — the municipal network-extent variants.
#
# The second boundary family, and the one that clips to an administrative
# boundary rather than to a distance chosen by the analyst:
#
#   mu0  the municipality of Porto Alegre  (IBGE 2025, CD_MUN 4314902)
#   mu1  the municipality + a 1 km collar
#   full the delivered extent, unchanged — the third member of the sequence
#
# Everything else follows the negative-buffer family (src/42): whole segments
# only, R rebuilt on each clipped network with the enumeration order recorded,
# the solver of record at the submitted parameters and budget, GAUS Lines v1.1
# for CC/BC/FK, informational acceptance, `intersects` at 500 m and
# nearest-segment at the points, and the same two comparisons — the variant
# against the submitted columns, and the submitted columns on the same cells,
# so a difference is the extent alone.
#
# One rule is specific to this family: each variant is reduced to the largest
# connected component of its own routing graph before R is rebuilt and before
# the solver runs, so the family measures the extent rather than the
# fragmentation a clip can cause. The raw component sizes are recorded either
# way.
#
# The weight specifications are `revision.boundary_sensitivity.municipal.w_specs`
# (the band of the submitted tables and the proposed kNN matrices);
# `headline_w` names the headline one.
#
# Run:  make boundary-muni   (Rscript src/43_boundary_muni.R)
# Writes: outputs/tables/stage4_boundary_muni_*.csv and the report section
#         reports/stage4_boundary_muni_section.md, which
#         src/42_boundary_sensitivity.R splices into
#         reports/stage4_boundary_sensitivity.md ahead of the negative-buffer
#         sections. A standalone copy is reports/stage4_boundary_muni.md.
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "boundary_muni")
capture_env_r(ctx)

AGG <- "aggregated"
PARENT <- ctx$cfg$revision$boundary_sensitivity
# Not modifyList(): it recurses into a sub-list and merges it by name, and the
# two families' `input_files` / `variants` are unnamed lists, so the parent's
# entries would survive the override. A flat, top-level replacement is what the
# `--family` flag does on the Python side (`family_cfg`).
BND    <- PARENT
for (.k in names(PARENT$municipal)) BND[[.k]] <- PARENT$municipal[[.k]]
SFX    <- paste0("_", BND$table_suffix)
W_SPECS <- unlist(BND$w_specs)
HEAD_W  <- BND$headline_w
if (!HEAD_W %in% names(W_SPECS))
  stop("municipal.headline_w '", HEAD_W, "' is not a key of municipal.w_specs",
       call. = FALSE)
SUB_ROLE <- "submitted_band"      # the w_specs key of the submitted tables' band
BOUND  <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
ALPHA  <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
CTRL   <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
STD    <- isTRUE(ctx$cfg$revision$standardize_coefficients)
MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
ACCESS   <- unlist(ctx$cfg$measures$accessibility)
INTERM   <- unlist(ctx$cfg$measures$intermediation)
STAT <- vapply(MEASURES, function(m)
  ctx$cfg$grid$aggregation[[if (m %in% ACCESS) "accessibility" else "intermediation"]],
  character(1))
N_REF_SEGMENTS <- as.integer(ctx$cfg$inputs$segment_centralities$expected_features)
# The segment count the manuscript (Sec. 4.1) gives for the municipal network.
PAPER_N_SEGMENTS <- 29978L
CONV_TOL <- 1e-8

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c")
  log(x + ctx$cfg$transforms$log_plus_c) else log(x)

variants <- vapply(BND$input_files, function(r) as.character(r$label), character(1))
if (!length(variants))
  stop("revision.boundary_sensitivity.municipal.input_files is empty", call. = FALSE)
log_msg(ctx, sprintf("municipal variants: %s", paste(variants, collapse = ", ")))
log_msg(ctx, sprintf("weight specifications: %s (headline: %s)",
                     paste(sprintf("%s = %s", names(W_SPECS), W_SPECS),
                           collapse = ", "), HEAD_W))

# =============================================================================
# 1. The aggregated sample, and each variant's covered subset
# =============================================================================

# Dependent variable: the rebuilt one (`dependent_variable.proposed_source`), as in every
# other revised table; the centralities are the frozen ones (see proposed_dv_layer()).
Land <- proposed_dv_layer(ctx, "aggregated")
crs_expected <- ctx$cfg$repro$crs_epsg
if (is.na(st_crs(Land)$epsg) || st_crs(Land)$epsg != crs_expected)
  Land <- st_transform(Land, crs_expected)
if (isTRUE(st_is_longlat(Land)))
  stop("layer is in geographic coordinates", call. = FALSE)
Land$cell_id <- Land[[ctx$cfg$grid$frozen_500m$id_column]]

bpath <- file.path(REPO, BND$aggregated_layer)
if (!file.exists(bpath))
  stop("missing ", BND$aggregated_layer,
       " -- run `make boundary-muni` (src/23_boundary_layers.py --family muni) first",
       call. = FALSE)
B <- st_drop_geometry(st_read(bpath, quiet = TRUE))
log_msg(ctx, sprintf("municipal boundary layer: %d cells, %d columns",
                     nrow(B), ncol(B)))
Land <- merge(Land, B[, setdiff(names(B), c("row_index", "col_index"))],
              by = "cell_id", all.x = TRUE, sort = FALSE)

map <- ctx$cfg$measures$columns[[AGG]]
for (canon in names(map)) {
  srcn <- map[[canon]]
  if (!is.null(Land[[srcn]]) && srcn != canon) Land[[canon]] <- Land[[srcn]]
}
# apply_exclusions() also sorts on the tie-break key before any W is built.
Land <- apply_exclusions(ctx, Land, AGG)
target_n <- ctx$cfg$exclusions$targets[[paste0(AGG, "_n")]]
log_msg(ctx, sprintf("N = %d (submitted aggregated N = %d) %s", nrow(Land),
                     target_n,
                     if (nrow(Land) == target_n) "MATCH" else "*** MISMATCH ***"))
if (nrow(Land) != target_n)
  stop("aggregated N does not match the submitted sample; stopping", call. = FALSE)

for (v in c("land_value", MEASURES, "plot_area"))
  Land[[paste0("ln_", v)]] <- lg(Land[[v]])
pts_all <- suppressWarnings(st_point_on_surface(Land))
coords_all <- st_coordinates(pts_all)[, 1:2]
D_all <- st_drop_geometry(Land)

sample_rows <- function(label) {
  cov <- D_all[[paste0(label, "_covered")]]
  ok  <- !is.na(cov) & cov == 1L
  for (m in MEASURES) {
    v <- D_all[[sprintf("%s_%s_%s", label, m, STAT[[m]])]]
    ok <- ok & !is.na(v) & v > 0
  }
  which(ok)
}
SAMPLES <- list(full = list(extent = "reference", scope = "full extent",
                            rows = seq_len(nrow(D_all)), source = "submitted"))
for (lb in variants) {
  rws <- sample_rows(lb)
  count_step(ctx, sprintf("%s: cells entering the analysis", lb), length(rws),
             sprintf("of the %d analysis cells (%.1f%%)", nrow(D_all),
                     100 * length(rws) / nrow(D_all)))
  SAMPLES[[paste0("submitted@", lb)]] <- list(
    extent = "reference", scope = sprintf("restricted to %s cells", lb),
    rows = rws, source = "submitted", ref = lb)
  SAMPLES[[lb]] <- list(extent = lb, scope = sprintf("restricted to %s cells", lb),
                        rows = rws, source = lb, ref = lb)
}

col_of <- function(src, m)
  if (src == "submitted") map[[m]] else sprintf("%s_%s_%s", src, m, STAT[[m]])

# =============================================================================
# 2. Correlations and the headline SARAR, per sample x measure x W
# =============================================================================

rows <- list()
for (snm in names(SAMPLES)) {
  S  <- SAMPLES[[snm]]
  Ds <- D_all[S$rows, , drop = FALSE]
  cs <- coords_all[S$rows, , drop = FALSE]
  Ws <- list()
  for (wnm in names(W_SPECS)) {
    sp   <- W_SPECS[[wnm]]
    keep <- connected_idx(ctx, cs, sp)
    Ws[[wnm]] <- list(spec = sp, keep = keep,
                      W = build_weights(ctx, cs[keep, , drop = FALSE], spec_name = sp),
                      D = Ds[keep, , drop = FALSE])
  }
  for (m in MEASURES) {
    cn <- col_of(S$source, m)
    v  <- Ds[[cn]]
    if (is.null(v)) { log_msg(ctx, "!! missing column ", cn); next }
    ok <- !is.na(v) & v > 0
    pear <- cor(Ds$ln_land_value[ok], lg(v[ok]), method = "pearson")
    spea <- cor(Ds$land_value[ok], v[ok], method = "spearman")
    for (wnm in names(Ws)) {
      dh <- Ws[[wnm]]$D
      dh$ln_measure <- lg(dh[[cn]])
      okh <- is.finite(dh$ln_measure)
      # SD for the standardized beta, on exactly the rows the model is fitted on
      sdln <- sd(dh$ln_measure[okh])
      fit <- fit_spreg(ctx,
                       as.formula(sprintf("ln_land_value ~ %sln_measure", CTRL)),
                       dh[okh, , drop = FALSE], Ws[[wnm]]$W, "sarar")
      bb <- se <- pv <- lam <- rr <- r2 <- NA_real_; ab <- NA_integer_
      if (!is.null(fit) && sum(okh) == nrow(dh)) {
        zz  <- sarar_row(fit)
        bb  <- zz$co[["ln_measure"]]; se <- zz$se[["ln_measure"]]
        pv  <- zz$p[["ln_measure"]]
        lam <- zz$co[["lambda"]];     rr <- zz$co[["rho"]]
        r2  <- pseudo_r2(fit, dh$ln_land_value[okh])
        ab  <- as.integer(abs(abs(rr) - BOUND) < 1e-6)
      }
      rows[[length(rows) + 1]] <- data.frame(
        sample = snm, extent = S$extent, scope = S$scope, source = S$source,
        measure = m, column = cn, n_cells = length(S$rows), n_positive = sum(ok),
        pearson_ln_landvalue = pear, spearman_landvalue = spea, sd_ln = sdln,
        w_role = wnm, w_spec = Ws[[wnm]]$spec, n_sarar = sum(okh),
        sarar_beta = bb, sarar_se = se, sarar_p = pv,
        sarar_beta_std = if (STD) bb * sdln else NA_real_,
        lambda_lag = lam, rho_error = rr,
        error_at_bound = ab, pseudo_r2 = r2,
        stringsAsFactors = FALSE)
      log_msg(ctx, sprintf(
        "%-16s %-4s [%-14s W, N=%d]: r=%.4f rho_s=%.4f  b=%+.4f (se %.4f) std=%+.4f p=%.4f  lambda=%+.3f rho=%+.3f%s",
        snm, m, wnm, sum(okh), pear, spea, bb, se, bb * sdln, pv, lam, rr,
        if (isTRUE(ab == 1L)) " AT BOUND" else ""))
    }
  }
}
agg_tab <- do.call(rbind, rows)

# Sign check: each coefficient against the full-extent coefficient for the same
# measure under the same weight specification.
full_sign <- with(agg_tab[agg_tab$sample == "full", ],
                  setNames(sign(sarar_beta), paste(measure, w_role)))
agg_tab$sign_vs_full <- as.integer(
  sign(agg_tab$sarar_beta) == full_sign[paste(agg_tab$measure, agg_tab$w_role)])

# =============================================================================
# 3. Does the ranking of the five measures survive? (signed std beta)
# =============================================================================

rank_rows <- list()
for (wnm in names(W_SPECS)) for (snm in names(SAMPLES)) {
  sub <- agg_tab[agg_tab$sample == snm & agg_tab$w_role == wnm, ]
  if (!nrow(sub)) next
  o <- order(-sub$sarar_beta_std)
  rank_rows[[length(rank_rows) + 1]] <- data.frame(
    sample = snm, extent = sub$extent[1], scope = sub$scope[1], w_role = wnm,
    order = paste(sub$measure[o], collapse = " > "),
    top = sub$measure[o][1],
    top_std = sub$sarar_beta_std[o][1],
    n_significant = sum(sub$sarar_p < ALPHA, na.rm = TRUE),
    significant = paste(sort(sub$measure[sub$sarar_p < ALPHA & !is.na(sub$sarar_p)]),
                        collapse = ","),
    # the paper's two claims, scored explicitly rather than read off the order
    pc1_first = identical(sub$measure[o][1], "PC1"),
    access_beats_intermediation =
      min(sub$sarar_beta_std[sub$measure %in% ACCESS], na.rm = TRUE) >
      max(sub$sarar_beta_std[sub$measure %in% INTERM], na.rm = TRUE),
    stringsAsFactors = FALSE)
}
rank_tab <- do.call(rbind, rank_rows)

std_of <- function(snm, wnm) {
  sub <- agg_tab[agg_tab$sample == snm & agg_tab$w_role == wnm, ]
  setNames(sub$sarar_beta_std, sub$measure)[MEASURES]
}
sp_rank <- function(a, b) {
  ok <- is.finite(a) & is.finite(b)
  if (sum(ok) < 3) return(NA_real_)
  suppressWarnings(cor(a[ok], b[ok], method = "spearman"))
}
top_of <- function(v) { ok <- is.finite(v); if (!any(ok)) NA_character_ else
  MEASURES[ok][which.max(v[ok])] }
rank_agree <- list()
for (wnm in names(W_SPECS)) {
  bs <- std_of("full", wnm)
  for (snm in names(SAMPLES)) {
    vs  <- std_of(snm, wnm)
    ref <- SAMPLES[[snm]]$ref
    rs  <- if (!is.null(ref) && SAMPLES[[snm]]$source != "submitted")
             std_of(paste0("submitted@", ref), wnm) else NULL
    rank_agree[[length(rank_agree) + 1]] <- data.frame(
      sample = snm, extent = SAMPLES[[snm]]$extent, scope = SAMPLES[[snm]]$scope,
      w_role = wnm, n_measures = sum(is.finite(vs)),
      spearman_vs_full = sp_rank(bs, vs),
      same_top_vs_full = identical(top_of(vs), top_of(bs)),
      spearman_vs_restricted = if (is.null(rs)) NA_real_ else sp_rank(rs, vs),
      same_top_vs_restricted = if (is.null(rs)) NA else
        identical(top_of(vs), top_of(rs)),
      order = paste(MEASURES[order(-vs)], collapse = " > "),
      stringsAsFactors = FALSE)
  }
}
rank_agree <- do.call(rbind, rank_agree)

# =============================================================================
# 4. Segment-level movement (stage 2) and the point-level correlations
# =============================================================================

seg_move <- NULL
smp <- out_table(ctx, sprintf("stage2_boundary%s_segment_movement.csv", SFX))
if (file.exists(smp)) {
  seg_move <- read.csv(smp)
  log_msg(ctx, sprintf("segment/cell/point movement table: %d rows", nrow(seg_move)))
} else {
  log_msg(ctx, "!! no ", basename(smp), " -- run src/23_boundary_layers.py --family muni")
}

dis_tab <- NULL
ppath <- file.path(REPO, BND$points_table)
if (file.exists(ppath)) {
  P  <- read.csv(ppath)
  Pt <- st_drop_geometry(st_read(cfg_path(ctx, "frozen", "cents_disaggregated"),
                                 quiet = TRUE))
  if (nrow(P) != nrow(Pt))
    stop("point table has ", nrow(P), " rows, frozen layer ", nrow(Pt), call. = FALSE)
  P <- P[order(P$point_index), ]
  dmap <- ctx$cfg$measures$columns$disaggregated
  for (cn in setdiff(names(P), "point_index")) Pt[[paste0("bnd_", cn)]] <- P[[cn]]
  Pt <- apply_exclusions(ctx, Pt, "disaggregated")
  log_msg(ctx, sprintf("disaggregated N = %d (submitted %d)", nrow(Pt),
                       ctx$cfg$exclusions$targets$disaggregated_n))
  lv <- Pt[[dmap$land_value]]
  covered <- function(src) !is.na(Pt[[sprintf("bnd_%s_covered", src)]]) &
                           Pt[[sprintf("bnd_%s_covered", src)]] == 1L
  drows <- list()
  for (src in c("submitted", variants)) for (m in MEASURES) {
    v <- if (src == "submitted") Pt[[dmap[[m]]]] else Pt[[sprintf("bnd_%s_%s", src, m)]]
    if (is.null(v)) next
    covm <- if (src == "submitted") rep(TRUE, nrow(Pt)) else covered(src)
    ok <- covm & !is.na(v) & v > 0 & !is.na(lv) & lv > 0
    drows[[length(drows) + 1]] <- data.frame(
      extent = src, measure = m, n = sum(ok),
      pearson_ln_landvalue = cor(log(lv[ok]), lg(v[ok]), method = "pearson"),
      spearman_landvalue = cor(lv[ok], v[ok], method = "spearman"),
      note = "correlation only; no point-level SARAR is estimated here",
      stringsAsFactors = FALSE)
  }
  for (src in variants) for (m in MEASURES) {
    v <- Pt[[dmap[[m]]]]
    ok <- covered(src) & !is.na(v) & v > 0 & !is.na(lv) & lv > 0
    drows[[length(drows) + 1]] <- data.frame(
      extent = paste0("submitted@", src), measure = m, n = sum(ok),
      pearson_ln_landvalue = cor(log(lv[ok]), lg(v[ok]), method = "pearson"),
      spearman_landvalue = cor(lv[ok], v[ok], method = "spearman"),
      note = "submitted measure on the variant's covered points",
      stringsAsFactors = FALSE)
  }
  dis_tab <- do.call(rbind, drows)
}

# =============================================================================
# Write the tables
# =============================================================================

w <- function(df, stem) {
  p <- out_table(ctx, sprintf("stage4_boundary%s_%s.csv", SFX, stem))
  write.csv(df, p, row.names = FALSE); log_msg(ctx, "wrote ", p); p
}
w(agg_tab,    "aggregated")
w(rank_tab,   "measure_ranking")
w(rank_agree, "ranking_agreement")
if (!is.null(dis_tab)) w(dis_tab, "disaggregated_correlations")

# =============================================================================
# The report section
# =============================================================================

fmt <- function(x, d = 4) ifelse(is.na(x), "n/a", formatC(x, format = "f", digits = d))
pct <- function(x, d = 2) ifelse(is.na(x), "n/a",
                                 paste0(formatC(100 * x, format = "f", digits = d), " %"))
fmtn <- function(x) formatC(x, format = "d", big.mark = ",")
md_table <- function(df) {
  df[] <- lapply(df, function(x) gsub("|", " \\| ", as.character(x), fixed = TRUE))
  hdr <- paste0("| ", paste(names(df), collapse = " | "), " |")
  sep <- paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|")
  body <- apply(df, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  c(hdr, sep, body)
}
yesno <- function(x) ifelse(is.na(x), "—", ifelse(x, "yes", "**no**"))
getv <- function(df, q) { i <- match(q, df$quantity); if (is.na(i)) NA_real_ else df$value[i] }

vt  <- read.csv(out_table(ctx, sprintf("stage0_boundary%s_variants.csv", SFX)))
xc  <- read.csv(out_table(ctx, sprintf("stage0_boundary%s_extent_check.csv", SFX)))
cov <- read.csv(out_table(ctx, sprintf("stage2_boundary%s_layers.csv", SFX)))
mman <- file.path(REPO, BND$boundary$manifest)
muni_area <- vt$municipality_area_km2[1]

n_within <- getv(xc, "segments fully inside (within)")
n_inter  <- getv(xc, "segments touching (intersects)")
d_med    <- getv(xc, "distance beyond the municipal limit, p50 (m)")
d_max    <- getv(xc, "distance beyond the municipal limit, p100 (m)")
n_out    <- N_REF_SEGMENTS - n_within
n_2km    <- getv(xc, "outside segments within 2000 m of the limit")

mn <- if (file.exists(mman)) jsonlite::fromJSON(mman) else NULL
n_parts <- if (!is.null(mn)) mn$geometry$n_parts else NA_integer_
S <- c(
  "## Municipal variants",
  "",
  sprintf(paste0("*Generated by `src/43_boundary_muni.R` (`make boundary-muni`). ",
                 "Seed %s. Aggregated level. Variants built by ",
                 "`src/16_boundary_muni_variants.py`, PC by ",
                 "`src/15_boundary_pc_runs.py --family muni` with `prefcent` %s @ ",
                 "`%s`, CC/BC/FK by GAUS Lines v1.1 (Dalcin and Krafta 2021) run ",
                 "headlessly via `src/13_gaus_headless.py`. Weight specifications: ",
                 "%s; the headline is `%s`.*"),
          ctx$cfg$repro$seed, ctx$cfg$inputs$pc_solver$version,
          substr(ctx$cfg$inputs$pc_solver$upstream_commit, 1, 7),
          paste(sprintf("`%s` = `%s`", names(W_SPECS), W_SPECS), collapse = ", "),
          HEAD_W),
  "",
  paste0("The clip polygon is the IBGE *Malha Municipal Digital* 2025 for Rio ",
         "Grande do Sul, Porto Alegre `CD_MUN 4314902`. Clipping to it asks whether ",
         "the answer holds on the administrative unit the paper is about, and each ",
         "variant is kept connected (largest routing-graph component), so the ",
         "numbers here are the effect of the extent and not of fragmentation."),
  "",
  "### M0. The delivered network against the municipal polygon",
  "",
  sprintf("The municipality is **%.2f km²** over %s polygon part(s).",
          muni_area, n_parts),
  "",
  md_table(data.frame(
    quantity = xc$quantity, value = ifelse(abs(xc$value - round(xc$value)) < 1e-9,
                                           formatC(xc$value, format = "d", big.mark = ","),
                                           fmt(xc$value, 1)),
    share = ifelse(is.na(xc$share), "—", pct(xc$share)),
    `share of` = ifelse(xc$share_denominator == "" | is.na(xc$share_denominator),
                        "—", xc$share_denominator),
    `vs the manuscript's 29,978` = ifelse(is.na(xc$vs_paper_29978), "—",
                                          sprintf("%+d", as.integer(xc$vs_paper_29978))),
    check.names = FALSE, stringsAsFactors = FALSE)),
  "",
  sprintf(paste0("Section 4.1 of the manuscript gives the network as %s segments. ",
                 "Clipping the delivered file to the municipal polygon gives **%s** ",
                 "segments fully inside (%+.2f %% against %s) and **%s** touching ",
                 "(%+.2f %%)."),
          fmtn(PAPER_N_SEGMENTS), fmtn(n_within),
          100 * (n_within - PAPER_N_SEGMENTS) / PAPER_N_SEGMENTS,
          fmtn(PAPER_N_SEGMENTS), fmtn(n_inter),
          100 * (n_inter - PAPER_N_SEGMENTS) / PAPER_N_SEGMENTS),
  "",
  sprintf(paste0("%s of the %s delivered segments (%.1f %%) lie outside the ",
                 "municipality. They reach a median **%.0f m** and a maximum ",
                 "**%.1f km** beyond the municipal limit; %s of them (%.1f %%) are ",
                 "within 2 km of it."),
          fmtn(n_out), fmtn(N_REF_SEGMENTS), 100 * n_out / N_REF_SEGMENTS,
          d_med, d_max / 1000,
          fmtn(n_2km), 100 * n_2km / n_out),
  "",
  "### M1. The variants",
  "",
  paste0("A variant is the municipal polygon buffered outward by 0 m or 1,000 m, ",
         "and the clip rule is the negative-buffer family's: a segment is kept only ",
         "if it lies fully inside, geometries are never cut, `(node1, node2)` stays ",
         "stable and each variant is a subgraph of the reference network. R is ",
         "rebuilt on each clipped network and the enumeration order is recorded. ",
         "The full delivered extent is the third member of the sequence and needs ",
         "no run."),
  "",
  md_table(data.frame(
    variant = vt$label,
    extent = paste0("municipality + ", formatC(vt$outward_buffer_m, format = "d"), " m"),
    `area km2` = fmt(vt$area_km2, 2),
    `segments after clip` = vt$n_segments_clipped,
    `components as clipped` = vt$components_as_clipped,
    `component sizes` = vt$component_sizes_as_clipped,
    `dropped to reconnect` = vt$n_segments_dropped_by_connectedness,
    `segments analysed` = vt$n_segments,
    `share of reference segments` = pct(vt$segment_share_of_reference),
    `zones (R > 0)` = vt$n_zones_rebuilt,
    `sum R vs reference, same rows` = pct(vt$sum_R_rel_diff, 3),
    check.names = FALSE, stringsAsFactors = FALSE)),
  "",
  sprintf(paste0("**Connectedness.** The raw clip leaves %s components. Dropping ",
                 "components below %d segments on `%s` would remove %d component(s) ",
                 "and %d segment(s) and leave **%d** components, so the rule applied ",
                 "is the stronger one — keep the largest connected component — which ",
                 "costs %s segments (%s of the clipped network)."),
          paste(vt$components_as_clipped, collapse = " and "),
          vt$tiny_threshold_segments[1], vt$label[1],
          vt$n_components_below_tiny[1], vt$n_segments_below_tiny[1],
          vt$components_after_dropping_tiny[1],
          paste(vt$n_segments_dropped_by_connectedness, collapse = " and "),
          paste(pct(vt$n_segments_dropped_by_connectedness / vt$n_segments_clipped, 3),
                collapse = " and ")),
  "")
if (!is.null(vt$order_within_0.1pct)) {
  S <- c(S, "Order dependence of the rebuilt R:", "",
    md_table(data.frame(
      variant = vt$label,
      `R moves > 0.1 % on reversed order` = pct(1 - vt$order_within_0.1pct),
      `> 1 %` = pct(vt$order_share_above_1pct),
      `max` = pct(vt$order_max_rel, 1),
      `R within 0.1 % of reference` = pct(vt$R_within_0.1pct_of_reference),
      check.names = FALSE, stringsAsFactors = FALSE)), "")
}

# --- M2: the solver on a connected variant -----------------------------------
pcr <- NULL
pcrp <- out_table(ctx, sprintf("stage0_boundary%s_pc_runs.csv", SFX))
if (file.exists(pcrp)) {
  pcr <- read.csv(pcrp)
  pcr$converged <- pcr$final_step_norm < CONV_TOL
  full <- read.csv(out_table(ctx, "stage0_pc_runs.csv"))
  ref_pc1 <- full$final_step_norm[full$gamma == 1 & full$beta == 2][1]
  ref_pc2 <- full$final_step_norm[full$gamma == 0 & full$beta == 2][1]
  S <- c(S, "### M2. The solver on a connected variant", "",
    sprintf(paste0("On the reference network the submitted runs end at a final step ",
                   "norm of %s (γ = 1) and %s (γ = 0). On the municipal variants:"),
            formatC(ref_pc1, format = "e", digits = 3),
            formatC(ref_pc2, format = "e", digits = 3)),
    "",
    md_table(data.frame(
      variant = pcr$label, measure = pcr$measure, gamma = pcr$gamma,
      `graph components` = pcr$graph_components,
      `zones outside the largest` = pcr$n_zones_outside_largest_component,
      `final step norm` = formatC(pcr$final_step_norm, format = "e", digits = 3),
      converged = ifelse(pcr$converged, "yes", "**no**"),
      `sum mass / sum R` = fmt(pcr$sum_mass / pcr$sum_R, 6),
      check.names = FALSE, stringsAsFactors = FALSE)), "")
}

# --- M3: coverage -------------------------------------------------------------
S <- c(S, "### M3. Coverage", "",
  paste0("A 500 m cell is scored only where the variant puts the same number of ",
         "segments in it as the full network does (`intersects`), and an address ",
         "only where its nearest reference segment survives the clip."),
  "",
  md_table(data.frame(
    variant = cov$label, segments = cov$n_segments,
    `segment coverage` = pct(cov$segment_coverage),
    `cells touched` = cov$n_cells_touched,
    `cells fully covered` = cov$n_cells_fully_covered,
    `cell coverage` = pct(cov$cell_coverage),
    `points covered` = cov$n_points_covered,
    `point coverage` = pct(cov$point_coverage),
    check.names = FALSE, stringsAsFactors = FALSE)), "")
for (lb in variants) {
  n_an <- length(SAMPLES[[lb]]$rows)
  S <- c(S, sprintf(paste0("- **%s** enters the aggregated analysis with **N = %d** ",
                           "of the %d analysis cells (%.1f %%)."),
                    lb, n_an, nrow(D_all), 100 * n_an / nrow(D_all)))
}
S <- c(S, "",
  "### M4. How far each measure moves", "",
  paste0("Each variant's measure against its submitted column, on the segments ",
         "the two extents share. `within 1 %` is the share whose value is within ",
         "1 % of the submitted one; `spearman` is whether the order is the same."),
  "")
if (!is.null(seg_move)) {
  for (lvl in c("segments (common)", "cells (fully covered)", "points (covered)")) {
    sm <- seg_move[seg_move$level == lvl, ]
    if (!nrow(sm)) next
    S <- c(S, sprintf("At the **%s** level:", lvl), "",
      md_table(data.frame(
        variant = sm$label, measure = sm$measure, n = sm$n_common,
        spearman = fmt(sm$spearman, 4), `pearson (ln)` = fmt(sm$pearson_ln, 4),
        `within 1 %` = pct(sm$within_1pct),
        `median rel` = formatC(sm$median_rel, format = "e", digits = 2),
        `mean shift` = pct(sm$mean_rel_shift, 1),
        check.names = FALSE, stringsAsFactors = FALSE)), "")
  }
}
S <- c(S,
  "### M5. The headline SARAR and the ranking of the measures", "",
  paste0("`ln_land_value ~ ln_plot_area + ln_<measure>`, `sphet::spreg(model = ",
         "\"sarar\", het = TRUE)`, coordinate ties broken by the configured sort ",
         "key. Rows with extent `reference` restricted to a variant's cells are the ",
         "submitted measures on the same cells, so the difference between that row ",
         "and the variant row is the extent and nothing else. `std` is β × SD(ln ",
         "measure) on the fitted rows. `at_bound` = 1 marks a GMM error parameter on ",
         "the estimator's bound of ±", BOUND, "."),
  "")
tab4 <- data.frame(
  extent = agg_tab$extent, scope = agg_tab$scope, measure = agg_tab$measure,
  W = agg_tab$w_role, N = agg_tab$n_sarar,
  `Pearson ln` = fmt(agg_tab$pearson_ln_landvalue),
  beta_hat = fmt(agg_tab$sarar_beta), se = fmt(agg_tab$sarar_se),
  p = fmt(agg_tab$sarar_p, 4), std = fmt(agg_tab$sarar_beta_std),
  lambda = fmt(agg_tab$lambda_lag, 3), rho = fmt(agg_tab$rho_error, 3),
  at_bound = agg_tab$error_at_bound, pseudo_R2 = fmt(agg_tab$pseudo_r2),
  check.names = FALSE, stringsAsFactors = FALSE)
tab4 <- tab4[order(tab4$measure, tab4$W, tab4$extent), ]
S <- c(S, md_table(tab4), "",
  paste0("The paper's conclusion is an order over the measures, ranked on the ",
         "signed standardized β (higher = better), and two claims inside it: ",
         "**PC1 first**, and **accessibility (PC1, PC2, CC) above intermediation ",
         "(BC, FK)**. Both are scored explicitly."),
  "",
  md_table(data.frame(
    sample = rank_tab$sample, extent = rank_tab$extent, W = rank_tab$w_role,
    `ranking by std beta` = rank_tab$order, top = rank_tab$top,
    `top std` = fmt(rank_tab$top_std),
    `PC1 first` = yesno(rank_tab$pc1_first),
    `access > intermediation` = yesno(rank_tab$access_beats_intermediation),
    `significant at 5 %` = rank_tab$significant,
    check.names = FALSE, stringsAsFactors = FALSE)), "",
  paste0("Agreement of each ranking with two baselines. **vs restricted** is the ",
         "submitted measures on the *same* cells, so it isolates the extent."),
  "",
  md_table(data.frame(
    sample = rank_agree$sample, extent = rank_agree$extent, W = rank_agree$w_role,
    `ranking` = rank_agree$order,
    `rho vs full` = fmt(rank_agree$spearman_vs_full, 3),
    `same top vs full` = yesno(rank_agree$same_top_vs_full),
    `rho vs restricted` = fmt(rank_agree$spearman_vs_restricted, 3),
    `same top vs restricted` = yesno(rank_agree$same_top_vs_restricted),
    check.names = FALSE, stringsAsFactors = FALSE)), "")
if (!is.null(dis_tab)) {
  S <- c(S, "### M6. Correlation with land value at the address points", "",
    paste0("Reported because a correlation needs no weight matrix; no point-level ",
           "SARAR is estimated here."),
    "",
    md_table(data.frame(
      extent = dis_tab$extent, measure = dis_tab$measure, n = dis_tab$n,
      `Pearson ln` = fmt(dis_tab$pearson_ln_landvalue),
      Spearman = fmt(dis_tab$spearman_landvalue),
      check.names = FALSE, stringsAsFactors = FALSE)), "")
}

# --- M7: summary, built from this run's tables --------------------------------
smc <- if (!is.null(seg_move)) seg_move[seg_move$level == "segments (common)", ] else NULL
smq <- if (!is.null(seg_move)) seg_move[seg_move$level == "points (covered)", ] else NULL
ra_var <- rank_agree[rank_agree$sample %in% variants, ]
rt_var <- rank_tab[rank_tab$sample %in% variants, ]
fitted <- !is.na(agg_tab$sarar_beta)
is_sub <- agg_tab$w_role == SUB_ROLE
S <- c(S, "### M7. Summary", "",
  sprintf("- **Sign.** %d of %d fitted coefficients (5 measures × %d samples × %d weight specifications) keep the sign the same measure has at the full extent under the same weight specification.",
          sum(agg_tab$sign_vs_full[fitted] == 1L, na.rm = TRUE), sum(fitted),
          length(SAMPLES), length(W_SPECS)),
  if (!is.null(smc) && nrow(smc)) sprintf("- **Order of the network.** The lowest segment-level Spearman against the submitted column is ρ = %s (%s, %s); the highest is ρ = %s (%s, %s).%s",
          fmt(min(smc$spearman, na.rm = TRUE), 4),
          smc$label[which.min(smc$spearman)], smc$measure[which.min(smc$spearman)],
          fmt(max(smc$spearman, na.rm = TRUE), 4),
          smc$label[which.max(smc$spearman)], smc$measure[which.max(smc$spearman)],
          if (nrow(smq)) sprintf(" At the address segments the lowest is ρ = %s (%s, %s).",
            fmt(min(smq$spearman, na.rm = TRUE), 4),
            smq$label[which.min(smq$spearman)], smq$measure[which.min(smq$spearman)]) else "") else NULL,
  if (!is.null(smc) && nrow(smc)) sprintf("- **Levels.** The share of common segments within 1 %% of the submitted value runs from %s to %s; the mean level shift runs from %s to %s.",
          pct(min(smc$within_1pct, na.rm = TRUE)), pct(max(smc$within_1pct, na.rm = TRUE)),
          pct(min(smc$mean_rel_shift, na.rm = TRUE), 1),
          pct(max(smc$mean_rel_shift, na.rm = TRUE), 1)) else NULL,
  sprintf("- **The paper's two claims.** PC1 is ranked first in %d of the %d (variant × W) cells and accessibility ranks above intermediation in %d of %d.",
          sum(rt_var$pc1_first), nrow(rt_var),
          sum(rt_var$access_beats_intermediation), nrow(rt_var)),
  sprintf("- **Ranking against the same cells.** Spearman vs the restricted submitted ranking: %s. Same top measure: %s.",
          paste(sprintf("%s/%s %s", ra_var$extent, ra_var$w_role,
                        fmt(ra_var$spearman_vs_restricted, 3)), collapse = ", "),
          paste(sprintf("%s/%s %s", ra_var$extent, ra_var$w_role,
                        ifelse(ra_var$same_top_vs_restricted, "yes", "no")),
                collapse = ", ")),
  if (!is.null(pcr)) sprintf("- **Convergence.** %d of %d runs on the municipal variants end below the %s criterion; the variants' routing graphs have %s component(s); final step norms %s.",
          sum(pcr$converged), nrow(pcr), formatC(CONV_TOL, format = "g"),
          paste(pcr$graph_components[!duplicated(pcr$label)], collapse = " and "),
          paste(unique(formatC(pcr$final_step_norm, format = "e", digits = 2)),
                collapse = " / ")) else NULL,
  sprintf("- **Bound.** The GMM error parameter sits on the bound of ±%s in %d of the %d fits under `%s` and %d of %d under the other specifications.",
          fmt(BOUND, 1),
          sum(agg_tab$error_at_bound[is_sub], na.rm = TRUE),
          sum(!is.na(agg_tab$error_at_bound[is_sub])), W_SPECS[[SUB_ROLE]],
          sum(agg_tab$error_at_bound[!is_sub], na.rm = TRUE),
          sum(!is.na(agg_tab$error_at_bound[!is_sub]))),
  "",
  "### M8. Tables",
  "",
  sprintf("- `outputs/tables/stage0_boundary%s_extent_check.csv` — the delivered network against the municipal polygon", SFX),
  sprintf("- `outputs/tables/stage0_boundary%s_variants.csv` — extents, components, segment counts, rebuilt R", SFX),
  sprintf("- `outputs/tables/stage0_boundary%s_pc_runs.csv` — the PC runs", SFX),
  sprintf("- `outputs/tables/stage2_boundary%s_layers.csv`, `stage2_boundary%s_coverage.csv`, `stage2_boundary%s_segment_movement.csv`", SFX, SFX, SFX),
  sprintf("- `outputs/tables/stage4_boundary%s_aggregated.csv`, `stage4_boundary%s_measure_ranking.csv`, `stage4_boundary%s_ranking_agreement.csv`", SFX, SFX, SFX),
  if (!is.null(dis_tab)) sprintf("- `outputs/tables/stage4_boundary%s_disaggregated_correlations.csv`", SFX) else NULL,
  sprintf("- `%s` + `%s` — the polygon and its provenance",
          BND$boundary$polygon, BND$boundary$manifest),
  "")

frag <- out_report(ctx, "stage4_boundary_muni_section.md")
writeLines(S, frag)
log_msg(ctx, "wrote ", frag)

standalone <- c(
  "# Stage 4 — the municipal network-extent variants",
  "",
  paste0("This is the section `src/42_boundary_sensitivity.R` splices into ",
         "`reports/stage4_boundary_sensitivity.md`, kept separately so it can be ",
         "read on its own."),
  "", S)
rp <- out_report(ctx, "stage4_boundary_muni.md")
writeLines(standalone, rp)
log_msg(ctx, "wrote ", rp)
write_counts(ctx, sprintf("stage4_boundary%s_counts.csv", SFX))
finish(ctx)
