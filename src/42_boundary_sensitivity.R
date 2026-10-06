#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 — network-extent (boundary) sensitivity, aggregated level: the
# negative-buffer stress test.
#
# Reviewer 3 asks how much the five measures, and the conclusions drawn from
# them, depend on where the street network was cut off. This script is the
# stress-test half of that answer; src/43_boundary_muni.R is the municipal half
# and writes a report section that is spliced in at the top of this report.
#
# The variants are built by negative buffer (src/14_boundary_variants.py): the
# reference extent is the union of the 30 m buffers of every delivered segment
# with its interior rings dropped (the "footprint"), and a variant is that
# footprint buffered inward by -2,000 m or -4,000 m, keeping only the segments
# that lie fully inside. The static weight R is rebuilt on each clipped network
# and the enumeration order is recorded. PC1/PC2 come from the solver of record
# run on the clipped network (src/15_boundary_pc_runs.py); CC/BC/FK from GAUS
# Lines v1.1 itself, run headlessly under QGIS (src/13_gaus_headless.py).
#
# What is here:
#   1. Coverage — how much of the network, the grid and the address sample each
#      variant keeps. Every later number is conditional on this.
#   2. Segment-level movement — Spearman, Pearson and the share within 1 % of
#      each measure against its submitted column on the common segments.
#   3. Correlations with land value and the headline SARAR per variant, under
#      both weight specifications, on the surviving sample. The full-extent
#      analysis is re-run on the same restricted sample as well, so a difference
#      is the extent and not the sample.
#   4. Whether the ranking of the five measures (signed standardized beta,
#      higher = better) survives, scored separately from the coefficients.
#
# Run:  make boundary      (Rscript src/42_boundary_sensitivity.R)
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "boundary_sensitivity")
capture_env_r(ctx)

AGG    <- "aggregated"
BND    <- ctx$cfg$revision$boundary_sensitivity
W_SUB  <- ctx$cfg$weights$active[[AGG]]
W_PROP <- ctx$cfg$weights$proposed[[AGG]]
# Role labels, written into the `w_role` column of every table.
W_SPECS <- c(submitted_band = W_SUB, proposed = W_PROP)
BOUND  <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
ALPHA  <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
CTRL   <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
STD    <- isTRUE(ctx$cfg$revision$standardize_coefficients)
MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
STAT <- vapply(MEASURES, function(m)
  ctx$cfg$grid$aggregation[[if (m %in% ctx$cfg$measures$accessibility)
    "accessibility" else "intermediation"]], character(1))
# A preferential-centrality run counts as converged below this final step norm.
CONV_TOL <- 1e-8

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c")
  log(x + ctx$cfg$transforms$log_plus_c) else log(x)

variants <- vapply(BND$input_files, function(r) as.character(r$label), character(1))
if (!length(variants)) stop("revision.boundary_sensitivity.input_files is empty",
                            call. = FALSE)
log_msg(ctx, sprintf("variants: %s", paste(variants, collapse = ", ")))

# =============================================================================
# 1. The aggregated sample, and each variant's covered subset
# =============================================================================

Land <- st_read(cfg_path(ctx, "frozen", "cents_aggregated"), quiet = TRUE)
crs_expected <- ctx$cfg$repro$crs_epsg
if (is.na(st_crs(Land)$epsg) || st_crs(Land)$epsg != crs_expected)
  Land <- st_transform(Land, crs_expected)
if (isTRUE(st_is_longlat(Land)))
  stop("layer is in geographic coordinates", call. = FALSE)
Land$cell_id <- Land[[ctx$cfg$grid$frozen_500m$id_column]]

bpath <- file.path(REPO, BND$aggregated_layer)
if (!file.exists(bpath))
  stop("missing ", BND$aggregated_layer,
       " -- run `make boundary-layers` first", call. = FALSE)
B <- st_drop_geometry(st_read(bpath, quiet = TRUE))
log_msg(ctx, sprintf("boundary layer: %d cells, %d columns", nrow(B), ncol(B)))
Land <- merge(Land, B[, setdiff(names(B), c("row_index", "col_index"))],
              by = "cell_id", all.x = TRUE, sort = FALSE)

map <- ctx$cfg$measures$columns[[AGG]]
for (canon in names(map)) {
  srcn <- map[[canon]]
  if (!is.null(Land[[srcn]]) && srcn != canon) Land[[canon]] <- Land[[srcn]]
}
# apply_exclusions() also sorts the frame on the tie-break key, so every
# neighbour list below is independent of the file's row order.
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

# The samples this report estimates on. Each variant contributes two: the
# variant's own measures, and the submitted measures on the same cells -- so a
# difference between them is the extent and not the sample.
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
      # the SD that standardizes beta is taken on exactly the rows the model is
      # fitted on (a distance band can drop isolated cells)
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
        "%-18s %-4s [%s W, N=%d]: r=%.4f rho_s=%.4f  b=%+.4f (se %.4f) std=%+.4f p=%.4f  lambda=%+.3f rho=%+.3f%s",
        snm, m, wnm, sum(okh), pear, spea, bb, se, bb * sdln, pv, lam, rr,
        if (isTRUE(ab == 1L)) " AT BOUND" else ""))
    }
  }
}
agg_tab <- do.call(rbind, rows)

# =============================================================================
# 3. Does the ranking of the five measures survive?
# =============================================================================
# The paper's conclusion is an order over the measures, ranked on the signed
# standardized coefficient (higher = better). Rank the five within each
# (sample, W), and score each variant's order against the order it should
# reproduce.

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
    stringsAsFactors = FALSE)
}
rank_tab <- do.call(rbind, rank_rows)

# Two agreements per sample; the second isolates the extent.
#   vs full       = variant against the submitted analysis on all cells. Mixes
#                   the extent with the smaller sample.
#   vs restricted = variant against the submitted measures on the same cells.
#                   The sample is identical on both sides and only the network
#                   the measures were computed on differs.
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

# Sign check: each coefficient against the full-extent coefficient for the same
# measure under the same weight specification.
full_sign <- with(agg_tab[agg_tab$sample == "full", ],
                  setNames(sign(sarar_beta), paste(measure, w_role)))
agg_tab$sign_vs_full <- as.integer(
  sign(agg_tab$sarar_beta) == full_sign[paste(agg_tab$measure, agg_tab$w_role)])

# =============================================================================
# 4. Segment-level movement (from stage 2) and the point-level correlations
# =============================================================================

seg_move <- NULL
smp <- out_table(ctx, "stage2_boundary_segment_movement.csv")
if (file.exists(smp)) {
  seg_move <- read.csv(smp)
  log_msg(ctx, sprintf("segment/cell/point movement table: %d rows", nrow(seg_move)))
} else {
  log_msg(ctx, "!! no stage2_boundary_segment_movement.csv -- run `make boundary-layers`")
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
  # the submitted measures on each variant's covered points, so the comparison
  # is like for like
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
# Write
# =============================================================================

w <- function(df, fn) { p <- out_table(ctx, fn); write.csv(df, p, row.names = FALSE)
                        log_msg(ctx, "wrote ", p); p }
w(agg_tab,    "stage4_boundary_aggregated.csv")
w(rank_tab,   "stage4_boundary_measure_ranking.csv")
w(rank_agree, "stage4_boundary_ranking_agreement.csv")
if (!is.null(dis_tab)) w(dis_tab, "stage4_boundary_disaggregated_correlations.csv")

fmt <- function(x, d = 4) ifelse(is.na(x), "n/a", formatC(x, format = "f", digits = d))
pct <- function(x, d = 2) ifelse(is.na(x), "n/a",
                                 paste0(formatC(100 * x, format = "f", digits = d), " %"))
md_table <- function(df) {
  df[] <- lapply(df, function(x) gsub("|", " \\| ", as.character(x), fixed = TRUE))
  hdr <- paste0("| ", paste(names(df), collapse = " | "), " |")
  sep <- paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|")
  body <- apply(df, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  c(hdr, sep, body)
}
yesno <- function(x) ifelse(is.na(x), "—", ifelse(x, "yes", "**no**"))

vt  <- read.csv(out_table(ctx, "stage0_boundary_variants.csv"))
cov <- read.csv(out_table(ctx, "stage2_boundary_layers.csv"))
n_ref_segments <- as.integer(ctx$cfg$inputs$segment_centralities$expected_features)

L <- c(
  "# Stage 4 — network-extent (boundary) sensitivity",
  "",
  sprintf(paste0("*Generated by `src/42_boundary_sensitivity.R` (`make boundary`). ",
                 "Seed %s. Aggregated level. Two weight specifications: `%s` (the ",
                 "weight matrix of the submitted tables) and the proposed `%s`. ",
                 "Variants built by `src/14_boundary_variants.py`, PC by ",
                 "`src/15_boundary_pc_runs.py` with `prefcent` %s @ `%s`, CC/BC/FK ",
                 "by GAUS Lines v1.1 (Dalcin and Krafta 2021) run headlessly via ",
                 "`src/13_gaus_headless.py`.*"),
          ctx$cfg$repro$seed, W_SUB, W_PROP,
          ctx$cfg$inputs$pc_solver$version,
          substr(ctx$cfg$inputs$pc_solver$upstream_commit, 1, 7)),
  "",
  "## The negative-buffer stress test",
  "",
  paste0("The extent the delivered network covers is shrunk by a fixed distance ",
         "from every outer edge, every street not entirely inside is dropped, the ",
         "capacities are rebuilt on what is left, and all five centrality measures ",
         "are recomputed on each smaller network. The aggregated analysis is then ",
         "re-run on the cells that survive intact. The tables report how far each ",
         "measure moved, how far each coefficient moved, and whether the five ",
         "measures still come out in the same order."),
  "",
  "## 1. The variants",
  "",
  paste0("**The extent rule.** The reference extent is not a polygon anywhere in ",
         "the supplied material, so it is reconstructed: the union of the 30 m ",
         "buffers of all delivered segments — 30 m because that is the buffer that ",
         "defines the static weight R, so the extent and the capacity rule measure ",
         "the city at the same width. The inward buffer is applied to the union's ",
         "**footprint**, i.e. the union with its interior rings (the insides of ",
         "city blocks) dropped, because eroding from the inside of every block ",
         "would erase the network without shrinking the study area."),
  "",
  sprintf(paste0("Union of the 30 m buffers: **%.2f km²**. Footprint (interior ",
                 "rings dropped): **%.2f km²**. That footprint is what gets ",
                 "buffered inward."),
          vt$reference_union_area_km2[1], vt$reference_footprint_area_km2[1]),
  "",
  paste0("**The clip rule.** A segment is kept only if it lies fully inside the ",
         "variant polygon; geometries are never cut. Cutting would leave `dist` — a ",
         "stored column the solver reads — describing a segment that no longer ",
         "exists and would move the zone off the midpoint of its own street. ",
         "Dropping whole segments keeps `(node1, node2)` stable and makes each ",
         "variant a subgraph of the reference network."),
  "",
  paste0("**R is rebuilt.** R is the buildable area of a *non-overlapping* 30 m ",
         "buffer, so a segment's weight depends on which neighbours it competes ",
         "with for the space between them. Each variant's R is therefore rebuilt on ",
         "its own clipped network, and the segment enumeration order is recorded in ",
         "the manifest because R is not a pure function of the geometry."),
  "")
tab1 <- data.frame(
  variant = vt$label, `inward buffer` = paste0(fmt(vt$inward_buffer_m, 0), " m"),
  `area km2` = fmt(vt$area_km2, 2),
  `share of footprint` = pct(vt$area_share_of_reference),
  segments = vt$n_segments, `share of segments` = pct(vt$segment_share_of_reference),
  `zones (R > 0)` = vt$n_zones_rebuilt,
  `sum R vs reference, same rows` = pct(vt$sum_R_rel_diff, 3),
  `R within 0.1 % of reference` = pct(vt$R_within_0.1pct_of_reference),
  check.names = FALSE, stringsAsFactors = FALSE)
L <- c(L, md_table(tab1), "",
  paste0("The last two columns separate the two reasons a variant's R differs from ",
         "the delivered `area` column: the clip, and the rebuild's own disagreement ",
         "with the delivered column on the rows the two share."),
  "")
if (!is.null(vt$order_within_0.1pct)) {
  tab1b <- data.frame(
    variant = vt$label,
    `R moves > 0.1 % on reversed order` = pct(1 - vt$order_within_0.1pct),
    `> 1 %` = pct(vt$order_share_above_1pct),
    `max` = pct(vt$order_max_rel, 1),
    check.names = FALSE, stringsAsFactors = FALSE)
  L <- c(L, "### Order dependence of the rebuilt R, per variant", "",
    md_table(tab1b), "",
    paste0("Reversing the enumeration order and changing nothing else moves this ",
           "much of R. It is why the order is recorded, and it is a floor under how ",
           "precisely a boundary variant can be reproduced."),
    "")
}
# --- 1b. what the clip does to the routing graph and to the solver -----------
pcr <- NULL
pcrp <- out_table(ctx, "stage0_boundary_pc_runs.csv")
if (file.exists(pcrp)) {
  pcr <- read.csv(pcrp)
  pcr$converged <- pcr$final_step_norm < CONV_TOL
  full <- read.csv(out_table(ctx, "stage0_pc_runs.csv"))
  ref_pc1 <- full$final_step_norm[full$gamma == 1 & full$beta == 2][1]
  ref_pc2 <- full$final_step_norm[full$gamma == 0 & full$beta == 2][1]
  comp <- pcr$graph_components[!duplicated(pcr$label)]
  L <- c(L,
    "## 1b. The clip and the routing graph",
    "",
    paste0("Removing whole segments can disconnect the network. Zones in a ",
           "component of their own cannot reach any other zone, their row of the ",
           "interaction kernel is zero, and the fixed-point iteration behaves ",
           "differently from the reference run even though every parameter is the ",
           "submitted one."),
    "",
    md_table(data.frame(
      variant = pcr$label, measure = pcr$measure, gamma = pcr$gamma,
      `graph components` = pcr$graph_components,
      `zones outside the largest` = pcr$n_zones_outside_largest_component,
      `final step norm` = formatC(pcr$final_step_norm, format = "e", digits = 3),
      converged = ifelse(pcr$converged, "yes", "**no**"),
      `max density` = formatC(pcr$max_density, format = "g", digits = 6),
      `sum mass / sum R` = fmt(pcr$sum_mass / pcr$sum_R, 6),
      check.names = FALSE, stringsAsFactors = FALSE)), "",
    sprintf(paste0("On the reference network the same two runs end at a final step ",
                   "norm of %s (γ = 1) and %s (γ = 0). On the variants %d of %d ",
                   "runs end below %s; the clipped routing graphs have %s ",
                   "components. Where a run does not converge, the variant's PC ",
                   "columns are the 200-iterate of the submitted specification on a ",
                   "disconnected graph, and the movement reported below is the joint ",
                   "effect of less network and a cut network."),
            formatC(ref_pc1, format = "e", digits = 3),
            formatC(ref_pc2, format = "e", digits = 3),
            sum(pcr$converged), nrow(pcr), formatC(CONV_TOL, format = "g"),
            paste(comp, collapse = " and ")),
    "")
}

L <- c(L, "## 2. Coverage — what each variant keeps", "",
  paste0("Every number after this is conditional on coverage. A 500 m cell is ",
         "scored only where the variant puts the same number of segments in it as ",
         "the full network does (`intersects`); an address is scored only where its ",
         "nearest reference segment survives the clip. Both rules are the ",
         "acceptance test's own partial-network rules."),
  "")
tab2 <- data.frame(
  variant = cov$label, segments = cov$n_segments,
  `segment coverage` = pct(cov$segment_coverage),
  `cells touched` = cov$n_cells_touched,
  `cells fully covered` = cov$n_cells_fully_covered,
  `cell coverage` = pct(cov$cell_coverage),
  `points covered` = cov$n_points_covered,
  `point coverage` = pct(cov$point_coverage),
  check.names = FALSE, stringsAsFactors = FALSE)
L <- c(L, md_table(tab2), "")
for (lb in variants) {
  n_an <- length(SAMPLES[[lb]]$rows)
  L <- c(L, sprintf(paste0("- **%s** enters the aggregated analysis with **N = %d** ",
                           "of the %d analysis cells (%.1f %%)."),
                    lb, n_an, nrow(D_all), 100 * n_an / nrow(D_all)))
}
L <- c(L, "",
  "## 3. How far each measure moves",
  "",
  paste0("Each variant's measure against its submitted column, **on the segments ",
         "the two extents share**, before any model. `within 1 %` is the share of ",
         "common segments whose value is within 1 % of the submitted one; ",
         "`spearman` is whether the order is the same."),
  "")
if (!is.null(seg_move)) {
  sm <- seg_move[seg_move$level == "segments (common)", ]
  L <- c(L, md_table(data.frame(
    variant = sm$label, measure = sm$measure, `n common` = sm$n_common,
    spearman = fmt(sm$spearman, 4), `pearson (ln)` = fmt(sm$pearson_ln, 4),
    `within 1 %` = pct(sm$within_1pct), `median rel` = formatC(sm$median_rel, format = "e", digits = 2),
    `mean shift` = pct(sm$mean_rel_shift, 1),
    check.names = FALSE, stringsAsFactors = FALSE)), "")
  sc <- seg_move[seg_move$level == "cells (fully covered)", ]
  L <- c(L, "Same, at the 500 m cells that are fully covered:", "",
    md_table(data.frame(
      variant = sc$label, measure = sc$measure, `n cells` = sc$n_common,
      spearman = fmt(sc$spearman, 4), `within 1 %` = pct(sc$within_1pct),
      `mean shift` = pct(sc$mean_rel_shift, 1),
      check.names = FALSE, stringsAsFactors = FALSE)), "")
  sq <- seg_move[seg_move$level == "points (covered)", ]
  L <- c(L,
    "And at the address points whose reference segment survives, where the sample lives:",
    "",
    md_table(data.frame(
      variant = sq$label, measure = sq$measure, `n points` = sq$n_common,
      spearman = fmt(sq$spearman, 4), `within 1 %` = pct(sq$within_1pct),
      `mean shift` = pct(sq$mean_rel_shift, 1),
      check.names = FALSE, stringsAsFactors = FALSE)), "")
  pc1s <- sm[sm$measure == "PC1", ]; pc1q <- sq[sq$measure == "PC1", ]
  pc1q <- pc1q[match(pc1s$label, pc1q$label), ]
  L <- c(L,
    sprintf(paste0("PC1's Spearman against its submitted column is %s over all ",
                   "common segments and %s over the segments the addresses sit on ",
                   "(%s). Across all measures at the points, the mean level shift ",
                   "runs from %s to %s."),
            paste(fmt(pc1s$spearman, 2), collapse = " / "),
            paste(fmt(pc1q$spearman, 2), collapse = " / "),
            paste(pc1s$label, collapse = " / "),
            pct(min(sq$mean_rel_shift, na.rm = TRUE), 1),
            pct(max(sq$mean_rel_shift, na.rm = TRUE), 1)),
    "")
}
L <- c(L,
  "## 4. Correlation with land value and the headline SARAR",
  "",
  paste0("`ln_land_value ~ ln_plot_area + ln_<measure>`, `sphet::spreg(model = ",
         "\"sarar\", het = TRUE)`. `reference` rows use the submitted measures; ",
         "`restricted to <variant> cells` with extent `reference` is the same ",
         "submitted measures on the same cells the variant is estimated on, so the ",
         "difference between that row and the variant row is the extent and nothing ",
         "else. `std` is β × SD(ln measure) on the fitted rows. `at_bound` = 1 marks ",
         "a GMM error parameter on the estimator's bound of ±", BOUND, "."),
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
L <- c(L, md_table(tab4), "",
  "## 5. Does the ranking of the five measures survive?",
  "",
  paste0("The paper's conclusion is an order over the measures, ranked on the ",
         "signed standardized coefficient (higher = better), so it is scored on its ",
         "own rather than inferred from the coefficient table."),
  "",
  md_table(data.frame(
    sample = rank_tab$sample, extent = rank_tab$extent, W = rank_tab$w_role,
    `ranking by std beta` = rank_tab$order, top = rank_tab$top,
    `top std` = fmt(rank_tab$top_std),
    `significant at 5 %` = rank_tab$significant,
    check.names = FALSE, stringsAsFactors = FALSE)), "",
  paste0("Agreement of each ranking with two baselines. **vs full extent** mixes ",
         "the extent with the smaller sample; **vs restricted** is the submitted ",
         "measures on the *same* cells, so it isolates the extent."),
  "",
  md_table(data.frame(
    sample = rank_agree$sample, extent = rank_agree$extent, W = rank_agree$w_role,
    `ranking` = rank_agree$order,
    `ρ vs full` = fmt(rank_agree$spearman_vs_full, 3),
    `same top vs full` = yesno(rank_agree$same_top_vs_full),
    `ρ vs restricted` = fmt(rank_agree$spearman_vs_restricted, 3),
    `same top vs restricted` = yesno(rank_agree$same_top_vs_restricted),
    check.names = FALSE, stringsAsFactors = FALSE)), "")
if (!is.null(dis_tab)) {
  L <- c(L, "## 6. Correlation with land value at the address points", "",
    paste0("Reported because a correlation needs no weight matrix; no point-level ",
           "SARAR is estimated here. Restricted to the addresses whose reference ",
           "segment survives the clip."),
    "",
    md_table(data.frame(
      extent = dis_tab$extent, measure = dis_tab$measure, n = dis_tab$n,
      `Pearson ln` = fmt(dis_tab$pearson_ln_landvalue),
      Spearman = fmt(dis_tab$spearman_landvalue),
      check.names = FALSE, stringsAsFactors = FALSE)), "")
}

# ---- summary, every sentence built from this run's tables --------------------
sm <- if (!is.null(seg_move)) seg_move[seg_move$level == "segments (common)", ] else NULL
fitted <- !is.na(agg_tab$sarar_beta)
n_sign_kept <- sum(agg_tab$sign_vs_full[fitted] == 1L, na.rm = TRUE)
ra_var <- rank_agree[rank_agree$sample %in% variants, ]
rk_var <- rank_tab[rank_tab$sample %in% variants, ]
variant_line <- function(lb) {
  a <- ra_var[ra_var$sample == lb, ]
  r <- rk_var[rk_var$sample == lb, ]
  r <- r[match(a$w_role, r$w_role), ]
  sprintf("- **%s** (N = %d cells): %s.", lb, length(SAMPLES[[lb]]$rows),
          paste(sprintf("under `%s` Spearman vs restricted %s, same top measure: %s, %d of 5 measures significant at %.0f %%",
                        a$w_role, fmt(a$spearman_vs_restricted, 3),
                        ifelse(is.na(a$same_top_vs_restricted), "n/a",
                               ifelse(a$same_top_vs_restricted, "yes", "no")),
                        r$n_significant, 100 * ALPHA),
                collapse = "; "))
}
L <- c(L, "## Summary", "",
  sprintf("- **Sign.** %d of %d fitted coefficients keep the sign the same measure has at the full extent under the same weight specification.",
          n_sign_kept, sum(fitted)),
  if (!is.null(sm)) sprintf("- **Order of the network, measure by measure.** Segment-level Spearman against the submitted column ranges from ρ = %s (%s, %s) to ρ = %s (%s, %s).",
          fmt(min(sm$spearman, na.rm = TRUE), 4),
          sm$label[which.min(sm$spearman)], sm$measure[which.min(sm$spearman)],
          fmt(max(sm$spearman, na.rm = TRUE), 4),
          sm$label[which.max(sm$spearman)], sm$measure[which.max(sm$spearman)]) else NULL,
  if (!is.null(sm)) sprintf("- **Levels.** The share of common segments within 1 %% of the submitted value runs from %s to %s. Each of the five measures sums over reachable lines, so removing lines moves values far from the cut as well; this is why the ranking is reported separately from the level.",
          pct(min(sm$within_1pct, na.rm = TRUE)), pct(max(sm$within_1pct, na.rm = TRUE))) else NULL,
  "- **Ranking of the measures, against the submitted measures on the same cells:**",
  vapply(variants, variant_line, character(1)),
  if (!is.null(dis_tab)) sprintf("- **PC1 and land value at the address points.** %s",
          paste(vapply(variants, function(lb) sprintf(
            "%s: Spearman %s on the submitted measure and %s on the variant measure, same addresses",
            lb,
            fmt(dis_tab$spearman_landvalue[dis_tab$extent == paste0("submitted@", lb) & dis_tab$measure == "PC1"], 3),
            fmt(dis_tab$spearman_landvalue[dis_tab$extent == lb & dis_tab$measure == "PC1"], 3)),
            character(1)), collapse = "; ")) else NULL,
  if (!is.null(pcr)) sprintf("- **Convergence.** %d of %d preferential-centrality runs on the variants end below the %s criterion within the 200-iteration budget; the variants' routing graphs have %s components (§1b).",
          sum(pcr$converged), nrow(pcr), formatC(CONV_TOL, format = "g"),
          paste(pcr$graph_components[!duplicated(pcr$label)], collapse = " and ")) else NULL,
  sprintf("- **Bound.** The GMM error parameter sits on the bound of ±%s in %d of the %d fits under `%s` and %d of %d under `%s`.",
          fmt(BOUND, 1),
          sum(agg_tab$error_at_bound[agg_tab$w_role == "submitted_band"], na.rm = TRUE),
          sum(!is.na(agg_tab$error_at_bound[agg_tab$w_role == "submitted_band"])),
          W_SUB,
          sum(agg_tab$error_at_bound[agg_tab$w_role == "proposed"], na.rm = TRUE),
          sum(!is.na(agg_tab$error_at_bound[agg_tab$w_role == "proposed"])),
          W_PROP),
  "",
  "## Tables",
  "",
  "- `outputs/tables/stage0_boundary_variants.csv` — the extents, areas, segment counts and rebuilt R",
  "- `outputs/tables/stage0_boundary_pc_runs.csv` — the PC runs on each variant",
  "- `outputs/tables/stage2_boundary_layers.csv`, `stage2_boundary_coverage.csv`",
  "- `outputs/tables/stage2_boundary_segment_movement.csv` — measure movement at three levels",
  "- `outputs/tables/stage4_boundary_aggregated.csv`",
  "- `outputs/tables/stage4_boundary_measure_ranking.csv`, `stage4_boundary_ranking_agreement.csv`",
  if (!is.null(dis_tab)) "- `outputs/tables/stage4_boundary_disaggregated_correlations.csv`" else NULL,
  "")

# ---- splice in the municipal family ------------------------------------------
# src/43_boundary_muni.R writes its section to a fragment rather than to its own
# report, so that one generated document carries both families. It goes first,
# ahead of the negative-buffer sections.
frag <- out_report(ctx, "stage4_boundary_muni_section.md")
if (file.exists(frag)) {
  i <- match("## The negative-buffer stress test", L)
  n_cells_var <- vapply(variants, function(lb) length(SAMPLES[[lb]]$rows), integer(1))
  L <- append(L, c(
    readLines(frag, warn = FALSE),
    "---",
    ""), after = i - 1)
  L[match("## The negative-buffer stress test", L) + 2] <- paste0(
    L[match("## The negative-buffer stress test", L) + 2], " ",
    sprintf(paste0("Unlike the municipal variants above, it asks *how big* the study ",
                   "area has to be. It keeps %s of the analysis cells and %s of the ",
                   "addresses%s."),
            paste(pct(n_cells_var / nrow(D_all), 0), collapse = " / "),
            paste(pct(cov$point_coverage[match(variants, cov$label)], 0), collapse = " / "),
            if (is.null(pcr)) "" else sprintf(", and its clipped routing graphs have %s components",
              paste(pcr$graph_components[!duplicated(pcr$label)], collapse = " / "))))
  log_msg(ctx, "spliced in ", frag)
} else {
  log_msg(ctx, "!! no stage4_boundary_muni_section.md -- run `make boundary-muni` ",
          "first if the municipal family is supposed to be in this report")
}

rp <- out_report(ctx, "stage4_boundary_sensitivity.md")
writeLines(L, rp)
log_msg(ctx, "wrote ", rp)
write_counts(ctx, "stage4_boundary_counts.csv")
finish(ctx)
