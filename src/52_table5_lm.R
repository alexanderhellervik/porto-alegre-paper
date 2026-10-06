#!/usr/bin/env Rscript
# =============================================================================
# Stage 4 -- the LM test battery behind Table 5.
#
# Table 5 of the revision is estimated on the rebuilt dependent variable under
# the revision's weight matrices (`weights.proposed`: kNN k = 6 at the cell
# level, k = 8 at the address level, ties decided by
# `weights.identical_points`). The four LM rows of Table 5 are computed here on
# the frame its coefficients were estimated on; the W sweep's LM suite runs on
# the same dependent variable (`dependent_variable.proposed_source`) and is
# used as a cross-check.
#
# What it does. Nothing but the LM battery. Same specification as Table 5
# (`ln_land_value ~ ln_plot_area + ln_<measure>`, five measures), same W, same
# exclusion order, same tie-break, on both dependent-variable sources so that
# the effect of the swap is visible:
#
#   frozen   -- `Preco` / `Puni` as shipped in the frozen layers (what every
#               submitted number was computed on).
#   rebuilt  -- stage 1's rebuild from the raw ITBI + registry records
#               (`dependent_variable.proposed_source`).
#
# and applies Anselin's decision rules (lm_rule() in common.R) per cell, so
# the table can say whether SARAR is the model the tests select.
#
# No estimation happens here: the LM suite is an OLS diagnostic, and Table 5's
# coefficients come from src/48_rebuilt_dv.R.
#
# The data loading repeats src/48_rebuilt_dv.R's checks (id alignment, CRS,
# frozen-column agreement), so the frames these tests run on are the frames
# those fits ran on. Every filter goes through apply_exclusions(), every W
# through build_weights(), the suite through run_lm_tests() -- the same calls
# the gate and the W sweep make -- so the rows on the sweep's source must
# reproduce the W sweep's, and the script checks that against
# `stage4_w_choice_model_selection.csv` when that file exists (run `make
# w-choice` first).
#
# Compute. Twenty OLS fits and twenty LM suites, seconds. Single process;
# `OMP_NUM_THREADS` is set by the Makefile target.
#
# Run:  Rscript src/52_table5_lm.R            (or `make table5-lm`)
# =============================================================================

suppressPackageStartupMessages({
  library(sf); library(spdep)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

ctx <- init(4, "table5_lm")
capture_env_r(ctx)

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
SOURCES  <- c("frozen", "rebuilt")
LEVELS   <- c("disaggregated", "aggregated")
DV       <- ctx$cfg$dependent_variable
W_SPEC   <- list(disaggregated = ctx$cfg$weights$proposed$disaggregated,
                 aggregated    = ctx$cfg$weights$proposed$aggregated)
ALPHA    <- as.numeric(ctx$cfg$revision$lm_selection$alpha)
BOTH_ROB <- ctx$cfg$revision$lm_selection$both_robust_significant
CTRL     <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""
frm      <- function(m) as.formula(sprintf("ln_land_value ~ %sln_%s", CTRL, m))

if (!identical(DV$proposed_source, "rebuilt"))
  stop("dependent_variable.proposed_source is '", DV$proposed_source,
       "'; this script runs the LM battery on `rebuilt` beside `frozen`",
       call. = FALSE)
log_msg(ctx, sprintf("dependent variable: frozen (the gate's) and %s (Table 5's)",
                     DV$proposed_source))
log_msg(ctx, sprintf("W: disaggregated '%s' (k = %s), aggregated '%s' (k = %s); tie-break '%s'",
                     W_SPEC$disaggregated, ctx$cfg$weights[[W_SPEC$disaggregated]]$k,
                     W_SPEC$aggregated, ctx$cfg$weights[[W_SPEC$aggregated]]$k,
                     ctx$cfg$weights$identical_points$handling))
log_msg(ctx, sprintf("LM decision rules: alpha = %.2f, both robust significant -> '%s'",
                     ALPHA, BOTH_ROB))

zh <- ctx$cfg$transforms$zero_handling
lg <- function(x) if (zh == "log_plus_c") log(x + ctx$cfg$transforms$log_plus_c) else log(x)
add_logs <- function(D) {
  for (v in c("land_value", MEASURES, "plot_area")) D[[paste0("ln_", v)]] <- lg(D[[v]])
  D
}
canon <- function(L, level) {
  map <- ctx$cfg$measures$columns[[level]]
  for (cn in names(map)) {
    srcn <- map[[cn]]
    if (!is.null(L[[srcn]]) && srcn != cn) L[[cn]] <- L[[srcn]]
  }
  L
}

# =============================================================================
# 1. Load both levels, both sources  (src/48_rebuilt_dv.R section 1)
# =============================================================================

load_disaggregated <- function() {
  L <- st_read(cfg_path(ctx, "frozen", "cents_disaggregated"), quiet = TRUE)
  if (is.na(st_crs(L)$epsg) || st_crs(L)$epsg != ctx$cfg$repro$crs_epsg)
    L <- st_transform(L, ctx$cfg$repro$crs_epsg)
  if (isTRUE(st_is_longlat(L))) stop("layer is in geographic coordinates", call. = FALSE)

  p <- file.path(REPO, DV$rebuilt_path)
  if (!file.exists(p))
    stop("stage-1 output missing: ", p, " -- run `make dv` first", call. = FALSE)
  RB <- read.csv(p, stringsAsFactors = FALSE)
  if (nrow(RB) != nrow(L) || !all(RB$id == L$id))
    stop(sprintf("%s is not id-aligned with the frozen layer (%d vs %d rows)",
                 DV$rebuilt_path, nrow(RB), nrow(L)), call. = FALSE)
  if (max(abs(RB$Preco_frozen - L$Preco), na.rm = TRUE) > 1e-6)
    stop("the stage-1 file's `Preco_frozen` does not match the layer's `Preco`",
         call. = FALSE)
  log_msg(ctx, sprintf("disaggregated: %d records, id-aligned; rebuilt vs frozen `Preco` max |delta| = %.4g",
                       nrow(L), max(abs(RB$Preco - L$Preco), na.rm = TRUE)))

  mk <- function(src) {
    Z <- L
    if (src == "rebuilt") { Z$Preco <- RB$Preco; Z$Puni <- RB$Puni }
    Z <- canon(Z, "disaggregated")
    log_msg(ctx, sprintf("-- funnel [disaggregated / %s]", src))
    E <- apply_exclusions(ctx, Z, "disaggregated")
    list(source = src, level = "disaggregated",
         D = add_logs(st_drop_geometry(E)), coords = st_coordinates(E)[, 1:2])
  }
  setNames(lapply(SOURCES, mk), SOURCES)
}

load_aggregated <- function() {
  srcs <- ctx$cfg$revision$maup$sources
  keys <- vapply(srcs, function(s) as.character(s$key), character(1))
  rec  <- srcs[[match("500m_headline", keys)]]
  if (is.null(rec)) stop("revision.maup.sources has no '500m_headline' record",
                         call. = FALSE)

  FZ <- st_read(cfg_path(ctx, "frozen", "cents_aggregated"), quiet = TRUE)
  RB <- st_read(file.path(REPO, rec$path), quiet = TRUE)
  for (nm in c("FZ", "RB")) {
    A <- get(nm)
    if (is.na(st_crs(A)$epsg) || st_crs(A)$epsg != ctx$cfg$repro$crs_epsg)
      assign(nm, st_transform(A, ctx$cfg$repro$crs_epsg))
  }
  if (nrow(FZ) != nrow(RB) || !all(FZ$id == RB$cell_id))
    stop("the rebuilt 500 m layer is not cell-aligned with the frozen one",
         call. = FALSE)
  RB$id <- RB$cell_id                       # the tie-break key at this level
  log_msg(ctx, sprintf("aggregated: %d cells, id-aligned; rebuilt vs frozen `Puni_mean` max |delta| = %.4g",
                       nrow(FZ), max(abs(RB$Puni_mean - RB$Puni_mean_frozen), na.rm = TRUE)))

  mk <- function(src) {
    Z <- canon(if (src == "frozen") FZ else RB, "aggregated")
    log_msg(ctx, sprintf("-- funnel [aggregated / %s]", src))
    E <- apply_exclusions(ctx, Z, "aggregated")
    list(source = src, level = "aggregated",
         D = add_logs(st_drop_geometry(E)),
         coords = suppressWarnings(
           st_coordinates(st_point_on_surface(st_geometry(E)))[, 1:2]))
  }
  setNames(lapply(SOURCES, mk), SOURCES)
}

DATA <- list(disaggregated = load_disaggregated(), aggregated = load_aggregated())
for (lv in LEVELS) for (src in SOURCES)
  log_msg(ctx, sprintf("%s / %-7s : N = %d", lv, src, nrow(DATA[[lv]][[src]]$D)))

# =============================================================================
# 2. The battery
# =============================================================================

Ws <- list()
for (lv in LEVELS) for (src in SOURCES)
  Ws[[paste(lv, src)]] <- suppressWarnings(
    build_weights(ctx, DATA[[lv]][[src]]$coords, spec_name = W_SPEC[[lv]]))

t0   <- Sys.time()
rows <- list()
for (lv in LEVELS) for (src in SOURCES) for (m in MEASURES) {
  d  <- DATA[[lv]][[src]]
  lt <- run_lm_tests(frm(m), d$D, Ws[[paste(lv, src)]])
  sel <- lm_rule(lt, ALPHA, BOTH_ROB)
  if (is.null(lt)) {
    log_msg(ctx, sprintf("!! LM suite failed: %s / %s / %s", lv, src, m))
    lt <- setNames(rep(NA_real_, 8), c("lm_err", "lm_lag", "rlm_err", "rlm_lag",
                                       "lm_err_p", "lm_lag_p", "rlm_err_p", "rlm_lag_p"))
  }
  rows[[length(rows) + 1]] <- data.frame(
    level = lv, source = src, w_spec = W_SPEC[[lv]],
    k = ctx$cfg$weights[[W_SPEC[[lv]]]]$k,
    n = nrow(d$D), measure = m,
    lm_err = lt[["lm_err"]], lm_err_p = lt[["lm_err_p"]],
    lm_lag = lt[["lm_lag"]], lm_lag_p = lt[["lm_lag_p"]],
    rlm_err = lt[["rlm_err"]], rlm_err_p = lt[["rlm_err_p"]],
    rlm_lag = lt[["rlm_lag"]], rlm_lag_p = lt[["rlm_lag_p"]],
    selected = sel[["model"]], basis = sel[["basis"]],
    stringsAsFactors = FALSE)
}
lm_res <- do.call(rbind, rows)
log_msg(ctx, sprintf("%d LM suites in %.1f s", nrow(lm_res),
                     as.numeric(difftime(Sys.time(), t0, units = "secs"))))
for (i in seq_len(nrow(lm_res)))
  log_msg(ctx, sprintf("  %-13s %-7s %-4s  LM-err %10.2f (p %.3g)  LM-lag %10.2f (p %.3g)  rLM-err %8.2f (p %.3g)  rLM-lag %8.2f (p %.3g)  -> %s",
    lm_res$level[i], lm_res$source[i], lm_res$measure[i],
    lm_res$lm_err[i], lm_res$lm_err_p[i], lm_res$lm_lag[i], lm_res$lm_lag_p[i],
    lm_res$rlm_err[i], lm_res$rlm_err_p[i], lm_res$rlm_lag[i], lm_res$rlm_lag_p[i],
    toupper(format(lm_res$selected[i]))))

sel_tab <- table(lm_res$source, lm_res$selected)
log_msg(ctx, "selection by source: ",
        paste(apply(expand.grid(s = rownames(sel_tab), m = colnames(sel_tab)), 1,
                    function(r) sprintf("%s %s x%d", r[["s"]], toupper(r[["m"]]),
                                        sel_tab[r[["s"]], r[["m"]]])),
              collapse = "; "))
n_not_sarar <- sum(!lm_res$selected %in% "sarar")
if (n_not_sarar > 0)
  log_msg(ctx, sprintf("!! the decision rules do NOT select SARAR in %d of %d cells",
                       n_not_sarar, nrow(lm_res)))

# --- rebuilt against frozen, cell by cell ------------------------------------
cmp_rows <- list()
for (lv in LEVELS) for (m in MEASURES) {
  a <- lm_res[lm_res$level == lv & lm_res$source == "frozen"  & lm_res$measure == m, ]
  b <- lm_res[lm_res$level == lv & lm_res$source == "rebuilt" & lm_res$measure == m, ]
  if (!nrow(a) || !nrow(b)) next
  st <- c("lm_err", "lm_lag", "rlm_err", "rlm_lag")
  r <- data.frame(level = lv, measure = m, w_spec = a$w_spec, k = a$k,
                  n_frozen = a$n, n_rebuilt = b$n, stringsAsFactors = FALSE)
  for (s in st) {
    r[[paste0(s, "_frozen")]]  <- a[[s]]
    r[[paste0(s, "_rebuilt")]] <- b[[s]]
    r[[paste0("rel_d_", s)]]   <- (b[[s]] - a[[s]]) / abs(a[[s]])
  }
  r$selected_frozen  <- a$selected
  r$selected_rebuilt <- b$selected
  r$selection_change <- if (identical(a$selected, b$selected)) "no" else "yes"
  cmp_rows[[length(cmp_rows) + 1]] <- r
}
cmp <- do.call(rbind, cmp_rows)
log_msg(ctx, sprintf("frozen vs rebuilt: %d of %d cells change the selected model; largest relative move in any statistic %.2f %%",
                     sum(cmp$selection_change == "yes"), nrow(cmp),
                     100 * max(abs(unlist(cmp[grep("^rel_d_", names(cmp))])), na.rm = TRUE)))

# --- reproduction check: the rows on the sweep's source must equal its rows ---
# `knn_k6` and `knn_k8` are the same matrices as the proposed specs, so the
# arm on the sweep's dependent variable is the same suite on the same sample
# under the same W and must land on the same numbers.
sw <- file.path(REPO, ctx$cfg$paths$outputs, "tables",
                "stage4_w_choice_model_selection.csv")
repro_done <- FALSE
repro_note <- character(0)
if (file.exists(sw)) {
  S <- read.csv(sw, stringsAsFactors = FALSE)
  ref <- list(disaggregated = "knn_k8", aggregated = "knn_k6")
  for (lv in LEVELS) {
    A <- lm_res[lm_res$level == lv & lm_res$source == DV$proposed_source, ]
    B <- S[S$level == lv & S$w_spec == ref[[lv]], ]
    if (!nrow(A) || nrow(B) != length(MEASURES)) next
    B <- B[match(A$measure, B$measure), ]
    d <- max(abs(c(A$lm_err - B$lm_err, A$lm_lag - B$lm_lag,
                   A$rlm_err - B$rlm_err, A$rlm_lag - B$rlm_lag)))
    same_sel <- identical(A$selected, B$selected)
    repro_note <- c(repro_note,
      sprintf("%s vs `%s` (N = %d): max |delta| = %.3g%s, selection %s",
              lv, ref[[lv]], B$n[1], d, if (d == 0) " (bit-identical)" else "",
              if (same_sel) "identical" else "DIFFERENT"))
    log_msg(ctx, "reproduction check, ", DV$proposed_source, " source ",
            repro_note[length(repro_note)])
  }
  repro_done <- length(repro_note) > 0
}
if (!repro_done)
  log_msg(ctx, "!! stage4_w_choice_model_selection.csv not found or incomplete -- reproduction check not run (run `make w-choice` first)")

# =============================================================================
# 3. Write
# =============================================================================

p <- out_table(ctx, "table5_lm_rebuilt.csv")
write.csv(lm_res, p, row.names = FALSE)
log_msg(ctx, "wrote ", p)
p2 <- out_table(ctx, "table5_lm_rebuilt_comparison.csv")
write.csv(cmp, p2, row.names = FALSE)
log_msg(ctx, "wrote ", p2)

# --- the report --------------------------------------------------------------
# Table 5's convention for a statistic cell: one decimal at 10 or more, three
# significant figures below, `***` for p < 0.001, `**` for p < 0.01, `*` for
# p < 0.05.
fmt_stat <- function(x) ifelse(is.na(x), "",
  ifelse(abs(x) >= 10, formatC(x, format = "f", digits = 1),
         formatC(signif(x, 3), format = "g", digits = 3)))
stars <- function(p) ifelse(is.na(p), "",
  ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", ""))))
cell <- function(x, p) paste0(fmt_stat(x), stars(p))
fmt_p <- function(p) ifelse(is.na(p), "",
  ifelse(p < 0.001, "<0.001", formatC(p, format = "f", digits = 3)))

ROWNAMES <- c(lm_err = "LM-error", lm_lag = "LM-lag",
              rlm_err = "robust LM-error", rlm_lag = "robust LM-lag")
block <- function(lv, src, with_p = FALSE) {
  s <- lm_res[lm_res$level == lv & lm_res$source == src, ]
  s <- s[match(c("CC", "PC1", "PC2", "BC", "FK"), s$measure), ]
  out <- c("| | **CC** | **PC1** | **PC2** | **BC** | **FK** |",
           "|---|---|---|---|---|---|")
  for (st in names(ROWNAMES))
    out <- c(out, sprintf("| %s | %s |", ROWNAMES[[st]],
                          paste(cell(s[[st]], s[[paste0(st, "_p")]]), collapse = " | ")))
  if (with_p)
    for (st in names(ROWNAMES))
      out <- c(out, sprintf("| *p*, %s | %s |", ROWNAMES[[st]],
                            paste(fmt_p(s[[paste0(st, "_p")]]), collapse = " | ")))
  out <- c(out, sprintf("| selected model | %s |",
                        paste(toupper(format(s$selected)), collapse = " | ")))
  out <- c(out, sprintf("| N | %s |", paste(format(s$n, big.mark = ","), collapse = " | ")))
  out
}

all_sarar <- all(lm_res$selected %in% "sarar")
# cells where all four statistics are significant at ALPHA
all4 <- with(lm_res, lm_err_p < ALPHA & lm_lag_p < ALPHA &
                     rlm_err_p < ALPHA & rlm_lag_p < ALPHA)
n_all4 <- sum(all4, na.rm = TRUE)
n_dis <- function(src) lm_res$n[lm_res$level == "disaggregated" & lm_res$source == src][1]
n_agg <- function(src) lm_res$n[lm_res$level == "aggregated" & lm_res$source == src][1]
same_agg_n <- identical(n_agg("frozen"), n_agg("rebuilt"))
md <- character(0)
say <- function(...) md <<- c(md, ...)

say("# Table 5 — the LM battery on the rebuilt dependent variable", "",
  sprintf("Generated by `src/52_table5_lm.R` (`make table5-lm`). Seed %s; CRS EPSG:%s.",
          ctx$cfg$repro$seed, ctx$cfg$repro$crs_epsg),
  "",
  sprintf("This run computes the four LM rows of Table 5 (both blocks) on the sample Table 5's coefficients are estimated on: the **rebuilt** dependent variable, N = %s at the address level (N = %s on the frozen variable). The W sweep's LM suite (`outputs/tables/stage4_w_choice_model_selection.csv`) runs on the `%s` variable and serves as the cross-check below.",
          format(n_dis("rebuilt"), big.mark = ","), format(n_dis("frozen"), big.mark = ","),
          DV$proposed_source),
  "", "---", "", "## 1. Settings", "",
  "| setting | value | config |", "|---|---|---|",
  sprintf("| dependent variable | `%s`, compared against `frozen` | `dependent_variable.proposed_source` |",
          DV$proposed_source),
  sprintf("| W, aggregated | `%s`, kNN k = %s, style `%s` | `weights.proposed.aggregated` |",
          W_SPEC$aggregated, ctx$cfg$weights[[W_SPEC$aggregated]]$k,
          ctx$cfg$weights[[W_SPEC$aggregated]]$style),
  sprintf("| W, disaggregated | `%s`, kNN k = %s, style `%s` | `weights.proposed.disaggregated` |",
          W_SPEC$disaggregated, ctx$cfg$weights[[W_SPEC$disaggregated]]$k,
          ctx$cfg$weights[[W_SPEC$disaggregated]]$style),
  sprintf("| coordinate ties | `%s`, key `[%s]` | `weights.identical_points` |",
          ctx$cfg$weights$identical_points$handling,
          paste(unlist(ctx$cfg$weights$identical_points$tie_break_key$disaggregated),
                collapse = ", ")),
  sprintf("| exclusions (address level) | %s, in that order | `exclusions.order` |",
          paste(unlist(ctx$cfg$exclusions$order$disaggregated), collapse = " → ")),
  sprintf("| specification | `ln_land_value ~ %sln_<measure>`, five measures | `models` |", CTRL),
  sprintf("| test suite | `spdep::%s`, `RSerr` / `RSlag` / `adjRSerr` / `adjRSlag` | `run_lm_tests()`, `src/common.R` |",
          if ("lm.RStests" %in% getNamespaceExports("spdep")) "lm.RStests" else "lm.LMtests"),
  sprintf("| decision rules | Anselin, α = %.2f; both robust significant → **%s** | `revision.lm_selection` |",
          ALPHA, toupper(as.character(BOTH_ROB))),
  sprintf("| N | %s address level (frozen %s), %s cell level (frozen %s) |  |",
          format(n_dis("rebuilt"), big.mark = ","), format(n_dis("frozen"), big.mark = ","),
          format(n_agg("rebuilt"), big.mark = ","), format(n_agg("frozen"), big.mark = ",")),
  "",
  "The LM suite is an OLS diagnostic: no estimator runs in this script. Statistics are printed as in Table 5 — one decimal at 10 or more, three significant figures below, `***` for *p* < 0.001, `**` for *p* < 0.01, `*` for *p* < 0.05.",
  "", "---", "", "## 2. The two blocks, rebuilt dependent variable", "",
  sprintf("### 2.1 Disaggregated (%s, k = %s), N = %s", W_SPEC$disaggregated,
          ctx$cfg$weights[[W_SPEC$disaggregated]]$k,
          format(n_dis("rebuilt"), big.mark = ",")),
  "", block("disaggregated", "rebuilt", with_p = TRUE), "",
  sprintf("### 2.2 Aggregated (%s, k = %s), N = %s", W_SPEC$aggregated,
          ctx$cfg$weights[[W_SPEC$aggregated]]$k,
          format(n_agg("rebuilt"), big.mark = ",")),
  "", block("aggregated", "rebuilt", with_p = TRUE), "",
  "---", "", "## 3. The same battery on the frozen dependent variable", "",
  sprintf("Same W, same specification, same exclusion order; only the dependent variable changes. At the cell level the analysis sample is %s under both sources (N = %s frozen, %s rebuilt). At the address level N is %s frozen and %s rebuilt: the %s×IQR fence is computed on the dependent variable itself, so the sample moves with it.",
          if (same_agg_n) "the same size" else "of different size",
          format(n_agg("frozen"), big.mark = ","), format(n_agg("rebuilt"), big.mark = ","),
          format(n_dis("frozen"), big.mark = ","), format(n_dis("rebuilt"), big.mark = ","),
          format(ctx$cfg$exclusions$iqr_outliers$multiplier)),
  "", "### 3.1 Disaggregated, frozen", "", block("disaggregated", "frozen"), "",
  "### 3.2 Aggregated, frozen", "", block("aggregated", "frozen"), "",
  "### 3.3 Cell by cell", "",
  "| level | measure | LM-error F → R | LM-lag F → R | robust LM-error F → R | robust LM-lag F → R | selected F → R |",
  "|---|---|---|---|---|---|---|")
for (i in seq_len(nrow(cmp)))
  say(sprintf("| %s | %s | %s → %s | %s → %s | %s → %s | %s → %s | %s |",
      cmp$level[i], cmp$measure[i],
      fmt_stat(cmp$lm_err_frozen[i]), fmt_stat(cmp$lm_err_rebuilt[i]),
      fmt_stat(cmp$lm_lag_frozen[i]), fmt_stat(cmp$lm_lag_rebuilt[i]),
      fmt_stat(cmp$rlm_err_frozen[i]), fmt_stat(cmp$rlm_err_rebuilt[i]),
      fmt_stat(cmp$rlm_lag_frozen[i]), fmt_stat(cmp$rlm_lag_rebuilt[i]),
      sprintf("%s → %s", toupper(format(cmp$selected_frozen[i])),
              toupper(format(cmp$selected_rebuilt[i])))))
RD  <- as.matrix(cmp[grep("^rel_d_", names(cmp))])
hit <- which(abs(RD) == max(abs(RD), na.rm = TRUE), arr.ind = TRUE)[1, ]
say("",
  sprintf("Largest relative movement in any of the %d statistics: **%.2f %%** (%s %s, %s). Cells where the swap changes the selected model: **%d of %d**.",
    length(RD), 100 * max(abs(RD), na.rm = TRUE), cmp$level[hit[["row"]]], cmp$measure[hit[["row"]]],
    ROWNAMES[[sub("^rel_d_", "", colnames(RD)[hit[["col"]]])]],
    sum(cmp$selection_change == "yes"), nrow(cmp)),
  "",
  if (repro_done)
    sprintf("**Reproduction check.** The `%s` rows recomputed against the W sweep's rows under the same matrices (`knn_k8`, `knn_k6`): %s.",
            DV$proposed_source, paste(repro_note, collapse = "; "))
  else
    "**Reproduction check not run**: `outputs/tables/stage4_w_choice_model_selection.csv` was not found. Run `make w-choice` before `make table5-lm` to compare these rows with the W sweep's.",
  "", "---", "", "## 4. What the decision rules select", "",
  sprintf("Anselin's rules applied per cell at α = %.2f: neither LM significant → OLS; exactly one → that model; both → decide on the robust pair; both robust significant → `%s`.",
          ALPHA, BOTH_ROB),
  "",
  if (all_sarar)
    sprintf("**SARAR is selected in all %d cells** (both sources, both levels).", nrow(lm_res))
  else
    sprintf("**SARAR is not selected in every cell.** %d of the %d cells select a different model (%s).",
            n_not_sarar, nrow(lm_res),
            paste(sprintf("%s/%s/%s → %s", lm_res$level[!lm_res$selected %in% "sarar"],
                          lm_res$source[!lm_res$selected %in% "sarar"],
                          lm_res$measure[!lm_res$selected %in% "sarar"],
                          toupper(format(lm_res$selected[!lm_res$selected %in% "sarar"]))),
                  collapse = "; ")),
  "",
  sprintf("Rebuilt arm alone: SARAR in %d of %d cells. All four statistics are significant at α = %.2f in %d of the %d cells.",
          sum(lm_res$source == "rebuilt" & lm_res$selected %in% "sarar"),
          sum(lm_res$source == "rebuilt"), ALPHA, n_all4, nrow(lm_res)),
  "",
  if (n_all4 > 0 && identical(BOTH_ROB, "sarar"))
    sprintf("In the %d cell(s) where both robust statistics are significant the rule reaches its last branch, where the configured choice (`both_robust_significant: %s`) applies: the tests indicate that both a lag and an error term are needed, and fitting them together (rather than picking the larger robust statistic) is the configured convention.",
            n_all4, BOTH_ROB) else NULL,
  "", "---", "", "## 5. Outputs", "",
  "| file | what |", "|---|---|",
  sprintf("| `outputs/tables/table5_lm_rebuilt.csv` | the %d suites: four statistics, four p-values, selected model and basis |",
          nrow(lm_res)),
  "| `outputs/tables/table5_lm_rebuilt_comparison.csv` | frozen against rebuilt, per (level, measure) |",
  "| `outputs/tables/table5_lm_counts.csv` | the exclusion funnel of this run |",
  "")

pr <- out_report(ctx, "table5_lm_rebuilt.md")
writeLines(md, pr)
log_msg(ctx, "wrote ", pr)

write_counts(ctx, "table5_lm_counts.csv")
log_msg(ctx, strrep("=", 74))
log_msg(ctx, sprintf("SARAR selected in %d of %d cells (%d of %d in the rebuilt arm)",
                     sum(lm_res$selected %in% "sarar"), nrow(lm_res),
                     sum(lm_res$source == "rebuilt" & lm_res$selected %in% "sarar"),
                     sum(lm_res$source == "rebuilt")))
finish(ctx)
