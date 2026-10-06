#!/usr/bin/env Rscript
# Stage 3 -- validation gate.
#
# Re-estimates the submitted results from the frozen inputs and scores every
# figure against the targets in config.yaml (`validation:`). Estimation is GMM
# SARAR via sphet::spreg, through fit_spreg() in common.R, as in the submitted
# analysis.
#
# Scored per level: N, Table 3 descriptives, all ten Table 4 correlations, and
# for every model in Table 5 the four LM statistics, the Area and centrality
# coefficients and the pseudo-R2. The LM statistics depend on W but not on the
# GMM fit, so they identify the W that produced the submitted table.
#
# Every W spec in weights.gate_specs[<level>] is run; only the spec named in
# weights.active[<level>] counts toward the gate tally. At the cell level the
# active W is the 6,100 m distance band; at the address level the paper's
# stated kNN k = 300 does not reproduce the submitted LM statistics, the
# 7,450 m distance band does, and that band is what the gate scores. An error
# parameter rho on sphet's nlminb bound (+-0.9) is flagged: such a fit is not
# an interior estimate.
#
# A target whose figure could not be computed (a failed LM suite or SARAR fit)
# is counted as a miss, so the tally always has the same denominator.
#
# Output, per level run: outputs/tables/stage3_validation_results_<level>.csv
# and stage3_counts_<level>.csv.
#
# Runtime and memory: the 7,450 m band on the 7,767 address points averages
# about 4,700 neighbours per point (~36 million non-zero weights). Its LM suite
# takes a few minutes per measure, but each SARAR fit takes one to two hours
# on a single core and several GiB of RAM, so the address level takes about
# 9 hours in all. The cell level takes under a minute.
#
# Fast mode (`--fast`): the SARAR fits under the W specs listed in
# `weights.gate_slow_specs` are skipped. Their LM statistics are still
# computed and scored; the skipped SARAR figures are reported as not run and
# left out of the tally (the log says how many), rather than counted as misses.
#
# Run:  Rscript src/30_validation_gate.R [disaggregated|aggregated|both] [--fast]

suppressPackageStartupMessages({
  library(sf); library(spdep); library(sphet)
})

source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

args  <- commandArgs(trailingOnly = TRUE)
FAST  <- "--fast" %in% args
args  <- setdiff(args, "--fast")
which_level <- if (length(args)) args[1] else "both"
levels_to_run <- if (which_level == "both")
  c("aggregated", "disaggregated") else which_level
if (!all(levels_to_run %in% c("aggregated", "disaggregated")))
  stop("level must be one of: aggregated | disaggregated | both", call. = FALSE)

ctx <- init(3, "validation_gate")
capture_env_r(ctx)
SLOW <- if (FAST) unlist(ctx$cfg$weights$gate_slow_specs) else character(0)
if (FAST)
  log_msg(ctx, "fast mode: SARAR fits skipped under ", paste(SLOW, collapse = ", "))

MEASURES <- c("PC1", "PC2", "CC", "BC", "FK")
results  <- list()   # tidy long: one row per level x w_spec x model x statistic

# ---- helpers ----------------------------------------------------------------

# Table 3 descriptives. `pareto` (top-20 % sum / bottom-80 % sum) is reported
# unscored: Table 3's Hierarchy column is recomputed by
# src/51_table3_hierarchy.py.
descriptives <- function(x) {
  c(relative_range = (max(x) - min(x)) / mean(x),
    cv = sd(x) / mean(x) * 100,
    pareto = {
      s <- sort(x, decreasing = TRUE)
      top <- ceiling(0.2 * length(s))
      sum(s[seq_len(top)]) / sum(s[-seq_len(top)])
    })
}

add <- function(level, w_spec, model, measure, stat, value, target = NA_real_) {
  results[[length(results) + 1]] <<- data.frame(
    level = level, w_spec = w_spec, model = model, measure = measure,
    statistic = stat, value = as.numeric(value), target = as.numeric(target),
    stringsAsFactors = FALSE)
}

target_of <- function(block, level, m) {
  b <- ctx$cfg$validation[[block]][[level]]
  if (is.null(b) || is.null(b[[m]])) NA_real_ else as.numeric(b[[m]])
}

fmt_cmp <- function(label, v, t, digits = 4) {
  if (is.na(t)) sprintf("%s=%.*f", label, digits, v) else
    sprintf("%s=%.*f (submitted %.*f, d=%+.*f)", label, digits, v, digits, t, digits, v - t)
}

# Every scored figure: (config block, statistic name in the results, model,
# whether it depends on W). The tally is built from this list, so a figure
# that could not be computed still counts -- as a miss.
SCORED <- data.frame(
  block     = c("mean_value", "relative_range", "cv", "correlations",
                "lm_err", "lm_lag", "rlm_err", "rlm_lag",
                "coef_area", "coef_measure", "pseudo_r2"),
  statistic = c("mean", "relative_range", "cv", "pearson_r",
                "lm_err", "lm_lag", "rlm_err", "rlm_lag",
                "coef_area", "coef_measure", "pseudo_r2"),
  model     = c("-", "-", "-", "-", "LM", "LM", "LM", "LM",
                "SARAR", "SARAR", "SARAR"),
  by_w      = c(FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE)

BOUND <- as.numeric(ctx$cfg$weights$spreg_parameter_bound)
TOL   <- ctx$cfg$validation$tolerance

# ---- main loop --------------------------------------------------------------

for (level in levels_to_run) {

  results <- list()
  log_msg(ctx, strrep("-", 70))
  log_msg(ctx, "LEVEL: ", level)

  key  <- if (level == "disaggregated") "cents_disaggregated" else "cents_aggregated"
  path <- cfg_path(ctx, "frozen", key)
  Land <- st_read(path, quiet = TRUE)

  crs_expected <- ctx$cfg$repro$crs_epsg
  if (is.na(st_crs(Land)$epsg) || st_crs(Land)$epsg != crs_expected) {
    log_msg(ctx, sprintf("!! CRS is %s, expected EPSG:%s -- transforming",
                         st_crs(Land)$epsg, crs_expected))
    Land <- st_transform(Land, crs_expected)
  }
  if (isTRUE(st_is_longlat(Land)))
    stop("layer is in geographic coordinates; kNN would be wrong", call. = FALSE)

  # Rename config-mapped columns to canonical names.
  map <- ctx$cfg$measures$columns[[level]]
  for (canon in names(map)) {
    src <- map[[canon]]
    if (!is.null(Land[[src]]) && src != canon) Land[[canon]] <- Land[[src]]
  }

  # --- exclusions, in the config-declared order -----------------------------
  ctx$counts <- ctx$counts[0, ]
  Land <- apply_exclusions(ctx, Land, level)

  target_n <- ctx$cfg$exclusions$targets[[paste0(level, "_n")]]
  got_n <- nrow(Land)
  add(level, "-", "-", "-", "N", got_n, target_n)
  log_msg(ctx, sprintf("N = %d  (target %d) %s", got_n, target_n,
                       if (got_n == target_n) "MATCH" else "*** MISMATCH ***"))

  # --- log transforms -------------------------------------------------------
  zh <- ctx$cfg$transforms$zero_handling
  lg <- function(x) if (zh == "log_plus_c")
    log(x + ctx$cfg$transforms$log_plus_c) else log(x)
  for (v in c("land_value", MEASURES, "plot_area"))
    Land[[paste0("ln_", v)]] <- lg(Land[[v]])

  # --- descriptive statistics ----------------------------------------------
  for (m in MEASURES) {
    d <- descriptives(Land[[m]])
    add(level, "-", "-", m, "mean",           mean(Land[[m]]),     target_of("mean_value", level, m))
    add(level, "-", "-", m, "relative_range", d["relative_range"], target_of("relative_range", level, m))
    add(level, "-", "-", m, "cv",             d["cv"],             target_of("cv", level, m))
    add(level, "-", "-", m, "pareto_index",   d["pareto"])
  }

  # --- correlations (Table 4, all five measures) ----------------------------
  log_msg(ctx, "-- Pearson correlations with ln(land value) --")
  for (m in MEASURES) {
    r <- cor(Land$ln_land_value, Land[[paste0("ln_", m)]],
             use = "complete.obs", method = "pearson")
    t <- target_of("correlations", level, m)
    add(level, "-", "-", m, "pearson_r", r, t)
    log_msg(ctx, "  ", m, " ", fmt_cmp("r", r, t))
  }

  # --- spatial weights: every gate spec, active one scored ------------------
  pts <- if (all(st_geometry_type(Land) == "POINT")) Land else
           suppressWarnings(st_point_on_surface(Land))
  coords <- st_coordinates(pts)[, 1:2]
  D <- st_drop_geometry(Land)

  active_spec <- ctx$cfg$weights$active[[level]]
  gate_specs  <- unlist(ctx$cfg$weights$gate_specs[[level]])
  if (!active_spec %in% gate_specs)
    stop(sprintf("weights.active.%s = '%s' is not listed in weights.gate_specs.%s",
                 level, active_spec, level), call. = FALSE)
  ctrl <- if (isTRUE(ctx$cfg$models$controls_all_models)) "ln_plot_area + " else ""

  for (ws in gate_specs) {
    log_msg(ctx, strrep("=", 70))
    log_msg(ctx, sprintf("W spec '%s'%s", ws,
                         if (ws == active_spec) "  [ACTIVE - scored]" else "  [comparison]"))
    W <- build_weights(ctx, coords, spec_name = ws)

    for (m in MEASURES) {
      form <- as.formula(sprintf("ln_land_value ~ %sln_%s", ctrl, m))
      log_msg(ctx, sprintf("-- %s : %s", m, deparse(form)))

      if (isTRUE(ctx$cfg$models$lm_tests)) {
        lt <- run_lm_tests(form, D, W)
        if (is.null(lt)) {
          log_msg(ctx, "   LM suite FAILED")
        } else {
          for (nm in c("lm_err", "lm_lag", "rlm_err", "rlm_lag")) {
            t <- target_of(nm, level, m)
            add(level, ws, "LM", m, nm, lt[[nm]], t)
            add(level, ws, "LM", m, paste0(nm, "_p"), lt[[paste0(nm, "_p")]])
          }
          log_msg(ctx, "   ", fmt_cmp("LMerr", lt[["lm_err"]], target_of("lm_err", level, m), 1),
                  "  ", fmt_cmp("RLMlag", lt[["rlm_lag"]], target_of("rlm_lag", level, m), 2))
        }
      }

      if (ws %in% SLOW) {
        log_msg(ctx, "   SARAR not run (fast mode)")
        next
      }
      fit <- fit_spreg(ctx, form, D, W, "sarar")
      if (is.null(fit)) next

      z  <- sarar_row(fit)
      co <- z$co; se <- z$se
      for (nm in names(co)) {
        add(level, ws, "SARAR", m, paste0("coef_", nm), co[[nm]])
        add(level, ws, "SARAR", m, paste0("se_", nm), se[[nm]])
      }
      add(level, ws, "SARAR", m, "coef_area", co[["ln_plot_area"]], target_of("coef_area", level, m))
      add(level, ws, "SARAR", m, "coef_measure", co[[paste0("ln_", m)]], target_of("coef_measure", level, m))

      # Only the error parameter rho is bounded (nlminb, +-0.9).
      at_bound <- abs(abs(co[["rho"]]) - BOUND) < 1e-6
      add(level, ws, "SARAR", m, "error_at_bound", as.numeric(at_bound))

      r2 <- pseudo_r2(fit, D$ln_land_value)
      add(level, ws, "SARAR", m, "pseudo_r2", r2, target_of("pseudo_r2", level, m))

      log_msg(ctx, "   ", fmt_cmp("Area", co[["ln_plot_area"]], target_of("coef_area", level, m)),
              "  ", fmt_cmp("beta", co[[paste0("ln_", m)]], target_of("coef_measure", level, m)))
      log_msg(ctx, sprintf("   lag(lambda)=%.4f  error(rho)=%.4f%s  ",
                           co[["lambda"]], co[["rho"]],
                           if (at_bound) "  *** rho AT nlminb BOUND -- not an interior estimate ***" else ""),
              fmt_cmp("pseudo-R2", r2, target_of("pseudo_r2", level, m)))
    }
  }

  # --- assemble and score this level ----------------------------------------
  res <- do.call(rbind, results)

  # Add a row with value NA for every configured target that was not computed
  # (a failed fit or LM suite), so it is scored as a miss.
  miss_rows <- list()
  n_not_run <- 0L
  for (i in seq_len(nrow(SCORED))) {
    sc_ <- SCORED[i, ]
    for (m in MEASURES) {
      t <- target_of(sc_$block, level, m)
      if (is.na(t)) next
      wsp <- if (sc_$by_w) active_spec else "-"
      if (sc_$model == "SARAR" && wsp %in% SLOW) {
        n_not_run <- n_not_run + 1L
        next
      }
      have <- any(res$w_spec == wsp & res$measure == m & res$statistic == sc_$statistic &
                  !is.na(res$value))
      if (!have)
        miss_rows[[length(miss_rows) + 1]] <- data.frame(
          level = level, w_spec = wsp, model = sc_$model, measure = m,
          statistic = sc_$statistic, value = NA_real_, target = t,
          stringsAsFactors = FALSE)
    }
  }
  if (length(miss_rows)) {
    log_msg(ctx, sprintf("!! %d scored figure(s) could not be computed; counted as misses",
                         length(miss_rows)))
    res <- rbind(res, do.call(rbind, miss_rows))
  }

  res$delta      <- res$value - res$target
  res$within_tol <- ifelse(is.na(res$target), NA,
                           !is.na(res$delta) &
                           abs(res$delta) <= pmax(as.numeric(TOL$abs),
                                                  abs(res$target) * as.numeric(TOL$rel)))
  res$scored <- !is.na(res$target) & (res$w_spec == "-" | res$w_spec == active_spec)

  p <- out_table(ctx, sprintf("stage3_validation_results_%s.csv", level))
  write.csv(res, p, row.names = FALSE)
  log_msg(ctx, "wrote ", p)
  write_counts(ctx, sprintf("stage3_counts_%s.csv", level))

  sc <- res[res$scored, ]
  log_msg(ctx, strrep("=", 70))
  log_msg(ctx, sprintf("GATE %-13s (W = %s): %d of %d scored figures within tolerance%s",
                       level, active_spec, sum(sc$within_tol), nrow(sc),
                       if (n_not_run) sprintf(" (fast mode: %d SARAR figures not run)",
                                              n_not_run) else ""))
  miss <- sc[!sc$within_tol, ]
  if (nrow(miss)) {
    log_msg(ctx, "  misses:")
    for (i in seq_len(nrow(miss)))
      log_msg(ctx, sprintf("    %-4s %-14s %12.4f  target %12.4f  delta %+10.4f",
                           miss$measure[i], miss$statistic[i], miss$value[i],
                           miss$target[i], miss$delta[i]))
  }
}

finish(ctx)
