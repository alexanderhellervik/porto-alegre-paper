# =============================================================================
# Table 2 of the article: Pearson correlations between the five centrality
# measures (natural logs) on the address-level analysis sample.
#
# The sample is the 7,775 records of the rebuilt dependent variable after the
# exclusions of config.yaml (`apply_exclusions`); the measures are the frozen
# ones attached to the point layer. Only the measures enter the correlations,
# so the version of the dependent variable matters only through the sample.
#
# Run:  make table2   (Rscript src/53_table2_correlations.R)
# Writes: outputs/tables/stage4_table2_correlations.csv  (long form: a, b, pearson, n)
#         outputs/tables/stage4_table2_matrix.csv        (the 5 x 5 matrix)
# =============================================================================
suppressPackageStartupMessages({ library(sf) })
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])),
                 "common.R"))
ctx <- init(4, "table2_correlations")
capture_env_r(ctx)

L <- proposed_dv_layer(ctx, "disaggregated")
L <- apply_exclusions(ctx, L, "disaggregated")
log_msg(ctx, sprintf("N = %d on the rebuilt dependent variable (submitted sample %d)",
                     nrow(L), ctx$cfg$exclusions$targets$disaggregated_n))

MEAS <- c("CC", "PC1", "PC2", "BC", "FK")
map <- ctx$cfg$measures$columns$disaggregated
M <- sapply(MEAS, function(k) log(as.numeric(L[[map[[k]]]])))
if (any(!is.finite(M))) stop("non-finite log measure in the analysis sample", call. = FALSE)
R <- cor(M, method = "pearson")
pv <- outer(seq_along(MEAS), seq_along(MEAS), Vectorize(function(i, j)
  if (i == j) 0 else cor.test(M[, i], M[, j])$p.value))

long <- do.call(rbind, lapply(seq_along(MEAS), function(i) do.call(rbind, lapply(seq_along(MEAS), function(j)
  if (j < i) data.frame(a = MEAS[i], b = MEAS[j], pearson = R[i, j], p = pv[i, j], n = nrow(L),
                        stringsAsFactors = FALSE) else NULL))))
p1 <- out_table(ctx, "stage4_table2_correlations.csv"); write.csv(long, p1, row.names = FALSE)
log_msg(ctx, "wrote ", p1)
mat <- data.frame(measure = MEAS, round(R, 6), check.names = FALSE)
p2 <- out_table(ctx, "stage4_table2_matrix.csv"); write.csv(mat, p2, row.names = FALSE)
log_msg(ctx, "wrote ", p2)
log_msg(ctx, sprintf("max p-value among the ten pairs: %.3g", max(long$p)))
write_counts(ctx, "stage4_table2_counts.csv")
finish(ctx)
