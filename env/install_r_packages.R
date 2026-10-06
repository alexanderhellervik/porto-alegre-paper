# Install the R packages the scripts in src/ load.
#
#   Rscript env/install_r_packages.R            # install what is missing
#   PIN_VERSIONS=1 Rscript env/install_r_packages.R
#                                               # install the recorded versions
#
# The versions are those of the runs of record (env/r_packages.csv,
# env/sessionInfo.txt, R 4.6.1). By default missing packages are installed
# from the configured CRAN mirror at whatever version it serves, and any
# difference from the recorded version is reported. With PIN_VERSIONS=1 every
# package whose installed version differs is reinstalled at the recorded
# version with remotes::install_version() (from source; `sf` needs the GDAL,
# GEOS and PROJ development headers).
#
# Packages go to the first entry of .libPaths(), normally R_LIBS_USER.

recorded <- c(
  sf         = "1.1-2",    # spatial data (GDAL/GEOS/PROJ bindings)
  spdep      = "1.4-2",    # neighbour lists, weights, LM tests
  sphet      = "2.1-1",    # GMM SARAR estimation (spreg)
  spatialreg = "1.4-3",    # ML spatial models, impacts
  yaml       = "2.3.12",   # config.yaml
  jsonlite   = "2.0.0",    # run manifests
  digest     = "0.6.39",   # input checksums in src/45_maup.R
  mgcv       = "1.9-4",    # smooth surface in src/50_floor_price.R
  ggplot2    = "4.0.3",    # figures in src/50_floor_price.R
  cowplot    = "1.2.0",
  scales     = "1.4.0"
)

if (is.null(getOption("repos")) || identical(getOption("repos")[["CRAN"]], "@CRAN@"))
  options(repos = c(CRAN = "https://cloud.r-project.org"))

installed_version <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) return(NA_character_)
  as.character(utils::packageVersion(p))
}
# packageVersion() normalises "1.4-2" to "1.4.2"; compare on that form.
norm <- function(v) as.character(numeric_version(gsub("-", ".", v)))

pin <- identical(Sys.getenv("PIN_VERSIONS"), "1")
missing <- names(recorded)[is.na(vapply(names(recorded), installed_version, ""))]

if (pin) {
  if (!requireNamespace("remotes", quietly = TRUE)) utils::install.packages("remotes")
  for (p in names(recorded)) {
    have <- installed_version(p)
    if (is.na(have) || norm(have) != norm(recorded[[p]])) {
      message(sprintf("installing %s %s", p, recorded[[p]]))
      remotes::install_version(p, version = recorded[[p]], upgrade = "never")
    }
  }
} else if (length(missing)) {
  message("installing: ", paste(missing, collapse = ", "))
  utils::install.packages(missing)
}

have <- vapply(names(recorded), installed_version, "")
status <- data.frame(
  package  = names(recorded),
  recorded = unname(recorded),
  installed = unname(have),
  match    = !is.na(have) & norm(ifelse(is.na(have), "0", have)) == norm(unname(recorded)),
  stringsAsFactors = FALSE
)
print(status, row.names = FALSE)
if (any(is.na(have))) {
  stop("not installed: ", paste(status$package[is.na(have)], collapse = ", "),
       call. = FALSE)
}
if (!all(status$match))
  message("Some versions differ from the runs of record; results may differ in ",
          "the last digits. PIN_VERSIONS=1 installs the recorded versions.")
if (requireNamespace("sf", quietly = TRUE)) print(sf::sf_extSoftVersion())
