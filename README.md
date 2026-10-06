# Land value and spatial centrality in Porto Alegre — reproduction package

Code, data and configuration to reproduce every table of the revised
manuscript and its Online Resource, and Figure 5, of *Comparing the performance of spatial centrality measures
through predicting urban land values: Evidence from Porto Alegre, Brazil*,
together with a validation gate that re-estimates the tables of the
**submitted** (first-round) version from the same inputs and scores them
against the printed values.

---

## 1. Quick start

Requirements: Linux or macOS, GNU make and bash, Python 3.12, R ≥ 4.3 (the
runs of record used R 4.6.1), and the GDAL, GEOS and PROJ libraries with their
development headers (needed to build the R package `sf`; on Debian/Ubuntu:
`libgdal-dev libgeos-dev libproj-dev libudunits2-dev`). QGIS is needed only for
the GAUS Lines runs (§6).

```bash
# Python environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# R packages (installs whatever is missing; PIN_VERSIONS=1 installs the
# recorded versions instead)
Rscript env/install_r_packages.R

# check the shipped data against data/SHA256SUMS
make verify

# a first result in under a minute: the validation gate at the cell level
make gate-agg

make help       # every target, with its stage and measured runtime
```

The Makefile calls `python` and `Rscript` by default. With the virtual
environment activated nothing else is needed; otherwise pass the interpreters
explicitly, e.g. `make dv PY=/path/to/venv/bin/python R=/path/to/Rscript`.

### What `make all` does, and what it does not

`make all` runs every step that needs no manual intervention, in dependency
order: the input inventory, the dependent-variable rebuild, the spatial layers,
the validation gate, every revision analysis on the 500 m grid and the address
points, the MAUP analysis, the preferential-centrality parameter grid, and the
network and preferential-centrality runs for the boundary variants. On the
machine of record (16 cores, 96 GiB) this takes about 11 hours, of which about
9 are the point-level validation gate (see below); `make pc-runs` alone needs
about 54 GiB of RAM.

**A faster run.** `make all FAST=1` (or `make gate FAST=1`) skips the one slow
step: the SARAR fits of the point-level gate under the 7,450 m distance band,
whose ~4,700 neighbours per point make each fit take one to two hours on a
single core. The band's LM statistics, which are what identify it as the
matrix behind the submitted Table 5, are still computed and scored; the
skipped SARAR figures are reported as not run and left out of the tally. With
`FAST=1`, `make all` takes about 3 hours. The switch changes nothing else:
every other table, including every table of the revised manuscript, is
identical with or without it.

It does **not** run GAUS Lines. The frozen CC, BC and FK columns are inputs
and are never recomputed; GAUS Lines is needed only for the boundary-extent
sensitivity (§5.3 of the manuscript), where each clipped network takes hours
on a single core. After `make all` finishes it prints the remaining steps:

```bash
make gaus-fetch
make gaus-native FILE=data/interim/boundary_<v>_edges.gpkg \
     LAYER=boundary_<v>_edges LABEL=boundary_<v>     # v = bn2000 bn4000 mu0 mu1
make boundary-layers
make boundary-muni
```

`make all` also does not run the checks that confirm the frozen inputs
themselves (`make profile`, `make rebuild-areas`, `make prefcent-bundle`,
`make pc-accept`, `make gaus-native` on the full network, ~13 h); run them
individually if you want to re-derive those inputs.

---

## 2. What is where

```
config.yaml            every analytical choice (conventions, weights, filters, tolerances)
Makefile               one target per step, with its stage and runtime
requirements.txt       Python packages at the versions of the runs of record
src/                   the pipeline: Python for data preparation and geometry,
                       R for estimation
data/raw/              the ITBI transactions and the unit registry
data/frozen/           the layers the submitted analysis ran on: address points,
                       500 m grid, 65,357 street segments, base network,
                       IBGE municipal polygon
data/SHA256SUMS        checksums of everything in data/raw, data/frozen and env/wheels
data/interim/          intermediate files and per-run manifests (regenerable, not tracked)
outputs/tables/        every table, as CSV (regenerable, not tracked, except the
                       purpose mapping stage0b_purpose_mapping.csv, an input to stage 1)
outputs/figures/       Figure 5 and its single panels (regenerable, not tracked)
reports/               generated Markdown reports (regenerable, not tracked)
logs/                  one log per run, and the environment of each run in logs/env/
env/                   the environment of the runs of record, the R installer and
                       the prefcent 0.1.0 wheel
tools/GAUS/            GAUS Lines v1.1, fetched by `make gaus-fetch`, not redistributed (§6)
```

See `data/README.md` for the contents, provenance and terms of every input.

---

## 3. Manuscript item → make target → output file

Runtimes were measured on the machine of record (16 cores, 96 GiB) with
`compute.max_workers` at 16. All paths are under `outputs/tables/` unless
stated otherwise.

### Main text

| manuscript item | `make` target | output file(s) | runtime |
|---|---|---|---|
| **Table 3**, unit values and the five measures: mean, median, relative range, Distribution (CV) and Hierarchy (the Pareto ratio of §4.4), analysis sample N = 7,775 | `rebuilt-dv` (submitted values: `gate`) | `stage4_rebuilt_dv_table3.csv` (rows `level = disaggregated`, `measure` × `statistic`; the printed values are column `rebuilt`, unit values in BRL per m²; submitted: `stage3_validation_results_{aggregated,disaggregated}.csv`) | 2 min |
| Hierarchy check on the submitted 7,767-record sample, beside the submitted column (not a printed table) | `table3-hierarchy` | `table3_hierarchy_faria.csv` | seconds |
| **Table 2**, correlations between the log measures, analysis sample | `table2` | `stage4_table2_correlations.csv`, `stage4_table2_matrix.csv` | 10 s |
| **Table 4**, address points and 500 m: Pearson (logs) and Spearman | `rebuilt-dv` (submitted values: `gate`) | `stage4_rebuilt_dv_table4.csv` (Spearman in `spearman_rebuilt`, `spearman_frozen`) | 2 min |
| **Table 4**, 250 m and 1,000 m | `maup` | `stage4_maup_correlations.csv` | 40 min |
| **Table 5**, coefficients, pseudo-R² and block-CV RMSE, both levels | `rebuilt-dv` | `stage4_rebuilt_dv_fits.csv` (rows `source = rebuilt`), `stage4_rebuilt_dv_cv_rmse.csv` (SARAR-coefficient trend, the primary row = column `rmse_sarar_trend_recal`; OLS trend, the check = `rmse_ols`; benchmarks in rows `null_training_block_mean` and `null_plot_area_only`). The manuscript's ρ (lag, Wy) and λ (error, Wu) are the CSV columns `lambda_lag` and `rho_error`: sphet names them the other way round | 2 min |
| **Table 5**, the four LM rows | `table5-lm` | `table5_lm_rebuilt.csv` (rows `source = rebuilt`) | seconds |
| **Figure 5**, floor price (US$ per m² of floor area) and development density, four 500 m hexagon panels (§5.2) | `floor-price` | `outputs/figures/floor_price/floor_price_density_panels.png` and `.pdf` (vector); single panels in the same directory | 4 min |
| **§5.3**, additional controls and property type | `controls-types` | see Tables S6–S7 below | 2 min |
| **§5.3**, preferential-centrality parameters | `pc-runs` → `pc-beta-layers` → `pc-beta` | see Table S4 below | ~5 min per β + 1 min + 2 min |
| **§5.3** and **Table S11**, extent of the network: the municipality and the municipality plus 1 km | `boundary-muni-variants` → `gaus-native` (mu0, mu1) → `boundary-muni` | `stage4_boundary_muni_{aggregated,measure_ranking,ranking_agreement,disaggregated_correlations}.csv`, `reports/stage4_boundary_muni.md` | ~30 min + ~30 min + GAUS Lines (hours per variant) |
| Negative-buffer extents (a stress test, not reported in the article) | `boundary-variants` → `boundary-pc` → `gaus-native` (bn2000, bn4000) → `boundary-layers` → `boundary` | `stage4_boundary_{aggregated,measure_ranking,ranking_agreement,disaggregated_correlations}.csv`, `reports/stage4_boundary_sensitivity.md` | ~35 min + GAUS Lines (hours per variant) |
| **Table 6**, MAUP: 250 m and 500 m under kNN k = 6 (§5.4) | `maup` | `stage4_maup_headline.csv` (benchmarks in `rmse_null_mean`, `rmse_null_plot_area_only`), `stage4_maup_sign_table.csv`, `stage4_maup_cv_rmse.csv` | 40 min |

`boundary-muni` needs the negative-buffer layers too, because its last step
rewrites the negative-buffer report with the municipal section; the full order
is `boundary-variants`, `boundary-pc`, `boundary-muni-variants`, `gaus-native`
for bn2000, bn4000, mu0 and mu1, `boundary-layers`, `boundary-muni`.

Table 1 (the definitions of the measures) and Figures 1–4 are not produced by
this package; Figures 1–4 are as in the first submission.

### Online Resource

| Online Resource item | `make` target | output file(s) | runtime |
|---|---|---|---|
| **Section S1**, network measures and aggregation: graph, cost rule, settings, grids and cell assignment | `gaus-fetch` → `gaus-native` (→ `gaus-accept`); grids and cell counts: `join`, `maup` | `data/interim/gaus_headless_<label>.gpkg` and its manifest; `reports/gaus_acceptance/`; `stage2_segment_multiplicity.csv`, `stage4_maup_n_cells.csv` | ~13 h on the full network |
| **Section S2**, preferential centrality: network, static weights, iteration | `rebuild-areas`, `prefcent-bundle` → `pc-accept`, `pc-runs` | `data/interim/pc_run_<label>.gpkg`; `reports/pc_acceptance/`; the recipe in `data/frozen/network/README.md` | 8 min; 1 min; ~5 min per β |
| **Section S3**, the dependent variable: formula, conventions, reproduction | `dv` | `data/interim/dependent_variable_headline.csv`, `stage1_*.csv`, `reports/stage1_dependent_variable.md` | 14 s |
| **Section S4**, notes to Table 5 and the weight-matrix sweep | `w-choice`, `rebuilt-dv` | `stage4_w_choice_neighbours.csv`, `stage4_w_choice_rule.csv`, `stage4_rebuilt_dv_fits.csv` | 13 min; 2 min |
| **Section S5**, sensitivity analyses (the full text of §5.3) | see Tables S4–S7 and S11 | | |
| **Table S0**, run index | — | not produced (an index of the rows below) | |
| **Table S1**, cell-level W sweep | `w-choice` | `stage4_w_choice_sweep.csv` (rows `level = aggregated`); neighbour geometry in `stage4_w_choice_neighbours.csv` | 13 min |
| **Table S2**, address-level W sweep | `w-choice` | `stage4_w_choice_sweep.csv` (rows `level = disaggregated`) | 13 min |
| §4.5 segment–cell multiplicity (29,871 assignments, 24,720 segments, 4,878 in more than one cell) | `join` | `stage2_segment_multiplicity.csv` | < 1 min |
| §4.6.2 cell adjacency audit (22 isolated cells; 381 of 812 with the sixth neighbour at the lattice spacing) | `w-sweep` | `stage4_w_choice_adjacency.csv` | 3 min |
| Online Resource S3 plot-area agreement (geolayer vs registry, 99.8 %) | `dv` | `stage1_plot_area_agreement.csv` | < 1 min |
| **Table S3**, MAUP, three cell sizes × comparison matrices | `maup` | `stage4_maup_w_sensitivity.csv` (all sizes × matrices), `stage4_maup_w_summary.csv`; the address row is Table 5 (`stage4_rebuilt_dv_fits.csv`, `source = rebuilt`) | 40 min |
| **Table S4**, PC β × γ sensitivity | `pc-runs` → `pc-beta-layers` → `pc-beta` | `stage4_pc_beta_aggregated.csv`, `stage4_pc_beta_rank_stability.csv`, `stage4_pc_beta_convergence.csv` | ~5 min per β + 1 min + 2 min |
| **Table S5**, analysis-sample sensitivity, fourteen address-level variants plus three cell-level tie-break rows (`level = aggregated`), among them the strict parking exclusion (`parking_strict`, both N's: `n` with the IQR fence recomputed, `n_fence_held` with the baseline fence) and the two parking conventions (`parking_in_multiplier_only`, `parking_in_both`) | `sample-sens` | `stage4_sample_sensitivity.csv`, `stage4_sample_sensitivity_detail.csv`, `stage4_sample_sensitivity_counts.csv` | 1 min |
| **Table S6**, additional controls, five measures × five specifications | `controls-types` | `stage4_controls_fits.csv`, `stage4_controls_summary.csv`, `stage4_controls_rank_stability.csv`, `stage4_controls_alternate_definitions.csv` | 2 min |
| **Table S7**, property-type split | `controls-types` | `stage4_types_fits.csv`, `stage4_types_summary.csv`, `stage4_types_neighbours.csv`, `stage4_types_groups.csv` | 2 min |
| **Table S8**, block vs random folds, SARAR-coefficient and OLS trends | `w-choice` | `stage4_w_choice_cv_rmse.csv` (`scheme` = `spatial_block` / `random_kfold`; benchmarks included) | 13 min |
| **Table S9**, decomposition along floor price × density | `floor-price` | `floor_price_channel_fits.csv`, `floor_price_channel_split.csv` | 4 min |
| **Table S10**, spatial pattern of floor price, density and the dependent variable | `floor-price` | `floor_price_spatial_pattern.csv` | 4 min |
| **Table S11**, extent of the network | `boundary-muni` (after `boundary-muni-variants` and `gaus-native` for mu0, mu1) | `stage4_boundary_muni_{aggregated,measure_ranking,ranking_agreement}.csv`, `reports/stage4_boundary_muni.md` | ~30 min + GAUS Lines |
| **Table S12**, block-CV RMSE at 250 m, 500 m and 1,000 m, k = 6 | `maup` | `stage4_maup_cv_rmse.csv` (`scheme = spatial_block`; benchmarks `null_mean`, `null_plot_area_only`; SARAR trend = `rmse_sarar_trend_recal`) | 40 min |

Figures A1–A3 of the Online Resource are unchanged from the first submission
and are not regenerated by this package.

### The validation gate

| item | `make` target | output file(s) | runtime |
|---|---|---|---|
| The submitted Tables 3–5, re-estimated and scored | `gate` (or `gate-agg`, `gate-dis`) | `stage3_validation_results_{aggregated,disaggregated}.csv`, `stage3_counts_{aggregated,disaggregated}.csv`; the tally and the misses in `logs/` | ~9 h (`FAST=1`: ~1 h; `gate-agg` alone: ~30 s) |

Every pipeline script writes a log under `logs/`, and most also write a counts table
(`*_counts.csv`, the number of records in and out of every filter). The
targets `inventory`, `profile`, `dv`, `join`, `pc-accept`, `gaus-accept`,
`pc-beta`, `boundary`, `boundary-muni`, `rebuilt-dv`, `controls-types`,
`floor-price` and `table5-lm` also write a Markdown report under `reports/`;
the others record their results in their CSV files and their log.

---

## 4. The conventions, in one page

Each of these is set in `config.yaml`; the scripts read them from there.

**The dependent variable** is improved-property value per m² of land (the
article's "unit value"; the column `Puni`), in BRL; the article reports levels
in US$ at the frozen layer's rate of 5.3633 BRL per US$ (in logs only the
intercept changes). One formula, address by address:

```
Puni(a) = mean(P(a)) * U(a) / A(a)
```

* `P(a)` — the `Price` of every ITBI transaction at address `a`, after
  byte-identical duplicate rows are dropped, **excluding** parking and garage
  transactions. A **blank** purpose is kept and treated as non-parking.
  *Exception (only-parking fallback):* where every transaction at the address
  is parking, the parking prices are used rather than dropping the address.
* `U(a)` — the number of rows for `a` in the unit registry, deduplicated,
  **excluding** parking and garage units. Address-level rows with no unit
  number ("parent" rows) **are** counted. *Exception (parent-only rule):* an
  address whose registry rows are *all* parent rows is one plot, `U = 1`.
  *Exception:* where the non-parking selection is empty, all rows are used.
* `A(a)` — `Terreno`, the plot area on the georeferenced address layer.
* The statistic is the **arithmetic mean** over transactions — not the median,
  not weighted by floor area. The multiplier is a **unit count**, not floor
  area.
* An address with no usable transaction leaves the sample at the positivity
  filter.

Two alternates are carried through the MAUP analysis and the address-level
sample sensitivity: `parking_in_multiplier_only` and `parking_in_both`.

Every table of the revised manuscript, the Online Resource tables included,
is computed on this rebuilt variable (`dependent_variable.proposed_source`);
the frozen column is used by the validation gate and, for comparison, beside
the rebuilt one by `make rebuilt-dv` and `make table5-lm`.

**Exclusion order, which differs by level.** At the point level: positivity
**then** the 1.5 × IQR fence, the fence computed on the surviving subsample
(9,030 → 8,167 → 7,767 on the frozen dependent variable; 7,775 on the rebuilt
one). Reversing the two steps gives 7,714 on the frozen variable and 7,712 on
the rebuilt one (Online Resource, Table S5). At the cell level: positivity only;
adding the IQR fence there would drop N to 753.

**Tie-break.** 165 of the 7,767 points share coordinates with another point
(53 clusters, the largest holding 24). For such points `knearneigh` has exactly
tied neighbour lists, and its answer depends on row order. Before any
neighbour list is built, the analysis frame is sorted by the layer's own record
id (a radix sort, so the locale's collation cannot affect it). This does not
remove the ties; it makes the way they fall a function of the data rather than
of how the file was written.

**Weights.** Row-standardised (`style = "W"`) throughout; the LM tests and the
models use the same matrix. Coordinates are projected (EPSG:31982, SIRGAS 2000
/ UTM 22S) before any distance is computed.

* `weights.proposed` — the revised manuscript's matrices: kNN **k = 6** at the
  cell level (`proposed_aggregated_k6`; six nearest neighbours approximate the
  first-order neighbourhood of an interior hexagon) and **k = 8** at the point
  level (`proposed_disaggregated_k8`).
* `weights.active` — the matrices that reproduce the submitted tables, which
  the validation gate scores: a 6,100 m distance band at the cell level
  (`band6100_aggregated`) and a 7,450 m distance band at the point level
  (`band7450_disaggregated`). The submitted text states kNN k = 300
  (`stated_knn300`); at the cell level that is not what produced the submitted
  Table 5, and at the point level it does not reproduce the submitted LM
  statistics, whereas the 7,450 m band reproduces all twenty. The gate runs
  `stated_knn300` alongside for comparison.

**Measures.** Closeness (CC), betweenness (BC) and Freeman–Krafta (FK),
computed with GAUS Lines; preferential centrality with agglomeration (PC1,
γ = 1) and without it (PC2, γ = 0), β = 2. Accessibility (CC, PC1, PC2) is
averaged over a cell's segments, intermediation (BC, FK) summed.

**Estimator.** SARAR by generalized spatial two-stage least squares (GS2SLS)
with heteroskedasticity-robust inference, via `sphet::spreg`
(`model = "sarar"`, `het = TRUE`, default instruments). Every model carries `ln_plot_area` as a control. Model choice
per (W, measure) follows Anselin's LM decision rules at α = 0.05: neither test
significant → OLS; one significant → that model; both significant → the robust
pair decides, and when both robust tests are significant the model is SARAR.

**Performance criterion.** Measures are ranked on the **standardized β** (the
covariates are z-standardised, the response stays in logs) and on the
**spatial block-CV RMSE** (10 k-means blocks on the coordinates; a random
k-fold leaks neighbours across folds under this much residual
autocorrelation). Pseudo-R², λ, ρ, the parameter-bound flag and N are reported
but not used to rank. Predictions use the estimated trend only, with W rebuilt
inside each training set: the SARAR-coefficient trend (the primary row; its
intercept recalibrated to the training mean, column `rmse_sarar_trend_recal`)
and the OLS trend (a check, `rmse_ols`). Two benchmarks are scored on the same
blocks: the no-predictor benchmark, the training-fold mean
(`null_training_block_mean`; `null_mean` in the MAUP tables), and OLS on log
plot area alone (`null_plot_area_only`); the gap between a model and the
second is what the centrality measure adds.

**The validation gate.** `make gate` re-estimates the submitted Tables 3–5
from the frozen inputs and scores each figure against the value printed in the
submitted manuscript, within `|Δ| ≤ max(0.005, 0.01·|target|)`. Every W in
`weights.gate_specs` is estimated and written out; only the one in
`weights.active` counts toward the tally, and the `scored` column of
`stage3_validation_results_<level>.csv` marks it. The tally and every miss are
printed to the log; misses are reported, not adjusted for.

The point-level gate is the slowest step of the package: each SARAR fit under
the 7,450 m band takes one to two hours, about 9 hours for the five measures.
`make gate FAST=1` (or `make gate-dis FAST=1`) skips those fits
(`weights.gate_slow_specs`) and scores everything else, including the band's
LM statistics, in about an hour; the log states how many SARAR figures were
not run. `make gate-agg` runs the cell level alone in about 30 seconds.

---

## 5. Environment

The runs of record were made with:

| | version | recorded in |
|---|---|---|
| R | 4.6.1 | `env/sessionInfo.txt` |
| R packages | `sf`, `spdep`, `sphet`, `spatialreg`, `yaml`, `jsonlite`, `digest`, `mgcv`, `ggplot2`, `cowplot`, `scales` | `env/install_r_packages.R`, `env/r_packages.csv` |
| Python | 3.12 | `env/python_version.txt` |
| Python packages | `numpy`, `scipy`, `pandas`, `geopandas`, `shapely`, `pyproj`, `pyogrio`, `PyYAML`, `openpyxl`, `RapidFuzz`, `prefcent` | `requirements.txt`, `env/pip_freeze.txt` |
| system libraries | GDAL 3.8.4, GEOS 3.12.1, PROJ 9.4.0, QGIS 3.34.4 (Ubuntu 24.04) | `env/apt-packages.txt` |
| `prefcent`, the preferential-centrality solver | 0.1.0 | `pip install prefcent==0.1.0`; source <https://github.com/prefcent/prefcent>, tag `v0.1.0`; archived at <https://doi.org/10.5281/zenodo.22643614>; the wheel is also in `env/wheels/` for offline installs |

The public `prefcent` 0.1.0 reproduces the frozen PC1 and PC2 columns exactly
on all 65,357 segments (`make pc-runs ARGS="--only b2"`, then
`make pc-accept FILE=data/interim/pc_run_b2.gpkg`).

Every run records its own environment in `logs/env/` (session information,
`.libPaths()`, the GDAL/GEOS/PROJ versions `sf` links against, installed R
packages, `pip freeze`). `env/` is changed only by `make env`, which copies the
current record there.

`compute.max_workers` in `config.yaml` caps every worker pool and the BLAS and
OpenMP threads underneath R and numpy. It ships as **8**; set it to the number
of CPUs you can use. Inside a container, `nproc` and `free` report the host
rather than the container's quota; read `/sys/fs/cgroup/cpu.max` and
`/sys/fs/cgroup/memory.max` (or your scheduler's limits) instead.

Memory: `make pc-runs` builds a dense 56,162 × 56,162 float64 kernel
(23.5 GiB), peaking near 53.5 GiB with working copies; 32 GiB of RAM is the
minimum for that target, which runs one β at a time. The point-level gate
under the 7,450 m band (about 4,700 neighbours per point) needs several GiB
and about 9 hours on a single core (see `FAST=1` above).
Nothing else needs more than a few GiB.

---

## 6. GAUS

CC, BC and FK are **GAUS Lines v1.1** measures — Graph Analysis of Urban
Systems, the QGIS plugin by Dalcin and Krafta (2021),
<https://github.com/gkdalcin/GAUS>. This package runs the plugin itself,
headless under QGIS, for the reference network and for every boundary variant.

The upstream repository carries no licence file, so the plugin is not
redistributed here. Fetch it:

```bash
make gaus-fetch      # clones https://github.com/gkdalcin/GAUS into tools/GAUS/
make gaus-native     # runs GAUS Lines v1.1 on the reference network
```

`make gaus-fetch` checks out the pinned commit
`488739cecfc8ccf96a379d985af18e0ba7811344` and verifies
`v1.1/GAUS Lines v1.1.py` against the SHA-256 in
`inputs.gaus_headless.script_sha256`. The script is byte-identical at every
upstream commit from `38a76b7` (July 2021) to the pinned one, so it is the
file the frozen CC, BC and FK were computed with. Every run also records the
SHA-256 of the script it executed in
`data/interim/gaus_headless_<label>_manifest.json`. `make gaus-native` stops
with a message if the plugin has not been fetched.

**Two Python interpreters.** `qgis.core` lives in the system Python; the
analysis environment has geopandas and no qgis. `src/13_gaus_headless.py` runs
in the analysis environment and calls `qgis_process` with the analysis
environment removed from the child's `PATH`
(`inputs.gaus_headless.path_for_qgis`): `qgis_process` embeds a Python whose
`sys.prefix` follows `PATH`, and with a virtual environment first it cannot
import `qgis` and reports the algorithm as unknown. For the same reason qgis
should not be installed into the analysis environment. The QGIS paths in
`inputs.gaus_headless` are those of the Debian/Ubuntu package; adjust them for
another installation.

**Files written outside the repository.** To register the plugin with QGIS
processing, `make gaus-native` copies the script into the QGIS user profile
(`inputs.gaus_headless.profile_scripts_dir`, by default
`~/.local/share/QGIS/QGIS3/profiles/default/processing/scripts`).

**The plugin writes into its input layer** (its columns are added directly to
the layer's data source), so the driver always runs it on a copy under
`data/interim/gaus_headless_work/` and never on the files in `data/frozen/`.

GAUS Lines is single-threaded pure Python. It takes about 13 hours on the full
65,357-segment network and hours per boundary variant; run it detached.

---

## 7. Licence and terms

The code — `src/`, `Makefile`, `config.yaml`, `env/install_r_packages.R` and
the scripts in `data/frozen/network/` — is released under the MIT licence;
see `LICENSE`.

The data are covered per source, as set out in `data/README.md`: the ITBI
transactions and the unit registry come from the Porto Alegre municipality;
the municipal polygon is IBGE's, under IBGE's own terms; the street network
and the segment layer derive from OpenStreetMap (© OpenStreetMap contributors)
and are distributed under the Open Database License (ODbL) 1.0; the address
layers and the centrality columns are the authors'.

GAUS Lines v1.1 is not redistributed here and is covered by the terms its
authors set. `prefcent` is distributed under its own licence (MIT) from PyPI.

---

## 8. Citation

If you use this package, please cite the article and the two tools that
compute the centrality measures:

* Lucato de Aguilar, R., and A. Hellervik. Comparing the performance of
  spatial centrality measures through predicting urban land values: Evidence
  from Porto Alegre, Brazil. Under review at *Networks and Spatial Economics*.
* Hellervik, A. (2026). prefcent v0.1.0 [software]. Zenodo.
  <https://doi.org/10.5281/zenodo.22643614>
* Dalcin, G., and R. Krafta (2021). GAUS: Graph Analysis of Urban Systems
  [QGIS plugin]. <https://github.com/gkdalcin/GAUS>
