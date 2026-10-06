# Reproduction package — Land value vs. spatial centrality (Porto Alegre)
#
# Requires GNU make (3.81 or later) and bash. `make help` lists the targets
# with their measured runtimes; README.md maps each manuscript table and
# figure to the target that makes it and the file it lands in.
#
# `make all` runs every step that needs no manual intervention, in dependency
# order: the dependent-variable rebuild, the stage-2 layers, the validation
# gate on the submitted tables, every stage-4 analysis on the 500 m grid and
# the address points, MAUP, the PC parameter grid, and the network and PC runs
# for the boundary variants. It stops before the GAUS Lines runs on the
# boundary variants (hours each, single core) and prints the remaining steps:
#
#   make gaus-native FILE=data/interim/boundary_<v>_edges.gpkg \
#        LAYER=boundary_<v>_edges LABEL=boundary_<v>   # v = bn2000 bn4000 mu0 mu1
#   make boundary-layers
#   make boundary-muni        # also rewrites the negative-buffer report (42)
#
# Stage numbers in the help text (0-4) follow the pipeline: 0 inputs and
# centrality runs, 1 dependent variable, 2 spatial layers, 3 validation gate,
# 4 revision analyses.

SHELL := /bin/bash

# ---------------------------------------------------------------------------
# Interpreters. `PY` must be a Python 3.12 with the packages in
# requirements.txt, and without qgis -- see `gaus-native` below. The runs of
# record used a virtualenv that was first on PATH, so a bare `python`
# resolved to it; if yours is elsewhere, override on the command line or in
# the environment:
#     make dv PY=/path/to/venv/bin/python
#
# `R` must be Rscript for R >= 4.3 (the runs of record used 4.6.x,
# env/sessionInfo.txt) with the packages installed by env/install_r_packages.R.
# This Makefile does not set R_LIBS_USER, so R's own default library applies.
# Override the same way:  make gate-agg R=/path/to/Rscript
# ---------------------------------------------------------------------------
PY ?= python
R  ?= Rscript

# compute.max_workers, read from config.yaml rather than from `nproc`, which
# inside a container reports the host and not the CPU quota. It caps the
# worker pools and the BLAS/OpenMP threads underneath R and numpy. Raise it in
# config.yaml, not here.
WORKERS := $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('config.yaml'))['compute']['max_workers'])" 2>/dev/null)
NEED_WORKERS = @test -n "$(WORKERS)" || { \
  echo "could not read compute.max_workers from config.yaml with PY=$(PY)"; \
  echo "(is PyYAML installed in that interpreter? see requirements.txt)"; exit 2; }
THREADS = OMP_NUM_THREADS=$(WORKERS) OPENBLAS_NUM_THREADS=$(WORKERS) MKL_NUM_THREADS=$(WORKERS)

# GAUS Lines v1.1 is not redistributed (its upstream repository has no licence
# file); `make gaus-fetch` clones it. Evaluated only by the GAUS targets.
gaus_cfg = $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('config.yaml'))['inputs']['gaus_headless']['$(1)'] or '')")
GAUS_REPO   = $(call gaus_cfg,upstream)
GAUS_PIN    = $(call gaus_cfg,upstream_commit)
GAUS_SHA256 = $(call gaus_cfg,script_sha256)
GAUS_DIR    = $(call gaus_cfg,local_clone)
GAUS_SCRIPT = $(call gaus_cfg,script_source)

.DEFAULT_GOAL := help
.PHONY: help all inventory profile prefcent-bundle pc-runs pc-beta-layers \
        pc-beta rebuild-areas pc-accept gaus-fetch gaus-native gaus-accept \
        boundary-variants boundary-pc boundary-layers boundary \
        boundary-muni-variants boundary-muni dv join gate gate-agg gate-dis \
        revision revision-agg maup w-choice w-sweep sample-sens rebuilt-dv \
        controls-types floor-price table2 table3-hierarchy table5-lm env verify \
        lint clean distclean

help:  ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-22s\033[0m %s\n",$$1,$$2}'
	@echo ""
	@echo "  compute.max_workers: $(if $(WORKERS),$(WORKERS),(unreadable -- check PY))"

## --- the whole chain ---------------------------------------------------------
# `join` brings in `dv`, and the input manifest is built by `inventory`.
# `w-choice` runs before `rebuilt-dv`, `sample-sens` and `table5-lm`, whose
# reproduction checks read its tables; `gate` runs before `maup`, which reads
# the point-level gate results. `pc-runs` needs about 54 GiB of RAM.
# `make all FAST=1` passes FAST=1 to `gate` (see there).
all:  ## every automatic step, in order (~11 h; FAST=1 ~3 h, see `gate`)
	$(MAKE) inventory
	$(MAKE) join
	$(MAKE) table2
	$(MAKE) gate
	$(MAKE) table3-hierarchy
	$(MAKE) revision-agg
	$(MAKE) w-choice
	$(MAKE) rebuilt-dv
	$(MAKE) sample-sens
	$(MAKE) table5-lm
	$(MAKE) controls-types
	$(MAKE) floor-price
	$(MAKE) maup
	$(MAKE) pc-runs
	$(MAKE) pc-beta-layers
	$(MAKE) pc-beta
	$(MAKE) boundary-variants
	$(MAKE) boundary-pc
	$(MAKE) boundary-muni-variants
	@echo ""
	@echo "Automatic steps done. The boundary sensitivity needs GAUS Lines on each"
	@echo "variant (hours each, single core; run detached), then two more targets:"
	@echo "  make gaus-fetch"
	@echo "  make gaus-native FILE=data/interim/boundary_<v>_edges.gpkg \\"
	@echo "       LAYER=boundary_<v>_edges LABEL=boundary_<v>    # v = bn2000 bn4000 mu0 mu1"
	@echo "  make boundary-layers"
	@echo "  make boundary-muni"

## --- stage 0 -----------------------------------------------------------------
inventory:  ## stage 0 — inventory and checksum all inputs (~1 min)
	$(PY) src/00_inventory.py

profile: inventory  ## stage 0 — full data profile; regenerates the purpose mapping (~3 min)
	$(PY) src/00_profile.py

# The prefcent verification run in data/frozen/network -> a segment layer that
# `make pc-accept` can score.
prefcent-bundle:  ## stage 0 — verification run -> segment GeoPackage for pc-accept (~1 min)
	$(PY) src/09_prefcent_bundle_to_segments.py $(ARGS)

# New preferential-centrality runs with the solver of record (prefcent 0.1.0)
# on `inputs.base_network`: one GeoPackage per beta, all gammas. The dense
# 56,162 x 56,162 float64 kernel is 23.5 GiB, so 32 GiB of RAM is the floor
# and the runs are made one beta at a time.
#   make pc-runs ARGS="--only b2"   /   ARGS="--list"
pc-runs:  ## stage 0 — the PC (gamma, beta) grid with the solver of record (~5 min per beta at 16 workers; peak 53.5 GiB)
	$(PY) src/11_pc_runs.py $(ARGS)

pc-beta-layers:  ## stage 2 — the registered PC runs onto the 500 m cells and the points (~1 min)
	$(PY) src/21_pc_beta_layers.py $(ARGS)

# Rebuild the static weight R from the network alone and score it against the
# shipped `area` column, forward and in reversed segment order.
rebuild-areas:  ## stage 0 — rebuild R from the network, score against the shipped `area` (~8 min)
	$(PY) src/12_rebuild_areas.py $(ARGS)

# Acceptance test for a new PC run. Exit 0 = PASS / PASS WITH WARNING,
# 1 = FAIL. A file is registered in revision.pc_sensitivity.input_files or
# revision.boundary_sensitivity.input_files only after this test.
pc-accept:  ## acceptance-test a PC run: make pc-accept FILE=... [ARGS=...] (~4 s)
	@test -n "$(FILE)" || { echo "usage: make pc-accept FILE=<candidate> [ARGS=...]"; exit 2; }
	$(PY) src/06_pc_acceptance.py "$(FILE)" $(ARGS)

## --- GAUS Lines v1.1 (CC, BC, FK) --------------------------------------------
# GAUS Lines v1.1 (Dalcin and Krafta 2021). Its upstream repository has no
# licence file, so this package fetches it instead of redistributing it:
# `gaus-fetch` clones the repository, checks out
# `inputs.gaus_headless.upstream_commit` and verifies the script against
# `inputs.gaus_headless.script_sha256`.
gaus-fetch:  ## fetch GAUS Lines v1.1 into tools/GAUS at the pinned commit and verify it
	@test -n "$(GAUS_PIN)" || { echo "inputs.gaus_headless.upstream_commit is not set"; exit 2; }
	@if [ -d "$(GAUS_DIR)/.git" ]; then \
	  echo "updating $(GAUS_DIR)"; git -C "$(GAUS_DIR)" fetch --quiet origin; \
	else \
	  echo "cloning $(GAUS_REPO) -> $(GAUS_DIR)"; \
	  git clone --quiet "$(GAUS_REPO)" "$(GAUS_DIR)"; \
	fi
	git -C "$(GAUS_DIR)" checkout --quiet "$(GAUS_PIN)"
	@test -f "$(GAUS_SCRIPT)" || { \
	  echo "$(GAUS_SCRIPT) not found at commit $(GAUS_PIN)."; exit 2; }
	@echo "$(GAUS_SHA256)  $(GAUS_SCRIPT)" | sha256sum -c - || { \
	  echo "the fetched script does not match inputs.gaus_headless.script_sha256"; exit 2; }
	@echo "GAUS ready: $(GAUS_SCRIPT)"

# Runs GAUS Lines v1.1 itself, headless under QGIS 3.34 -- the source of CC,
# BC and FK at any extent.
#
# Two interpreters are involved. `qgis.core` lives in the system Python; the
# analysis environment ($(PY)) has geopandas and no qgis. The driver runs in
# the analysis environment and calls `qgis_process` with that environment
# removed from the child's PATH (`inputs.gaus_headless.path_for_qgis`): the
# embedded interpreter's sys.prefix follows PATH, and with a virtualenv first
# it cannot import qgis and reports the algorithm as unknown. Keep qgis out of
# the analysis environment. The paths in `inputs.gaus_headless` are those of a
# Debian/Ubuntu QGIS package; adjust them for another installation.
#
# The plugin writes its columns into its input layer, so the driver always
# runs it on a copy under inputs.gaus_headless.work_dir.
#
#   make gaus-native                                  # the reference network
#   make gaus-native FILE=data/interim/boundary_mu0_edges.gpkg \
#        LAYER=boundary_mu0_edges LABEL=boundary_mu0
gaus-native: ## stage 0 — GAUS Lines v1.1 under QGIS: single core, ~13 h on the full network, hours per boundary variant
	@test -f "$(GAUS_SCRIPT)" || { \
	  echo "GAUS Lines v1.1 is missing: $(GAUS_SCRIPT)"; \
	  echo "It is not redistributed (no upstream licence). Run: make gaus-fetch"; exit 2; }
	$(PY) src/13_gaus_headless.py $(if $(FILE),--input "$(FILE)") \
	  $(if $(LAYER),--layer "$(LAYER)") $(if $(LABEL),--label "$(LABEL)") $(ARGS)

gaus-accept:  ## acceptance-test a GAUS run: make gaus-accept FILE=... [ARGS=...] (~3 s)
	@test -n "$(FILE)" || { echo "usage: make gaus-accept FILE=<candidate> [ARGS=...]"; exit 2; }
	$(PY) src/08_gaus_acceptance.py "$(FILE)" $(ARGS)

## --- boundary-extent sensitivity ---------------------------------------------
# Two families of variants. Negative buffers of the network's own footprint
# (bn2000, bn4000) and the municipal polygon with and without a 1 km collar
# (mu0, mu1). The order is:
#   make boundary-variants boundary-pc          # bn*: networks, R, PC runs
#   make boundary-muni-variants                 # mu*: networks, R
#   make gaus-native ... for each of bn2000 bn4000 mu0 mu1 (see above)
#   make boundary-layers                        # bn* onto cells and points
#   make boundary-muni                          # mu*: PC runs, tests, layers,
#                                               # models, both reports
boundary-variants:  ## stage 0 — negative-buffer extents, clipped networks, rebuilt R (~10 min)
	$(PY) src/14_boundary_variants.py $(ARGS)

boundary-pc:  ## stage 0 — the solver of record on each negative-buffer variant (~20 min)
	$(PY) src/15_boundary_pc_runs.py $(ARGS)

boundary-layers:  ## stage 2 — negative-buffer variants onto the 500 m cells and the address points (~2 min)
	$(PY) src/23_boundary_layers.py $(ARGS)

boundary:  ## stage 4 — negative-buffer boundary sensitivity, a stress test not reported in the article (~3 min)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/42_boundary_sensitivity.R

boundary-muni-variants:  ## stage 0 — municipal extents (IBGE 2025), clipped networks, rebuilt R (~30 min)
	$(PY) src/16_boundary_muni_variants.py $(ARGS)

# Needs `boundary-muni-variants`, GAUS-native on data/interim/boundary_mu{0,1}_edges.gpkg
# and `boundary-layers`. The last step rewrites the negative-buffer report so
# that it includes the municipal section.
boundary-muni:  ## stage 0-4 — municipal variants: PC runs, acceptance tests, layers, models (~30 min)
	$(NEED_WORKERS)
	$(PY) src/15_boundary_pc_runs.py --family muni --n-jobs $(WORKERS)
	$(PY) src/06_pc_acceptance.py data/interim/pc_boundary_mu0.gpkg \
	  --informational --label boundary_mu0
	$(PY) src/06_pc_acceptance.py data/interim/pc_boundary_mu1.gpkg \
	  --informational --label boundary_mu1
	$(PY) src/08_gaus_acceptance.py data/interim/gaus_headless_boundary_mu0.gpkg \
	  --informational --label boundary_mu0
	$(PY) src/08_gaus_acceptance.py data/interim/gaus_headless_boundary_mu1.gpkg \
	  --informational --label boundary_mu1
	$(PY) src/23_boundary_layers.py --family muni
	$(THREADS) $(R) src/43_boundary_muni.R
	$(THREADS) $(R) src/42_boundary_sensitivity.R

## --- stages 1-2 --------------------------------------------------------------
# On a fresh clone `dv` builds the input manifest itself if `inventory` has
# not run.
dv:  ## stage 1 — rebuild the dependent variable from the raw CSVs (~14 s)
	$(PY) src/10_dependent_variable.py

join: dv  ## stage 2 — attach centralities, aggregate to hexagons (~2 s on top of dv)
	$(PY) src/20_join_aggregate.py

## --- stage 3: the validation gate --------------------------------------------
# Reproduces the submitted Tables 3-5 from the frozen layers. At the cell level
# the 6,100 m band reproduces the submitted Table 5; at the point level the
# paper's stated k = 300 does not, and the gate scores the 7,450 m band that
# does (config.yaml, `weights`).
# The point-level gate fits a SARAR under the 7,450 m distance band, whose
# ~4,700 neighbours per point make each fit take one to two hours: about 9 h
# in all. FAST=1 skips those SARAR fits (weights.gate_slow_specs) and still
# scores the band's LM statistics: `make gate FAST=1` takes about an hour.
GATE_FLAGS := $(if $(filter 1,$(FAST)),--fast,)

gate:  ## stage 3 — validation gate, both levels (~9 h; FAST=1 ~1 h)
	$(R) src/30_validation_gate.R both $(GATE_FLAGS)

gate-agg:  ## stage 3 — validation gate, cell level only (~30 s)
	$(R) src/30_validation_gate.R aggregated

gate-dis:  ## stage 3 — validation gate, point level, three W specs on 7,767 points (~9 h; FAST=1 ~1 h)
	$(R) src/30_validation_gate.R disaggregated $(GATE_FLAGS)

# The stage-4 targets below read the rebuilt dependent variable and the 500 m
# layer built by `join`; on a fresh clone this file rule runs `join` first.
STAGE2_LAYER := data/interim/aggregated_500m_headline.gpkg
$(STAGE2_LAYER):
	$(MAKE) join


## --- stage 4: the revision ---------------------------------------------------
revision: revision-agg  ## stage 4 — alias of revision-agg

revision-agg:  ## stage 4 — cell-level revision analyses: LM selection, W sweep, block CV (~4 min)
	$(R) src/40_revision.R aggregated

maup: $(STAGE2_LAYER)  ## stage 4 — MAUP at 250 m / 500 m / 1,000 m; Tables 4, 6, S3, S12 (~40 min)
	$(R) src/45_maup.R

w-choice: $(STAGE2_LAYER)  ## stage 4 — W-choice sweep + block CV, both levels; Tables S1, S2, S8 (~13 min)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/40_revision.R --mode both both

w-sweep: $(STAGE2_LAYER)  ## stage 4 — the W-choice sweep only, both levels (~2.5 min)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/40_revision.R --mode w_sweep both

sample-sens: $(STAGE2_LAYER)  ## stage 4 — analysis-sample sensitivity, Table S5 (~1 min)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/47_sample_sensitivity.R

table2: $(STAGE2_LAYER)  ## stage 4 — Table 2: Pearson correlations between the log measures, analysis sample (~10 s)
	$(THREADS) $(R) src/53_table2_correlations.R

rebuilt-dv: $(STAGE2_LAYER)  ## stage 4 — Tables 3, 4 and 5 on the rebuilt dependent variable, frozen alongside (~2 min)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/48_rebuilt_dv.R

pc-beta: $(STAGE2_LAYER)  ## stage 4 — PC (gamma, beta) sensitivity, cell level, Table S4 (~2 min)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/41_pc_beta_sensitivity.R

# Log built area and log unit count as controls, and the property-type split.
# The Python step builds the two controls per address from the registry
# through stage 1's own address key; the R step estimates.
controls-types: $(STAGE2_LAYER)  ## stage 4 — extra controls + property-type split, Tables S6, S7 (~2 min)
	$(NEED_WORKERS)
	$(PY) src/49_built_area.py
	$(THREADS) PIPELINE_MAX_WORKERS=$(WORKERS) $(R) src/49_controls_types.R

# The floor-price / density decomposition and the property-type floor-price
# maps. Reuses the controls table of `controls-types`.
floor-price: $(STAGE2_LAYER)  ## stage 4 — floor price vs density: Figure 5, Tables S9, S10 (~4 min)
	$(NEED_WORKERS)
	@test -f data/interim/controls_built_area.csv || $(PY) src/49_built_area.py
	$(THREADS) PIPELINE_MAX_WORKERS=$(WORKERS) $(R) src/50_floor_price.R

table3-hierarchy:  ## stage 4 — Hierarchy check on the submitted 7,767 sample (printed Table 3 comes from rebuilt-dv) (seconds)
	$(PY) src/51_table3_hierarchy.py

# The four LM rows of Table 5 on the rebuilt dependent variable under the
# revised matrices, with the frozen-variable rows beside them. An OLS
# diagnostic: no spatial model is estimated.
table5-lm: $(STAGE2_LAYER)  ## stage 4 — Table 5's LM rows on the rebuilt DV, k = 6 / k = 8 (seconds)
	$(NEED_WORKERS)
	$(THREADS) $(R) src/52_table5_lm.R

## --- housekeeping ------------------------------------------------------------
# Every run records its environment under logs/env/. `make env` records it
# afresh and copies it into env/, the tracked record of the runs of record.
env:  ## record sessionInfo(), R packages and pip freeze, and copy them into env/
	$(R) -e 'source("src/common.R"); ctx <- init(0, "env"); capture_env_r(ctx)'
	$(PY) -c "import sys;sys.path.insert(0,'src');import common;common.capture_env(common.init(0,'env'))"
	cp logs/env/* env/

verify:  ## check data/ and env/wheels against the tracked data/SHA256SUMS
	$(PY) src/00_inventory.py --verify

lint:  ## ruff check and ruff format --check on src/
	ruff check src/ && ruff format --check src/

# Removes everything `make` produces. data/interim holds the GAUS Lines and PC
# outputs, which take hours to days to regenerate; `clean` asks before
# deleting them unless FORCE=1. The tracked purpose mapping
# (outputs/tables/stage0b_purpose_mapping.csv) is kept.
clean:  ## remove derived outputs (asks before deleting data/interim; FORCE=1 skips)
	@if [ "$(FORCE)" != 1 ] && [ -n "$$(ls -A data/interim 2>/dev/null | grep -v '^.gitkeep$$')" ]; then \
	  read -r -p "delete data/interim (GAUS and PC runs included)? [y/N] " a; \
	  [ "$$a" = y ] || { echo "kept data/interim"; exit 1; }; \
	fi
	@mkdir -p data/interim outputs/tables outputs/figures logs reports
	find data/interim -mindepth 1 ! -name .gitkeep -delete
	find outputs/tables outputs/figures -mindepth 1 ! -name .gitkeep \
	  ! -name stage0b_purpose_mapping.csv -delete
	rm -rf logs/* reports/*
	@touch data/interim/.gitkeep outputs/tables/.gitkeep logs/.gitkeep reports/.gitkeep

distclean: clean  ## also remove the fetched GAUS clone
	rm -rf "$(GAUS_DIR)"
