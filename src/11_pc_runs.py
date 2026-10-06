#!/usr/bin/env python
"""Stage 0 — produce preferential-centrality runs with the solver of record.

    make pc-runs                       # every run in revision.pc_sensitivity.runs
    make pc-runs ARGS="--only b2"      # one beta only
    python src/11_pc_runs.py --list

`inputs.pc_solver` (`prefcent` 0.1.0, Hellervik 2026; version, tag and commit
recorded there and copied into every manifest) is the solver of record, and
`inputs.base_network` is the input of record. This script rebuilds the routing
graph from the base network exactly as `inputs.pc_solver.graph` describes it,
builds the decay kernel, and evolves the model under
`inputs.pc_solver.submitted_run` with only `beta` and `gamma` moved. Everything
else — d0, kappa = 0, a0 = R, omega = 1, the L1 renormalisation and the fixed
200-iteration budget — is held at the submitted run's values, and the script
refuses a config that asks for anything prefcent 0.1.0 would not actually do.
A run produced here therefore differs from the submitted run in its declared
parameters and nothing else.

The graph rule is not restated here. `extract_network_from_shapefile.extract()`
ships in the reconstruction bundle (`inputs.pc_solver.reconstruction_bundle`, a
frozen read-only input) and is imported, not copied: two copies of the
midpoint-split rule would eventually disagree, and the disagreement would look
like a solver difference. Its sha256 goes in the manifest.

Output, one GeoPackage per beta, columns named by the authors' batch-driver
convention (`cd{run}g{gamma}b{beta}k{int(100*Dp)}`, decimal points dropped):

    beta 2.0   cd1g1b2k0 / ca1g1b2k0        gamma 1     the submitted PC1
               cd4g0b2k0 / ca4g0b2k0        gamma 0     the submitted PC2
               cd106g05b2k0 / ca106g05b2k0  gamma 0.5
    beta 1.5   cd101g1b15k0                 gamma 1
               cd102g0b15k0                 gamma 0
               cd105g05b15k0                gamma 0.5
    beta 2.25  cd108g1b225k0                gamma 1
               cd110g0b225k0                gamma 0
               cd109g05b225k0               gamma 0.5
    beta 2.5   cd103g1b25k0                 gamma 1
               cd104g0b25k0                 gamma 0
               cd107g05b25k0                gamma 0.5

One label is one kernel: gamma is free within a label, beta is not.

The beta = 2.0 file carries the submitted run indices because it *is* the
submitted run recomputed here; `make pc-accept` scores it against the submitted
columns and must PASS. The variants take run indices from 101 up, a block the
authors' earlier batches never used, so no variant column name can collide with
a submitted one.

**Memory.** The dense kernel is 56,162 x 56,162 float64 = 23.5 GiB. One beta is
built at a time and freed before the next; the runs of record peaked at about
53.5 GiB, so 64 GiB of RAM is a comfortable minimum. Peak memory is sampled from
the cgroup (Linux cgroup v2) throughout and recorded per run. Worker and thread
counts come from `compute.max_workers`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent

# BLAS reads its thread count at load time, so `compute.max_workers` has to be in
# the environment before numpy is imported. Inside a container `nproc` reports
# the host's cores rather than the container's quota, and a BLAS pool sized from
# it would oversubscribe underneath an otherwise well-behaved process count.
_MAX_WORKERS = int(
    yaml.safe_load((REPO / "config.yaml").read_text())["compute"]["max_workers"]
)
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, str(_MAX_WORKERS))

import geopandas as gpd
import numpy as np
import pandas as pd
import prefcent as pc

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import common

# Inside a container `free`/`nproc` report the host; the cgroup files report the
# container's own limits.
CGROUP_PEAK = common.CGROUP_PEAK
CGROUP_MAX = common.CGROUP_MAX
MemoryWatch = common.MemoryWatch
read_int = common.read_cgroup_int


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `write_layer()` stamps repro.gpkg_last_change and VACUUMs, so the .gpkg bytes
# are a function of the data alone. Reuse it rather than restating it.
stage2 = _load_module("stage2", SRC / "20_join_aggregate.py")


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git dir
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pc_column_name(family: str, run: int, gamma: float, beta: float, dp: float) -> str:
    """`cd{run}g{gamma}b{beta}k{int(100*Dp)}` with the decimal points dropped.

    Transcribed from the authors' batch driver, which is the convention's
    only written source: `f"cd{idx}g{gamma}b{beta}k{int(dp*100)}"
    .replace(".", "")`. So beta 1.5 becomes `b15`, beta 2.0 becomes `b2`.
    """
    g = f"{gamma:g}"
    b = f"{beta:g}"
    return f"{family}{run}g{g}b{b}k{int(dp * 100)}".replace(".", "")


def _summary_from_manifests(
    ctx: common.Context, labels: list[str], fresh: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per configured (beta, gamma) run, from this run or from its manifest."""
    have = {r["label"] for r in fresh}
    rows = list(fresh)
    for label in labels:
        if label in have:
            continue
        man = ctx.interim(f"pc_run_{label}_manifest.json")
        if not man.exists():
            ctx.log.warning(
                "no manifest for label '%s' — it is missing from the run table; "
                'run `make pc-runs ARGS="--only %s"`',
                label,
                label,
            )
            continue
        m = json.loads(man.read_text())
        for r in m["runs"]:
            rows.append(
                {k: v for k, v in r.items() if k != "prefcent_manifest"}
                | {
                    "kernel_wall_seconds": m["kernel"]["wall_seconds"],
                    "n_jobs": m["kernel"]["n_jobs"],
                }
            )
        ctx.log.info("run table: '%s' read back from %s", label, man.name)
    return sorted(rows, key=lambda r: (float(r["beta"]), -float(r["gamma"])))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="11_pc_runs.py",
        description="Produce PC runs with the solver of record (prefcent).",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="run only these beta labels (default: every label in the config grid)",
    )
    ap.add_argument(
        "--list", action="store_true", help="print the configured run grid and exit"
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="rebuild outputs/tables/stage0_pc_runs.csv from the manifests already "
        "on disk, without running anything",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Dijkstra workers (default: compute.max_workers)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "pc_runs")
    cfg = ctx.cfg

    solver = cfg["inputs"]["pc_solver"]
    submitted = solver["submitted_run"]
    graph_rule = solver["graph"]
    net_cfg = cfg["inputs"]["base_network"]
    sens = cfg["revision"]["pc_sensitivity"]
    runs_cfg = sens.get("runs") or []
    if not runs_cfg:
        ctx.log.error("revision.pc_sensitivity.runs is empty — nothing to run")
        return 2

    n_jobs = int(args.n_jobs or cfg["compute"]["max_workers"])
    ceiling = int(cfg["compute"]["host_ceiling"])
    if n_jobs > ceiling:
        ctx.log.warning(
            "n_jobs=%d exceeds compute.host_ceiling=%d, the worker cap the runs of "
            "record used; continuing",
            n_jobs,
            ceiling,
        )

    # prefcent 0.1.0 has no switch for these four settings: `pc.Model` with no
    # closure applies no density penalty (kappa = 0), `evolve()` with no start
    # begins from the capacities R, every step renormalises the mass to sum(R)
    # (L1), and with no tolerance it runs the full iteration budget. The call
    # below relies on exactly that, so a config asking for anything else is
    # refused rather than recorded in a manifest it would not describe.
    implicit = {
        "kappa": 0.0,
        "a0": "R",
        "normalization": "l1_2024",
        "stop": "budget_only",
    }
    for k, want in implicit.items():
        got = submitted[k]
        if (float(got) != want) if k == "kappa" else (str(got) != want):
            raise SystemExit(
                f"inputs.pc_solver.submitted_run.{k} = {got!r}, but prefcent "
                f"{pc.__version__} as called here implements only {want!r}"
            )

    labels = sorted({str(r["label"]) for r in runs_cfg})
    if args.list:
        for r in runs_cfg:
            print(
                f"{r['label']:>4}  beta={r['beta']:<4} gamma={r['gamma']:<3} "
                f"run={r['run']:<4} {r['density_column']} / {r['mass_column']}  ({r['role']})"
            )
        return 0
    wanted = [] if args.summary_only else (list(args.only) if args.only else labels)
    unknown = [x for x in wanted if x not in labels]
    if unknown:
        raise SystemExit(f"unknown label(s) {unknown}; configured: {labels}")

    mem_max = read_int(CGROUP_MAX)
    ctx.log.info(
        "compute: max_workers=%d (cgroup memory.max = %s GiB, host_ceiling=%d)",
        n_jobs,
        f"{mem_max / 2**30:.1f}" if mem_max else "unknown",
        ceiling,
    )

    # -- the network, and the graph rule that is not restated here ------------
    bundle = REPO / solver["reconstruction_bundle"]
    extractor_path = bundle / "extract_network_from_shapefile.py"
    if not extractor_path.exists():
        ctx.log.error("graph extractor missing from the bundle: %s", extractor_path)
        return 2
    extractor = _load_module("pc_extract", extractor_path)

    net_path = REPO / net_cfg["path"]
    if not net_path.exists():
        ctx.log.error("inputs.base_network.path does not exist: %s", net_path)
        return 2
    net_sha = sha256(net_path)

    t0 = time.time()
    net, meta = extractor.extract(
        str(net_path),
        node_cols=tuple(net_cfg["columns"]["segment_id"]),
        dist_col=net_cfg["columns"]["length_m"],
        speed_col=net_cfg["columns"]["speed_kmh"],
        area_col=net_cfg["columns"]["static_weight"],
    )
    ctx.log.info(
        "extracted %s in %.1fs: %d segments, %d graph nodes, %d graph edges, %d zones",
        net_path.name,
        time.time() - t0,
        meta["n_segments"],
        meta["graph_nodes"],
        meta["graph_edges"],
        meta["n_zones"],
    )
    ctx.count("base network segments", meta["n_segments"], net_cfg["path"])
    ctx.count("zones (area > 0)", meta["n_zones"])
    ctx.count("zero-capacity segments", meta["zero_capacity_segments"], "area == 0")

    # counts declared in config are a contract, not a comment
    declared = net_cfg["counts"]
    for key, got in (
        ("segments", meta["n_segments"]),
        ("zones", meta["n_zones"]),
        ("zero_capacity_segments", meta["zero_capacity_segments"]),
    ):
        if int(declared[key]) != int(got):
            ctx.log.error(
                "base network %s = %d, config declares %d — refusing to run on an "
                "input that is not the one of record",
                key,
                got,
                declared[key],
            )
            return 1
    if abs(float(declared["sum_R"]) - float(meta["sum_R"])) > 1e-6:
        ctx.log.error(
            "sum_R = %.8f, config declares %.8f", meta["sum_R"], declared["sum_R"]
        )
        return 1
    for k in ("graph_nodes", "graph_edges"):
        if int(graph_rule[k]) != int(meta[k]):
            ctx.log.error(
                "%s = %d, inputs.pc_solver.graph declares %d", k, meta[k], graph_rule[k]
            )
            return 1
    ctx.log.info("base network matches every count declared in config")

    seg = gpd.read_file(net_path, layer=net_cfg["layer"])
    id_cols = list(net_cfg["columns"]["segment_id"])
    area = np.zeros(meta["n_segments"], dtype=np.float64)
    area[net["zone_rows"]] = net["R"]
    if not np.array_equal(area, seg[net_cfg["columns"]["static_weight"]].to_numpy()):
        ctx.log.error("the extracted R and the layer's own `area` column disagree")
        return 1

    landscape = pc.Landscape(net["R"], labels=net["zone_ids"])
    summary_rows: list[dict[str, Any]] = []
    written: list[str] = []

    for label in wanted:
        group = [r for r in runs_cfg if str(r["label"]) == label]
        beta = float(group[0]["beta"])
        if any(float(r["beta"]) != beta for r in group):
            raise SystemExit(f"label '{label}' mixes betas; one label is one kernel")
        out_stem = f"pc_run_{label}"
        ctx.log.info("=" * 74)
        ctx.log.info(
            "beta = %g — %d run(s): %s",
            beta,
            len(group),
            ", ".join(f"gamma={r['gamma']:g}" for r in group),
        )

        with MemoryWatch() as mem:
            before = mem.mark()
            t1 = time.time()
            kernel = pc.kernels.from_graph(
                net["graph_edges"],
                net["graph_costs"],
                meta["graph_nodes"],
                sources=net["zone_graph_index"],
                directed=False,
                n_jobs=n_jobs,
                beta=beta,
                d0=float(submitted["d0"]),
                cost_units=meta["cost_units"],
                self_interaction=bool(submitted["self_interaction"]),
            )
            kernel_seconds = time.time() - t1
            after = mem.mark()
            ctx.log.info(
                "kernel built in %.1fs: %s %s, is_symmetric=%s "
                "(cgroup memory %.1f -> %.1f GiB)",
                kernel_seconds,
                kernel.shape,
                kernel.dtype,
                kernel.is_symmetric,
                before,
                after,
            )

            out = pd.DataFrame({c: seg[c].to_numpy() for c in id_cols})
            run_records: list[dict[str, Any]] = []
            for spec in group:
                gamma = float(spec["gamma"])
                dens_col = str(spec["density_column"])
                mass_col = str(spec["mass_column"])
                expect = pc_column_name(
                    "cd", int(spec["run"]), gamma, beta, float(spec["dp"])
                )
                if dens_col != expect:
                    raise SystemExit(
                        f"config names the column `{dens_col}` but the naming "
                        f"convention makes it `{expect}` for run={spec['run']} "
                        f"gamma={gamma:g} beta={beta:g}"
                    )

                t2 = time.time()
                model = pc.Model(kernel, landscape, gamma=gamma)
                res = model.evolve(
                    max_iter=int(submitted["max_iter"]),
                    omega=float(submitted["omega"]),
                )
                solve_seconds = time.time() - t2
                mem.mark()
                # confirm the solver did what `implicit` above says it does
                if (
                    model.closure.name != "identity"
                    or res.start_convention != "capacity"
                    or res.status.name != "FIXED_BUDGET"
                ):
                    raise SystemExit(
                        f"prefcent reports closure={model.closure.name}, "
                        f"start={res.start_convention}, status={res.status.name}; "
                        "expected identity / capacity / FIXED_BUDGET"
                    )

                mass = np.zeros(meta["n_segments"], dtype=np.float64)
                mass[net["zone_rows"]] = res.mass
                dens = np.zeros_like(mass)
                nz = area > 0
                dens[nz] = mass[nz] / area[nz]
                out[dens_col] = dens
                out[mass_col] = mass

                ctx.log.info(
                    "gamma=%g %s in %.1fs after %d iterations, final step norm %.4e",
                    gamma,
                    res.status.name,
                    solve_seconds,
                    res.iterations,
                    res.final_step_norm,
                )
                ctx.log.info(
                    "   %s / %s: sum mass %.6f (sum R %.6f), max density %.6g",
                    dens_col,
                    mass_col,
                    float(mass.sum()),
                    float(area.sum()),
                    float(dens.max()),
                )
                rec = {
                    "label": label,
                    "role": spec["role"],
                    "run": int(spec["run"]),
                    "gamma": gamma,
                    "beta": beta,
                    "d0": float(submitted["d0"]),
                    "kappa": float(submitted["kappa"]),
                    "dp": float(spec["dp"]),
                    "a0": submitted["a0"],
                    "omega": float(submitted["omega"]),
                    "normalization": submitted["normalization"],
                    "max_iter": int(submitted["max_iter"]),
                    "stop": submitted["stop"],
                    "self_interaction": bool(submitted["self_interaction"]),
                    "density_column": dens_col,
                    "mass_column": mass_col,
                    "status": res.status.name,
                    "iterations": int(res.iterations),
                    "final_step_norm": float(res.final_step_norm),
                    "criterion": str(res.criterion),
                    "start_convention": str(res.start_convention),
                    "matvec_count": int(res.matvec_count),
                    "sum_mass": float(mass.sum()),
                    "sum_R": float(area.sum()),
                    # the localisation the report reads: at gamma = 1 a steep
                    # kernel concentrates the mass, and `max_density` is the
                    # number that says how far
                    "max_density": float(dens.max()),
                    "max_mass": float(mass.max()),
                    "solve_wall_seconds": round(solve_seconds, 1),
                    "prefcent_manifest_sha256": res.manifest.fingerprint_sha256,
                    "prefcent_manifest": res.manifest.to_dict(),
                }
                run_records.append(rec)
                summary_rows.append(
                    {k: v for k, v in rec.items() if k != "prefcent_manifest"}
                    | {
                        "kernel_wall_seconds": round(kernel_seconds, 1),
                        "n_jobs": n_jobs,
                    }
                )
                del model, res

            kernel_shape = tuple(int(x) for x in kernel.shape)
            kernel_dtype = str(kernel.dtype)
            kernel_symmetric = bool(kernel.is_symmetric)
            del kernel
            peak_gib = mem.peak_gib

        # The sampled peak is a property of the LABEL -- one kernel and its
        # solves -- and is only known once the kernel has been freed, so it is
        # written back onto this label's rows rather than guessed inside the loop.
        for rec in run_records:
            rec["label_peak_gib"] = round(peak_gib, 2)
        for row in summary_rows:
            if row["label"] == label:
                row["label_peak_gib"] = round(peak_gib, 2)

        # `area` is echoed back so the acceptance test can check R too.
        out[net_cfg["columns"]["static_weight"]] = area
        gdf = gpd.GeoDataFrame(
            out, geometry=seg.geometry.reset_index(drop=True), crs=seg.crs
        )
        gpkg = stage2.write_layer(ctx, gdf, f"{out_stem}.gpkg")
        written.append(str(gpkg.relative_to(REPO)))

        manifest = {
            "label": label,
            "produced_by": "src/11_pc_runs.py",
            "what": (
                f"preferential-centrality runs at beta = {beta:g} on the base network "
                "of record, every other parameter of the submitted run held fixed"
            ),
            "solver": {
                "package": solver["package"],
                "declared_version": solver["version"],
                "installed_version": str(pc.__version__),
                "upstream": solver["upstream"],
                "upstream_commit": solver["upstream_commit"],
                "upstream_tag": solver["upstream_tag"],
                "wheel": solver["wheel"],
                "wheel_sha256": solver["wheel_sha256"],
            },
            "input": {
                "path": net_cfg["path"],
                "layer": net_cfg.get("layer"),
                "sha256": net_sha,
                "crs": str(seg.crs),
                "derived_from": net_cfg["derived_from"],
                "derived_from_sha256": net_cfg["derived_from_sha256"],
                "n_segments": meta["n_segments"],
                "n_zones": meta["n_zones"],
                "zero_capacity_segments": meta["zero_capacity_segments"],
                "sum_R": meta["sum_R"],
            },
            "graph": {
                "rule": graph_rule["rule"],
                "extractor": str(extractor_path.relative_to(REPO)),
                "extractor_sha256": sha256(extractor_path),
                "graph_nodes": meta["graph_nodes"],
                "graph_edges": meta["graph_edges"],
                "cost_units": meta["cost_units"],
                "speed_reference_kmh": meta["speed_reference_kmh"],
                "midpoint_rule": meta["midpoint_rule"],
                "duplicate_edge_rule": graph_rule["duplicate_edge_rule"],
                "connector_cost": graph_rule["connector_cost"],
                # R is not a pure function of geometry. These runs use the
                # shipped `area`; a partial-network variant rebuilds it and
                # records the enumeration order it used.
                "static_weight_source": "shipped `area` column of inputs.base_network",
                "static_weight_rebuilt": False,
                "segment_enumeration_order": (
                    "row order of inputs.base_network, which is the row order of "
                    "edges_combined.shp = the export's own `G.edges` order"
                ),
            },
            "kernel": {
                "form": "(cost + d0)^(-beta), zero diagonal",
                "shape": list(kernel_shape),
                "dtype": kernel_dtype,
                "is_symmetric": kernel_symmetric,
                "beta": beta,
                "d0": float(submitted["d0"]),
                "self_interaction": bool(submitted["self_interaction"]),
                "wall_seconds": round(kernel_seconds, 1),
                "n_jobs": n_jobs,
                "dense_gib": round(kernel_shape[0] * kernel_shape[1] * 8 / 2**30, 2),
            },
            "resources": {
                "cgroup_memory_max_gib": round(mem_max / 2**30, 2) if mem_max else None,
                "cgroup_memory_peak_sampled_gib": round(peak_gib, 2),
                "cgroup_memory_peak_since_boot_gib": round(
                    (read_int(CGROUP_PEAK) or 0) / 2**30, 2
                ),
                "max_workers": n_jobs,
                "thread_env": {
                    v: os.environ.get(v)
                    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
                "total_wall_seconds": round(
                    kernel_seconds + sum(r["solve_wall_seconds"] for r in run_records),
                    1,
                ),
            },
            "runs": run_records,
            "columns_written": [
                c for r in group for c in (r["density_column"], r["mass_column"])
            ]
            + [net_cfg["columns"]["static_weight"]],
            "naming_convention": (
                "cd{run}g{gamma}b{beta}k{int(100*Dp)}, decimal points dropped, "
                "from the authors' batch driver"
            ),
            "git_hash": git_hash(),
            "seed": int(cfg["repro"]["seed"]),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "geopandas": gpd.__version__,
            "outputs": {"gpkg": str(gpkg.relative_to(REPO))},
        }
        man_path = ctx.interim(f"{out_stem}_manifest.json")
        man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        ctx.log.info("wrote %s", man_path.relative_to(REPO))
        ctx.log.info(
            "beta %g done: kernel %.1fs, solves %s, sampled peak %.1f GiB",
            beta,
            kernel_seconds,
            " + ".join(f"{r['solve_wall_seconds']:.1f}s" for r in run_records),
            peak_gib,
        )

    # The summary table covers the whole configured grid, not just this
    # invocation: `make pc-runs ARGS="--only b2"` must not silently drop the
    # other betas' rows from a tracked table. Rows for labels this run did not
    # touch are read back from their manifests.
    summary_rows = _summary_from_manifests(ctx, labels, summary_rows)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ctx.table("stage0_pc_runs.csv"), index=False)
    ctx.log.info("wrote %s", ctx.table("stage0_pc_runs.csv").relative_to(REPO))
    ctx.write_counts("stage0_pc_runs_counts.csv")
    for w in written:
        ctx.log.info("next: make pc-accept FILE=%s", w)
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
