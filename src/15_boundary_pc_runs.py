#!/usr/bin/env python
"""Stage 0 — run the solver of record on each boundary-extent variant.

    make boundary-pc
    make boundary-pc ARGS="--only bn2000"
    python src/15_boundary_pc_runs.py --list

The counterpart of `src/11_pc_runs.py` for the boundary sensitivity. There, the
network is fixed at `inputs.base_network` and (gamma, beta) move; here (gamma,
beta) are fixed at `inputs.pc_solver.submitted_run` — gamma in {1, 0}, beta = 2,
d0 = 5000, a0 = R, omega = 1, L1 renormalisation, exactly 200 iterations —
and the network moves. That is the whole design of the experiment: one thing
changes, and it is the extent.

The variant networks are built by `src/14_boundary_variants.py` and
`src/16_boundary_muni_variants.py` (`revision.boundary_sensitivity.variants[].network`).
They carry the same five columns `inputs.base_network` does, `(node1, node2)`
unchanged, and an `area` column that has been rebuilt on the clipped network.
This script therefore does not check the base network's declared counts: a variant is
a different network by construction. What it does check is that the layer is a
strict subgraph of the network of record — every `(node1, node2)` present, and
the geometry and `dist` bit-identical on every surviving row — because that is
what makes a difference downstream the extent and nothing else.

The column names are the submitted run's names (`cd1g1b2k0` / `cd4g0b2k0`): the
convention encodes (run, gamma, beta, Dp) and none of those moved. The extent
lives in the file name and in the manifest, which is also what lets
`make pc-accept` score a variant against the submitted columns under its own
defaults, on the segments the two extents share.

Writes, per variant:
    data/interim/pc_boundary_<label>.gpkg        node1, node2, cd*/ca*, area
    data/interim/pc_boundary_<label>_manifest.json
    outputs/tables/stage0_boundary_pc_runs.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent

# As in src/11_pc_runs.py: BLAS reads its thread count at load time, so
# `compute.max_workers` has to be in the environment before numpy is imported.
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
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import common


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage2 = _load_module("stage2", SRC / "20_join_aggregate.py")
runs11 = _load_module("pc_runs", SRC / "11_pc_runs.py")


def family_cfg(bcfg: dict, family: str | None) -> tuple[dict, str]:
    """Resolve `revision.boundary_sensitivity` for one variant family.

    Two families share this machinery and write separate tables.
    `negative_buffer` is the block itself; any other name is a sub-block laid
    over it, so the keys both families share -- the solver runs, the acceptance
    mode -- are stated once. Returns (config, table-name suffix).
    """
    if family in (None, "", "negative_buffer"):
        return bcfg, ""
    blocks = {k: v for k, v in bcfg.items() if isinstance(v, dict) and "variants" in v}
    sub = blocks.get(family)
    if sub is None:  # also accept the family by its table suffix ("muni")
        sub = next(
            (v for v in blocks.values() if str(v.get("table_suffix")) == family), None
        )
    if sub is None:
        names = sorted(
            {"negative_buffer", *blocks}
            | {str(v["table_suffix"]) for v in blocks.values() if "table_suffix" in v}
        )
        raise SystemExit(
            f"unknown variant family '{family}'; configured: {', '.join(names)}"
        )
    return {**bcfg, **sub}, "_" + str(sub["table_suffix"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="15_boundary_pc_runs.py",
        description="Run the solver of record on the boundary-extent variants.",
    )
    ap.add_argument("--only", nargs="*", default=None, metavar="LABEL")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument(
        "--family",
        default="negative_buffer",
        help="which variant family to run: `negative_buffer` (the default) or "
        "`muni` (the municipal polygon). The family decides which variants are "
        "read and which table the summary is written to.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "boundary_pc_runs")
    cfg = ctx.cfg
    solver = cfg["inputs"]["pc_solver"]
    submitted = solver["submitted_run"]
    graph_rule = solver["graph"]
    net_cfg = cfg["inputs"]["base_network"]
    bcfg, sfx = family_cfg(cfg["revision"]["boundary_sensitivity"], args.family)
    ctx.log.info("variant family: %s (tables suffixed '%s')", args.family, sfx)
    variants = bcfg["variants"] or []
    run_specs = bcfg["pc_runs"] or []
    if not variants or not run_specs:
        ctx.log.error("revision.boundary_sensitivity.variants / .pc_runs is empty")
        return 2
    if args.list:
        for v in variants:
            for r in run_specs:
                print(
                    f"{v['label']:>8}  gamma={r['gamma']:<3} beta={r['beta']:<5} "
                    f"{r['density_column']} ({r['measure']})"
                )
        return 0

    n_jobs = int(args.n_jobs or cfg["compute"]["max_workers"])
    ceiling = int(cfg["compute"]["host_ceiling"])
    if n_jobs > ceiling:
        raise SystemExit(
            f"n_jobs={n_jobs} exceeds compute.host_ceiling={ceiling}; "
            "lower --n-jobs or compute.max_workers"
        )
    mem_max = common.read_cgroup_int(common.CGROUP_MAX)
    ctx.log.info(
        "compute: max_workers=%d (cgroup memory.max = %s GiB)",
        n_jobs,
        f"{mem_max / 2**30:.1f}" if mem_max else "unknown",
    )

    bundle = REPO / solver["reconstruction_bundle"]
    extractor_path = bundle / "extract_network_from_shapefile.py"
    extractor = _load_module("pc_extract", extractor_path)

    # -- the network of record, for the subgraph check ------------------------
    ref = gpd.read_file(REPO / net_cfg["path"], layer=net_cfg["layer"])
    id_cols = list(net_cfg["columns"]["segment_id"])
    ref_key = {
        (a, b): i
        for i, (a, b) in enumerate(zip(ref[id_cols[0]], ref[id_cols[1]], strict=True))
    }
    ref_dist = ref[net_cfg["columns"]["length_m"]].astype(float).to_numpy()
    ref_wkb = ref.geometry.to_wkb()

    summary: list[dict[str, Any]] = []
    if args.only:
        known = {str(v["label"]) for v in variants}
        unknown = [x for x in args.only if x not in known]
        if unknown:
            raise SystemExit(f"unknown label(s) {unknown}; configured: {sorted(known)}")
    for spec in variants:
        label = str(spec["label"])
        if args.only and label not in set(args.only):
            continue
        net_path = REPO / spec["network"]
        if not net_path.exists():
            ctx.log.error(
                "variant network missing: %s — run `make boundary-variants` first",
                net_path,
            )
            return 1
        ctx.log.info("=" * 74)
        ctx.log.info("variant '%s': %s", label, spec["network"])

        seg = gpd.read_file(net_path, layer=spec["network_layer"])
        # -- the subgraph check -----------------------------------------------
        rows = [
            ref_key.get((a, b))
            for a, b in zip(seg[id_cols[0]], seg[id_cols[1]], strict=True)
        ]
        if any(r is None for r in rows):
            ctx.log.error(
                "%d of the %d variant segments carry a (node1, node2) that is not "
                "in the network of record — the clip did not keep the key stable",
                sum(r is None for r in rows),
                len(seg),
            )
            return 1
        rows_arr = np.asarray(rows, dtype=np.int64)
        d_dist = float(
            np.abs(
                seg[net_cfg["columns"]["length_m"]].astype(float).to_numpy()
                - ref_dist[rows_arr]
            ).max()
        )
        same_geom = int(
            (seg.geometry.to_wkb().to_numpy() == ref_wkb.to_numpy()[rows_arr]).sum()
        )
        if d_dist > 0 or same_geom != len(seg):
            ctx.log.error(
                "the variant is not a strict subgraph: max |Δdist| = %.6g, %d of %d "
                "geometries bit-identical",
                d_dist,
                same_geom,
                len(seg),
            )
            return 1
        ctx.log.info(
            "subgraph check OK: %d of %d segments of record, (node1, node2) stable, "
            "dist and geometry bit-identical on every one",
            len(seg),
            len(ref),
        )
        ctx.count(f"{label}: segments", len(seg), spec["network"])

        net, meta = extractor.extract(
            str(net_path),
            node_cols=tuple(id_cols),
            dist_col=net_cfg["columns"]["length_m"],
            speed_col=net_cfg["columns"]["speed_kmh"],
            area_col=net_cfg["columns"]["static_weight"],
        )
        ctx.log.info(
            "%s: %d segments, %d graph nodes, %d graph edges, %d zones, sum R %.6f",
            label,
            meta["n_segments"],
            meta["graph_nodes"],
            meta["graph_edges"],
            meta["n_zones"],
            meta["sum_R"],
        )
        ctx.count(f"{label}: zones (area > 0)", meta["n_zones"])

        # -- connectivity of the routing graph --------------------------------
        # A clip does not only remove segments, it can cut the network. Zones in
        # a component of their own are unreachable from every other zone, their
        # kernel row is zero, and the fixed-point iteration then behaves quite
        # differently from the submitted run -- so the component count is part
        # of the record, not a diagnostic.
        edges = np.asarray(net["graph_edges"])
        adj = sp.coo_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(meta["graph_nodes"], meta["graph_nodes"]),
        )
        n_comp, comp = connected_components(adj, directed=False)
        zone_comp = comp[np.asarray(net["zone_graph_index"])]
        zone_sizes = np.bincount(zone_comp, minlength=n_comp)
        n_zones_largest = int(zone_sizes.max())
        n_zones_outside = int(meta["n_zones"] - n_zones_largest)
        conn = {
            "graph_components": int(n_comp),
            "n_zones_in_largest_component": n_zones_largest,
            "n_zones_outside_largest_component": n_zones_outside,
            "zone_bearing_components": int((zone_sizes > 0).sum()),
        }
        if n_comp > 1:
            ctx.log.warning(
                "%s: the clip cuts the network — %d components, %d of the %d "
                "zones outside the largest. Those zones cannot reach any other "
                "zone, so their kernel row is zero; read the run's "
                "`final_step_norm` with that in mind",
                label,
                n_comp,
                n_zones_outside,
                meta["n_zones"],
            )
        ctx.count(
            f"{label}: routing-graph components",
            int(n_comp),
            f"{n_zones_outside} zone(s) outside the largest",
        )

        area = np.zeros(meta["n_segments"], dtype=np.float64)
        area[net["zone_rows"]] = net["R"]
        landscape = pc.Landscape(net["R"], labels=net["zone_ids"])

        out = pd.DataFrame({c: seg[c].to_numpy() for c in id_cols})
        run_records: list[dict[str, Any]] = []
        beta = float(run_specs[0]["beta"])
        if any(float(r["beta"]) != beta for r in run_specs):
            raise SystemExit("boundary_sensitivity.pc_runs mixes betas; one kernel")

        with common.MemoryWatch() as mem:
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
            ctx.log.info(
                "kernel built in %.1fs: %s %s (%.2f GiB dense)",
                kernel_seconds,
                kernel.shape,
                kernel.dtype,
                kernel.shape[0] * kernel.shape[1] * 8 / 2**30,
            )
            for r in run_specs:
                gamma = float(r["gamma"])
                expect = runs11.pc_column_name(
                    "cd", int(r["run"]), gamma, beta, float(r["dp"])
                )
                if str(r["density_column"]) != expect:
                    raise SystemExit(
                        f"config names `{r['density_column']}` but the column "
                        f"naming convention makes it `{expect}`"
                    )
                t2 = time.time()
                model = pc.Model(kernel, landscape, gamma=gamma)
                res = model.evolve(
                    max_iter=int(submitted["max_iter"]),
                    omega=float(submitted["omega"]),
                )
                solve_seconds = time.time() - t2
                mass = np.zeros(meta["n_segments"], dtype=np.float64)
                mass[net["zone_rows"]] = res.mass
                dens = np.zeros_like(mass)
                nz = area > 0
                dens[nz] = mass[nz] / area[nz]
                out[str(r["density_column"])] = dens
                out[str(r["mass_column"])] = mass
                ctx.log.info(
                    "gamma=%g %s in %.1fs after %d iterations, final step norm %.4e, "
                    "max density %.6g",
                    gamma,
                    res.status.name,
                    solve_seconds,
                    res.iterations,
                    res.final_step_norm,
                    float(dens.max()),
                )
                run_records.append(
                    {
                        "label": label,
                        "measure": r["measure"],
                        "run": int(r["run"]),
                        "gamma": gamma,
                        "beta": beta,
                        "d0": float(submitted["d0"]),
                        "dp": float(r["dp"]),
                        "a0": submitted["a0"],
                        "omega": float(submitted["omega"]),
                        "normalization": submitted["normalization"],
                        "max_iter": int(submitted["max_iter"]),
                        "stop": submitted["stop"],
                        "density_column": str(r["density_column"]),
                        "mass_column": str(r["mass_column"]),
                        "status": res.status.name,
                        "iterations": int(res.iterations),
                        "final_step_norm": float(res.final_step_norm),
                        "criterion": str(res.criterion),
                        "matvec_count": int(res.matvec_count),
                        "n_segments": meta["n_segments"],
                        "n_zones": meta["n_zones"],
                        **conn,
                        "sum_mass": float(mass.sum()),
                        "sum_R": float(area.sum()),
                        "max_density": float(dens.max()),
                        "solve_wall_seconds": round(solve_seconds, 1),
                        "prefcent_manifest_sha256": res.manifest.fingerprint_sha256,
                    }
                )
                del model, res
            kernel_shape = tuple(int(x) for x in kernel.shape)
            del kernel
            peak_gib = mem.peak_gib

        out[net_cfg["columns"]["static_weight"]] = area
        gdf = gpd.GeoDataFrame(
            out, geometry=seg.geometry.reset_index(drop=True), crs=seg.crs
        )
        gpkg = stage2.write_layer(ctx, gdf, f"pc_boundary_{label}.gpkg")

        vman_path = ctx.interim(f"boundary_{label}_manifest.json")
        vman = json.loads(vman_path.read_text()) if vman_path.exists() else {}
        for rec in run_records:
            rec["label_peak_gib"] = round(peak_gib, 2)
            rec["kernel_wall_seconds"] = round(kernel_seconds, 1)
            rec["n_jobs"] = n_jobs
            rec["inward_buffer_m"] = vman.get("extent", {}).get("inward_buffer_m")
            rec["extent_area_km2"] = vman.get("extent", {}).get("variant_area_km2")
        summary += run_records

        manifest = {
            "label": label,
            "produced_by": "src/15_boundary_pc_runs.py",
            "what": (
                "the submitted preferential-centrality run (gamma in {1, 0}, "
                "beta = 2, 200 iterations) recomputed on a boundary-extent variant "
                "of the network of record"
            ),
            "solver": {
                "package": solver["package"],
                "declared_version": solver["version"],
                "installed_version": str(pc.__version__),
                "upstream_commit": solver["upstream_commit"],
                "wheel_sha256": solver["wheel_sha256"],
            },
            "input": {
                "path": spec["network"],
                "layer": spec["network_layer"],
                "sha256": common.checksum(net_path),
                "built_by": vman.get(
                    "produced_by",
                    "src/14_boundary_variants.py or src/16_boundary_muni_variants.py",
                ),
                "variant_manifest": str(vman_path.relative_to(REPO)),
                "n_segments": meta["n_segments"],
                "n_zones": meta["n_zones"],
                "sum_R": meta["sum_R"],
                **conn,
                "is_strict_subgraph_of_record": True,
                "subgraph_check": {
                    "id_columns": id_cols,
                    "max_abs_dist_difference": d_dist,
                    "geometries_bit_identical": same_geom,
                },
            },
            "graph": {
                "rule": graph_rule["rule"],
                "extractor": str(extractor_path.relative_to(REPO)),
                "extractor_sha256": common.checksum(extractor_path),
                "graph_nodes": meta["graph_nodes"],
                "graph_edges": meta["graph_edges"],
                "cost_units": meta["cost_units"],
                "duplicate_edge_rule": graph_rule["duplicate_edge_rule"],
                # the variant rebuilt R, and this is the enumeration order it used
                "static_weight_source": (
                    "rebuilt on the clipped network by "
                    + str((bundle / "rebuild_segment_areas.py").relative_to(REPO))
                ),
                "static_weight_rebuilt": True,
                "segment_enumeration_order": vman.get("static_weight", {}).get(
                    "segment_enumeration_order"
                ),
            },
            "kernel": {
                "form": "(cost + d0)^(-beta), zero diagonal",
                "shape": list(kernel_shape),
                "beta": beta,
                "d0": float(submitted["d0"]),
                "wall_seconds": round(kernel_seconds, 1),
                "n_jobs": n_jobs,
                "dense_gib": round(kernel_shape[0] * kernel_shape[1] * 8 / 2**30, 2),
            },
            "resources": {
                "cgroup_memory_max_gib": round(mem_max / 2**30, 2) if mem_max else None,
                "cgroup_memory_peak_sampled_gib": round(peak_gib, 2),
                "total_wall_seconds": round(
                    kernel_seconds + sum(r["solve_wall_seconds"] for r in run_records),
                    1,
                ),
            },
            "runs": run_records,
            "git_hash": runs11.git_hash(),
            "seed": int(cfg["repro"]["seed"]),
            "outputs": {"gpkg": str(gpkg.relative_to(REPO))},
        }
        man = ctx.interim(f"pc_boundary_{label}_manifest.json")
        man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        ctx.log.info("wrote %s", man.relative_to(REPO))
        ctx.log.info(
            'next: make pc-accept FILE=%s ARGS="--informational --label boundary_%s"',
            gpkg.relative_to(REPO),
            label,
        )

    if summary:
        pd.DataFrame(summary).to_csv(
            ctx.table(f"stage0_boundary{sfx}_pc_runs.csv"), index=False
        )
        ctx.log.info(
            "wrote %s",
            ctx.table(f"stage0_boundary{sfx}_pc_runs.csv").relative_to(REPO),
        )
    ctx.write_counts(f"stage0_boundary{sfx}_pc_runs_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
