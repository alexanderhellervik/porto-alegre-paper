#!/usr/bin/env python
"""Stage 0 — build the municipal boundary-extent variants.

    make boundary-muni-variants
    make boundary-muni-variants ARGS="--only mu0"
    python src/16_boundary_muni_variants.py --list

The second boundary-variant family, and the one the paper reports. Where
`src/14_boundary_variants.py` shrinks the extent the delivered network happens to
cover by a fixed distance (the negative-buffer family), this script clips the
network to the administrative unit the paper is about: the Porto Alegre
municipal polygon from the IBGE Malha Municipal Digital 2025 (`CD_MUN 4314902`).

What changes, and what does not
-------------------------------
Unchanged from the negative-buffer family, so the two are comparable line for
line:

  * the clip rule — a segment is kept only if it lies fully inside the variant
    polygon; geometries are never cut, `(node1, node2)` stays stable and the
    variant network is a strict subgraph of the network of record;
  * the R rebuild — R is the buildable area of a non-overlapping 30 m buffer,
    so it is rebuilt on each clipped network with the bundle's own
    `rebuild_segment_areas.py`, forward and reversed, and the segment enumeration
    order is recorded in the manifest;
  * the solver, the parameters and the budget — `src/15_boundary_pc_runs.py
    --family muni` runs the submitted (gamma, beta, d0, a0, omega, 200 iterations)
    on the clipped network and nothing else moves.

New in this family:

  * The polygon has a meaning. "The municipality of Porto Alegre" is a boundary
    a reader can check; "the network's own footprint eroded by 2 km" is a
    construction. A variant is that polygon buffered outward by
    `outward_buffer_m` >= 0 — mu0 = the municipality, mu1 = a 1 km collar — and
    the full delivered extent is the third, un-clipped member of the sequence.
  * Connectedness is a rule (`connectedness` in the config). Dropping whole
    segments does not only shrink the network, it can cut it, and on a cut graph
    the submitted run's fixed point is not reached within its 200-iteration
    budget. With `largest_component`, each variant is reduced to the largest
    connected component of its own routing graph before R is rebuilt and before
    the solver runs. The raw component sizes are measured and recorded either
    way, and the effect of dropping only components smaller than
    `tiny_component_segments` is reported beside them as a diagnostic.

The clip is done in the network's own CRS (EPSG:32722), not the other way
round: `src/15_boundary_pc_runs.py` checks that every surviving geometry is
bit-identical to the network of record, and reprojecting the network would break
that. The polygon is what moves (EPSG:4674 -> EPSG:32722).
`data/interim/poa_municipality.gpkg` is written in EPSG:31982, the project's
metric CRS, for every other consumer.

Writes:
    data/interim/poa_municipality.gpkg            the dissolved polygon, EPSG:31982
    data/interim/poa_municipality_manifest.json   source sha256, CD_MUN, area
    data/interim/boundary_<label>_network.gpkg    layer `boundary_<label>_network`:
        node1, node2, dist, speed, area (rebuilt R), geometry — what
        src/15_boundary_pc_runs.py feeds prefcent
    data/interim/boundary_<label>_edges.gpkg      layer `boundary_<label>_edges`:
        every attribute of the surviving rows, the input for GAUS Lines v1.1
        (`make gaus-native`)
    data/interim/boundary_<label>_manifest.json
    data/interim/boundary_muni_extents.gpkg
    outputs/tables/stage0_boundary_muni_variants.csv
    outputs/tables/stage0_boundary_muni_extent_check.csv   the network vs the
        municipality: how much lies inside, and how far outside the rest reaches
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
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


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(common.REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


def graph_components(node1: np.ndarray, node2: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of the routing graph, labelled per segment.

    The graph is the one `extract_network_from_shapefile.py` builds: the nodes
    are the `(node1, node2)` endpoint ids and a segment is an edge between them.
    Returns (n_components, component label of each segment).
    """
    nodes = pd.unique(np.concatenate([node1, node2]))
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    a = idx.reindex(node1).to_numpy()
    b = idx.reindex(node2).to_numpy()
    m = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes), len(nodes)))
    n_comp, lab = connected_components(m, directed=False)
    return int(n_comp), lab[a]


def component_report(seg_comp: np.ndarray, tiny: int) -> dict[str, Any]:
    """Component sizes in segments, plus what the `< tiny` rule would do."""
    sizes = np.sort(np.bincount(seg_comp))[::-1]
    sizes = sizes[sizes > 0]
    small = sizes[sizes < tiny]
    return {
        "n_components": int(sizes.size),
        "component_sizes": [int(x) for x in sizes[:12]],
        "n_segments_largest_component": int(sizes[0]),
        "n_segments_outside_largest": int(sizes[1:].sum()),
        "tiny_threshold_segments": int(tiny),
        "n_components_below_tiny": int(small.size),
        "n_segments_in_components_below_tiny": int(small.sum()),
        # does dropping only the small components leave a connected network?
        "components_left_after_dropping_tiny": int((sizes >= tiny).sum()),
        "tiny_rule_reconnects": bool((sizes >= tiny).sum() == 1),
    }


def extract_municipality(ctx: common.Context, bcfg: dict[str, Any]) -> dict[str, Any]:
    """Pull CD_MUN out of the IBGE mesh, dissolve, write it, and record it."""
    b = bcfg["boundary"]
    src = common.REPO / b["path"]
    if not src.exists():
        raise SystemExit(f"the municipal mesh is missing: {src}")
    mesh = gpd.read_file(src)
    ctx.count("IBGE mesh: municipalities read", len(mesh), b["path"])
    declared = int(b["source_crs_epsg"])
    if mesh.crs is None or mesh.crs.to_epsg() != declared:
        raise SystemExit(f"the mesh declares EPSG:{mesh.crs}, config says {declared}")
    sel = mesh[mesh[b["id_column"]].astype(str) == str(b["cd_mun"])]
    if len(sel) != 1:
        raise SystemExit(
            f"{b['id_column']} == {b['cd_mun']} selects {len(sel)} rows, expected 1"
        )
    got_name = str(sel[b["name_column"]].iloc[0])
    if got_name != str(b["nm_mun"]):
        raise SystemExit(f"{b['cd_mun']} is '{got_name}', config says '{b['nm_mun']}'")

    target = int(ctx.cfg["repro"]["crs_epsg"])
    poly31982 = shapely.union_all(sel.to_crs(target).geometry.values)
    area_km2 = float(shapely.area(poly31982)) / 1e6
    parts = [float(shapely.area(p)) / 1e6 for p in shapely.get_parts(poly31982)]
    parts.sort(reverse=True)
    declared_km2 = float(sel["AREA_KM2"].iloc[0]) if "AREA_KM2" in sel else float("nan")
    ctx.log.info(
        "%s (%s): %.4f km2 measured in EPSG:%d over %d part(s) %s; the mesh's own "
        "AREA_KM2 column says %.3f km2 (%+.3f %%)",
        b["nm_mun"],
        b["cd_mun"],
        area_km2,
        target,
        len(parts),
        [round(p, 4) for p in parts],
        declared_km2,
        100 * (area_km2 - declared_km2) / declared_km2,
    )
    ctx.count("municipality: area (m2)", round(area_km2 * 1e6), f"EPSG:{target}")
    ctx.count("municipality: polygon parts", len(parts))

    out = gpd.GeoDataFrame(
        {
            b["id_column"]: [str(b["cd_mun"])],
            b["name_column"]: [b["nm_mun"]],
            "area_km2": [area_km2],
            "n_parts": [len(parts)],
        },
        geometry=[poly31982],
        crs=f"EPSG:{target}",
    )
    poly_path = stage2.write_layer(ctx, out, Path(b["polygon"]).name)

    bundle = sorted(src.parent.glob("*"))
    manifest = {
        "produced_by": "src/16_boundary_muni_variants.py",
        "what": (
            "the Porto Alegre municipal polygon, dissolved from the IBGE Malha "
            "Municipal Digital 2025 for Rio Grande do Sul and reprojected to the "
            "project's metric CRS; the clip polygon of the municipal variant "
            "family."
        ),
        "source": {
            "path": b["path"],
            "sha256": common.checksum(src),
            "bundle_sha256": common.checksum_dir(src.parent.resolve())[0],
            "bundle_files": {f.name: common.checksum(f) for f in bundle if f.is_file()},
            "n_municipalities": len(mesh),
            "crs": str(mesh.crs),
            "columns": list(mesh.columns),
        },
        "selection": {
            "id_column": b["id_column"],
            "cd_mun": str(b["cd_mun"]),
            "name_column": b["name_column"],
            "nm_mun": got_name,
            "n_rows_selected": 1,
        },
        "geometry": {
            "crs": f"EPSG:{target}",
            "area_m2": area_km2 * 1e6,
            "area_km2": area_km2,
            "n_parts": len(parts),
            "part_areas_km2": parts,
            "mesh_declared_area_km2": declared_km2,
            "area_rel_diff_vs_mesh_column": (area_km2 - declared_km2) / declared_km2,
            "bounds": [float(x) for x in shapely.bounds(poly31982)],
        },
        "outputs": {"polygon_gpkg": str(poly_path.relative_to(common.REPO))},
        "git_hash": git_hash(),
        "seed": int(ctx.cfg["repro"]["seed"]),
        "geopandas": gpd.__version__,
        "shapely": shapely.__version__,
    }
    man = ctx.interim(Path(b["manifest"]).name)
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    ctx.log.info("wrote %s", man.relative_to(common.REPO))
    return {
        "polygon_31982": poly31982,
        "area_km2": area_km2,
        "parts_km2": parts,
        "manifest": manifest,
        "sel": sel,
    }


def extent_check(
    ctx: common.Context, gdf: gpd.GeoDataFrame, muni: Any, tiny: int
) -> list[dict[str, Any]]:
    """Does the delivered network stop at the municipal limit + ~2 km?

    The paper (Sec. 4.1) says the network was *"divided into 29,978 segments,
    except for the buffer of approximately 2 km"*. This measures both halves of
    that sentence against the municipal polygon.
    """
    gv = gdf.geometry.values
    within = shapely.within(gv, muni)
    inter = shapely.intersects(gv, muni)
    mid = shapely.line_interpolate_point(gv, 0.5, normalized=True)
    midin = shapely.intersects(mid, muni)
    ilen = shapely.length(shapely.intersection(gv, muni))
    majority = ilen > 0.5 * shapely.length(gv)
    paper_n = 29978

    rows: list[dict[str, Any]] = []
    for rule, mask in (
        ("fully inside (within)", within),
        ("touching (intersects)", inter),
        ("midpoint inside", midin),
        ("majority of length inside", majority),
    ):
        rows.append(
            {
                "quantity": f"segments {rule}",
                "value": int(mask.sum()),
                "share": float(mask.mean()),
                "share_denominator": f"the {len(gdf):,} delivered segments",
                "vs_paper_29978": int(mask.sum()) - paper_n,
            }
        )
        ctx.log.info(
            "segments %-26s %6d (%.2f %% of %d)  vs the paper's %d: %+d",
            rule,
            int(mask.sum()),
            100 * mask.mean(),
            len(gdf),
            paper_n,
            int(mask.sum()) - paper_n,
        )

    out = ~within
    d = shapely.distance(gv[out], muni)
    ctx.log.info(
        "the %d segments not fully inside reach a median %.0f m and a maximum "
        "%.0f m beyond the municipal limit; %.2f %% of them are within 2 km of it",
        int(out.sum()),
        float(np.median(d)),
        float(d.max()),
        100 * float((d <= 2000).mean()),
    )
    ctx.count("segments outside the municipality", int(out.sum()), "not fully within")
    ctx.count("buffer width: max (m)", round(float(d.max())), "measured")
    for q in (10, 25, 50, 75, 90, 95, 99, 100):
        rows.append(
            {
                "quantity": f"distance beyond the municipal limit, p{q:02d} (m)",
                "value": float(np.percentile(d, q)),
                "share": float("nan"),
                "share_denominator": "",
                "vs_paper_29978": float("nan"),
            }
        )
    for t in (1000, 2000, 4000, 6000, 10000):
        rows.append(
            {
                "quantity": f"outside segments within {t} m of the limit",
                "value": int((d <= t).sum()),
                "share": float((d <= t).mean()),
                "share_denominator": f"the {int(out.sum()):,} segments outside",
                "vs_paper_29978": float("nan"),
            }
        )
    n_comp, seg_comp = graph_components(
        gdf[ctx.cfg["inputs"]["base_network"]["columns"]["segment_id"][0]].to_numpy(),
        gdf[ctx.cfg["inputs"]["base_network"]["columns"]["segment_id"][1]].to_numpy(),
    )
    rep = component_report(seg_comp, tiny)
    ctx.log.info(
        "the full delivered network is %s: %d routing-graph component(s)",
        "connected" if n_comp == 1 else "not connected",
        n_comp,
    )
    ctx.count("full network: routing-graph components", n_comp)
    rows.append(
        {
            "quantity": "routing-graph components, full delivered network",
            "value": int(rep["n_components"]),
            "share": float("nan"),
            "share_denominator": "",
            "vs_paper_29978": float("nan"),
        }
    )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="16_boundary_muni_variants.py",
        description="Build the municipal boundary-extent variants.",
    )
    ap.add_argument("--only", nargs="*", default=None, metavar="LABEL")
    ap.add_argument("--list", action="store_true", help="print the variant grid")
    ap.add_argument(
        "--extent-check-only",
        action="store_true",
        help="extract and record the polygon and measure the delivered network "
        "against it, then stop — no clip, no R rebuild, no variant layers",
    )
    ap.add_argument(
        "--no-reversed",
        action="store_true",
        help="skip the reversed-order R rebuild that measures the order dependence",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "boundary_muni_variants")
    cfg = ctx.cfg
    parent = cfg["revision"]["boundary_sensitivity"]
    bcfg = parent["municipal"]
    variants = bcfg["variants"] or []
    if not variants:
        ctx.log.error("revision.boundary_sensitivity.municipal.variants is empty")
        return 2
    if args.list:
        for v in variants:
            print(f"{v['label']:>6}  outward_buffer_m={v['outward_buffer_m']:<8}")
        return 0
    wanted = [str(v["label"]) for v in variants]
    if args.only:
        unknown = [x for x in args.only if x not in wanted]
        if unknown:
            raise SystemExit(f"unknown label(s) {unknown}; configured: {wanted}")
        wanted = [x for x in wanted if x in set(args.only)]
    tiny = int(bcfg["tiny_component_segments"])
    rule = str(bcfg["connectedness"])
    if rule not in ("largest_component", "as_clipped"):
        raise SystemExit(f"unknown connectedness rule '{rule}'")

    # -- the polygon ----------------------------------------------------------
    muni = extract_municipality(ctx, bcfg)
    poly31982 = muni["polygon_31982"]

    # -- the network of record ------------------------------------------------
    seg_cfg = cfg["inputs"]["segment_centralities"]
    src = common.REPO / seg_cfg["path"]
    if not src.exists():
        ctx.log.error("inputs.segment_centralities.path does not exist: %s", src)
        return 2
    gdf = gpd.read_file(src)
    ctx.count("segments read", len(gdf), seg_cfg["path"])
    source_crs = str(gdf.crs)
    if gdf.crs is None or gdf.crs.is_geographic:
        raise SystemExit(f"the file of record is not in a metric CRS: {gdf.crs}")
    # The polygon moves into the network's CRS, never the network into the
    # polygon's: src/15_boundary_pc_runs.py requires bit-identical geometries.
    muni_net = shapely.union_all(
        muni["sel"].to_crs(gdf.crs).geometry.values  # type: ignore[arg-type]
    )
    ctx.log.info(
        "clip polygon reprojected %s -> %s: %.4f km2 (EPSG:%s reference %.4f km2, "
        "%+.2e relative)",
        f"EPSG:{bcfg['boundary']['source_crs_epsg']}",
        source_crs,
        shapely.area(muni_net) / 1e6,
        cfg["repro"]["crs_epsg"],
        muni["area_km2"],
        (shapely.area(muni_net) / 1e6 - muni["area_km2"]) / muni["area_km2"],
    )

    checks = extent_check(ctx, gdf, muni_net, tiny)
    pd.DataFrame(checks).to_csv(
        ctx.table("stage0_boundary_muni_extent_check.csv"), index=False
    )
    if args.extent_check_only:
        ctx.log.info(
            "--extent-check-only: stopping after %s",
            ctx.table("stage0_boundary_muni_extent_check.csv").relative_to(common.REPO),
        )
        ctx.write_counts("stage0_boundary_muni_extent_check_counts.csv")
        ctx.finish()
        return 0

    bundle = common.REPO / cfg["inputs"]["pc_solver"]["reconstruction_bundle"]
    rebuild_script = bundle / "rebuild_segment_areas.py"
    if not rebuild_script.exists():
        ctx.log.error("rebuild script missing from the bundle: %s", rebuild_script)
        return 2
    rebuild = _load_module("rebuild_segment_areas", rebuild_script)

    net_cols = cfg["inputs"]["base_network"]["columns"]
    id_cols = list(net_cols["segment_id"])
    reference_R = gdf[net_cols["static_weight"]].astype(float).to_numpy()

    rows: list[dict[str, Any]] = []
    extent_rows: list[dict[str, Any]] = [
        {
            "label": "municipality",
            "outward_buffer_m": 0.0,
            "area_m2": float(shapely.area(poly31982)),
            "area_km2": muni["area_km2"],
            "n_parts": len(muni["parts_km2"]),
            "n_segments": -1,
            "geometry": poly31982,
        }
    ]

    for spec in variants:
        label = str(spec["label"])
        if label not in wanted:
            continue
        d = float(spec["outward_buffer_m"])
        if d < 0:
            raise SystemExit(f"variant '{label}': outward_buffer_m must be >= 0")
        ctx.log.info("=" * 74)
        poly = muni_net if d == 0 else shapely.buffer(muni_net, d)
        area = float(shapely.area(poly))
        n_parts = int(shapely.get_num_geometries(poly))

        keep = shapely.within(gdf.geometry.values, poly)
        n_clipped = int(keep.sum())
        if n_clipped == 0:
            ctx.log.error("variant '%s' retains no segment", label)
            return 1
        ctx.log.info(
            "%s: municipality + %.0f m -> area %.4f km2, %d part(s); %d of %d "
            "segments fully inside (%.2f %%)",
            label,
            d,
            area / 1e6,
            n_parts,
            n_clipped,
            len(gdf),
            100 * n_clipped / len(gdf),
        )

        # -- connectedness ----------------------------------------------------
        sub_idx = np.where(keep)[0]
        _n_comp, seg_comp = graph_components(
            gdf[id_cols[0]].to_numpy()[sub_idx], gdf[id_cols[1]].to_numpy()[sub_idx]
        )
        rep = component_report(seg_comp, tiny)
        ctx.log.info(
            "%s: the clip leaves %d routing-graph component(s), sizes %s; the "
            "'< %d segments' rule would drop %d component(s) / %d segment(s) and "
            "leave %d component(s) -- %s",
            label,
            rep["n_components"],
            rep["component_sizes"],
            tiny,
            rep["n_components_below_tiny"],
            rep["n_segments_in_components_below_tiny"],
            rep["components_left_after_dropping_tiny"],
            "connected" if rep["tiny_rule_reconnects"] else "still disconnected",
        )
        ctx.count(
            f"{label}: routing-graph components as clipped",
            rep["n_components"],
            f"{rep['n_segments_outside_largest']} segment(s) outside the largest",
        )
        if rule == "largest_component" and rep["n_components"] > 1:
            sizes = np.bincount(seg_comp)
            biggest = int(sizes.argmax())
            keep = keep.copy()
            keep[sub_idx[seg_comp != biggest]] = False
            ctx.log.info(
                "%s: reduced to the largest component -- %d of the %d clipped "
                "segments dropped (%.3f %%), %d remain, 1 component",
                label,
                n_clipped - int(keep.sum()),
                n_clipped,
                100 * (n_clipped - int(keep.sum())) / n_clipped,
                int(keep.sum()),
            )
        n_keep = int(keep.sum())
        ctx.count(
            f"{label}: segments retained",
            n_keep,
            f"municipality + {d:.0f} m, {rule}",
        )
        sub = gdf.loc[keep].copy()
        # the analysed network must be connected, whatever the rule said
        n_comp_final, comp_final = graph_components(
            sub[id_cols[0]].to_numpy(), sub[id_cols[1]].to_numpy()
        )
        rep_final = component_report(comp_final, tiny)
        if rule == "largest_component" and n_comp_final != 1:
            ctx.log.error(
                "%s: the largest-component reduction left %d components",
                label,
                n_comp_final,
            )
            return 1

        # -- R rebuilt on the clipped network ----------------------------------
        order_note = (
            "row order of inputs.segment_centralities (= the export's `G.edges` "
            "order) restricted to the segments fully inside the municipal polygon "
            "and to its largest routing-graph component"
            if rule == "largest_component"
            else "row order of inputs.segment_centralities restricted to the "
            "segments fully inside the municipal polygon"
        )
        sub = sub.reset_index(drop=False).rename(columns={"index": "_row_of_record"})
        with common.MemoryWatch() as mem:
            t1 = time.time()
            R_fwd = rebuild.rebuild_areas(sub, use_impediment=True)
            fwd_seconds = time.time() - t1
            fwd_peak = mem.peak_gib
        rev_stats: dict[str, Any] = {}
        if not args.no_reversed:
            with common.MemoryWatch() as mem:
                t1 = time.time()
                R_rev = rebuild.rebuild_areas(
                    sub.iloc[::-1].reset_index(drop=True), use_impediment=True
                )[::-1].copy()
                rev_seconds = time.time() - t1
                rev_peak = mem.peak_gib
            both = (R_fwd > 0) & (R_rev > 0)
            rel = np.abs(R_rev[both] - R_fwd[both]) / R_fwd[both]
            rev_stats = {
                "reversed_wall_seconds": round(rev_seconds, 1),
                "reversed_peak_gib": round(rev_peak, 2),
                "order_n_both_positive": int(both.sum()),
                "order_within_0.1pct": float((rel < 1e-3).mean()),
                "order_share_above_1pct": float((rel >= 1e-2).mean()),
                "order_max_rel": float(rel.max()) if rel.size else 0.0,
                # net change in the number of zones (forward minus reversed)
                "order_zone_count_net_diff": int((R_fwd > 0).sum() - (R_rev > 0).sum()),
                # segments that are a zone in one order and not in the other
                "order_zone_set_symmetric_diff": int(
                    ((R_fwd > 0) != (R_rev > 0)).sum()
                ),
            }
            ctx.log.info(
                "%s order dependence: reversing the enumeration order moves %.2f %% "
                "of R by more than 0.1 %%, %.2f %% by more than 1 %%, max %.1f %%",
                label,
                100 * (1 - rev_stats["order_within_0.1pct"]),
                100 * rev_stats["order_share_above_1pct"],
                100 * rev_stats["order_max_rel"],
            )

        ref_here = reference_R[keep]
        both = (R_fwd > 0) & (ref_here > 0)
        rel_ref = np.abs(R_fwd[both] - ref_here[both]) / ref_here[both]
        ctx.log.info(
            "%s: R rebuilt in %.1fs — %d zones (reference on the same rows: %d), "
            "sum R %.6g (reference %.6g, %+.3f %%), within 0.1 %% of the reference "
            "value on %.2f %% of common zones",
            label,
            fwd_seconds,
            int((R_fwd > 0).sum()),
            int((ref_here > 0).sum()),
            R_fwd.sum(),
            ref_here.sum(),
            100 * (R_fwd.sum() - ref_here.sum()) / ref_here.sum(),
            100 * float((rel_ref < 1e-3).mean()),
        )

        # -- the two layers ---------------------------------------------------
        net = gpd.GeoDataFrame(
            {
                id_cols[0]: sub[id_cols[0]].to_numpy(),
                id_cols[1]: sub[id_cols[1]].to_numpy(),
                net_cols["length_m"]: sub[net_cols["length_m"]]
                .astype(float)
                .to_numpy(),
                net_cols["speed_kmh"]: sub[net_cols["speed_kmh"]]
                .astype(float)
                .to_numpy(),
                net_cols["static_weight"]: R_fwd,
            },
            geometry=sub.geometry.to_numpy(),
            crs=gdf.crs,
        )
        net_path = stage2.write_layer(ctx, net, f"boundary_{label}_network.gpkg")

        edges = sub.drop(columns=["_row_of_record"]).copy()
        edges[net_cols["static_weight"]] = R_fwd
        edges_path = stage2.write_layer(ctx, edges, f"boundary_{label}_edges.gpkg")

        extent_rows.append(
            {
                "label": label,
                "outward_buffer_m": d,
                "area_m2": area,
                "area_km2": area / 1e6,
                "n_parts": n_parts,
                "n_segments": n_keep,
                "geometry": (
                    poly
                    if str(gdf.crs) == f"EPSG:{cfg['repro']['crs_epsg']}"
                    else gpd.GeoSeries([poly], crs=gdf.crs)
                    .to_crs(cfg["repro"]["crs_epsg"])
                    .iloc[0]
                ),
            }
        )
        row = {
            "label": label,
            "outward_buffer_m": d,
            "extent_rule": "municipal polygon (IBGE 2025, CD_MUN 4314902) buffered "
            "outward",
            "clip_rule": bcfg["clip_rule"],
            "connectedness_rule": rule,
            "municipality_area_km2": muni["area_km2"],
            "area_m2": area,
            "area_km2": area / 1e6,
            "area_share_of_municipality": area / (muni["area_km2"] * 1e6),
            "n_parts": n_parts,
            "n_segments_clipped": n_clipped,
            "n_segments": n_keep,
            "n_segments_dropped_by_clip": len(gdf) - n_clipped,
            "n_segments_dropped_by_connectedness": n_clipped - n_keep,
            "segment_share_of_reference": n_keep / len(gdf),
            "components_as_clipped": rep["n_components"],
            "component_sizes_as_clipped": "|".join(
                str(x) for x in rep["component_sizes"]
            ),
            "n_segments_outside_largest": rep["n_segments_outside_largest"],
            "tiny_threshold_segments": tiny,
            "n_components_below_tiny": rep["n_components_below_tiny"],
            "n_segments_below_tiny": rep["n_segments_in_components_below_tiny"],
            "components_after_dropping_tiny": rep[
                "components_left_after_dropping_tiny"
            ],
            "tiny_rule_reconnects": rep["tiny_rule_reconnects"],
            "components_analysed": rep_final["n_components"],
            "n_zones_rebuilt": int((R_fwd > 0).sum()),
            "n_zones_reference_same_rows": int((ref_here > 0).sum()),
            "sum_R_rebuilt": float(R_fwd.sum()),
            "sum_R_reference_same_rows": float(ref_here.sum()),
            "sum_R_rel_diff": float((R_fwd.sum() - ref_here.sum()) / ref_here.sum()),
            "R_within_0.1pct_of_reference": float((rel_ref < 1e-3).mean()),
            "R_median_rel_vs_reference": float(np.median(rel_ref)),
            "R_max_rel_vs_reference": float(rel_ref.max()),
            "forward_wall_seconds": round(fwd_seconds, 1),
            "forward_peak_gib": round(fwd_peak, 2),
            "segment_enumeration_order": order_note,
            **rev_stats,
        }
        rows.append(row)

        manifest = {
            "label": label,
            "produced_by": "src/16_boundary_muni_variants.py",
            "what": (
                "a municipal boundary-extent variant: the Porto Alegre municipal "
                f"polygon buffered outward by {d:.0f} m, with every segment not "
                "fully inside dropped"
                + (
                    ", the result reduced to its largest routing-graph component,"
                    if rule == "largest_component"
                    else ","
                )
                + " and R rebuilt on what is left"
            ),
            "extent": {
                "rule": "municipal_polygon_buffered_outward",
                "municipality": {
                    "cd_mun": str(bcfg["boundary"]["cd_mun"]),
                    "nm_mun": bcfg["boundary"]["nm_mun"],
                    "source": bcfg["boundary"]["path"],
                    "polygon": bcfg["boundary"]["polygon"],
                    "area_km2": muni["area_km2"],
                },
                "outward_buffer_m": d,
                # the negative-buffer family's key, so the two tables line up
                "inward_buffer_m": -d,
                "variant_area_m2": area,
                "variant_area_km2": area / 1e6,
                "area_share_of_municipality": area / (muni["area_km2"] * 1e6),
                "n_parts": n_parts,
                "crs": source_crs,
            },
            "clip": {
                "rule": bcfg["clip_rule"],
                "predicate": "shapely.within(segment, variant_polygon)",
                "geometry_modified": False,
                "n_segments_in": len(gdf),
                "n_segments_after_clip": n_clipped,
                "n_segments_kept": n_keep,
                "n_segments_dropped": len(gdf) - n_keep,
                "id_columns": id_cols,
                "id_columns_stable": True,
            },
            "connectedness": {
                "rule": rule,
                "as_clipped": rep,
                "as_analysed": rep_final,
                "why": (
                    "dropping whole segments can cut the routing graph, and on a "
                    "cut graph the submitted run's fixed point is not reached in "
                    "its 200-iteration budget; keeping the largest component makes "
                    "the municipal family measure the extent and not the "
                    "fragmentation"
                ),
            },
            "static_weight": {
                "rebuilt": True,
                "why": "R is a non-overlapping buffer area, so it depends on the "
                "neighbours a segment competes with",
                "script": str(rebuild_script.relative_to(common.REPO)),
                "script_sha256": common.checksum(rebuild_script),
                "segment_enumeration_order": order_note,
                "buffer_m": rebuild.BUFFER,
                "sample_spacing_m": rebuild.MIN_SPACING,
                "min_zone_area_m2": rebuild.MIN_AREA,
                "use_impediment": True,
                **{
                    k: v
                    for k, v in row.items()
                    if k.startswith(("R_", "order_", "sum_R"))
                },
            },
            "input": {
                "path": seg_cfg["path"],
                "sha256": common.checksum(src),
                "n_features": len(gdf),
                "crs": source_crs,
            },
            "outputs": {
                "network_gpkg": str(net_path.relative_to(common.REPO)),
                "edges_gpkg": str(edges_path.relative_to(common.REPO)),
            },
            "git_hash": git_hash(),
            "seed": int(cfg["repro"]["seed"]),
            "python": sys.version.split()[0],
            "geopandas": gpd.__version__,
            "shapely": shapely.__version__,
        }
        man = ctx.interim(f"boundary_{label}_manifest.json")
        man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        ctx.log.info("wrote %s", man.relative_to(common.REPO))
        ctx.log.info(
            "next: python src/15_boundary_pc_runs.py --family muni --only %s", label
        )

    ext = gpd.GeoDataFrame(
        extent_rows, geometry="geometry", crs=f"EPSG:{cfg['repro']['crs_epsg']}"
    )
    stage2.write_layer(ctx, ext, "boundary_muni_extents.gpkg")
    pd.DataFrame(rows).to_csv(
        ctx.table("stage0_boundary_muni_variants.csv"), index=False
    )
    ctx.log.info(
        "wrote %s",
        ctx.table("stage0_boundary_muni_variants.csv").relative_to(common.REPO),
    )
    ctx.write_counts("stage0_boundary_muni_variants_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
