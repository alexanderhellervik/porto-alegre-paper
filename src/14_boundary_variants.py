#!/usr/bin/env python
"""Stage 0 — build the boundary-extent variants by negative buffer.

    make boundary-variants
    make boundary-variants ARGS="--only bn2000"
    python src/14_boundary_variants.py --list

Reviewer 3 asks how much the five measures, and the conclusions drawn from them,
depend on where the street network was cut off. `revision.boundary_sensitivity`
is that sensitivity. This script builds the variant extents and the variant
networks; `src/15_boundary_pc_runs.py` runs the solver on them and
`src/13_gaus_headless.py` (GAUS Lines v1.1 itself) recomputes CC / BC / FK.

The extent, and why it is the 30 m buffer union
-----------------------------------------------
The reference extent is not a polygon anywhere in the supplied material — it is
whatever region the 65,357 delivered segments happen to cover. It is therefore
*reconstructed*, and the reconstruction has to be stated rather than assumed:

    extent    = the union of the 30 m buffers of every segment
    footprint = that union with its interior rings dropped

30 m is not an arbitrary width. It is the buffer that defines the static weight
R (`rebuild_segment_areas.BUFFER`), so the extent polygon and the capacity rule
measure the city at the same width, and a segment is inside the extent exactly
when the strip of land it capitalises is. A concave hull was the alternative and
was rejected: its shape depends on a free parameter of its own, it bridges the
Guaiba and the unbuilt gaps that the network genuinely does not cover, and it
would have to be tuned to look right.

**Why the inward buffer is applied to the footprint and not to the raw union.**
The union is a 30 m-wide web with one exterior ring and ~13,400 interior rings,
and every interior ring is the inside of a city block — tens of metres across. A
-2,000 m buffer erodes from *every* boundary, so it would grow each block
interior by 2 km and erase the whole network long before it had shrunk the study
area at all. That is an artefact of the ring topology, not a boundary effect.
Dropping the interior rings first gives the region the network covers, which is
what "the extent of the study area" means and what the referee is asking about;
the inward buffer then pulls that region in from its outer edge. Both areas are
reported: the union's own (the buildable strip) and the footprint's.

A variant is the footprint buffered inward by `inward_buffer_m` (negative),
which shrinks the study area from every outer edge at once — the referee's
question — rather than from one side, as a bbox clip does.

The clip rule
-------------
**A segment is kept only if it lies fully inside the variant polygon**; segments
are never cut. Clipping geometries would (i) leave `dist` — a stored column the
solver reads, not a recomputed length — describing a segment that no longer
exists, (ii) move the zone off the midpoint of its own street, which is how the
solver's routing graph places zones (`inputs.pc_solver.graph.zone_position`), and
(iii) give GAUS a set of dangling ends that `touches` would connect differently.
Dropping whole segments keeps `(node1, node2)` stable and every surviving row
bit-identical to the reference one, so the variant network is a subgraph of the
reference network and every difference downstream is the extent and nothing else.

R is rebuilt, not carried across
--------------------------------
R is the buildable area of a non-overlapping 30 m buffer, so a segment's weight
depends on which neighbours it competes with. Clip the network and a boundary
segment stops losing half its strip to a neighbour that is no longer there. Each
variant's R is therefore rebuilt on the clipped network with the bundle's own
`rebuild_segment_areas.py`, and the **segment enumeration order** is recorded in
the manifest, because R is not a pure function of the geometry. The
reversed-order pass is run on every variant for the same reason, so the size of
the order dependence is a measured number on the variant too.

Writes, per variant `<label>`:
    data/interim/boundary_<label>_network.gpkg   layer `boundary_<label>_network`
        — node1, node2, dist, speed, area (rebuilt R), geometry. The
        `inputs.base_network` analogue: what src/15_boundary_pc_runs.py feeds
        the solver.
    data/interim/boundary_<label>_edges.gpkg     layer `boundary_<label>_edges`
        — every attribute of the surviving rows of the file of record, the
        input for GAUS Lines v1.1 (`make gaus-native`).
    data/interim/boundary_<label>_manifest.json
    data/interim/boundary_extents.gpkg           layer `boundary_extents` — the
        reference extent and every variant polygon, for the map and the record.
    outputs/tables/stage0_boundary_variants.csv
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="14_boundary_variants.py",
        description="Build the negative-buffer boundary-extent variants.",
    )
    ap.add_argument("--only", nargs="*", default=None, metavar="LABEL")
    ap.add_argument("--list", action="store_true", help="print the variant grid")
    ap.add_argument(
        "--no-reversed",
        action="store_true",
        help="skip the reversed-order R rebuild that measures the order dependence",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "boundary_variants")
    cfg = ctx.cfg
    bcfg = cfg["revision"]["boundary_sensitivity"]
    ecfg = bcfg["extent"]
    variants = bcfg["variants"] or []
    if not variants:
        ctx.log.error("revision.boundary_sensitivity.variants is empty")
        return 2
    if args.list:
        for v in variants:
            print(f"{v['label']:>8}  inward_buffer_m={v['inward_buffer_m']:<8}")
        return 0
    wanted = [str(v["label"]) for v in variants]
    if args.only:
        unknown = [x for x in args.only if x not in wanted]
        if unknown:
            raise SystemExit(f"unknown label(s) {unknown}; configured: {wanted}")
        wanted = [x for x in wanted if x in set(args.only)]

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

    # -- the reference extent -------------------------------------------------
    buf_m = float(ecfg["buffer_m"])
    qs = int(ecfg["quad_segs"])
    t0 = time.time()
    union = shapely.union_all(shapely.buffer(gdf.geometry.values, buf_m, quad_segs=qs))
    union_area = float(shapely.area(union))
    n_rings = sum(shapely.get_num_interior_rings(g) for g in shapely.get_parts(union))
    # The footprint: the same union with its interior rings dropped. See the
    # module docstring — the rings are block interiors, and eroding from them is
    # not what shrinking a study area means.
    reference_extent = shapely.union_all(
        shapely.polygons(
            [shapely.get_exterior_ring(g) for g in shapely.get_parts(union)]
        )
    )
    extent_seconds = time.time() - t0
    ref_area = float(shapely.area(reference_extent))
    ctx.log.info(
        "reference extent in %.1fs: union of the %.0f m buffers of %d segments = "
        "%.4f km2 over %d part(s) with %d interior ring(s); footprint (rings "
        "dropped) = %.4f km2 over %d part(s)",
        extent_seconds,
        buf_m,
        len(gdf),
        union_area / 1e6,
        shapely.get_num_geometries(union),
        n_rings,
        ref_area / 1e6,
        shapely.get_num_geometries(reference_extent),
    )
    ctx.count("reference extent: union area (m2)", round(union_area), ecfg["rule"])
    ctx.count(
        "reference extent: footprint area (m2)",
        round(ref_area),
        "interior rings dropped",
    )

    rows: list[dict[str, Any]] = []
    extent_rows: list[dict[str, Any]] = [
        {
            "label": "reference_union",
            "inward_buffer_m": 0.0,
            "area_m2": union_area,
            "area_km2": union_area / 1e6,
            "n_parts": int(shapely.get_num_geometries(union)),
            "n_segments": len(gdf),
            "geometry": union,
        },
        {
            "label": "reference",
            "inward_buffer_m": 0.0,
            "area_m2": ref_area,
            "area_km2": ref_area / 1e6,
            "n_parts": int(shapely.get_num_geometries(reference_extent)),
            "n_segments": len(gdf),
            "geometry": reference_extent,
        },
    ]

    bundle = common.REPO / cfg["inputs"]["pc_solver"]["reconstruction_bundle"]
    rebuild_script = bundle / "rebuild_segment_areas.py"
    if not rebuild_script.exists():
        ctx.log.error("rebuild script missing from the bundle: %s", rebuild_script)
        return 2
    rebuild = _load_module("rebuild_segment_areas", rebuild_script)

    net_cols = cfg["inputs"]["base_network"]["columns"]
    id_cols = list(net_cols["segment_id"])
    reference_R = gdf[net_cols["static_weight"]].astype(float).to_numpy()

    for spec in variants:
        label = str(spec["label"])
        if label not in wanted:
            continue
        d = float(spec["inward_buffer_m"])
        if d >= 0:
            raise SystemExit(f"variant '{label}': inward_buffer_m must be negative")
        ctx.log.info("=" * 74)
        poly = shapely.buffer(reference_extent, d, quad_segs=qs)
        area = float(shapely.area(poly))
        n_parts = int(shapely.get_num_geometries(poly))
        if area <= 0:
            ctx.log.error(
                "variant '%s': the %.0f m inward buffer empties the extent", label, d
            )
            return 1

        # `within`, not `intersects`: whole segments only.
        keep = shapely.within(gdf.geometry.values, poly)
        sub = gdf.loc[keep].copy()
        n_keep = int(keep.sum())
        ctx.log.info(
            "%s: inward %.0f m -> area %.4f km2 (%.2f %% of reference), %d part(s); "
            "%d of %d segments fully inside (%.2f %%)",
            label,
            d,
            area / 1e6,
            100 * area / ref_area,
            n_parts,
            n_keep,
            len(gdf),
            100 * n_keep / len(gdf),
        )
        ctx.count(f"{label}: segments retained", n_keep, f"inward buffer {d:.0f} m")
        if n_keep == 0:
            ctx.log.error("variant '%s' retains no segment", label)
            return 1

        # -- R rebuilt on the clipped network ----------------------------------
        # Enumeration order = the surviving rows in the file-of-record row order,
        # which is the export's own `G.edges` order with the dropped rows removed.
        order_note = (
            "row order of inputs.segment_centralities (= the export's `G.edges` "
            "order) restricted to the segments fully inside the variant polygon"
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

        # how far the rebuilt R moves from the shipped `area` on the same rows —
        # part of it is the clip, part is the rebuild's own disagreement with the
        # shipped column on the full network (`make rebuild-areas`).
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
                "inward_buffer_m": d,
                "area_m2": area,
                "area_km2": area / 1e6,
                "n_parts": n_parts,
                "n_segments": n_keep,
                "geometry": poly,
            }
        )
        row = {
            "label": label,
            "inward_buffer_m": d,
            "extent_rule": ecfg["rule"],
            "clip_rule": bcfg["clip_rule"],
            "reference_union_area_km2": union_area / 1e6,
            "reference_footprint_area_km2": ref_area / 1e6,
            "area_m2": area,
            "area_km2": area / 1e6,
            "area_share_of_reference": area / ref_area,
            "n_parts": n_parts,
            "n_segments": n_keep,
            "n_segments_dropped": len(gdf) - n_keep,
            "segment_share_of_reference": n_keep / len(gdf),
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
            "produced_by": "src/14_boundary_variants.py",
            "what": (
                "a boundary-extent variant: the reference extent (union of the "
                f"{buf_m:.0f} m segment buffers) buffered inward by {d:.0f} m, with "
                "every segment not fully inside dropped and R rebuilt on what is left"
            ),
            "extent": {
                "rule": ecfg["rule"],
                "buffer_m": buf_m,
                "interior_rings_dropped": n_rings,
                "reference_union_area_m2": union_area,
                "reference_union_area_km2": union_area / 1e6,
                "reference_area_m2": ref_area,
                "reference_area_km2": ref_area / 1e6,
                "inward_buffer_m": d,
                "variant_area_m2": area,
                "variant_area_km2": area / 1e6,
                "area_share_of_reference": area / ref_area,
                "n_parts": n_parts,
                "crs": source_crs,
            },
            "clip": {
                "rule": bcfg["clip_rule"],
                "predicate": "shapely.within(segment, variant_polygon)",
                "geometry_modified": False,
                "n_segments_in": len(gdf),
                "n_segments_kept": n_keep,
                "n_segments_dropped": len(gdf) - n_keep,
                "id_columns": id_cols,
                "id_columns_stable": True,
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
        ctx.log.info("next: python src/15_boundary_pc_runs.py --only %s", label)

    ext = gpd.GeoDataFrame(extent_rows, geometry="geometry", crs=gdf.crs)
    stage2.write_layer(ctx, ext, "boundary_extents.gpkg")
    pd.DataFrame(rows).to_csv(ctx.table("stage0_boundary_variants.csv"), index=False)
    ctx.log.info(
        "wrote %s", ctx.table("stage0_boundary_variants.csv").relative_to(common.REPO)
    )
    ctx.write_counts("stage0_boundary_variants_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
