"""Recompute the segment capacities (the static weight R) from the network alone.

The submitted `area` column is the area of a **non-overlapping 30 m buffer**
around each eligible street segment, less the road surface itself. This module
reimplements that construction, as done by the (unpublished) model-building
code of the submitted run:

  * every eligible segment gets a 30 m buffer and a zone at its midpoint;
  * overlaps are removed with a Voronoi partition, not by pairwise difference:
    points are sampled every 5 m along every eligible segment, one global
    Voronoi diagram is built over all of them, each segment claims the cells of
    its own points, and that union is clipped to its own 30 m buffer. Two
    segments therefore split the space between them along the perpendicular
    bisector, and no area is counted twice;
  * the road surface of every physical, non-tunnel segment (a buffer whose
    width depends on the road class, `IMPEDIMENT`) is subtracted, so what
    remains is buildable area.

A segment is eligible if its `highway` class is in `ALLOWED_HIGHWAY`, its speed
is at most `MAX_SPEED` and it is neither a bridge nor a tunnel -- the filter of
the submitted model. A zone whose area ends below `MIN_AREA` is dropped, which
is why the submitted column's smallest positive value is 10.018 and why 2,959
otherwise-eligible segments carry area 0.

The input must carry the columns `highway`, `speed`, `road_type`, `tunnel`,
`bridge` and `drive` as well as the geometry, i.e. `edges_combined.shp`; the
base network GeoPackage does not have them.

    python rebuild_segment_areas.py data/frozen/segments/edges_combined.shp \
        [--bbox xmin,ymin,xmax,ymax] [--out areas.npz]
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Zone eligibility of the submitted model: allowed road classes, maximum
# speed (km/h) and the attributes that exclude a segment.
ALLOWED_HIGHWAY = (
    "primary", "secondary", "tertiary", "unclassified", "residential", "service",
    "living street", "living_street", "pedestrian", "bicycle road", "track", "road",
)
MAX_SPEED = 61
FORBIDDEN_ATTRS = ("tunnel", "bridge")
BUFFER = 30.0        # zone buffer, metres
EPSILON = 1.0        # keep off the exact endpoints
MIN_SPACING = 5.0    # point every 5 m along the line
MIN_AREA = 10.0      # below this the zone is deleted

# The road surface itself: buffer width in metres by road class. Subtracted from
# the zone buffers so that what remains is buildable area; applied over every
# physical non-tunnel segment, motorways included.
IMPEDIMENT = {
    "motorway": 30, "trunk": 10, "primary": 10, "secondary": 10, "tertiary": 5,
    "unclassified": 5, "residential": 5, "service": 5, "motorway_link": 10,
    "trunk_link": 10, "primary_link": 10, "secondary_link": 10, "tertiary_link": 5,
    "motorway_junction": 10, "living street": 5, "living_street": 5, "pedestrian": 5,
    "bicycle road": 5, "track": 0, "bus_guideway": 5, "raceway": 30, "road": 5,
    "construction": 5, "escape": 5, "turning_circle": 5, "dismantled": 0, "planned": 0,
    "rest_area": 5, "razed": 0, "unsurfaced": 5, "bus_stop": 0, "yes": 5, "disused": 0,
    "cycleway": 5, "corridor": 0, "footway": 0, "path": 0, "steps": 0, "services": 0,
    "elevator": 0, "escalator": 0, "desire_path": 0, "disused_path": 0,
    "emergency_access_point": 0, "footway-disabled": 0, "crossing": 0, "@ser": 0,
    "via_ferrata": 0,
}
IMPEDIMENT_DEFAULT = 5   # width for a road class not listed above


def truthy(series) -> np.ndarray:
    """Boolean attribute test; the shapefile stores the flags as strings."""
    s = series.astype(str).str.strip().str.lower()
    return s.isin(("true", "yes", "1", "t")).to_numpy()


def eligible(gdf) -> np.ndarray:
    hw = gdf["highway"].astype(str).to_numpy()
    ok = np.isin(hw, ALLOWED_HIGHWAY) & (gdf["speed"].astype(float).to_numpy() <= MAX_SPEED)
    for attr in FORBIDDEN_ATTRS:
        if attr in gdf.columns:
            ok &= ~truthy(gdf[attr])
    if "drive" in gdf.columns:            # segments not open to driving are excluded
        ok &= ~(gdf["drive"].astype(str).str.strip().str.lower() == "false").to_numpy()
    return ok


def impediment_parts(gdf) -> list[Any]:
    """The road surfaces, one buffered polygon per segment (unioned lazily, per zone).

    A single global union over 65k buffers is neither necessary nor affordable: each zone
    buffer meets only its own neighbourhood, so the parts are indexed and unioned per
    zone. The result is identical -- difference against a union equals difference against
    the members that actually intersect.
    """
    keep = np.ones(len(gdf), dtype=bool)
    if "road_type" in gdf.columns:
        keep &= (gdf["road_type"].astype(str) == "physical").to_numpy()
    if "tunnel" in gdf.columns:
        keep &= ~truthy(gdf["tunnel"])
    hw = gdf["highway"].astype(str).to_numpy()
    widths = np.array([IMPEDIMENT.get(h, IMPEDIMENT_DEFAULT) for h in hw], dtype=float)
    return [g.buffer(w) for g, w, k in
            zip(gdf.geometry, widths, keep, strict=True)
            if k and w > 0 and g is not None and not g.is_empty and g.length > 0]


def sample_points(lines) -> tuple[list[tuple[float, float]], list[int]]:
    """Points every MIN_SPACING along each line, EPSILON in from both ends."""
    points: list[tuple[float, float]] = []
    origins: list[int] = []
    for idx, line in enumerate(lines):
        n = int((line.length - 2 * EPSILON) / MIN_SPACING)
        for i in range(1, n):
            p = line.interpolate(EPSILON + i * MIN_SPACING)
            points.append((p.x, p.y))
            origins.append(idx)
        a = line.interpolate(EPSILON)
        b = line.interpolate(line.length - EPSILON)
        points.append((a.x, a.y))
        origins.append(idx)
        points.append((b.x, b.y))
        origins.append(idx)
    return points, origins


def dedupe(points, origins):
    """First occurrence wins, both exactly and at whole-metre rounding."""
    seen_exact: dict[tuple[float, float], int] = {}
    seen_round: set[tuple[int, int]] = set()
    for p, o in zip(points, origins, strict=True):
        r = (round(p[0]), round(p[1]))
        if p not in seen_exact and r not in seen_round:
            seen_exact[p] = o
            seen_round.add(r)
    return list(seen_exact.keys()), list(seen_exact.values())


def finite_voronoi_polygons(vor: Voronoi) -> list[Polygon]:
    """Voronoi regions as polygons, with the unbounded regions closed off."""
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    radius = float(np.ptp(vor.points))
    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    regions = []
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            regions.append(vertices)
            continue
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        regions.append(np.array(new_region)[np.argsort(angles)].tolist())
    return [Polygon(np.array([new_vertices[v] for v in r])) for r in regions]


def rebuild_areas(gdf, *, use_impediment: bool = True, verbose: bool = True) -> np.ndarray:
    """Return the non-overlapping buffer area per row; 0 where the row is not a zone."""
    sel = eligible(gdf)
    rows = np.flatnonzero(sel)
    lines = list(gdf.geometry.iloc[rows])
    if verbose:
        print(f"  eligible segments: {len(rows)} of {len(gdf)}", flush=True)

    t = time.time()
    pts, origins = sample_points(lines)
    pts, origins = dedupe(pts, origins)
    if verbose:
        print(f"  {len(pts)} sample points after dedupe ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    vor = Voronoi(np.array(pts))
    cells = finite_voronoi_polygons(vor)
    if verbose:
        print(f"  voronoi + finite cells: {time.time()-t:.1f}s", flush=True)

    t = time.time()
    buffers = [ln.buffer(BUFFER) for ln in lines]
    by_origin: dict[int, list[Polygon]] = {}
    for cell, o in zip(cells, origins, strict=True):
        by_origin.setdefault(o, []).append(cell)

    area = np.zeros(len(gdf), dtype=np.float64)
    trimmed: dict[int, Any] = {}
    for k, (o, polys) in enumerate(by_origin.items()):
        if verbose and k and k % 10000 == 0:
            print(f"    clipped {k} of {len(by_origin)} ({time.time()-t:.0f}s)", flush=True)
        buf = buffers[o]
        if buf.is_empty or not buf.is_valid:
            continue
        parts = [p.intersection(buf) for p in polys
                 if not p.is_empty and p.is_valid and p.intersects(buf)]
        if parts:
            geom = unary_union(parts)
            trimmed[rows[o]] = geom
            area[rows[o]] = geom.area
    if verbose:
        print(f"  clip + union: {time.time()-t:.1f}s", flush=True)

    if use_impediment:
        t = time.time()
        parts = impediment_parts(gdf)
        if parts:
            tree = STRtree(parts)
            for k, i in enumerate(np.flatnonzero(area > 0)):
                geom = trimmed[i]
                hits = tree.query(geom)
                if len(hits):
                    geom = geom.difference(unary_union([parts[j] for j in hits]))
                area[i] = geom.area
                if verbose and k and k % 10000 == 0:
                    print(f"    impediment {k} zones ({time.time()-t:.0f}s)", flush=True)
        if verbose:
            print(f"  impediment subtraction: {time.time()-t:.1f}s", flush=True)

    dropped = int(((area > 0) & (area < MIN_AREA)).sum())
    area[area < MIN_AREA] = 0.0
    if verbose:
        print(f"  dropped below {MIN_AREA} m²: {dropped}; zones: {int((area>0).sum())}",
              flush=True)
    return area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("network")
    ap.add_argument("--layer", default=None)
    ap.add_argument("--bbox", default=None, help="xmin,ymin,xmax,ymax — windowed run")
    ap.add_argument("--reference-col", default="area", help="compare against, if present")
    ap.add_argument("--out", default=None, help="write areas to this .npz")
    ap.add_argument("--no-impediment", action="store_true",
                    help="skip the road-surface subtraction (use_impediment=False)")
    args = ap.parse_args()

    kw = {"layer": args.layer} if args.layer else {}
    gdf = gpd.read_file(args.network, **kw)
    if args.bbox:
        xmin, ymin, xmax, ymax = (float(v) for v in args.bbox.split(","))
        gdf = gdf.cx[xmin:xmax, ymin:ymax].reset_index(drop=True)
        print(f"windowed to {len(gdf)} segments")
    t0 = time.time()
    area = rebuild_areas(gdf, use_impediment=not args.no_impediment)
    print(f"total {time.time()-t0:.1f}s")

    if args.reference_col in gdf.columns:
        ref = gdf[args.reference_col].astype(float).to_numpy()
        both = (area > 0) & (ref > 0)
        rel = np.abs(area[both] - ref[both]) / ref[both]
        print(f"\nvs `{args.reference_col}`: both-positive {both.sum()}, "
              f"zone-set agreement {np.mean((area>0)==(ref>0)):.4%}")
        print(f"   median rel {np.median(rel):.4g}  p90 {np.percentile(rel,90):.4g}  "
              f"max {rel.max():.4g}   within 1% {np.mean(rel<1e-2):.2%}")
    if args.out:
        np.savez(args.out, area=area)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
