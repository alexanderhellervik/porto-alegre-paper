"""Build the routing graph of a preferential-centrality run from a segment file.

The segment file written by the solver (`edges_combined.shp`, or the base
network GeoPackage derived from it) carries the whole segment network: `node1`,
`node2`, `dist`, `speed` and the capacity column `area`. This rebuilds the
routing graph from those five columns.

The graph rests on three properties of the model that built the submitted run:

  * a zone sits at the **midpoint** of its own segment;
  * its connector to the road has **dist exactly 0**;
  * splitting a segment preserves its length exactly.

So zone-to-zone cost is the shortest path over the segment network with every
segment split at its midpoint. Midpoints are inserted on *all* segments, not
just zone-bearing ones: subdividing an edge cannot change shortest paths, and
it keeps the index arithmetic trivial. Zero-capacity segments stay in the graph
and carry routes without being zones.

Node space: segment endpoints occupy 0..n_ends-1, midpoints
n_ends..n_ends+n_segments-1, so the midpoint of row i is always `n_ends + i`.

Self-loop segments (`node1 == node2`, 53 of them in the submitted network)
contribute two identical undirected half-edges. prefcent resolves duplicate
edges by the **minimum**, giving cost/2; summing them would double the cost of
reaching 47 zones.

    python extract_network_from_shapefile.py <segments.shp|.gpkg> \
        [--out data/interim/prefcent_reconstruction/paper_network]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import numpy as np
import pyogrio

# The repository root (this file lives in data/frozen/network/). Paths inside
# it are recorded relative to it, so the metadata carries no machine paths.
REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Cost units of the solver that built the submitted run: a cost of 1000
# represents driving 1000 m at 60 km/h, i.e. one minute, and the start-and-end
# penalty d0 is expressed in the same units (5000 = five minutes).
SPEED_REFERENCE_KMH = 60.0
COST_UNITS = "metre-equivalents at 60 km/h (1000 per minute)"


def repo_relative(path: str) -> str:
    """`path` relative to the repository root when it lies inside it."""
    full = os.path.abspath(path)
    if os.path.commonpath([full, REPO]) == REPO:
        return os.path.relpath(full, REPO)
    return path


def _node_index(n1: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Map endpoint labels to 0-based indices, whatever dtype they arrive in."""
    both = np.concatenate([n1, n2])
    try:
        both = both.astype(np.int64)
    except (TypeError, ValueError):
        both = both.astype(str)
    uniq, inv = np.unique(both, return_inverse=True)
    m = len(n1)
    return inv[:m].astype(np.int64), inv[m:].astype(np.int64), len(uniq)


def extract(
    path: str,
    *,
    node_cols: tuple[str, str] = ("node1", "node2"),
    dist_col: str = "dist",
    speed_col: str = "speed",
    area_col: str = "area",
    reference_cols: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (arrays, metadata) for the routing graph of the segment file `path`.

    `arrays` holds the graph edges and half-edge costs, the graph index, row and
    id of every zone, the static weights R of the zones and, for each column in
    `reference_cols`, that column as `ref_<name>`. `metadata` records the source
    file (repository-relative), its sha256 and the graph's counts.
    """
    cols = list(dict.fromkeys([*node_cols, dist_col, speed_col, area_col, *reference_cols]))
    df = pyogrio.read_dataframe(path, read_geometry=False, columns=cols)
    info = pyogrio.read_info(path)
    n_seg = len(df)

    dist = df[dist_col].astype(np.float64).to_numpy()
    speed = df[speed_col].astype(np.float64).to_numpy()
    area = df[area_col].astype(np.float64).to_numpy()
    for name, arr in ((dist_col, dist), (speed_col, speed), (area_col, area)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
    if np.any(dist < 0):
        raise ValueError(f"{dist_col} must be ≥ 0")
    if np.any(speed <= 0):
        raise ValueError(f"{speed_col} must be > 0 (cost divides by it)")
    if np.any(area < 0):
        raise ValueError(f"{area_col} must be ≥ 0")

    e1, e2, n_ends = _node_index(
        df[node_cols[0]].to_numpy(), df[node_cols[1]].to_numpy()
    )
    mid = n_ends + np.arange(n_seg, dtype=np.int64)
    cost = dist / (speed / SPEED_REFERENCE_KMH)
    half = cost / 2.0

    edges = np.concatenate(
        [np.stack([e1, mid], axis=1), np.stack([e2, mid], axis=1)]
    ).astype(np.int64)
    weights = np.concatenate([half, half]).astype(np.float64)

    zone_rows = np.flatnonzero(area > 0.0)
    out: dict[str, Any] = {
        "graph_edges": edges,
        "graph_costs": weights,
        "zone_graph_index": mid[zone_rows],
        "zone_rows": zone_rows,
        "R": area[zone_rows],
        "segment_node1": df[node_cols[0]].to_numpy().astype(str),
        "segment_node2": df[node_cols[1]].to_numpy().astype(str),
        "zone_ids": np.array(
            [
                f"{a}_{b}"
                for a, b in zip(
                    df[node_cols[0]].to_numpy().astype(str)[zone_rows],
                    df[node_cols[1]].to_numpy().astype(str)[zone_rows],
                    strict=True,
                )
            ]
        ),
    }
    for c in reference_cols:
        out[f"ref_{c}"] = df[c].astype(np.float64).to_numpy()

    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    meta = {
        "source": repo_relative(path),
        "source_sha256": sha,
        "crs": str(info["crs"]),
        "n_segments": int(n_seg),
        "n_endpoints": int(n_ends),
        "graph_nodes": int(n_ends + n_seg),
        "graph_edges": int(edges.shape[0]),
        "n_zones": int(zone_rows.size),
        "zero_capacity_segments": int((area == 0.0).sum()),
        "self_loop_segments": int((e1 == e2).sum()),
        "self_loop_segments_with_zone": int(((e1 == e2) & (area > 0.0)).sum()),
        "sum_R": float(area[zone_rows].sum()),
        "speeds": sorted({float(s) for s in speed}),
        "cost_units": COST_UNITS,
        "speed_reference_kmh": SPEED_REFERENCE_KMH,
        "midpoint_rule": "zone at fraction 0.5 of its segment; connector cost 0",
        "reference_columns": list(reference_cols),
    }
    return out, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shapefile")
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "data", "interim", "prefcent_reconstruction",
                             "paper_network"),
    )
    ap.add_argument("--reference-cols", nargs="*", default=[])
    ap.add_argument("--area-col", default="area")
    args = ap.parse_args()

    out, meta = extract(
        args.shapefile,
        area_col=args.area_col,
        reference_cols=tuple(args.reference_cols),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out + ".npz", **out)
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    for k in ("n_segments", "n_endpoints", "graph_nodes", "graph_edges", "n_zones",
              "zero_capacity_segments", "self_loop_segments_with_zone", "sum_R"):
        print(f"  {k}: {meta[k]}")
    print(f"wrote {args.out}.npz ({os.path.getsize(args.out + '.npz')/2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
