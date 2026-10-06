"""Stage 2 — joins and aggregation onto the hexagon grid.

Three jobs, in order:

1. **Characterise the frozen 500 m grid.** Recover its lattice (CRS, orientation,
   origin, spacing, cell area) from `cents_aggregated.gpkg`, record it under
   `grid:` in `config.yaml`, and verify the recorded parameters regenerate the
   821 frozen cells.
2. **Test what the frozen `*_mean` / `*_sum` columns were aggregated from.**
   Hypothesis A: the frozen address points. Hypothesis B: the street segments of
   `inputs.segment_centralities`. The test is direct — join points to cells,
   aggregate every plausible way, compare cell by cell — and the segment layer
   is then checked against the per-cell unit counts the frozen columns imply.
3. **Produce the stage-2 aggregated dependent variable** from stage 1's rebuilt
   `Puni` variants on the 500 m grid and on the 250 m / 1,000 m MAUP grids,
   with the centralities of the MAUP grids aggregated from the segment layer.

Aggregation rule (paper §4.5; `grid.aggregation` in the config):
  accessibility measures (PC1, PC2, CC) -> MEAN
  intermediation measures (BC, FK)      -> SUM
  land value, plot area                 -> MEAN

Nothing here makes an analytical decision that is not in `config.yaml`.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import prep

MEASURES = ["PC1", "PC2", "CC", "BC", "FK"]

# frozen aggregated column stem per logical measure, from measures.columns.aggregated
FROZEN_STEM = {
    "PC1": "cd1g1b2k0",
    "PC2": "cd4g0b2k0",
    "CC": "GgAcc0",
    "BC": "GgBtw0",
    "FK": "GgCen0",
}


# ---------------------------------------------------------------------------
# grid geometry
# ---------------------------------------------------------------------------


def hexagon(cx: float, cy: float, circumradius: float, orientation: str) -> Polygon:
    """One regular hexagon centred on (cx, cy).

    `flat_top`: two horizontal edges (top and bottom), vertices due east and west.
    `pointy_top`: the 30-degree rotation of that.
    """
    if orientation == "flat_top":
        phase = 0.0
    elif orientation == "pointy_top":
        phase = math.pi / 6
    else:
        raise ValueError(f"unknown grid orientation: {orientation}")
    return Polygon(
        [
            (
                cx + circumradius * math.cos(phase + k * math.pi / 3),
                cy + circumradius * math.sin(phase + k * math.pi / 3),
            )
            for k in range(6)
        ]
    )


def lattice_centre(
    gcfg: dict[str, Any], spacing: float, row: np.ndarray, col: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Centre coordinates of cell (row, col) on a grid of the given spacing.

    Anchored at `grid.origin_x/origin_y`, the centre of cell (0, 0), at *any*
    spacing: the 250 m and 1,000 m MAUP grids share the 500 m grid's origin,
    orientation and CRS.
    """
    col_dx = spacing * math.sqrt(3.0) / 2.0  # = 1.5 x circumradius
    cx = gcfg["origin_x"] + np.asarray(col) * col_dx
    cy = (
        gcfg["origin_y"]
        - np.asarray(row) * spacing
        - (np.asarray(col) % 2) * (spacing / 2.0)
    )
    return cx, cy


def build_grid(
    gcfg: dict[str, Any],
    spacing: float,
    bounds: tuple[float, float, float, float],
    clip_to: Any | None = None,
) -> gpd.GeoDataFrame:
    """Tessellate `bounds` with hexagons on the config lattice, then clip."""
    circumradius = spacing / math.sqrt(3.0)
    col_dx = spacing * math.sqrt(3.0) / 2.0
    minx, miny, maxx, maxy = bounds
    c0 = math.floor((minx - circumradius - gcfg["origin_x"]) / col_dx)
    c1 = math.ceil((maxx + circumradius - gcfg["origin_x"]) / col_dx)
    r0 = math.floor((gcfg["origin_y"] - maxy - spacing) / spacing)
    r1 = math.ceil((gcfg["origin_y"] - miny + spacing) / spacing)
    cols = np.repeat(np.arange(c0, c1 + 1), r1 - r0 + 1)
    rows = np.tile(np.arange(r0, r1 + 1), c1 - c0 + 1)
    cx, cy = lattice_centre(gcfg, spacing, rows, cols)
    g = gpd.GeoDataFrame(
        {
            "row_index": rows,
            "col_index": cols,
            "centre_x": cx,
            "centre_y": cy,
            "geometry": [
                hexagon(x, y, circumradius, gcfg["orientation"]) for x, y in zip(cx, cy)
            ],
        },
        crs=f"EPSG:{gcfg['crs_epsg']}",
    )
    if clip_to is not None:
        g["geometry"] = g.geometry.intersection(clip_to)
        g = g[~g.geometry.is_empty & (g.geometry.area >= gcfg["min_cell_area_m2"])]
    return g.reset_index(drop=True)


def characterise_frozen_grid(
    ctx: common.Context, agg: gpd.GeoDataFrame
) -> dict[str, Any]:
    """Recover the lattice of the frozen 500 m grid from its own geometry."""
    gcfg = ctx.cfg["grid"]
    spacing = float(gcfg["spacing_m"])
    circumradius = spacing / math.sqrt(3.0)
    full_area = 3.0 * math.sqrt(3.0) / 2.0 * circumradius**2

    bbox_cx = (agg["left"].to_numpy() + agg["right"].to_numpy()) / 2.0
    bbox_cy = (agg["top"].to_numpy() + agg["bottom"].to_numpy()) / 2.0
    is_full = np.isclose(agg.geometry.area.to_numpy(), full_area, rtol=1e-9)
    ctx.count(
        "grid: unclipped (full-area) cells",
        int(is_full.sum()),
        f"{int((~is_full).sum())} clipped at the study-area edge",
    )

    cx, cy = lattice_centre(
        gcfg, spacing, agg["row_index"].to_numpy(), agg["col_index"].to_numpy()
    )
    centre_offset = np.hypot(cx - bbox_cx, cy - bbox_cy)
    gen = gpd.GeoSeries(
        [hexagon(x, y, circumradius, gcfg["orientation"]) for x, y in zip(cx, cy)],
        crs=agg.crs,
        index=agg.index,
    )
    hausdorff = np.array(
        [a.hausdorff_distance(b) for a, b in zip(gen[is_full], agg.geometry[is_full])]
    )
    tol = float(gcfg["validation"]["geometry_tol_m"])
    outside = np.array(
        [
            b.difference(a.buffer(tol)).area
            for a, b in zip(gen[~is_full], agg.geometry[~is_full])
        ]
    )

    # the frozen `Area` column is compared with the ellipsoidal area
    geod = Geod(ellps="WGS84")
    geod_area = np.array(
        [abs(geod.geometry_area_perimeter(g)[0]) for g in agg.to_crs(4326).geometry]
    )
    area_col = gcfg["frozen_500m"]["area_column"]
    geod_match = float(
        np.max(np.abs(geod_area - agg[area_col].to_numpy()) / agg[area_col].to_numpy())
    )

    return {
        "crs_epsg": int(gcfg["crs_epsg"]),
        "orientation": gcfg["orientation"],
        "spacing_m": spacing,
        "circumradius_m": circumradius,
        "edge_length_m": circumradius,
        "across_flats_m": spacing,
        "across_corners_m": 2 * circumradius,
        "col_dx_m": spacing * math.sqrt(3.0) / 2.0,
        "row_dy_m": -spacing,
        "odd_col_dy_m": -spacing / 2.0,
        "origin_x": float(gcfg["origin_x"]),
        "origin_y": float(gcfg["origin_y"]),
        "n_cells": len(agg),
        "n_full_cells": int(is_full.sum()),
        "n_clipped_cells": int((~is_full).sum()),
        "planar_cell_area_m2": full_area,
        "planar_cell_area_ha": full_area / 1e4,
        "ellipsoidal_cell_area_ha_median": float(np.median(geod_area) / 1e4),
        "ellipsoidal_vs_frozen_Area_max_rel": geod_match,
        "max_centroid_offset_m": float(centre_offset.max()),
        "max_hausdorff_full_cells_m": float(hausdorff.max()) if len(hausdorff) else 0.0,
        "max_clipped_area_outside_generated_m2": float(outside.max())
        if len(outside)
        else 0.0,
    }


# ---------------------------------------------------------------------------
# the point -> cell join
# ---------------------------------------------------------------------------


def points_from(df: pd.DataFrame, xcol: str, ycol: str, crs: Any) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.drop(columns=[c for c in ("geometry",) if c in df.columns]),
        geometry=gpd.points_from_xy(df[xcol], df[ycol]),
        crs=crs,
    )


def join_points(
    ctx: common.Context, pts: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, label: str
) -> gpd.GeoDataFrame:
    """Point-in-polygon join, with the unmatched counted, never dropped silently."""
    j = gpd.sjoin(pts, grid[["cell_id", "geometry"]], how="left", predicate="within")
    j = j.drop(columns="index_right", errors="ignore")
    # `within` needs the point in a cell's interior, so a point exactly on a
    # shared edge matches no cell and is counted as outside below. A point can
    # only match two cells if cells overlap; keep the first match in that case.
    j = j[~j.index.duplicated(keep="first")]
    n_out = int(j["cell_id"].isna().sum())
    ctx.count(
        f"join {label}: points inside a cell",
        int(len(j) - n_out),
        f"{n_out} of {len(j)} outside every cell",
    )
    return j


# ---------------------------------------------------------------------------
# hypothesis A — were the frozen columns aggregated from the address points?
# ---------------------------------------------------------------------------


def role(ctx: common.Context, src: str) -> str:
    m = ctx.cfg["measures"]
    if src in m["accessibility"]:
        return "accessibility"
    if src in m["intermediation"]:
        return "intermediation"
    return {"Puni": "land_value", "Terreno": "plot_area"}.get(src, "")


def universes(ctx: common.Context, pt: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    """The three point universes the frozen columns could have been built on."""
    ex = ctx.cfg["exclusions"]
    m = pd.Series(True, index=pt.index)
    for c in ex["positivity"]["disaggregated"]:
        m &= pt[c].notna() & (pt[c] > 0)
    u_pos = pt[m]
    v = u_pos["Puni"]
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    k = float(ex["iqr_outliers"]["multiplier"])
    u_smp = u_pos[(v >= q1 - k * (q3 - q1)) & (v <= q3 + k * (q3 - q1))]
    out = {
        f"all_{len(pt)}": pt,
        f"positive_{len(u_pos)}": u_pos,
        f"sample_{len(u_smp)}": u_smp,
    }
    for name, d in out.items():
        ctx.count(f"hypA universe: {name}", len(d))
    return out


def match_shares(
    ours: pd.Series, frozen: pd.Series, exact_tol: float, within_tol: float
) -> dict[str, Any]:
    c = pd.DataFrame({"o": ours, "f": frozen}).dropna()
    if c.empty:
        return {
            "n": 0,
            "exact": np.nan,
            "within": np.nan,
            "median_rel": np.nan,
            "median_ratio": np.nan,
        }
    denom = c["f"].abs().replace(0.0, np.nan)
    rel = (c["o"] - c["f"]).abs() / denom
    return {
        "n": len(c),
        "exact": float((rel < exact_tol).mean()),
        "within": float((rel < within_tol).mean()),
        "median_rel": float(rel.median()),
        "median_ratio": float((c["o"] / c["f"].replace(0.0, np.nan)).median()),
    }


def hypothesis_a(
    ctx: common.Context, pt: gpd.GeoDataFrame, agg: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the points every plausible way; score against the frozen columns."""
    gcfg = ctx.cfg["grid"]
    vcfg = gcfg["validation"]
    exact_tol, within_tol = float(vcfg["exact_rel_tol"]), float(vcfg["within_rel_tol"])
    rule = gcfg["aggregation"]
    frozen = agg.set_index("cell_id")

    targets: list[tuple[str, str, str]] = []
    for m in MEASURES:
        for stat in ("mean", "sum"):
            targets.append((m, f"{FROZEN_STEM[m]}_{stat}", stat))
    targets += [
        ("Puni", "Puni_mean", "mean"),
        ("Puni", "Puni_mean", "sum"),
        ("Puni", "Puni_median", "median"),
        ("Terreno", "Terreno_mean", "mean"),
        ("Terreno", "Terreno_mean", "sum"),
    ]

    rows = []
    for uname, U in universes(ctx, pt).items():
        for pos_name, (xc, yc) in gcfg["point_positions"].items():
            j = join_points(
                ctx, points_from(U, xc, yc, agg.crs), agg, f"hypA {uname}/{pos_name}"
            ).dropna(subset=["cell_id"])
            for dedupe in ("none", "by_snapped_point"):
                d = (
                    j.drop_duplicates(subset=["cell_id", "nearest_x", "nearest_y"])
                    if dedupe == "by_snapped_point"
                    else j
                )
                gr = d.groupby("cell_id")
                for src, fcol, stat in targets:
                    if fcol not in frozen.columns:
                        continue
                    rows.append(
                        {
                            "universe": uname,
                            "point_position": pos_name,
                            "dedupe": dedupe,
                            "source_column": src,
                            "frozen_column": fcol,
                            "aggregation": stat,
                            "is_paper_rule": stat == rule.get(role(ctx, src)),
                            **match_shares(
                                getattr(gr[src], stat)(),
                                frozen[fcol],
                                exact_tol,
                                within_tol,
                            ),
                        }
                    )
    sweep = pd.DataFrame(rows)

    # `sum / mean` of a frozen column IS the number of units that were aggregated
    stems = sorted(
        {c.rsplit("_", 1)[0] for c in frozen.columns if c.endswith("_sum")}
        & {c.rsplit("_", 1)[0] for c in frozen.columns if c.endswith("_mean")}
    )
    imp = pd.DataFrame(
        {
            s: frozen[f"{s}_sum"] / frozen[f"{s}_mean"].replace(0.0, np.nan)
            for s in stems
        }
    )
    consensus = imp.round().mode(axis=1)[0]

    ref = join_points(
        ctx,
        points_from(
            pt, *gcfg["point_positions"][gcfg["active_point_position"]], agg.crs
        ),
        agg,
        "implied-N reference",
    ).dropna(subset=["cell_id"])
    gr = ref.groupby("cell_id")
    implied = pd.DataFrame(
        {
            "implied_units": consensus,
            "max_integer_deviation": (imp - imp.round()).abs().max(axis=1),
            "measures_disagreeing": (imp.round().sub(consensus, axis=0).abs() > 0).sum(
                axis=1
            ),
            "n_address_points": gr.size().reindex(frozen.index).fillna(0).astype(int),
            "n_distinct_snapped_points": (
                ref.drop_duplicates(["cell_id", "nearest_x", "nearest_y"])
                .groupby("cell_id")
                .size()
                .reindex(frozen.index)
                .fillna(0)
                .astype(int)
            ),
        }
    )
    implied.index.name = "cell_id"
    implied.attrs["n_distinct_snapped_points"] = int(
        pt.groupby(["nearest_x", "nearest_y"]).ngroups
    )
    implied.attrs["n_distinct_centrality_tuples"] = int(pt.groupby(MEASURES).ngroups)
    implied.attrs["n_column_pairs"] = len(stems)
    return sweep, implied.reset_index()


# ---------------------------------------------------------------------------
# segment-level centralities
# ---------------------------------------------------------------------------

SEG_IDX = "_seg_idx"


def load_segment_centralities(
    ctx: common.Context, spec: dict[str, Any] | None = None
) -> gpd.GeoDataFrame:
    """Read the segment layer named by `inputs.segment_centralities`, in grid CRS.

    The frozen centrality columns are segment aggregates, not point aggregates
    (section 2 of the stage-2 report), so the centralities of any other grid are
    built from this layer and never from the address points.

    `spec` overrides the config block with one of the same shape. That is how a
    PC beta variant (`revision.pc_sensitivity.input_files`) or a boundary variant
    reaches the grid: the aggregation is the submitted one either way
    (`intersects`), and running a variant through a second implementation of it
    would make the variant and the submitted column incomparable for a reason
    that has nothing to do with the variant.
    """
    spec = spec if spec is not None else ctx.cfg["inputs"]["segment_centralities"]
    src = Path(spec["path"])
    if not src.is_absolute():
        src = common.REPO / src
    if not src.exists():
        raise FileNotFoundError(f"segment centrality layer not found: {src}")
    seg = gpd.read_file(src, layer=spec["layer"]).to_crs(
        f"EPSG:{ctx.cfg['grid']['crs_epsg']}"
    )
    return seg.reset_index(names=SEG_IDX)


def assign_segments(
    ctx: common.Context,
    seg: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    rule: str,
    label: str,
) -> pd.DataFrame:
    """Map segments to cells under `rule`; returns (`_seg_idx`, `cell_id`) pairs.

    `intersects` is the submitted rule: a segment belongs to **every** cell its
    geometry touches, so a segment crossing a boundary is counted in both. The
    point rules assign each segment to exactly one cell and are kept because
    they are the obvious alternatives the acceptance tests score against it.
    """
    if rule == "intersects":
        j = gpd.sjoin(
            seg[[SEG_IDX, "geometry"]],
            grid[["cell_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        out = j[[SEG_IDX, "cell_id"]].reset_index(drop=True)
    elif rule in ("midpoint", "centroid", "representative_point"):
        if rule == "midpoint":
            geom = seg.geometry.interpolate(0.5, normalized=True)
        elif rule == "centroid":
            geom = seg.geometry.centroid
        else:
            geom = seg.geometry.representative_point()
        pts = gpd.GeoDataFrame(seg[[SEG_IDX]], geometry=geom, crs=seg.crs)
        j = join_points(ctx, pts, grid, f"{label} ({rule})")
        out = j.loc[j["cell_id"].notna(), [SEG_IDX, "cell_id"]].reset_index(drop=True)
    elif rule == "length_majority":
        j = gpd.overlay(
            seg[[SEG_IDX, "geometry"]],
            grid[["cell_id", "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        j["_len"] = j.geometry.length
        out = (
            j.sort_values("_len", ascending=False)
            .drop_duplicates(SEG_IDX)[[SEG_IDX, "cell_id"]]
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"unknown inputs.segment_centralities.assignment_rule: {rule}")
    ctx.count(
        f"{label}: segment-cell pairs ({rule})",
        len(out),
        f"{out[SEG_IDX].nunique()} distinct segments, {out['cell_id'].nunique()} cells",
    )
    return out


def aggregate_segment_centralities(
    ctx: common.Context,
    grid: gpd.GeoDataFrame,
    size_m: int,
    spec: dict[str, Any] | None = None,
    label: str | None = None,
) -> gpd.GeoDataFrame:
    """Aggregate segment-level centralities onto `grid`, per `grid.aggregation`.

    `spec` overrides `inputs.segment_centralities` for a run that is not the file
    of record — a PC beta or boundary variant. Everything else, including the
    `intersects` assignment rule and the mean/sum split by measure role, is
    unchanged, which is the point.
    """
    spec = spec if spec is not None else ctx.cfg["inputs"]["segment_centralities"]
    seg = load_segment_centralities(ctx, spec)
    pairs = assign_segments(
        ctx,
        seg,
        grid,
        spec["assignment_rule"],
        label or f"segments@{size_m}m",
    )
    cols = [c for k, c in spec["columns"].items() if k in MEASURES]
    j = pairs.merge(seg[[SEG_IDX] + cols], on=SEG_IDX, how="left")
    ctx.count(f"segments aggregated at {size_m} m", len(j))

    rule = ctx.cfg["grid"]["aggregation"]
    gr = j.groupby("cell_id")
    out = {"n_segments": gr.size()}
    for logical, colname in spec["columns"].items():
        if logical not in MEASURES:
            continue
        stat = rule[role(ctx, logical)]
        out[f"{logical}_{stat}"] = getattr(gr[colname], stat)()
    return grid.merge(pd.DataFrame(out).reset_index(), on="cell_id", how="left")


# ---------------------------------------------------------------------------
# aggregation of the point-based variables
# ---------------------------------------------------------------------------


def aggregate_points(
    ctx: common.Context,
    j: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    value_cols: dict[str, str],
) -> pd.DataFrame:
    """Cell means, medians and counts for the point-based variables.

    The rule is the one section 2 of the stage-2 report tests against the
    frozen `Puni_mean`: a plain mean over every address record in the cell,
    with no pre-filter.
    """
    gr = j.dropna(subset=["cell_id"]).groupby("cell_id")
    out = pd.DataFrame({"n_addresses": gr.size()})
    for name, col in value_cols.items():
        stat = ctx.cfg["grid"]["aggregation"][role(ctx, name)]
        out[f"{name}_{stat}"] = getattr(gr[col], stat)()
        out[f"{name}_median"] = gr[col].median()
        out[f"{name}_count"] = gr[col].count()
    return out.reindex(grid["cell_id"]).reset_index()


def write_layer(ctx: common.Context, gdf: gpd.GeoDataFrame, name: str) -> Path:
    """Write a GeoPackage whose bytes depend only on the data.

    A GeoPackage records when it was written in `gpkg_contents.last_change`, so
    two identical writes hash differently and the manifest would drift on every
    run. Stamping `repro.gpkg_last_change` and VACUUMing removes that.
    """
    p = ctx.interim(name)
    p.unlink(missing_ok=True)
    gdf.to_file(p, driver="GPKG", layer=Path(name).stem)
    con = sqlite3.connect(p)
    try:
        con.execute(
            "UPDATE gpkg_contents SET last_change = ?",
            (ctx.cfg["repro"]["gpkg_last_change"],),
        )
        con.commit()
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()
    ctx.log.info("wrote %s (%d rows)", p.relative_to(common.REPO), len(gdf))
    return p


def segment_count_check(
    ctx: common.Context, agg: gpd.GeoDataFrame, implied: pd.DataFrame
) -> dict[str, Any]:
    """Segments per 500 m cell under the configured rule vs the implied counts.

    A cell without an implied count agrees when no segment lands in it. This
    feeds the report only; it writes no table and no count.
    """
    spec = ctx.cfg["inputs"]["segment_centralities"]
    seg = load_segment_centralities(ctx)
    rule = spec["assignment_rule"]
    if rule != "intersects":
        return {"rule": rule, "checked": False}
    j = gpd.sjoin(
        seg[[SEG_IDX, "geometry"]],
        agg[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    cnt = j.groupby("cell_id").size().reindex(agg["cell_id"]).fillna(0).astype(int)
    target = implied.set_index("cell_id")["implied_units"].reindex(agg["cell_id"])
    agree = (cnt.to_numpy() == target.to_numpy()) | (
        target.isna().to_numpy() & (cnt.to_numpy() == 0)
    )
    return {
        "rule": rule,
        "checked": True,
        "n_segments": len(seg),
        "n_pairs": int(cnt.sum()),
        "n_cells": len(agg),
        "n_cells_agree": int(agree.sum()),
    }


def segment_multiplicity(
    ctx: common.Context, seg: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, size_m: int
) -> dict[str, Any]:
    """How many cells each street segment is assigned to on one grid.

    Under the `intersects` rule a segment belongs to every cell its geometry
    touches, so the number of segment-cell assignments exceeds the number of
    distinct segments. Reported per grid size; writes no count.
    """
    rule = ctx.cfg["inputs"]["segment_centralities"]["assignment_rule"]
    if rule != "intersects":
        return {"grid_m": size_m, "assignment_rule": rule}
    j = gpd.sjoin(
        seg[[SEG_IDX, "geometry"]],
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    per_seg = j.groupby(SEG_IDX).size()
    per_cell = j.groupby("cell_id").size()
    per_cell_all = per_cell.reindex(grid["cell_id"]).fillna(0)
    return {
        "grid_m": size_m,
        "assignment_rule": rule,
        "n_cells": len(grid),
        "n_cells_with_segments": int(per_cell.size),
        "n_segments_in_file": len(seg),
        "n_distinct_segments_assigned": int(per_seg.size),
        "n_assignments": len(j),
        "n_segments_in_1_cell": int((per_seg == 1).sum()),
        "n_segments_in_2_cells": int((per_seg == 2).sum()),
        "n_segments_in_3plus_cells": int((per_seg >= 3).sum()),
        "n_segments_in_more_than_1_cell": int((per_seg > 1).sum()),
        "mean_segments_per_cell_with_segments": float(per_cell.mean()),
        "median_segments_per_cell_with_segments": float(per_cell.median()),
        "mean_segments_per_cell_all": float(per_cell_all.mean()),
        "median_segments_per_cell_all": float(per_cell_all.median()),
    }


def positivity_breakdown(
    df: pd.DataFrame, cmap: dict[str, str], logical: list[str]
) -> dict[str, int]:
    """Why the cells that fail the aggregated positivity filter fail it."""
    ok = pd.Series(True, index=df.index)
    for k in logical:
        ok &= df[cmap[k]].notna() & (df[cmap[k]] > 0)
    fail = df[~ok]
    cent = [cmap[m] for m in MEASURES]
    return {
        "n_fail": len(fail),
        "no_centrality": int(fail[cent].isna().all(axis=1).sum()),
        "bc_zero": int((fail[cmap["BC"]] == 0).sum()),
        "fk_zero": int((fail[cmap["FK"]] == 0).sum()),
        "pc_zero": int(((fail[cmap["PC1"]] == 0) & (fail[cmap["PC2"]] == 0)).sum()),
        "land_value_fail": int(
            (~(fail[cmap["land_value"]].notna() & (fail[cmap["land_value"]] > 0))).sum()
        ),
    }


def positive_count(df: pd.DataFrame, cols: list[str]) -> int:
    m = pd.Series(True, index=df.index)
    for c in cols:
        if c not in df.columns:
            return -1
        m &= df[c].notna() & (df[c] > 0)
    return int(m.sum())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ctx = common.init(2, "join_aggregate")
    gcfg = ctx.cfg["grid"]
    vcfg = gcfg["validation"]
    exact_tol, within_tol = float(vcfg["exact_rel_tol"]), float(vcfg["within_rel_tol"])
    crs = f"EPSG:{gcfg['crs_epsg']}"
    cmap = ctx.cfg["measures"]["columns"]["aggregated"]

    # -- inputs --------------------------------------------------------------
    agg = gpd.read_file(ctx.path("frozen", "cents_aggregated"))
    pt = gpd.read_file(ctx.path("frozen", "cents_disaggregated"))
    if str(agg.crs) != crs or str(pt.crs) != crs:
        raise RuntimeError(
            f"CRS mismatch: grid={agg.crs}, points={pt.crs}, config={crs}"
        )
    ctx.log.info("CRS confirmed on both frozen layers: %s", agg.crs)
    agg = agg.rename(columns={gcfg["frozen_500m"]["id_column"]: "cell_id"})
    ctx.count("frozen 500 m grid cells", len(agg))
    ctx.count("frozen address points", len(pt))

    # -- 1. grid characterisation and regeneration ---------------------------
    params = characterise_frozen_grid(ctx, agg)
    ctx.log.info(
        "grid: %s hexagons, edge %.4f m, centres %.1f m apart, planar area %.4f ha",
        params["orientation"],
        params["edge_length_m"],
        params["spacing_m"],
        params["planar_cell_area_ha"],
    )
    ctx.log.info(
        "regeneration: max centroid offset %.3e m, max Hausdorff (unclipped) %.3e m",
        params["max_centroid_offset_m"],
        params["max_hausdorff_full_cells_m"],
    )

    # the polygon that clipped the frozen grid is not among the inputs: use the
    # frozen grid's own union as the study area, for every cell size
    study_area = agg.geometry.union_all()
    regen = build_grid(
        gcfg, float(gcfg["spacing_m"]), study_area.bounds, clip_to=study_area
    )
    ctx.count(
        "grid: regenerated 500 m cells",
        len(regen),
        "clipped to the frozen grid's union",
    )
    frozen_keys = set(zip(agg.row_index.astype(int), agg.col_index.astype(int)))
    regen_keys = set(zip(regen.row_index.astype(int), regen.col_index.astype(int)))
    params.update(
        regenerated_n_cells=len(regen),
        regenerated_keys_identical=bool(frozen_keys == regen_keys),
        regenerated_keys_only_in_frozen=len(frozen_keys - regen_keys),
        regenerated_keys_only_in_regenerated=len(regen_keys - frozen_keys),
    )
    pd.DataFrame([{"parameter": k, "value": v} for k, v in params.items()]).to_csv(
        ctx.table("stage2_grid_parameters.csv"), index=False
    )

    # -- 2. hypothesis A ------------------------------------------------------
    sweep, implied = hypothesis_a(ctx, pt, agg)
    sweep.to_csv(ctx.table("stage2_aggregation_validation.csv"), index=False)
    implied.to_csv(ctx.table("stage2_implied_segment_counts.csv"), index=False)
    best = (
        sweep.sort_values("within", ascending=False)
        .groupby(["frozen_column", "aggregation"], as_index=False)
        .head(1)
        .sort_values(["source_column", "aggregation"])
    )
    best.to_csv(ctx.table("stage2_aggregation_best.csv"), index=False)

    total_segments = int(implied["implied_units"].sum())
    ctx.count(
        "units implied by frozen sum/mean (all cells)",
        total_segments,
        f"vs {len(pt)} address points",
    )
    thresh = float(vcfg["hypothesis_a_within_share"])
    paper_rule_best = best[best.is_paper_rule].set_index("source_column")
    centralities_point_based = bool(
        all(paper_rule_best.loc[m, "within"] >= thresh for m in MEASURES)
    )
    land_point_based = bool(
        all(paper_rule_best.loc[s, "within"] >= thresh for s in ("Puni", "Terreno"))
    )
    ctx.log.info(
        "hypothesis A: land value / plot area %s; centralities %s",
        "holds" if land_point_based else "refuted",
        "hold (point-aggregated)"
        if centralities_point_based
        else "refuted (segment-aggregated)",
    )

    # -- 3. the stage-2 aggregated dependent variable -------------------------
    variants = list(ctx.cfg["dependent_variable"]["variants"])
    dv: dict[str, pd.DataFrame] = {}
    for v in variants:
        d = pd.read_csv(ctx.interim(f"dependent_variable_{v}.csv"))
        if (
            len(d) != len(pt)
            or not (d["Endereco"].to_numpy() == pt["Endereco"].to_numpy()).all()
        ):
            raise RuntimeError(
                f"stage-1 output '{v}' is not row-aligned with the frozen point layer "
                "(stage 1 writes one row per frozen record, in layer order)"
            )
        dv[v] = d
        ctx.count(f"stage-1 rebuilt records: {v}", len(d))

    xc, yc = gcfg["point_positions"][gcfg["active_point_position"]]
    base = pt[["feature_x", "feature_y", "nearest_x", "nearest_y"]].copy()

    # frozen columns carried onto the 500 m output, with the three that collide
    # with our rebuilt names given a _frozen suffix
    rename_frozen = {
        "Puni_mean": "Puni_mean_frozen",
        "Puni_median": "Puni_median_frozen",
        "Terreno_mean": "Terreno_mean_frozen",
        "US$": "USD_mean_frozen",
        "Número": "seq_frozen",
        "Area": "cell_area_ellipsoidal_m2",
    }
    drop_frozen = [
        "geometry",
        "row_index",
        "col_index",
        "left",
        "top",
        "right",
        "bottom",
    ]
    carry = agg.drop(columns=drop_frozen).rename(columns=rename_frozen)

    written: dict[str, Path] = {}
    cell_counts: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    dv_validation: list[dict[str, Any]] = []
    sizes = [int(gcfg["spacing_m"])] + [int(s) for s in gcfg["maup"]["sizes_m"]]
    multiplicity: list[dict[str, Any]] = []
    seg_all = load_segment_centralities(ctx)

    for size in sizes:
        if size == int(gcfg["spacing_m"]):
            grid = agg[["cell_id", "row_index", "col_index", "geometry"]].copy()
        else:
            grid = build_grid(gcfg, float(size), study_area.bounds, clip_to=study_area)
            grid["cell_id"] = (
                grid.row_index.astype(int).astype(str)
                + "_"
                + grid.col_index.astype(int).astype(str)
            )
            grid = grid[["cell_id", "row_index", "col_index", "geometry"]]
        grid["cell_area_m2"] = grid.geometry.area
        ctx.count(f"grid {size} m: cells", len(grid))
        multiplicity.append(segment_multiplicity(ctx, seg_all, grid, size))

        for v in variants:
            src = base.assign(
                Puni=dv[v]["Puni"].to_numpy(), Terreno=dv[v]["Terreno"].to_numpy()
            )
            j = join_points(ctx, points_from(src, xc, yc, crs), grid, f"{size} m / {v}")
            unmatched.append(
                {
                    "grid_m": size,
                    "variant": v,
                    "n_points": len(j),
                    "n_outside_all_cells": int(j["cell_id"].isna().sum()),
                }
            )
            out = grid.merge(
                aggregate_points(ctx, j, grid, {"Puni": "Puni", "Terreno": "Terreno"}),
                on="cell_id",
                how="left",
            )
            out["n_addresses"] = out["n_addresses"].fillna(0).astype(int)

            has_centralities = False
            if size == int(gcfg["spacing_m"]):
                out = out.merge(carry, on="cell_id", how="left")
                has_centralities = True
                dv_validation.append(
                    {
                        "variant": v,
                        "grid_m": size,
                        **match_shares(
                            out.set_index("cell_id")["Puni_mean"],
                            out.set_index("cell_id")["Puni_mean_frozen"],
                            exact_tol,
                            within_tol,
                        ),
                    }
                )
            else:
                # the frozen centralities are segment aggregates, so the MAUP
                # grids take theirs from the segment layer
                out = aggregate_segment_centralities(ctx, out, size)
                has_centralities = True

            name = f"aggregated_{size}m_{v}.gpkg"
            written[name.removesuffix(".gpkg")] = write_layer(ctx, out, name)
            cell_counts.append(
                {
                    "grid_m": size,
                    "variant": v,
                    "n_cells": len(out),
                    "n_cells_with_addresses": int((out.n_addresses > 0).sum()),
                    "n_cells_empty": int((out.n_addresses == 0).sum()),
                    "n_cells_positive_land_value": int(
                        (out["Puni_mean"].notna() & (out["Puni_mean"] > 0)).sum()
                    ),
                    "n_addresses_assigned": int(out.n_addresses.sum()),
                    "median_addresses_per_cell": float(out.n_addresses.median()),
                    "has_centrality_columns": has_centralities,
                }
            )

    pd.DataFrame(cell_counts).to_csv(ctx.table("stage2_cell_counts.csv"), index=False)
    mult = pd.DataFrame(multiplicity).sort_values("grid_m")
    mult.to_csv(ctx.table("stage2_segment_multiplicity.csv"), index=False)
    for r in mult.itertuples():
        ctx.log.info(
            "segments @%d m: %d assignments of %d distinct segments; %d in more than one cell",
            r.grid_m,
            r.n_assignments,
            r.n_distinct_segments_assigned,
            r.n_segments_in_more_than_1_cell,
        )
    pd.DataFrame(unmatched).to_csv(
        ctx.table("stage2_unmatched_points.csv"), index=False
    )
    dvv = pd.DataFrame(dv_validation)
    dvv.to_csv(ctx.table("stage2_dv_validation.csv"), index=False)

    # submitted aggregated N, on the frozen columns and on the rebuilt headline
    pos_logical = ctx.cfg["exclusions"]["positivity"]["aggregated"]
    frozen_pos = positive_count(agg, [cmap[k] for k in pos_logical])
    head = gpd.read_file(ctx.interim(f"aggregated_500m_{variants[0]}.gpkg"))
    rebuilt_pos = positive_count(
        head, ["Puni_mean", "Terreno_mean"] + [cmap[m] for m in MEASURES]
    )
    ctx.count("aggregated cells passing positivity (frozen columns)", frozen_pos)
    ctx.count("aggregated cells passing positivity (rebuilt headline)", rebuilt_pos)

    seg_check = segment_count_check(ctx, agg, implied)
    pos_why = positivity_breakdown(agg, cmap, pos_logical)

    man = common.write_manifest(ctx, written, "stage2_manifest.json")
    ctx.write_counts("stage2_counts.csv")

    write_report(
        ctx,
        params=params,
        sweep=sweep,
        best=best,
        implied=implied,
        total_segments=total_segments,
        centralities_point_based=centralities_point_based,
        land_point_based=land_point_based,
        dvv=dvv,
        cell_counts=pd.DataFrame(cell_counts),
        unmatched=pd.DataFrame(unmatched),
        frozen_pos=frozen_pos,
        rebuilt_pos=rebuilt_pos,
        digests=json.loads(man.read_text()),
        n_points=len(pt),
        seg_check=seg_check,
        pos_why=pos_why,
        cmap=cmap,
    )
    ctx.finish()
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * float(x):.1f} %"


def write_report(ctx: common.Context, **kw: Any) -> Path:
    p = kw["params"]
    imp = kw["implied"]
    sweep = kw["sweep"]
    best_all = kw["best"]
    paper = best_all[best_all.is_paper_rule].set_index("source_column")
    tgt = ctx.cfg["exclusions"]["targets"]
    vcfg = ctx.cfg["grid"]["validation"]
    tol = f"{100 * float(vcfg['within_rel_tol']):g} %"
    spacing = float(p["spacing_m"])
    sc = kw["seg_check"]
    why = kw["pos_why"]
    cmap = kw["cmap"]
    cc = kw["cell_counts"]
    um = kw["unmatched"]
    n_cells = int(p["n_cells"])

    def best_of(fcol: str, stat: str) -> pd.Series:
        r = best_all[(best_all.frozen_column == fcol) & (best_all.aggregation == stat)]
        return r.iloc[0]

    universes = sorted(
        {u for u in sweep["universe"]}, key=lambda u: -int(u.rsplit("_", 1)[1])
    )
    universe_list = " / ".join(f"{int(u.rsplit('_', 1)[1]):,}" for u in universes)

    # -- headline -------------------------------------------------------------
    seg_sentence = (
        f"Under the `{sc['rule']}` rule the segment layer "
        "(`inputs.segment_centralities`) puts exactly that many segments in "
        f"{sc['n_cells_agree']} of the {sc['n_cells']} cells."
        if sc["checked"]
        else ""
    )
    lattice_ok = bool(p["regenerated_keys_identical"])
    md: list[str] = [
        "# Stage 2 — joins and aggregation",
        "",
        (
            f"Produced by `src/20_join_aggregate.py` (seed {ctx.cfg['repro']['seed']}, "
            f"CRS EPSG:{ctx.cfg['grid']['crs_epsg']}). Every join and filter is counted "
            "in `outputs/tables/stage2_counts.csv`."
        ),
        "",
        (
            f"**Headline.** The frozen {spacing:g} m grid is a `{p['orientation']}` "
            "hexagon lattice whose parameters are recorded in `config.yaml`; "
            "regenerating it from them "
            + (
                f"reproduces the (row, col) keys of all {n_cells} cells."
                if lattice_ok
                else f"does not reproduce the keys of all {n_cells} cells."
            )
            + " The frozen `Puni_mean` and `Terreno_mean` columns "
            + ("are" if kw["land_point_based"] else "are not")
            + " plain means over the address points ("
            + f"{_pct(paper.loc['Puni', 'within'])} and "
            + f"{_pct(paper.loc['Terreno', 'within'])} of cells within {tol}). "
            + "The five frozen centrality columns "
            + ("are" if kw["centralities_point_based"] else "are not")
            + f": their `sum / mean` ratios count **{kw['total_segments']:,} "
            "aggregated units** in total. " + seg_sentence
        ),
        "",
        "---",
        "",
        f"## 1. The frozen {spacing:g} m grid",
        "",
        "| parameter | value |",
        "|---|---|",
    ]
    md += [
        f"| {a} | {b} |"
        for a, b in [
            ("CRS", f"EPSG:{p['crs_epsg']}"),
            (
                "orientation",
                f"`{p['orientation']}` — two horizontal edges, vertices due E and W",
            ),
            (
                "cells",
                (
                    f"{p['n_cells']} ({p['n_full_cells']} whole, {p['n_clipped_cells']} "
                    "clipped at the study-area edge)"
                ),
            ),
            (
                "edge length = circumradius",
                f"{p['edge_length_m']:.4f} m (= {spacing:g}/√3)",
            ),
            ("across flats = centre spacing", f"{p['across_flats_m']:.1f} m"),
            ("across corners = bbox width", f"{p['across_corners_m']:.4f} m"),
            ("column spacing Δx", f"{p['col_dx_m']:.4f} m (= {spacing / 2:g}√3)"),
            (
                "row spacing Δy",
                f"{p['row_dy_m']:.1f} m — `row_index` increases southward",
            ),
            ("odd-column offset", f"{p['odd_col_dy_m']:.1f} m"),
            (
                "origin = centre of cell (row 0, col 0)",
                f"({p['origin_x']:.4f}, {p['origin_y']:.4f})",
            ),
            (
                "planar cell area",
                f"{p['planar_cell_area_m2']:,.2f} m² = **{p['planar_cell_area_ha']:.4f} ha**",
            ),
            (
                "ellipsoidal cell area (median)",
                f"**{p['ellipsoidal_cell_area_ha_median']:.4f} ha**",
            ),
        ]
    ]
    nb_dist = math.hypot(p["col_dx_m"], spacing / 2.0)
    md += [
        "",
        (
            f"All six neighbours of a cell centre are {nb_dist:.4f} m away "
            f"(√({p['col_dx_m']:.4f}² + {spacing / 2:g}²)), the centre spacing."
        ),
        "",
        (
            "**Regeneration.** Rebuilding the lattice from the recorded parameters and "
            "clipping it to the frozen grid's own union gives:"
        ),
        "",
        (
            f"- max centroid offset over all {p['n_cells']} cells: "
            f"**{p['max_centroid_offset_m']:.2e} m**"
        ),
        (
            f"- max Hausdorff distance on the {p['n_full_cells']} unclipped cells: "
            f"**{p['max_hausdorff_full_cells_m']:.2e} m**"
        ),
        (
            "- clipped cells: max area outside the generated hexagon "
            f"{p['max_clipped_area_outside_generated_m2']:.2e} m²"
        ),
        (
            f"- regenerated cell count **{p['regenerated_n_cells']}**; (row, col) keys "
            f"identical to the frozen set: "
            f"**{'yes' if p['regenerated_keys_identical'] else 'no'}**"
        ),
        "",
        (
            "**Caveat.** The polygon that clipped the frozen grid is not among the "
            f"inputs, so the study-area polygon used here is the union of the {n_cells} "
            "frozen cells. The regeneration test is therefore a test of the *lattice*, "
            "not of the clip. Every other cell size is clipped to the same polygon, "
            "which is the property MAUP needs: the three grids cover exactly one area."
        ),
        "",
        (
            "**Cell area.** The planar area of a whole cell in "
            f"EPSG:{p['crs_epsg']} is {p['planar_cell_area_ha']:.4f} ha; the WGS84 "
            f"ellipsoidal area is {p['ellipsoidal_cell_area_ha_median']:.4f} ha "
            "(median), and the ellipsoidal area reproduces the frozen layer's own "
            f"`Area` column to {p['ellipsoidal_vs_frozen_Area_max_rel']:.1e} relative. "
            "Recomputing the area in the projected CRS gives the planar number."
        ),
        "",
        "---",
        "",
        "## 2. What were the frozen columns aggregated from?",
        "",
        (
            'Paper §4.5: *"land unit values and accessibility measures **from point '
            'observations** were averaged, while intermediation measures were summed"*. '
            "**Hypothesis A** takes that literally — the point observations are the "
            f"{kw['n_points']:,} addresses of the frozen disaggregated layer. "
            "**Hypothesis B** is that the aggregation ran over street segments."
        ),
        "",
        (
            f"Every combination of point universe ({universe_list}), point "
            "position (the address point `feature_x/y` vs the network-snapped point "
            "`nearest_x/y`), de-duplication (none vs one row per distinct snapped "
            "point) and aggregation (mean / sum / median) was joined to the "
            f"{n_cells} frozen cells and compared column by column — "
            f"{len(sweep)} scored combinations in "
            "`outputs/tables/stage2_aggregation_validation.csv`. Best variant for each "
            "frozen column:"
        ),
        "",
    ]
    b = best_all.copy()
    for c in ("exact", "within", "median_rel"):
        b[c] = (100 * b[c]).round(1)
    b["median_ratio"] = b["median_ratio"].round(3)
    md += prep.md_table(
        b[
            [
                "frozen_column",
                "aggregation",
                "is_paper_rule",
                "universe",
                "point_position",
                "dedupe",
                "n",
                "exact",
                "within",
                "median_rel",
                "median_ratio",
            ]
        ].rename(
            columns={
                "is_paper_rule": "paper rule?",
                "n": "cells",
                "exact": "exact %",
                "within": f"within {tol} ",
                "median_rel": "median rel err %",
                "median_ratio": "median ours/frozen",
            }
        )
    )

    # -- verdict, from the sweep ---------------------------------------------
    pm = best_of("Puni_mean", "mean")
    same = sweep[
        (sweep.frozen_column == "Puni_mean")
        & (sweep.aggregation == "mean")
        & (sweep.point_position == pm.point_position)
        & (sweep.dedupe == pm.dedupe)
    ].set_index("universe")["within"]
    by_universe = ", ".join(
        f"{int(u.rsplit('_', 1)[1]):,} points {_pct(same[u])}"
        for u in universes
        if u in same.index
    )
    med = best_of("Puni_median", "median")
    ter = best_of("Terreno_mean", "mean")
    land_note = (
        " The full address layer gives the best match, so no exclusion was "
        "applied before aggregating; the positivity filter applies to the "
        "*cells* afterwards."
        if pm.universe == universes[0]
        else ""
    )
    cent = paper.loc[MEASURES]
    top_m = str(cent["within"].astype(float).idxmax())
    others = ", ".join(
        f"{m} {_pct(cent.loc[m, 'within'])}" for m in MEASURES if m != top_m
    )
    best_any = float(sweep.loc[sweep.source_column.isin(MEASURES), "within"].max())
    md += [
        "",
        "### Verdict",
        "",
        (
            "- **Land value and plot area — hypothesis A "
            + ("holds" if kw["land_point_based"] else "is refuted")
            + ".** The best reproduction of `Puni_mean` is the plain mean of `Puni` "
            f"over the `{pm.universe}` universe at the `{pm.point_position}` "
            f"position (de-duplication: `{pm.dedupe}`): exact on {_pct(pm.exact)} of "
            f"the {int(pm.n)} cells and within {tol} on {_pct(pm.within)}. At that "
            f"position, by universe: {by_universe} within {tol}. `Puni_median` "
            f"(median) reaches {_pct(med.within)} and `Terreno_mean` (mean) "
            f"{_pct(ter.within)} within {tol}." + land_note
        ),
        (
            "- **The five centralities — hypothesis A "
            + ("holds" if kw["centralities_point_based"] else "is refuted")
            + ".** Under the paper's own rule (mean for the accessibility measures, "
            "sum for the intermediation ones) the best-reproduced column is "
            f"{top_m}: {_pct(cent.loc[top_m, 'within'])} of cells within {tol} and "
            f"{_pct(cent.loc[top_m, 'exact'])} exact; {others}. The median ratio "
            f"ours/frozen is {float(cent.loc['FK', 'median_ratio']):.2f} for FK and "
            f"{float(cent.loc['BC', 'median_ratio']):.2f} for BC. Across all "
            f"{len(sweep)} combinations, no centrality column exceeds "
            f"{_pct(best_any)} of cells within {tol}."
        ),
        "",
        "### The frozen layer says how many units it aggregated",
        "",
    ]
    has_data = imp["implied_units"].notna()
    max_dev = float(imp.loc[has_data, "max_integer_deviation"].max())
    n_agree = int((has_data & imp["measures_disagreeing"].eq(0)).sum())
    md += [
        (
            "For a column aggregated both ways, `sum / mean` is the count of units "
            "that went in. Across the "
            f"{imp.attrs['n_column_pairs']} `_sum`/`_mean` column pairs of the frozen "
            f"layer that ratio is at most {max_dev:.1e} from an integer in any cell, "
            f"and all pairs give the same integer in {n_agree} of the "
            f"{int(has_data.sum())} cells that carry data. The aggregation universe "
            "can therefore be counted:"
        ),
        "",
    ]
    md += prep.md_table(
        pd.DataFrame(
            [
                {
                    "quantity": "units aggregated, all cells",
                    "value": f"{kw['total_segments']:,}",
                },
                {
                    "quantity": "median units per cell",
                    "value": f"{imp.implied_units.median():.0f}",
                },
                {
                    "quantity": "max units in one cell",
                    "value": f"{imp.implied_units.max():.0f}",
                },
                {
                    "quantity": "address points, all cells",
                    "value": f"{int(imp.n_address_points.sum()):,}",
                },
                {
                    "quantity": "median address points per cell",
                    "value": f"{imp.n_address_points.median():.0f}",
                },
                {
                    "quantity": "distinct snapped points, summed over cells",
                    "value": f"{int(imp.n_distinct_snapped_points.sum()):,}",
                },
                {
                    "quantity": "median units per address point",
                    "value": f"{(imp.implied_units / imp.n_address_points.replace(0, np.nan)).median():.1f}",
                },
            ]
        )
    )
    md += [
        "",
        (
            f"The address layer has {kw['n_points']:,} rows on "
            f"{imp.attrs['n_distinct_snapped_points']:,} distinct snapped positions and "
            f"only {imp.attrs['n_distinct_centrality_tuples']:,} distinct centrality "
            f"tuples — at most "
            f"{imp.attrs['n_distinct_centrality_tuples'] / kw['total_segments']:.0%} of "
            f"the {kw['total_segments']:,} units the aggregation used. Summing BC or FK "
            "over addresses would additionally multiply every segment by the number of "
            "addresses on it."
        ),
        "",
        (
            "Per-cell targets are in `outputs/tables/stage2_implied_segment_counts.csv`. "
            "The number of segments landing in each cell must equal `implied_units`, "
            "which is an acceptance test for the segment join before any MAUP number "
            "is computed. "
            + (
                f"Under the `{sc['rule']}` rule — a segment counts in every cell it "
                f"touches — the {sc['n_segments']:,} segments of "
                "`inputs.segment_centralities` give "
                f"{sc['n_pairs']:,} segment–cell pairs and match the target in "
                f"{sc['n_cells_agree']} of {sc['n_cells']} cells."
                if sc["checked"]
                else f"The configured rule `{sc['rule']}` is not checked here."
            )
        ),
        "",
        "---",
        "",
        "## 3. The stage-2 aggregated dependent variable",
        "",
        (
            "For each of stage 1's rebuilt `Puni` variants the "
            f"{kw['n_points']:,} address records are joined to the {spacing:g} m cells "
            "and averaged under the rule section 2 recovered — plain mean, no "
            "pre-filter — and the frozen centrality columns are carried across "
            "unchanged (`*_frozen` suffixes mark the frozen land-value columns kept "
            "for comparison). Written to "
            f"`data/interim/aggregated_{spacing:g}m_<variant>.gpkg`, checksummed in "
            "`data/interim/stage2_manifest.json`."
        ),
        "",
        "Rebuilt cell means vs the frozen `Puni_mean`:",
        "",
    ]
    d = kw["dvv"].copy()
    d["exact %"] = (100 * d["exact"]).round(1)
    d[f"within {tol}"] = (100 * d["within"]).round(1)
    d["median rel err %"] = (100 * d["median_rel"]).map("{:.3g}".format)
    md += prep.md_table(
        d[
            ["variant", "grid_m", "n", "exact %", f"within {tol}", "median rel err %"]
        ].rename(columns={"n": "cells"})
    )
    base = cc[cc.grid_m == int(spacing)].iloc[0]
    out_base = um[um.grid_m == int(spacing)]
    n_out = int(out_base["n_outside_all_cells"].max())
    md += [
        "",
        (
            "Cells passing the aggregated positivity filter on the **frozen** "
            f"columns: **{kw['frozen_pos']}** against the submitted target "
            f"{tgt['aggregated_n']}"
            + (" — reproduced" if kw["frozen_pos"] == tgt["aggregated_n"] else "")
            + ". On the **rebuilt headline** land value with the frozen centralities: "
            f"**{kw['rebuilt_pos']}**."
        ),
        "",
        (
            f"Of the {why['n_fail']} cells the filter drops on the frozen columns, "
            f"{why['no_centrality']} carry no centrality values at all, "
            f"{why['bc_zero']} have `{cmap['BC']}` = 0, {why['fk_zero']} have "
            f"`{cmap['FK']}` = 0, {why['pc_zero']} have PC1 = PC2 = 0, and "
            f"{why['land_value_fail']} fail on land value (a cell can fall in more "
            f"than one group). {int(base['n_cells_empty'])} of the {int(base['n_cells'])} "
            "cells contain no address point, and "
            f"{n_out} address point(s) fall outside every cell."
        ),
        "",
        "---",
        "",
        "## 4. MAUP grids at 250 m and 1,000 m",
        "",
        (
            f"Both are generated on the {spacing:g} m grid's lattice — same origin, "
            "same orientation, same CRS, clipped to the same study-area polygon — "
            "and carry the point-based variables:"
        ),
        "",
    ]
    md += prep.md_table(cc)
    md += [
        "",
        (
            "Their centralities come from the **segment** layer "
            "(`inputs.segment_centralities`), not from the address points: "
            "`aggregate_segment_centralities()` assigns each segment to cells under "
            f"the `{sc['rule']}` rule and then applies `grid.aggregation`. "
            "Aggregating the address points instead would give a quantity that is not "
            "the one the frozen columns hold, and would do it worst for the "
            "intermediation measures, which sum over units."
        ),
        "",
        "Points falling outside every cell, by grid and variant:",
        "",
    ]
    md += prep.md_table(um)
    md += [
        "",
        "---",
        "",
        "## 5. Outputs",
        "",
        "| file | sha256 (first 16) |",
        "|---|---|",
    ]
    md += [
        f"| `data/interim/{k}.gpkg` | `{rec.get('sha256', '')[:16]}` |"
        for k, rec in sorted(kw["digests"].items())
    ]
    md += [
        "",
        (
            "Tables: `stage2_grid_parameters.csv`, `stage2_aggregation_validation.csv`, "
            "`stage2_aggregation_best.csv`, `stage2_implied_segment_counts.csv`, "
            "`stage2_dv_validation.csv`, `stage2_cell_counts.csv`, "
            "`stage2_segment_multiplicity.csv`, "
            "`stage2_unmatched_points.csv`, `stage2_counts.csv`."
        ),
        "",
    ]
    out = ctx.report("stage2_join_aggregate.md")
    out.write_text("\n".join(md) + "\n")
    ctx.log.info("wrote %s", out.relative_to(common.REPO))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
