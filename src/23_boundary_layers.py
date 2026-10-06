#!/usr/bin/env python
"""Stage 2 — put each boundary-extent variant on the 500 m grid and on the points.

    make boundary-layers
    make boundary-layers ARGS="--only bn2000"
    python src/23_boundary_layers.py --family muni

The boundary counterpart of `src/21_pc_beta_layers.py`. A variant carries all
five measures, not two: PC1 / PC2 come from `src/15_boundary_pc_runs.py` and
CC / BC / FK from GAUS Lines v1.1 itself (`src/13_gaus_headless.py`), run per
variant.

Everything reaches the grid and the points through the same two joins the
submitted columns went through — `intersects` at 500 m with the mean/sum split by
measure role, and nearest segment with the junction-tie rule. What a partial
network needs on top of that is coverage, and both restrictions are the ones the
acceptance test uses (`pc_acceptance.tier2.full_coverage_only`,
`tier3.restrict_to_covered_points`):

  * **cells**: a cell is scored only where the variant puts the same number of
    segments in it as the reference does. A cell the clip half-empties would
    otherwise report a mean over the surviving half and read as a large boundary
    effect that is really a missing denominator.
  * **points**: an address is scored only where its nearest reference segment
    survives the clip. Outside the extent an address snaps to whatever is nearest
    the new boundary, which is a statement about the clip and not about the
    measure.

It also scores each measure against its submitted column on the segments the
two extents share — Spearman and Pearson, plus the share within 1 % — which is
the segment-level half of the boundary-extent sensitivity.

Outputs (`revision.boundary_sensitivity.aggregated_layer` / `.points_table`; the
municipal family writes to its own paths and to `stage2_boundary_muni_*` tables):
    data/interim/boundary_aggregated_500m.gpkg   one row per frozen cell,
                                                 <label>_<measure>_<stat>
    data/interim/boundary_points.csv             one row per frozen point,
                                                 submitted_<measure>, <label>_<measure>
    outputs/tables/stage2_boundary_layers.csv
    outputs/tables/stage2_boundary_segment_movement.csv
    outputs/tables/stage2_boundary_coverage.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import common


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage2 = _load_module("stage2", SRC / "20_join_aggregate.py")


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


SEG_IDX = stage2.SEG_IDX
MEASURES = ["PC1", "PC2", "CC", "BC", "FK"]


def movement(got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """How far a variant's measure sits from the submitted one, on common rows."""
    ok = np.isfinite(got) & np.isfinite(ref)
    g, r = got[ok], ref[ok]
    if g.size < 3:
        return {"n_common": int(g.size)}
    both_zero = (g == 0) & (r == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(r != 0, np.abs(g - r) / np.abs(r), np.nan)
    rel = np.where(both_zero, 0.0, rel)
    live = np.isfinite(rel)
    gr = pd.Series(g).rank().to_numpy()
    rr = pd.Series(r).rank().to_numpy()
    return {
        "n_common": int(g.size),
        "spearman": float(np.corrcoef(gr, rr)[0, 1]),
        "pearson": float(np.corrcoef(g, r)[0, 1]),
        "pearson_ln": float(
            np.corrcoef(np.log(g[(g > 0) & (r > 0)]), np.log(r[(g > 0) & (r > 0)]))[
                0, 1
            ]
        )
        if ((g > 0) & (r > 0)).sum() > 2
        else float("nan"),
        "within_1pct": float((rel[live] < 0.01).mean()) if live.any() else float("nan"),
        "median_rel": float(np.median(rel[live])) if live.any() else float("nan"),
        "p90_rel": float(np.percentile(rel[live], 90)) if live.any() else float("nan"),
        "max_rel": float(rel[live].max()) if live.any() else float("nan"),
        "mean_variant": float(g.mean()),
        "mean_submitted": float(r.mean()),
        "mean_rel_shift": float((g.mean() - r.mean()) / r.mean())
        if r.mean() != 0
        else float("nan"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="23_boundary_layers.py",
        description="Aggregate the boundary variants to cells and points.",
    )
    ap.add_argument("--only", nargs="*", default=None, metavar="LABEL")
    ap.add_argument(
        "--family",
        default="negative_buffer",
        help="which variant family to aggregate: `negative_buffer` (the default) "
        "or `muni` (the municipal polygon).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(2, "boundary_layers")
    cfg = ctx.cfg
    gcfg = cfg["grid"]
    crs = f"EPSG:{gcfg['crs_epsg']}"
    bcfg, sfx = family_cfg(cfg["revision"]["boundary_sensitivity"], args.family)
    ctx.log.info("variant family: %s (tables suffixed '%s')", args.family, sfx)
    files = bcfg["input_files"] or []
    if not files:
        ctx.log.error("revision.boundary_sensitivity.input_files is empty")
        return 2
    if args.only:
        known = {str(f["label"]) for f in files}
        unknown = [x for x in args.only if x not in known]
        if unknown:
            raise SystemExit(f"unknown label(s) {unknown}; configured: {sorted(known)}")
        files = [f for f in files if str(f["label"]) in set(args.only)]

    seg_spec = cfg["inputs"]["segment_centralities"]
    rule = seg_spec["assignment_rule"]
    id_cols = list(seg_spec["columns"]["segment_id"])

    # -- the reference: the file of record, on the grid and on the points -----
    ref = stage2.load_segment_centralities(ctx)
    ctx.count("reference segments", len(ref), seg_spec["path"])
    ref_key = pd.MultiIndex.from_arrays([ref[c] for c in id_cols])

    agg = gpd.read_file(ctx.path("frozen", "cents_aggregated"))
    if str(agg.crs) != crs:
        raise RuntimeError(f"frozen grid CRS {agg.crs}, config {crs}")
    agg = agg.rename(columns={gcfg["frozen_500m"]["id_column"]: "cell_id"})
    grid = agg[["cell_id", "row_index", "col_index", "geometry"]].copy()
    ctx.count("frozen 500 m cells", len(grid))

    ref_pairs = stage2.assign_segments(ctx, ref, grid, rule, "reference@500m")
    ref_n = ref_pairs.groupby("cell_id").size().rename("ref_n_segments")

    pt = gpd.read_file(ctx.path("frozen", "cents_disaggregated"))
    ctx.count("frozen address points", len(pt))
    pos = gcfg["active_point_position"]
    xc, yc = gcfg["point_positions"][pos]
    P = gpd.GeoDataFrame(
        pd.DataFrame(index=pt.index),
        geometry=gpd.points_from_xy(pt[xc], pt[yc]),
        crs=crs,
    )
    # Which reference segment each address snaps to — computed once, on the full
    # network, and then used to decide which addresses a variant may be scored on.
    nb = gpd.sjoin_nearest(
        P, ref[[SEG_IDX, "geometry"]], how="left", distance_col="snap_m"
    )
    nb = nb[~nb.index.duplicated(keep="first")]
    ref_seg_of_point = nb[SEG_IDX].to_numpy()
    ctx.log.info(
        "reference snap: %d points, median %.3f m", len(nb), nb["snap_m"].median()
    )

    cells = grid.copy()
    cells["ref_n_segments"] = (
        ref_n.reindex(cells["cell_id"]).fillna(0).astype(int).to_numpy()
    )
    points = pd.DataFrame({"point_index": pt.index.to_numpy()})
    for m in MEASURES:
        points[f"submitted_{m}"] = (
            ref[seg_spec["columns"][m]].to_numpy()[ref_seg_of_point].astype(float)
        )

    processed: list[dict[str, Any]] = []
    moves: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    for rec in files:
        label = str(rec["label"])
        pc_path = common.REPO / rec["path"]
        gaus_path = common.REPO / rec["gaus"]
        for p in (pc_path, gaus_path):
            if not p.exists():
                ctx.log.error(
                    "missing input for '%s': %s — run `make boundary-pc` and "
                    "`make gaus-native` on the variant first",
                    label,
                    p,
                )
                return 1
        ctx.log.info("=" * 74)
        ctx.log.info("=== %s: PC %s + GAUS %s", label, rec["path"], rec["gaus"])

        pcv = gpd.read_file(pc_path, layer=rec["layer"]).to_crs(crs)
        gv = gpd.read_file(gaus_path, layer=rec["gaus_layer"]).to_crs(crs)
        gcols = cfg["gaus_acceptance"]["columns"]
        gv = gv[[*id_cols, gcols["CC"], gcols["BC"], gcols["FK"]]]
        merged = pcv.merge(gv, on=id_cols, how="left", validate="one_to_one")
        n_missing = int(merged[gcols["CC"]].isna().sum())
        if n_missing:
            ctx.log.error(
                "%d of the %d variant segments have no GAUS row — the two runs "
                "are not on the same clipped network",
                n_missing,
                len(merged),
            )
            return 1
        col_of = {
            "PC1": rec["PC1"],
            "PC2": rec["PC2"],
            "CC": gcols["CC"],
            "BC": gcols["BC"],
            "FK": gcols["FK"],
        }
        seg_path = ctx.interim(f"boundary_measures_{label}.gpkg")
        merged_out = merged[[*id_cols, *col_of.values(), "geometry"]].copy()
        stage2.write_layer(ctx, merged_out, seg_path.name)
        ctx.count(f"{label}: variant segments", len(merged), rec["path"])

        # -- segment-level movement against the submitted columns -------------
        var_key = pd.MultiIndex.from_arrays([merged[c] for c in id_cols])
        pos_in_ref = ref_key.get_indexer(var_key)
        if (pos_in_ref < 0).any():
            ctx.log.error("the variant carries a (node1, node2) not in the reference")
            return 1
        coverage.append(
            {
                "label": label,
                "n_reference_segments": len(ref),
                "n_variant_segments": len(merged),
                "segment_coverage": len(merged) / len(ref),
            }
        )
        for m in MEASURES:
            moves.append(
                {"label": label, "level": "segments (common)", "measure": m}
                | movement(
                    merged[col_of[m]].astype(float).to_numpy(),
                    ref[seg_spec["columns"][m]].astype(float).to_numpy()[pos_in_ref],
                )
            )

        # -- cells ------------------------------------------------------------
        spec = {
            "path": str(seg_path.relative_to(common.REPO)),
            "layer": seg_path.stem,
            "assignment_rule": rule,
            "columns": {"segment_id": id_cols, **col_of},
        }
        out = stage2.aggregate_segment_centralities(
            ctx, grid, int(gcfg["spacing_m"]), spec=spec, label=f"{label}@500m"
        )
        stat_of = {m: gcfg["aggregation"][stage2.role(ctx, m)] for m in MEASURES}
        var_n = out["n_segments"].fillna(0).astype(int).to_numpy()
        n_touched = int((var_n > 0).sum())
        full = var_n == cells["ref_n_segments"].to_numpy()
        n_full = int(full.sum())
        ctx.count(
            f"{label}: cells covered completely",
            n_full,
            f"of {len(grid)} ({n_full / len(grid):.2%}); "
            f"{n_touched} cells touched at all",
        )
        ren: dict[str, str] = {"n_segments": f"{label}_n_segments"}
        for m in MEASURES:
            ren[f"{m}_{stat_of[m]}"] = f"{label}_{m}_{stat_of[m]}"
        out = out.rename(columns=ren)
        # blank the incompletely covered cells: an aggregate over half a cell is
        # a missing denominator, not a boundary effect
        for c in ren.values():
            out.loc[~full, c] = np.nan
        out[f"{label}_covered"] = full.astype(int)
        cells = cells.merge(
            out[["cell_id", *ren.values(), f"{label}_covered"]].copy(),
            on="cell_id",
            how="left",
        )

        # -- points -----------------------------------------------------------
        # an address is scored only where its nearest reference segment survives
        kept = np.zeros(len(ref), dtype=bool)
        kept[pos_in_ref] = True
        keep_pt = kept[ref_seg_of_point]
        ctx.count(
            f"{label}: points whose reference segment survives",
            int(keep_pt.sum()),
            f"of {len(P)} ({keep_pt.mean():.2%})",
        )
        var_of_ref = np.full(len(ref), -1, dtype=np.int64)
        var_of_ref[pos_in_ref] = np.arange(len(merged))
        idx = var_of_ref[ref_seg_of_point]
        for m in MEASURES:
            v = np.full(len(P), np.nan)
            src = merged[col_of[m]].astype(float).to_numpy()
            v[keep_pt] = src[idx[keep_pt]]
            points[f"{label}_{m}"] = v
        points[f"{label}_covered"] = keep_pt.astype(int)
        for m in MEASURES:
            moves.append(
                {"label": label, "level": "points (covered)", "measure": m}
                | movement(
                    points[f"{label}_{m}"].to_numpy(),
                    np.where(keep_pt, points[f"submitted_{m}"].to_numpy(), np.nan),
                )
            )
        for m in MEASURES:
            a = cells[f"{label}_{m}_{stat_of[m]}"].to_numpy(dtype=float)
            # the submitted cell columns, as the analysis reads them
            pub_col = cfg["measures"]["columns"]["aggregated"][m]
            b = np.where(full, agg[pub_col].astype(float).to_numpy(), np.nan)
            moves.append(
                {"label": label, "level": "cells (fully covered)", "measure": m}
                | movement(a, b)
            )

        processed.append(
            {
                "label": label,
                "mode": rec["mode"],
                "role": rec["role"],
                "pc_path": rec["path"],
                "gaus_path": rec["gaus"],
                "pc_sha256": common.checksum(pc_path),
                "gaus_sha256": common.checksum(gaus_path),
                "n_segments": len(merged),
                "segment_coverage": len(merged) / len(ref),
                # counted before the incompletely covered cells are blanked
                "n_cells_touched": n_touched,
                "n_cells_fully_covered": n_full,
                "cell_coverage": n_full / len(grid),
                "n_points_covered": int(keep_pt.sum()),
                "point_coverage": float(keep_pt.mean()),
            }
        )

    # -- write ----------------------------------------------------------------
    gpkg = stage2.write_layer(ctx, cells, Path(bcfg["aggregated_layer"]).name)
    pts_path = common.REPO / bcfg["points_table"]
    pts_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(pts_path, index=False)
    ctx.log.info("wrote %s (%d points)", pts_path.relative_to(common.REPO), len(points))

    pd.DataFrame(processed).to_csv(
        ctx.table(f"stage2_boundary{sfx}_layers.csv"), index=False
    )
    pd.DataFrame(moves).to_csv(
        ctx.table(f"stage2_boundary{sfx}_segment_movement.csv"), index=False
    )
    pd.DataFrame(coverage).to_csv(
        ctx.table(f"stage2_boundary{sfx}_coverage.csv"), index=False
    )

    manifest = {
        "produced_by": "src/23_boundary_layers.py",
        "what": (
            "each boundary-extent variant's five measures put on the 500 m grid "
            "(`intersects`, restricted to fully covered cells) and on the "
            "frozen address points (nearest segment, restricted to points "
            "whose reference segment survives the clip)"
        ),
        "coverage_rules": {
            "cells": "full_coverage_only",
            "points": "restrict_to_covered_points",
        },
        "runs": processed,
        "outputs": {
            "aggregated": str(gpkg.relative_to(common.REPO)),
            "points": str(pts_path.relative_to(common.REPO)),
        },
        "seed": int(cfg["repro"]["seed"]),
    }
    man = ctx.interim(f"boundary{sfx}_layers_manifest.json")
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    ctx.log.info("wrote %s", man.relative_to(common.REPO))
    ctx.write_counts(f"stage2_boundary{sfx}_counts.csv")
    ctx.log.info(
        "next: %s",
        "make boundary  (Rscript src/42_boundary_sensitivity.R)"
        if not sfx
        else "make boundary-muni  (Rscript src/43_boundary_muni.R)",
    )
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
