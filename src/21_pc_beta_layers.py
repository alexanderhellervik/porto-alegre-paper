#!/usr/bin/env python
"""Stage 2 — put each registered PC beta run on the 500 m grid and on the points.

    make pc-beta-layers

`revision.pc_sensitivity.input_files` lists the beta runs that have been through
`make pc-accept`. This script takes each of them through the same two joins the
submitted columns went through, so that a beta variant and the submitted
PC1/PC2 are comparable for the one reason they are supposed to be:

  * **to the 500 m cells** — `intersects`, a segment belongs to every cell its
    geometry touches, accessibility aggregated by mean (`grid.aggregation`).
    It calls `aggregate_segment_centralities()` in `src/20_join_aggregate.py`
    with an overriding spec rather than re-implementing the rule.
  * **to the frozen address points** — nearest segment, with the junction tie
    rule: an address whose foot of perpendicular lands on a shared node is
    equidistant from two segments and the frozen join's tie-break is not
    observable, so the first match is taken and the ties are counted.

The `b2` run is the submitted run recomputed, so both joins are also a test:
its cell means must reproduce the frozen `cd1g1b2k0_mean` / `cd4g0b2k0_mean` and
its point values the frozen `PC1` / `PC2`. Those two agreements are reported and
are the evidence that a difference at beta 1.5 or 2.5 is the parameter and not
the plumbing.

Measure slots. A registered run names its columns in the slots `PC1` (the
gamma = 1 run), `PC2` (the gamma = 0 run: preferential centrality without
agglomeration) and the optional `PC3` (the gamma = 0.5 run at the same beta).
The slots are a convention
of `revision.pc_sensitivity` alone: `measures.accessibility` is untouched, so
nothing outside this sensitivity ever sees a third PC. Each slot is aggregated by
a separate call to `aggregate_segment_centralities()` through the `PC1` slot of
an overriding spec, which is what keeps every measure on the one submitted
aggregation rule (`intersects`, mean) instead of on a second implementation of
it.

Outputs (`revision.pc_sensitivity.aggregated_layer` / `.points_table`):
    data/interim/pc_beta_aggregated_500m.gpkg   one row per frozen cell, <label>_PC1_mean ...
    data/interim/pc_beta_points.csv             one row per frozen point, <label>_PC1 ...
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
SEG_IDX = stage2.SEG_IDX

# Slot order is report order. `PC3` (gamma = 0.5) is optional; a record that does
# not name it simply has two series.
MEAS_SLOTS = ("PC1", "PC2", "PC3")

# The two slots the frozen layers carry a submitted column for, and which the
# `reproduction` run is therefore scored against.
SUBMITTED_SLOTS = ("PC1", "PC2")


def measures_of(rec: dict[str, Any]) -> list[str]:
    """The slots this registered run actually names, in slot order."""
    return [m for m in MEAS_SLOTS if rec.get(m)]


def shares(
    got: np.ndarray, ref: np.ndarray, tight: float, wide: float
) -> dict[str, Any]:
    """Agreement on live rows; 0 vs 0 counts as agreement, not as 0/0."""
    ok = np.isfinite(got) & np.isfinite(ref)
    g, r = got[ok], ref[ok]
    both_zero = (g == 0) & (r == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(r != 0, np.abs(g - r) / np.abs(r), np.nan)
    rel = np.where(both_zero, 0.0, rel)
    live = np.isfinite(rel)
    if not live.any():
        return {"n": 0}
    v = rel[live]
    return {
        "n": int(live.sum()),
        "within_tight": float((v < tight).mean()),
        "within_wide": float((v < wide).mean()),
        "median_rel": float(np.median(v)),
        "max_rel": float(v.max()),
        "pearson_r": float(np.corrcoef(g[live], r[live])[0, 1]),
        "spearman_r": float(
            np.corrcoef(
                pd.Series(g[live]).rank().to_numpy(),
                pd.Series(r[live]).rank().to_numpy(),
            )[0, 1]
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="21_pc_beta_layers.py",
        description="Aggregate the registered PC beta runs to cells and points.",
    )
    ap.add_argument(
        "--only", nargs="*", default=None, metavar="LABEL", help="labels to process"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(2, "pc_beta_layers")
    cfg = ctx.cfg
    gcfg = cfg["grid"]
    vcfg = gcfg["validation"]
    tight, wide = float(vcfg["exact_rel_tol"]), float(vcfg["within_rel_tol"])
    crs = f"EPSG:{gcfg['crs_epsg']}"
    sens = cfg["revision"]["pc_sensitivity"]
    files = sens["input_files"] or []
    if not files:
        ctx.log.error(
            "revision.pc_sensitivity.input_files is empty — run `make pc-runs`, "
            "check each output with `make pc-accept`, then register it"
        )
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

    # -- the 500 m grid: the frozen cells themselves, as stage 2 uses them ----
    agg = gpd.read_file(ctx.path("frozen", "cents_aggregated"))
    if str(agg.crs) != crs:
        raise RuntimeError(f"frozen grid CRS {agg.crs}, config {crs}")
    agg = agg.rename(columns={gcfg["frozen_500m"]["id_column"]: "cell_id"})
    grid = agg[["cell_id", "row_index", "col_index", "geometry"]].copy()
    ctx.count("frozen 500 m cells", len(grid))

    pt = gpd.read_file(ctx.path("frozen", "cents_disaggregated"))
    ctx.count("frozen address points", len(pt))
    pos = gcfg["active_point_position"]
    xc, yc = gcfg["point_positions"][pos]
    P = gpd.GeoDataFrame(
        pd.DataFrame(index=pt.index),
        geometry=gpd.points_from_xy(pt[xc], pt[yc]),
        crs=crs,
    )

    cells = grid.copy()
    points = pd.DataFrame(
        {
            "point_index": pt.index.to_numpy(),
            "frozen_PC1": pt[cfg["measures"]["columns"]["disaggregated"]["PC1"]]
            .astype(float)
            .to_numpy(),
            "frozen_PC2": pt[cfg["measures"]["columns"]["disaggregated"]["PC2"]]
            .astype(float)
            .to_numpy(),
        }
    )
    checks: list[dict[str, Any]] = []
    processed: list[dict[str, Any]] = []

    for rec in files:
        label = str(rec["label"])
        path = common.REPO / rec["path"]
        if not path.exists():
            ctx.log.error("registered run is missing: %s", path)
            return 1
        meas = measures_of(rec)
        if not meas:
            raise SystemExit(f"registered run '{label}' names no measure column")
        base_spec = {
            "path": rec["path"],
            "layer": rec["layer"],
            "assignment_rule": rule,
        }
        ctx.log.info(
            "=== %s (beta %s, %s): %s, mode=%s",
            label,
            rec["beta"],
            rec["role"],
            " ".join(f"{m}={rec[m]}" for m in meas),
            rec["mode"],
        )

        # -- cells ------------------------------------------------------------
        # One call per slot, each through the spec's `PC1` slot: the rule
        # (`intersects`, mean) is the submitted one for every measure, and only
        # the column moves.
        n_seg_col = f"{label}_n_segments"
        for m in meas:
            spec = base_spec | {"columns": {"segment_id": id_cols, "PC1": rec[m]}}
            got = stage2.aggregate_segment_centralities(
                ctx, grid, int(gcfg["spacing_m"]), spec=spec, label=f"{label}|{m}@500m"
            )
            ren = {"PC1_mean": f"{label}_{m}_mean"}
            if m == meas[0]:
                ren["n_segments"] = n_seg_col
            got = got.rename(columns=ren)
            take = ["cell_id", *ren.values()]
            cells = cells.merge(got[take].copy(), on="cell_id", how="left")
            if m == meas[0]:
                out = got

        # -- points -----------------------------------------------------------
        seg = stage2.load_segment_centralities(ctx, base_spec | {"columns": {}})
        slim = seg[[SEG_IDX, "geometry"]].copy()
        nb = gpd.sjoin_nearest(P, slim, how="left", distance_col="snap_m")
        tied = int(nb.index.duplicated(keep=False).sum())
        one = nb[~nb.index.duplicated(keep="first")]
        vals = one[[SEG_IDX, "snap_m"]].merge(
            seg[[SEG_IDX, *[rec[m] for m in meas]]], on=SEG_IDX, how="left"
        )
        for m in meas:
            points[f"{label}_{m}"] = vals[rec[m]].to_numpy()
        ctx.count(
            f"{label}: points snapped to a segment",
            len(one),
            f"median {one['snap_m'].median():.3f} m, {tied} equidistant-tie rows",
        )

        processed.append(
            {
                "label": label,
                "beta": float(rec["beta"]),
                "role": rec["role"],
                "mode": rec["mode"],
                "path": rec["path"],
                "measures": "|".join(meas),
                **{f"{m}_column": rec[m] for m in meas},
                "sha256": common.checksum(path),
                "n_segments": len(seg),
                "n_cells_with_segments": int(out[n_seg_col].notna().sum()),
                "segment_cell_pairs": int(out[n_seg_col].fillna(0).sum()),
                "n_points": len(one),
                "n_tie_rows": tied,
                "snap_median_m": float(one["snap_m"].median()),
                "snap_max_m": float(one["snap_m"].max()),
            }
        )

        # -- the reproduction is also the plumbing test -----------------------
        if rec["role"] == "reproduction":
            for slot in SUBMITTED_SLOTS:
                frozen_col = cfg["measures"]["columns"]["aggregated"][slot]
                got = cells.set_index("cell_id")[f"{label}_{slot}_mean"]
                ref = agg.set_index("cell_id")[frozen_col].astype(float)
                checks.append(
                    {
                        "label": label,
                        "level": "cells (500 m, intersects)",
                        "measure": slot,
                        "against": frozen_col,
                        **shares(
                            got.to_numpy(),
                            ref.reindex(got.index).to_numpy(),
                            tight,
                            wide,
                        ),
                    }
                )
            for slot in SUBMITTED_SLOTS:
                checks.append(
                    {
                        "label": label,
                        "level": f"points (nearest segment, {pos})",
                        "measure": slot,
                        "against": f"frozen_{slot}",
                        **shares(
                            points[f"{label}_{slot}"].to_numpy(),
                            points[f"frozen_{slot}"].to_numpy(),
                            tight,
                            wide,
                        ),
                    }
                )
            for c in checks:
                if c["label"] != label:
                    continue
                ctx.log.info(
                    "   check %-34s %s vs %-16s n=%d within 0.1%% %.4f%% "
                    "within 1%% %.4f%% median rel %.3g max %.3g",
                    c["level"],
                    c["measure"],
                    c["against"],
                    c["n"],
                    100 * c["within_tight"],
                    100 * c["within_wide"],
                    c["median_rel"],
                    c["max_rel"],
                )

    # -- write ----------------------------------------------------------------
    agg_name = Path(sens["aggregated_layer"]).name
    gpkg = stage2.write_layer(ctx, cells, agg_name)
    pts_path = common.REPO / sens["points_table"]
    pts_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(pts_path, index=False)
    ctx.log.info("wrote %s (%d points)", pts_path.relative_to(common.REPO), len(points))

    proc = pd.DataFrame(processed)
    proc.to_csv(ctx.table("stage2_pc_beta_layers.csv"), index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(ctx.table("stage2_pc_beta_reproduction_checks.csv"), index=False)

    manifest = {
        "produced_by": "src/21_pc_beta_layers.py",
        "what": (
            "each registered PC run put on the 500 m grid (`intersects`) "
            "and on the frozen address points (nearest segment, junction "
            "ties counted), one series per measure slot (PC1 = gamma 1, "
            "PC2 = gamma 0, PC3 = gamma 0.5)"
        ),
        "measure_slots": {"PC1": "gamma 1", "PC2": "gamma 0", "PC3": "gamma 0.5"},
        "assignment_rule": rule,
        "aggregation": cfg["grid"]["aggregation"],
        "point_position": pos,
        "runs": processed,
        "reproduction_checks": checks,
        "outputs": {
            "aggregated": str(gpkg.relative_to(common.REPO)),
            "points": str(pts_path.relative_to(common.REPO)),
        },
        "seed": int(cfg["repro"]["seed"]),
    }
    man = ctx.interim("pc_beta_layers_manifest.json")
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    ctx.log.info("wrote %s", man.relative_to(common.REPO))
    ctx.write_counts("stage2_pc_beta_counts.csv")
    ctx.log.info("next: make pc-beta  (Rscript src/41_pc_beta_sensitivity.R)")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
