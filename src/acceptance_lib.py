"""Comparison helpers shared by the two acceptance tests.

`src/06_pc_acceptance.py` (preferential centrality) and
`src/08_gaus_acceptance.py` (GAUS Lines) score a candidate run against the
segment file of record and against the two frozen layers of the submitted
analysis. Both import the functions below rather than restating them, so the
two tests apply the same agreement rule, the same segment-to-cell assignment
and the same point-to-segment snapping.

- `shares()`             agreement shares that count 0 vs 0 as a match
- `md_table()`           a DataFrame as a Markdown table, for the reports
- `acceptance_test()`    per-cell segment counts under an assignment rule
- `hex_reproduction()`   aggregate segment columns into cells and score them
- `point_reproduction()` nearest-segment values at the frozen address points
- `PARAM_RE`             the `c[ad]<run>g<gamma>b<beta>k<100*Dp>` column pattern

This module is a library: it has no `main()` and no make target.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

# Stage 2 owns the grid lattice and the segment-assignment rules; reuse them
# rather than re-deriving either here.
_spec = importlib.util.spec_from_file_location(
    "stage2", Path(__file__).resolve().parent / "20_join_aggregate.py"
)
stage2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage2)

MEASURES = stage2.MEASURES
SEG_IDX = stage2.SEG_IDX
# Preferential-centrality column names: `ca` = mass, `cd` = density (mass / R),
# then run index, gamma, beta and int(100 * Dp), with decimal points dropped.
PARAM_RE = re.compile(r"^(c[ad])(\d+)g(\d+)b(\d+)k(\d+)$")


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------


def shares(ours: pd.Series, frozen: pd.Series, exact: float, within: float) -> dict:
    """Match shares that treat 0 vs 0 as a match rather than as 0/0 = NaN.

    Some frozen cells carry `GgBtw0_sum` = 0 and some a zero PC mean; a plain
    relative error scores those as 0/0 = NaN, i.e. as failures, and would
    understate the agreement of every column that has them.

    `exact` and `within` are the shares of rows below each relative tolerance;
    `median_ratio` is the median of ours / frozen.
    """
    c = pd.DataFrame({"o": ours, "f": frozen}).dropna()
    if c.empty:
        return {"n": 0, "exact": np.nan, "within": np.nan, "median_ratio": np.nan}
    both_zero = (c["o"] == 0) & (c["f"] == 0)
    den = c["f"].abs().replace(0.0, np.nan)
    rel = (c["o"] - c["f"]).abs() / den
    return {
        "n": len(c),
        "exact": float(((rel < exact) | both_zero).mean()),
        "within": float(((rel < within) | both_zero).mean()),
        "median_rel": float(np.nanmedian(rel)),
        "median_ratio": float((c["o"] / den).median()),
    }


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.6g}") -> list[str]:
    """Render `df` as Markdown table lines, floats through `floatfmt`."""
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in r
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return [head, rule, *rows]


# ---------------------------------------------------------------------------
# segment-to-cell assignment
# ---------------------------------------------------------------------------


def acceptance_test(
    ctx: common.Context,
    seg: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    target: pd.Series,
    rules: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Per-cell segment counts under each assignment rule vs a per-cell target.

    Returns one summary row per rule, the per-cell counts, and the
    (segment, cell) pairs each rule produced. A cell without a target counts as
    agreeing when no segment lands in it.
    """
    summary, per_cell, pairs = [], {"cell_id": cells["cell_id"]}, {}
    has_target = target.notna()
    for rule in rules:
        p = stage2.assign_segments(ctx, seg, cells, rule, "acceptance")
        pairs[rule] = p
        cnt = (
            p.groupby("cell_id").size().reindex(cells["cell_id"]).fillna(0).astype(int)
        )
        per_cell[f"n_{rule}"] = cnt.to_numpy()
        t = target.reindex(cells["cell_id"])
        agree = (cnt.to_numpy() == t.to_numpy()) | (
            (~has_target.reindex(cells["cell_id"]).to_numpy()) & (cnt.to_numpy() == 0)
        )
        summary.append(
            {
                "assignment_rule": rule,
                "segment_cell_pairs": int(cnt.sum()),
                "target_pairs": int(target.sum()),
                "distinct_segments": int(p[SEG_IDX].nunique()),
                "cells_exact": int(agree.sum()),
                "cells_total": len(cells),
                "cells_exact_share": float(agree.mean()),
                "max_abs_cell_diff": int(
                    np.nanmax(np.abs(cnt.to_numpy() - t.to_numpy()))
                ),
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(per_cell), pairs


# ---------------------------------------------------------------------------
# hexagon-level reproduction
# ---------------------------------------------------------------------------


def hex_reproduction(
    seg: gpd.GeoDataFrame,
    pairs: pd.DataFrame,
    frozen: pd.DataFrame,
    tol: tuple[float, float],
) -> pd.DataFrame:
    """Score every float segment column, aggregated into cells by mean / sum /
    median, against every frozen column with the matching suffix."""
    num = [
        c
        for c in seg.columns
        if c != SEG_IDX and seg[c].dtype.kind == "f" and c != "geometry"
    ]
    d = pairs.merge(seg[[SEG_IDX] + num], on=SEG_IDX, how="left")
    gr = d.groupby("cell_id")
    rows = []
    for stat in ("mean", "sum", "median"):
        agg = getattr(gr[num], stat)()
        for fc in frozen.columns:
            if not fc.endswith(f"_{stat}"):
                continue
            for c in num:
                rows.append(
                    {
                        "frozen_column": fc,
                        "aggregation": stat,
                        "segment_column": c,
                        **shares(agg[c].reindex(frozen.index), frozen[fc], *tol),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# point-level reproduction
# ---------------------------------------------------------------------------


def point_reproduction(
    ctx: common.Context,
    pt: gpd.GeoDataFrame,
    seg: gpd.GeoDataFrame,
    positions: dict[str, list[str]],
    tol: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each point's frozen measures with its nearest segment's columns.

    `positions` maps a label to the (x, y) column pair that places a point.
    Points equidistant from several segments are flagged as ties;
    `within_best_of_ties` lets such a point match any of its equidistant
    segments. Returns the per-(position, measure, column) scores and the snap
    distances.
    """
    num = [c for c in seg.columns if c != SEG_IDX and seg[c].dtype.kind == "f"]
    rows, snaps, nearest = [], [], {}
    for pos, (xc, yc) in positions.items():
        P = gpd.GeoDataFrame(
            pt[MEASURES].copy(),
            geometry=gpd.points_from_xy(pt[xc], pt[yc]),
            crs=seg.crs,
        )
        nb = gpd.sjoin_nearest(
            P, seg[[SEG_IDX, "geometry"]], how="left", distance_col="snap_m"
        )
        tied = set(nb.index[nb.index.duplicated(keep=False)])
        one = nb[~nb.index.duplicated(keep="first")]
        nearest[pos] = one[SEG_IDX].to_numpy()
        snaps.append(
            {
                "point_position": pos,
                "n_points": len(one),
                "n_with_equidistant_ties": len(tied),
                "snap_median_m": float(one["snap_m"].median()),
                "snap_p90_m": float(one["snap_m"].quantile(0.90)),
                "snap_max_m": float(one["snap_m"].max()),
            }
        )
        ctx.count(
            f"points snapped to nearest segment ({pos})",
            len(one),
            f"median {one['snap_m'].median():.3f} m, {len(tied)} equidistant ties",
        )
        m = one.reset_index(names="ptid").merge(
            seg[[SEG_IDX] + num], on=SEG_IDX, how="left"
        )
        m["tie"] = m["ptid"].isin(tied)
        # best-of-ties: does ANY equidistant segment carry the frozen value?
        allm = nb.reset_index(names="ptid").merge(
            seg[[SEG_IDX] + num], on=SEG_IDX, how="left"
        )
        for meas in MEASURES:
            for c in num:
                a, b = m[meas], m[c]
                bz = (a == 0) & (b == 0)
                rel = (a - b).abs() / b.abs().replace(0.0, np.nan)
                ok = (rel < tol[1]) | bz
                aa, bb = allm[meas], allm[c]
                bz2 = (aa == 0) & (bb == 0)
                ok2 = ((aa - bb).abs() / bb.abs().replace(0.0, np.nan) < tol[1]) | bz2
                rows.append(
                    {
                        "point_position": pos,
                        "measure": meas,
                        "segment_column": c,
                        "n": len(m),
                        "exact": float((((rel < tol[0]) | bz).fillna(False)).mean()),
                        "within": float(ok.fillna(False).mean()),
                        "within_best_of_ties": float(
                            ok2.fillna(False).groupby(allm["ptid"]).max().mean()
                        ),
                        "fails": int((~ok.fillna(False)).sum()),
                        "fails_at_tie": int((~ok.fillna(False) & m["tie"]).sum()),
                        "fails_frozen_zero": int((~ok.fillna(False) & (a == 0)).sum()),
                        "median_rel_err": float(np.nanmedian(rel)),
                    }
                )
    snap = pd.DataFrame(snaps)
    if len(nearest) == 2:
        a, b = list(nearest.values())
        n_diff = int((a != b).sum())
        ctx.count(
            "points whose nearest segment differs address vs snapped",
            n_diff,
            f"of {len(a)}",
        )
        snap["n_nearest_differs_between_positions"] = n_diff
    return pd.DataFrame(rows), snap
