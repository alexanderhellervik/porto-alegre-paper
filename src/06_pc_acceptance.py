#!/usr/bin/env python
"""Acceptance test for a preferential-centrality (PC) run.

A new PC run — the submitted run recomputed with `prefcent` (Hellervik 2026), a
beta variant, or a run on a different network extent — is scored against the
submitted PC1 and PC2 before it is used:

    make pc-accept FILE=<candidate>            # or
    python src/06_pc_acceptance.py <candidate> [--pc1-col ...] [--pc2-col ...]

Three tiers, all reported; the verdict is decided by tier 1.

  **Tier 1 — segment level** against the file of record
  (`inputs.segment_centralities`, `edges_combined.shp`). Candidate segments are
  matched to reference segments by an explicit, reported key: `(node1, node2)`,
  exact WKB, a rounded-coordinate key, nearest centroid within a tolerance, or
  row order. For PC1, PC2 (plus the `ca*` mass twins and `area` / `weight` when
  present): exact share, within 0.1 %, within 1 %, max |rel diff|, Pearson and
  Spearman r, and the share of zero-mass segments that agree.

  **Tier 2 — hexagon level** against the frozen 500 m layer: aggregate the
  candidate into the regenerated 500 m grid under the `intersects` rule of the
  submitted aggregation and score against `cd1g1b2k0_mean/_sum`,
  `cd4g0b2k0_mean/_sum`.

  **Tier 3 — point level** against the frozen 9,030-point layer: nearest-segment
  value vs the point's own PC1 / PC2, with junction ties allowed to match any
  equidistant segment.

A **partial network** — a boundary-extent variant covering less than the
reference — is expected, not a failure: coverage is reported, every statistic is
computed on the common segments, and tier 2 is restricted to fully covered
cells.

Thresholds and tolerances are read from `pc_acceptance:` in `config.yaml`.
Exit code 0 for PASS and PASS WITH WARNING, 1 for FAIL; `--informational`
always exits 0.

Everything it reads is read-only. It estimates nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import acceptance_lib as lib
import common

# The agreement rule, the cell assignment and the point snapping live in
# `acceptance_lib`, shared with the GAUS acceptance test so the two cannot
# drift apart.
stage2 = lib.stage2

SEG_IDX = stage2.SEG_IDX
PARAM_RE = lib.PARAM_RE

KEY_METHODS = (
    "id_cols",
    "geometry_wkb",
    "rounded_coords",
    "nearest_centroid",
    "row_order",
)


# ---------------------------------------------------------------------------
# loading a candidate
# ---------------------------------------------------------------------------


def read_candidate(
    path: Path,
    target_crs: str,
    *,
    layer: str | None = None,
    crs_override: int | None = None,
    wkt_column: str | None = None,
) -> tuple[gpd.GeoDataFrame, str]:
    """Read a shapefile / GeoPackage / CSV-with-WKT into the grid CRS.

    Returns the layer and the CRS it declared, so the report can say whether the
    CRS came from the file or from `--crs`. A CSV carries no CRS at all and must
    be told one: a distance-based comparison is never run on an implicit
    projection.
    """
    if path.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path)
        col = wkt_column
        if col is None:
            for cand in ("geometry", "wkt", "WKT", "geom", "the_geom"):
                if cand in df.columns:
                    col = cand
                    break
        if col is None:
            raise SystemExit(
                f"{path.name} is a CSV and no WKT column was found; pass "
                "--wkt-column (accepted inputs are a shapefile, a GeoPackage, "
                "or a CSV with a WKT geometry column)"
            )
        geom = gpd.GeoSeries([shapely_wkt.loads(s) for s in df[col]])
        gdf = gpd.GeoDataFrame(df.drop(columns=[col]), geometry=geom)
        declared = "none (CSV)"
        if crs_override is None:
            raise SystemExit(
                f"{path.name} is a CSV: its CRS is not recorded in the file. "
                "Pass --crs <epsg>."
            )
        gdf = gdf.set_crs(f"EPSG:{crs_override}")
    else:
        gdf = gpd.read_file(path, layer=layer)
        declared = str(gdf.crs)
        if crs_override is not None:
            gdf = gdf.set_crs(f"EPSG:{crs_override}", allow_override=True)
            declared = f"{declared} (overridden with --crs {crs_override})"
        elif gdf.crs is None:
            raise SystemExit(
                f"{path.name} declares no CRS. Pass --crs <epsg> — a "
                "distance-based comparison on an unknown projection is not a "
                "comparison."
            )
    return gdf.to_crs(target_crs).reset_index(names=SEG_IDX), declared


# ---------------------------------------------------------------------------
# tier 1a — segment identity
# ---------------------------------------------------------------------------


def _coord_key(gdf: gpd.GeoDataFrame, nd: int) -> pd.Series:
    """Direction-insensitive endpoint key at `nd` decimal places (metres).

    Two exports of the same network can disagree on a LineString's direction and
    on the last bits of a coordinate; the endpoints rounded to a stated tolerance
    plus the rounded length identify a segment across both.
    """
    out = []
    for g in gdf.geometry:
        c = np.round(np.asarray(g.coords, dtype=float)[:, :2], nd)
        a, b = tuple(c[0]), tuple(c[-1])
        if b < a:
            a, b = b, a
        out.append((a[0], a[1], b[0], b[1], round(g.length, nd)))
    return pd.Series(out, index=gdf.index)


def match_segments(
    ctx: common.Context,
    ref: gpd.GeoDataFrame,
    cand: gpd.GeoDataFrame,
    cfg: dict[str, Any],
    id_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match candidate segments to reference segments, best key first.

    Returns the (ref `_seg_idx`, cand `_seg_idx`) pairs and a record of which key
    was used, why the earlier ones were skipped, and the coverage reached.
    """
    nd = round(-np.log10(float(cfg["coord_round_m"])))
    tried: list[dict[str, Any]] = []
    for method in cfg["key_priority"]:
        if method not in KEY_METHODS:
            raise ValueError(f"unknown pc_acceptance.key_priority entry: {method}")
        note = ""
        pairs = None
        if method == "id_cols":
            if not id_cols:
                note = "no id columns given (--id-cols)"
            elif not all(c in ref.columns and c in cand.columns for c in id_cols):
                note = f"id columns {id_cols} not present in both files"
            else:
                rk = ref[id_cols].astype(str).agg("|".join, axis=1)
                ck = cand[id_cols].astype(str).agg("|".join, axis=1)
                pairs = _join_on_key(ref, cand, rk, ck)
        elif method == "geometry_wkb":
            pairs = _join_on_key(
                ref, cand, ref.geometry.to_wkb(), cand.geometry.to_wkb()
            )
        elif method == "rounded_coords":
            pairs = _join_on_key(ref, cand, _coord_key(ref, nd), _coord_key(cand, nd))
        elif method == "nearest_centroid":
            pairs = _join_nearest_centroid(
                ref, cand, float(cfg["nearest_centroid_tol_m"])
            )
        elif method == "row_order":
            if len(ref) != len(cand):
                note = f"row counts differ ({len(cand):,} vs {len(ref):,})"
            else:
                pairs = pd.DataFrame(
                    {
                        "ref_idx": ref[SEG_IDX].to_numpy(),
                        "cand_idx": cand[SEG_IDX].to_numpy(),
                    }
                )
        n = 0 if pairs is None else len(pairs)
        tried.append(
            {
                "key": method,
                "matched": n,
                "coverage_of_candidate": n / len(cand) if len(cand) else np.nan,
                "coverage_of_reference": n / len(ref) if len(ref) else np.nan,
                "note": note,
            }
        )
        ctx.count(f"segment key '{method}': matched", n, note)
        if n:
            info = {
                "key_used": method,
                "n_matched": n,
                "keys_tried": pd.DataFrame(tried),
                "coverage_of_candidate": n / len(cand),
                "coverage_of_reference": n / len(ref),
                "partial": n < len(ref),
            }
            return pairs, info
    return pd.DataFrame(columns=["ref_idx", "cand_idx"]), {
        "key_used": "none",
        "n_matched": 0,
        "keys_tried": pd.DataFrame(tried),
        "coverage_of_candidate": 0.0,
        "coverage_of_reference": 0.0,
        "partial": True,
    }


def _join_on_key(
    ref: gpd.GeoDataFrame, cand: gpd.GeoDataFrame, rk: pd.Series, ck: pd.Series
) -> pd.DataFrame:
    """1:1 join on a key, ambiguous keys dropped on both sides.

    A key value that repeats does not identify a segment; silently keeping the
    first row would invent a correspondence. Those rows are dropped and show up
    as missing coverage, which is visible, rather than as a wrong comparison,
    which is not.
    """
    r = pd.DataFrame({"k": rk.to_numpy(), "ref_idx": ref[SEG_IDX].to_numpy()})
    c = pd.DataFrame({"k": ck.to_numpy(), "cand_idx": cand[SEG_IDX].to_numpy()})
    r = r[~r["k"].duplicated(keep=False)]
    c = c[~c["k"].duplicated(keep=False)]
    return r.merge(c, on="k", how="inner")[["ref_idx", "cand_idx"]]


def _join_nearest_centroid(
    ref: gpd.GeoDataFrame, cand: gpd.GeoDataFrame, tol_m: float
) -> pd.DataFrame:
    rc = gpd.GeoDataFrame(
        ref[[SEG_IDX]].rename(columns={SEG_IDX: "ref_idx"}),
        geometry=ref.geometry.centroid,
        crs=ref.crs,
    )
    cc = gpd.GeoDataFrame(
        cand[[SEG_IDX]].rename(columns={SEG_IDX: "cand_idx"}),
        geometry=cand.geometry.centroid,
        crs=cand.crs,
    )
    j = gpd.sjoin_nearest(cc, rc, how="inner", max_distance=tol_m, distance_col="_d")
    j = j.sort_values("_d")
    j = j[~j["cand_idx"].duplicated(keep="first")]
    j = j[~j["ref_idx"].duplicated(keep="first")]
    return j[["ref_idx", "cand_idx"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# tier 1b — the numbers
# ---------------------------------------------------------------------------


def compare_series(
    cand: pd.Series, ref: pd.Series, tol: dict[str, float]
) -> dict[str, Any]:
    """Segment-level agreement, using `acceptance_lib.shares` for the shares.

    `shares` treats 0 vs 0 as a match rather than as 0/0 = NaN — which matters
    here, because the segments with `area` = 0 carry zero mass and hence PC = 0
    in every variant.
    """
    exact, tight, wide = tol["exact"], tol["tight"], tol["wide"]
    tight_s = lib.shares(cand, ref, exact, tight)
    wide_s = lib.shares(cand, ref, exact, wide)
    c = pd.DataFrame({"o": cand, "f": ref}).dropna()
    if c.empty:
        return {
            "n_common": 0, "exact": np.nan, "within_tight": np.nan,
            "within_wide": np.nan, "max_abs_rel_diff": np.nan,
            "pearson_r": np.nan, "spearman_r": np.nan,
            "n_ref_zero": 0, "zero_agree_share": np.nan,
        }  # fmt: skip
    both_zero = (c["o"] == 0) & (c["f"] == 0)
    den = c["f"].abs().replace(0.0, np.nan)
    rel = ((c["o"] - c["f"]).abs() / den).where(~both_zero, 0.0)
    refzero = c["f"] == 0
    return {
        "n_common": len(c),
        "exact": float(tight_s["exact"]),
        "within_tight": float(tight_s["within"]),
        "within_wide": float(wide_s["within"]),
        "max_abs_rel_diff": float(np.nanmax(rel.to_numpy())) if len(rel) else np.nan,
        "pearson_r": _corr(c["o"], c["f"]),
        "spearman_r": _corr(c["o"].rank(), c["f"].rank()),
        "n_ref_zero": int(refzero.sum()),
        "zero_agree_share": float((c.loc[refzero, "o"] == 0).mean())
        if int(refzero.sum())
        else np.nan,
    }


def _corr(a: pd.Series, b: pd.Series) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])


def declared_parameters(
    column: str, known: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """What the run's (γ, β, Dp) are declared to be, by config first, name second.

    `cd{run}g{gamma}b{beta}k{int(100*Dp)}` — so `cd1g1b2k1` announces Dp = 0.01
    in its own name. Checking it separates "the numbers differ" from "this is a
    density-penalty run".

    The convention drops the decimal point (the authors' batch driver:
    `f"...b{beta}...".replace(".", "")`), so β = 1.5 is written `b15` and the
    name alone cannot say whether that is 1.5 or 15. `measures.
    pc_variant_parameters` records the parameters of every column this package
    produces or reads, and is consulted first for that reason; the regex is the
    fallback for a column that is not listed there.
    """
    if known and column in known:
        rec = known[column]
        m = PARAM_RE.match(column)
        return {
            "family": m.group(1) if m else column[:2],
            "run": int(rec["run"]),
            "gamma": float(rec["gamma"]),
            "beta": float(rec["beta"]),
            "dp": float(rec["dp"]),
            "source": "config",
        }
    m = PARAM_RE.match(column)
    if not m:
        return {}
    return {
        "family": m.group(1),
        "run": int(m.group(2)),
        "gamma": float(m.group(3)),
        "beta": float(m.group(4)),
        "dp": int(m.group(5)) / 100.0,
        "source": "column name",
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4%}"


def write_report(ctx: common.Context, R: dict[str, Any], label: str) -> Path:
    L: list[str] = [
        f"# PC acceptance test — `{R['candidate_path']}`",
        "",
        (
            "*Generated by `src/06_pc_acceptance.py` "
            f"(`make pc-accept FILE={R['candidate_path']}`). "
            f"Seed {ctx.cfg['repro']['seed']}. Reference: "
            f"`{R['reference_path']}`. All inputs read-only.*"
        ),
        "",
        f"## Verdict: **{R['verdict']}**  (exit code {R['exit_code']})",
        "",
        R["verdict_note"],
        "",
    ]
    if R["banner"]:
        L += ["> " + R["banner"], ""]
    L += ["---", "", "## Tier 1 — segment level (decides the verdict)", ""]
    L += [
        (
            f"- candidate CRS as read: `{R['candidate_crs']}`, compared in "
            f"`EPSG:{ctx.cfg['grid']['crs_epsg']}`"
        ),
        (
            f"- segments: **{R['n_candidate']:,}** candidate vs "
            f"{R['n_reference']:,} reference "
            f"({R['n_candidate'] - R['n_reference']:+,})"
        ),
        (
            f"- total length: **{R['len_candidate_km']:,.1f} km** candidate vs "
            f"{R['len_reference_km']:,.1f} km reference "
            f"({R['len_ratio']:.4%} of the reference)"
        ),
        (
            f"- segment identity resolved by **`{R['key_used']}`**; "
            f"{R['n_matched']:,} segments matched = "
            f"{R['coverage_of_candidate']:.4%} of the candidate, "
            f"{R['coverage_of_reference']:.4%} of the reference"
        ),
        "",
        "Keys tried, in `pc_acceptance.key_priority` order:",
        "",
    ]
    L += lib.md_table(R["keys_tried"])
    L += ["", "Per-column agreement on the common segments:", ""]
    L += lib.md_table(R["tier1"])
    L += ["", R["tier1_note"], "", "### Declared parameters in the column names", ""]
    L += lib.md_table(R["params"])
    L += [
        "",
        R["dp_note"],
        "",
        "---",
        "",
        "## Tier 2 — hexagon level, frozen 500 m layer",
        "",
        R["tier2_note"],
        "",
    ]
    if R["tier2"] is not None:
        L += lib.md_table(R["tier2"])
        L += ["", "Worst cells (the 10 largest relative deviations overall):", ""]
        L += lib.md_table(
            R["tier2_worst"].sort_values("abs_rel_diff", ascending=False).head(10)
        )
    L += [
        "",
        "---",
        "",
        "## Tier 3 — point level, frozen 9,030-point layer",
        "",
        R["tier3_note"],
        "",
    ]
    if R["tier3"] is not None:
        L += lib.md_table(R["tier3"])
    L += [
        "",
        "---",
        "",
        "## Tables",
        "",
        f"- `outputs/tables/pc_acceptance_{label}_summary.csv`",
        f"- `outputs/tables/pc_acceptance_{label}_segments.csv`",
        f"- `outputs/tables/pc_acceptance_{label}_hex.csv`",
        f"- `outputs/tables/pc_acceptance_{label}_points.csv`",
        "",
        "The thresholds are `pc_acceptance:` in `config.yaml`.",
        "",
    ]
    p = ctx.path("reports") / "pc_acceptance" / f"{label}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    ctx.log.info("wrote %s", p.relative_to(common.REPO))
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="06_pc_acceptance.py",
        description=(
            "Acceptance test for a preferential-centrality run: does it "
            "reproduce the submitted PC1 and PC2 on the reference network?"
        ),
    )
    ap.add_argument("file", help="candidate PC run (.shp / .gpkg / .csv with WKT)")
    ap.add_argument("--layer", default=None, help="layer name (GeoPackage)")
    ap.add_argument("--pc1-col", default=None, help="candidate column for PC1")
    ap.add_argument("--pc2-col", default=None, help="candidate column for PC2")
    ap.add_argument(
        "--id-cols",
        nargs="*",
        default=None,
        help="segment id columns (e.g. node1 node2)",
    )
    ap.add_argument("--crs", type=int, default=None, help="EPSG code of the candidate")
    ap.add_argument("--wkt-column", default=None, help="WKT column name (CSV input)")
    ap.add_argument("--label", default=None, help="output stem (default: file stem)")
    ap.add_argument(
        "--informational",
        action="store_true",
        help=(
            "score a declared variant: every tier-1/2/3 number is computed and "
            "reported as the size of the parameter's effect, the verdict decides "
            "nothing and the exit code is always 0"
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "pc_acceptance")
    acfg = ctx.cfg["pc_acceptance"]
    gcfg = ctx.cfg["grid"]
    scfg = ctx.cfg["inputs"]["segment_centralities"]
    crs = f"EPSG:{gcfg['crs_epsg']}"
    t1cfg, t2cfg, t3cfg = acfg["tier1"], acfg["tier2"], acfg["tier3"]
    tol = {
        "exact": float(t1cfg["exact_rel_tol"]),
        "tight": float(t1cfg["tight_rel_tol"]),
        "wide": float(t1cfg["within_rel_tol"]),
    }

    cand_path = Path(args.file)
    if not cand_path.is_absolute():
        cand_path = common.REPO / cand_path
    if not cand_path.exists():
        raise SystemExit(f"candidate file not found: {cand_path}")
    label = args.label or cand_path.stem
    id_cols = list(args.id_cols) if args.id_cols is not None else list(acfg["id_cols"])

    # -- reference: the file of record ---------------------------------------
    ref = stage2.load_segment_centralities(ctx)
    ctx.count("reference segments", len(ref), str(scfg["path"]))
    cand, declared_crs = read_candidate(
        cand_path,
        crs,
        layer=args.layer,
        crs_override=args.crs,
        wkt_column=args.wkt_column,
    )
    ctx.count("candidate segments", len(cand), str(cand_path.name))

    pc_cols = {
        "PC1": args.pc1_col or acfg["columns"]["PC1"],
        "PC2": args.pc2_col or acfg["columns"]["PC2"],
    }
    ref_cols = {"PC1": scfg["columns"]["PC1"], "PC2": scfg["columns"]["PC2"]}
    missing = [m for m, c in pc_cols.items() if c not in cand.columns]

    # -- tier 1a: identity ----------------------------------------------------
    pairs, key = match_segments(ctx, ref, cand, acfg, id_cols)
    ctx.count(
        "segments common to candidate and reference",
        key["n_matched"],
        f"key '{key['key_used']}'",
    )

    # -- tier 1b: the numbers -------------------------------------------------
    rows: list[dict[str, Any]] = []
    if key["n_matched"]:
        # prefix both sides before merging: a column present on one side only
        # must not silently answer for the other (that would compare a column
        # with itself and report a perfect match)
        rt = (
            ref.drop(columns="geometry")
            .add_prefix("R__")
            .rename(columns={f"R__{SEG_IDX}": "ref_idx"})
        )
        ct = (
            cand.drop(columns="geometry")
            .add_prefix("C__")
            .rename(columns={f"C__{SEG_IDX}": "cand_idx"})
        )
        m = pairs.merge(rt, on="ref_idx", how="left").merge(
            ct, on="cand_idx", how="left"
        )

        def col(name: str, side: str) -> pd.Series | None:
            c = ("R__" if side == "ref" else "C__") + name
            return m[c] if c in m.columns else None

        todo: list[tuple[str, str, str]] = []
        for meas in ("PC1", "PC2"):
            todo.append((meas, ref_cols[meas], pc_cols[meas]))
        for meas in ("PC1", "PC2"):
            twin_r = "ca" + ref_cols[meas][2:]
            twin_c = "ca" + pc_cols[meas][2:]
            if twin_r in ref.columns and twin_c in cand.columns:
                todo.append((f"{meas} mass twin", twin_r, twin_c))
        for extra in acfg["extra_columns"]:
            if extra in ref.columns and extra in cand.columns:
                todo.append((extra, extra, extra))

        for meas, rc, cc in todo:
            a, b = col(rc, "ref"), col(cc, "cand")
            if a is None or b is None:
                rows.append(
                    {
                        "measure": meas,
                        "reference_column": rc,
                        "candidate_column": cc,
                        "status": "COLUMN MISSING",
                        "n_common": 0,
                    }
                )
                continue
            rec = {
                "measure": meas,
                "reference_column": rc,
                "candidate_column": cc,
                "status": "compared",
            }
            rec.update(compare_series(b.astype(float), a.astype(float), tol))
            rows.append(rec)
    tier1 = pd.DataFrame(rows)

    # -- declared parameters in the names ------------------------------------
    prm_rows = []
    expected = ctx.cfg["measures"]["pc_variant_parameters"]
    dp_flags: list[str] = []
    for meas in ("PC1", "PC2"):
        want = expected[ref_cols[meas]]
        got = declared_parameters(pc_cols[meas], expected)
        agree = (
            bool(got)
            and float(got["gamma"]) == float(want["gamma"])
            and float(got["beta"]) == float(want["beta"])
            and float(got["dp"]) == float(want["dp"])
        )
        prm_rows.append(
            {
                "measure": meas,
                "candidate_column": pc_cols[meas],
                "declared_gamma": got.get("gamma", ""),
                "declared_beta": got.get("beta", ""),
                "declared_dp": got.get("dp", ""),
                "submitted_gamma": want["gamma"],
                "submitted_beta": want["beta"],
                "submitted_dp": want["dp"],
                "name_matches_submitted": agree if got else "unparseable name",
            }
        )
        if got and not agree:
            dp_flags.append(
                f"`{pc_cols[meas]}` announces γ={got['gamma']:g}, β={got['beta']:g}, "
                f"Dp={got['dp']:g} under the column-naming convention, but the "
                f"submitted {meas} is γ={want['gamma']}, β={want['beta']}, "
                f"Dp={want['dp']}"
            )
    params = pd.DataFrame(prm_rows)
    dp_present = sorted(
        c
        for c in cand.columns
        if PARAM_RE.match(c) and declared_parameters(c, expected).get("dp", 0) != 0
    )
    dp_note = (
        "**Column-name check.** " + " · ".join(dp_flags)
        if dp_flags
        else "Both column names announce the submitted parameters."
    )
    if acfg["report_dp_columns"]:
        dp_note += "\n\n" + (
            "Density-penalty (Dp ≠ 0) columns present in the candidate, reported "
            "only and not compared: " + ", ".join(f"`{c}`" for c in dp_present)
            if dp_present
            else "No density-penalty (Dp ≠ 0) column is present in the candidate."
        )

    # -- verdict (tier 1 only) ------------------------------------------------
    pass_tight = float(t1cfg["pass_share_tight"])
    warn_wide = float(t1cfg["warn_share_within"])
    reasons: list[str] = []
    if missing:
        verdict = "FAIL"
        reasons.append(
            "the candidate has no column for "
            + ", ".join(f"{m} (looked for `{pc_cols[m]}`)" for m in missing)
        )
    elif not key["n_matched"]:
        verdict = "FAIL"
        reasons.append(
            "no candidate segment could be matched to a reference segment under "
            f"any key in {list(acfg['key_priority'])}"
        )
    else:
        core = tier1[tier1["measure"].isin(["PC1", "PC2"])]
        if (core["within_tight"] >= pass_tight).all():
            verdict = "PASS"
            if dp_flags:
                # the numbers reproduce but the export is named as a different
                # run: a labelling defect rather than a solver defect, flagged so
                # it does not pass silently
                verdict = "PASS WITH WARNING"
                reasons.extend(dp_flags)
        elif (core["within_wide"] >= warn_wide).all():
            verdict = "PASS WITH WARNING"
            reasons.append(
                f"PC1/PC2 clear {warn_wide:.1%} of common segments within "
                f"{tol['wide']:.1%} but not {pass_tight:.2%} within "
                f"{tol['tight']:.1%}"
            )
        else:
            verdict = "FAIL"
            reasons.extend(dp_flags)
            for _, r in core.iterrows():
                reasons.append(
                    f"{r['measure']} (`{r['candidate_column']}` vs "
                    f"`{r['reference_column']}`): "
                    f"{fmt_pct(r['within_tight'])} of {int(r['n_common']):,} "
                    f"common segments within {tol['tight']:.1%}, "
                    f"{fmt_pct(r['within_wide'])} within {tol['wide']:.1%}, "
                    f"max |rel diff| {r['max_abs_rel_diff']:.4g}, "
                    f"Pearson r {r['pearson_r']:.6f}"
                )

    # -- tier 2 ---------------------------------------------------------------
    tier2, tier2_worst, tier2_note = None, None, ""
    if key["n_matched"] and not missing:
        tier2, tier2_worst, tier2_note = run_tier2(ctx, cand, pc_cols, ref, t2cfg, tol)
    else:
        tier2_note = "Not run: tier 1 could not identify the candidate's columns."

    # -- tier 3 ---------------------------------------------------------------
    tier3, tier3_note = None, ""
    if key["n_matched"] and not missing:
        tier3, tier3_note = run_tier3(
            ctx, cand, pc_cols, t3cfg, tol, ref, set(pairs["ref_idx"].to_numpy())
        )
    else:
        tier3_note = "Not run: tier 1 could not identify the candidate's columns."

    # -- banner, notes, outputs ----------------------------------------------
    banner = ""
    if key["partial"]:
        banner = (
            f"**Partial network — statistics on {key['n_matched']:,} common "
            f"segments.** The candidate covers "
            f"{key['coverage_of_reference']:.2%} of the "
            f"{len(ref):,} reference segments. That is expected for a "
            "boundary-extent variant and is not itself a failure; every tier-1 "
            "number below is computed on the intersection, and tier 2 is "
            "restricted to fully covered cells."
        )
    exit_code = 1 if verdict == "FAIL" else 0
    verdict_note = (
        "PC1 and PC2 reproduce the submitted run"
        + (
            f" on the {key['n_matched']:,} segments this candidate shares with "
            "the reference network."
            if key["partial"]
            else " on the reference network."
        )
        if verdict == "PASS"
        else ("**" + verdict + ".** " + " · ".join(reasons))
    )

    # -- informational mode -------------------------------------------------
    # A declared parameter variant is expected to move the numbers. Scoring it
    # against the submitted columns is still worth doing -- the distance is the
    # sensitivity result -- but a FAIL verdict on a run that was never meant to
    # reproduce would be a category error.
    tier1_verdict = verdict
    if args.informational:
        exit_code = 0
        verdict_note = (
            f"**INFORMATIONAL.** Scored as a declared variant, not as a "
            f"reproduction: the tier-1 shares below measure **how far this run "
            f"moves from the submitted columns**, they do not judge it. "
            f"(Had this been scored as a reproduction the verdict would be "
            f"**{tier1_verdict}**"
            + (": " + " · ".join(reasons) if reasons else "")
            + ".) Informational mode never fails; it is meant for a run whose "
            "parameters or extent are declared to differ."
        )
        verdict = "INFORMATIONAL"
    tier1_note = (
        f"`exact` = within {tol['exact']:.0e}; `within_tight` = within "
        f"{tol['tight']:.1%}; `within_wide` = within {tol['wide']:.1%}. A segment "
        "where both sides are 0 counts as agreeing — the zero-mass segments "
        "(`area` = 0) carry PC = 0 in every variant, and scoring them as "
        "0/0 = NaN would understate every column. PASS needs every PC row at "
        f"`within_tight` >= {pass_tight:.2%}; PASS WITH WARNING needs "
        f"`within_wide` >= {warn_wide:.1%}."
    )

    cand_rel = (
        str(cand_path.relative_to(common.REPO))
        if cand_path.is_relative_to(common.REPO)
        else str(cand_path)
    )
    summary = pd.DataFrame(
        [
            {"item": "candidate", "value": cand_rel},
            {"item": "candidate_crs_as_read", "value": declared_crs},
            {"item": "reference", "value": str(scfg["path"])},
            {"item": "label", "value": label},
            {"item": "segment_key", "value": key["key_used"]},
            {"item": "n_candidate_segments", "value": len(cand)},
            {"item": "n_reference_segments", "value": len(ref)},
            {"item": "n_common_segments", "value": key["n_matched"]},
            {"item": "coverage_of_reference", "value": key["coverage_of_reference"]},
            {"item": "coverage_of_candidate", "value": key["coverage_of_candidate"]},
            {"item": "partial_network", "value": key["partial"]},
            {
                "item": "candidate_length_km",
                "value": float(cand.geometry.length.sum() / 1000),
            },
            {
                "item": "reference_length_km",
                "value": float(ref.geometry.length.sum() / 1000),
            },
            {"item": "dp_nonzero_columns_present", "value": ";".join(dp_present)},
            {
                "item": "mode",
                "value": "informational" if args.informational else "acceptance",
            },
            {"item": "tier1_verdict", "value": tier1_verdict},
            {"item": "verdict", "value": verdict},
            {"item": "exit_code", "value": exit_code},
            {"item": "reasons", "value": " | ".join(reasons)},
        ]
    )
    summary.to_csv(ctx.table(f"pc_acceptance_{label}_summary.csv"), index=False)
    tier1.to_csv(ctx.table(f"pc_acceptance_{label}_segments.csv"), index=False)
    (tier2 if tier2 is not None else pd.DataFrame()).to_csv(
        ctx.table(f"pc_acceptance_{label}_hex.csv"), index=False
    )
    (tier3 if tier3 is not None else pd.DataFrame()).to_csv(
        ctx.table(f"pc_acceptance_{label}_points.csv"), index=False
    )

    if args.informational:
        banner = (
            "**Informational run.** This candidate declares parameters or an "
            "extent that differ from the submitted run, so it is scored for the "
            "size of the difference, not for agreement. Nothing here can fail. "
        ) + banner

    write_report(
        ctx,
        {
            "candidate_path": cand_rel,
            "reference_path": str(scfg["path"]),
            "candidate_crs": declared_crs,
            "verdict": verdict,
            "verdict_note": verdict_note,
            "exit_code": exit_code,
            "banner": banner,
            "n_candidate": len(cand),
            "n_reference": len(ref),
            "len_candidate_km": float(cand.geometry.length.sum() / 1000),
            "len_reference_km": float(ref.geometry.length.sum() / 1000),
            "len_ratio": float(cand.geometry.length.sum() / ref.geometry.length.sum()),
            "key_used": key["key_used"],
            "n_matched": key["n_matched"],
            "coverage_of_candidate": key["coverage_of_candidate"],
            "coverage_of_reference": key["coverage_of_reference"],
            "keys_tried": key["keys_tried"],
            "tier1": tier1,
            "tier1_note": tier1_note,
            "params": params,
            "dp_note": dp_note,
            "tier2": tier2,
            "tier2_worst": tier2_worst,
            "tier2_note": tier2_note,
            "tier3": tier3,
            "tier3_note": tier3_note,
        },
        label,
    )
    ctx.write_counts(f"pc_acceptance_{label}_counts.csv")
    ctx.log.info("VERDICT: %s (exit %d)", verdict, exit_code)
    print(f"\nPC ACCEPTANCE [{label}]: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    ctx.finish()
    return exit_code


def run_tier2(
    ctx: common.Context,
    cand: gpd.GeoDataFrame,
    pc_cols: dict[str, str],
    ref: gpd.GeoDataFrame,
    t2cfg: dict[str, Any],
    tol: dict[str, float],
    frozen_stem: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Aggregate the candidate into the 500 m grid and score against the frozen
    layer, restricted to cells the candidate covers completely.

    `pc_cols` maps a logical measure to the candidate column; `frozen_stem` maps
    the same logical measure to the stem of the frozen layer's `*_mean` / `*_sum`
    pair. The default is the PC pair, which is what this script's own main()
    wants; `src/08_gaus_acceptance.py` passes the GAUS stems instead, so the two
    tests share one tier-2 implementation. The agreement band is
    `t2cfg["within_rel_tol"]`; `tol["exact"]` is the exact-match tolerance.
    """
    gcfg = ctx.cfg["grid"]
    band = float(t2cfg["within_rel_tol"])
    agg = gpd.read_file(ctx.path("frozen", "cents_aggregated")).rename(
        columns={gcfg["frozen_500m"]["id_column"]: "cell_id"}
    )
    study_area = agg.geometry.union_all()
    if t2cfg["grid_source"] == "regenerated":
        regen = stage2.build_grid(
            gcfg, float(gcfg["spacing_m"]), study_area.bounds, clip_to=study_area
        )
        cells = regen.merge(
            agg[["cell_id", "row_index", "col_index"]], on=["row_index", "col_index"]
        )[["cell_id", "geometry"]]
    elif t2cfg["grid_source"] == "frozen":
        cells = agg[["cell_id", "geometry"]].copy()
    else:
        raise ValueError(
            f"unknown pc_acceptance.tier2.grid_source: {t2cfg['grid_source']}"
        )
    ctx.count("tier 2: 500 m cells", len(cells), t2cfg["grid_source"])

    # the reference's own per-cell segment count on this grid is the coverage
    # yardstick: a cell is fully covered when the candidate puts the same number
    # of segments in it under the `intersects` rule
    ref_pairs = stage2.assign_segments(ctx, ref, cells, "intersects", "tier2 reference")
    ref_counts = (
        ref_pairs.groupby("cell_id")
        .size()
        .reindex(cells["cell_id"])
        .fillna(0)
        .astype(int)
    )
    acc, per_cell, pairs_by_rule = lib.acceptance_test(
        ctx, cand, cells, ref_counts, ["intersects"]
    )
    pairs = pairs_by_rule["intersects"]
    covered = per_cell.loc[
        per_cell["n_intersects"].to_numpy()
        == ref_counts.reindex(per_cell["cell_id"]).to_numpy(),
        "cell_id",
    ]
    ctx.count(
        "tier 2: fully covered cells",
        len(covered),
        f"of {len(cells)}; {int(acc.iloc[0]['segment_cell_pairs']):,} segment-cell pairs",
    )

    if frozen_stem is None:
        frozen_stem = {
            "PC1": ctx.cfg["inputs"]["segment_centralities"]["columns"]["PC1"],
            "PC2": ctx.cfg["inputs"]["segment_centralities"]["columns"]["PC2"],
        }
    keep = [c for c in agg.columns if c.endswith(("_mean", "_sum")) and any(
        c.startswith(s) for s in frozen_stem.values()
    )]  # fmt: skip
    frozen = agg.set_index("cell_id")[keep]
    if t2cfg["full_coverage_only"]:
        frozen = frozen.loc[frozen.index.isin(set(covered))]

    # score every candidate column against every frozen column; this also
    # cross-scores PC1 against PC2's frozen column, which catches a swap
    slim = cand[[SEG_IDX, "geometry"]].copy()
    for c in pc_cols.values():
        slim[c] = cand[c].astype(float)
    cross = lib.hex_reproduction(
        slim,
        pairs[pairs["cell_id"].isin(set(frozen.index))],
        frozen,
        (tol["exact"], band),
    )
    rows, worst = [], []
    d = pairs.merge(slim.drop(columns="geometry"), on=SEG_IDX, how="left")
    gr = d[d["cell_id"].isin(set(frozen.index))].groupby("cell_id")
    for meas, c in pc_cols.items():
        for stat in ("mean", "sum"):
            fc = f"{frozen_stem[meas]}_{stat}"
            if fc not in frozen.columns:
                continue
            ours = getattr(gr[c], stat)().reindex(frozen.index)
            s = lib.shares(ours, frozen[fc], tol["exact"], band)
            best = cross[cross.frozen_column == fc].sort_values(
                ["within", "exact"], ascending=False
            )
            rows.append(
                {
                    "measure": meas,
                    "frozen_column": fc,
                    "candidate_column": c,
                    "n_cells": s["n"],
                    "rel_tol": band,
                    "within_band": s["within"],
                    "median_ratio": s["median_ratio"],
                    "best_matching_candidate_column": (
                        best.iloc[0]["segment_column"] if len(best) else ""
                    ),
                }
            )
            den = frozen[fc].abs().replace(0.0, np.nan)
            rel = (ours - frozen[fc]).abs() / den
            for cid, v in (
                rel.sort_values(ascending=False)
                .head(int(t2cfg["worst_cells_listed"]))
                .items()
            ):
                worst.append(
                    {
                        "measure": meas, "frozen_column": fc, "cell_id": int(cid),
                        "candidate": float(ours.get(cid, np.nan)),
                        "frozen": float(frozen[fc].get(cid, np.nan)),
                        "abs_rel_diff": float(v),
                    }
                )  # fmt: skip
    t2 = pd.DataFrame(rows)
    thr = float(t2cfg["pass_share"])
    ok = bool((t2["within_band"] >= thr).all()) if len(t2) else False
    note = (
        f"Candidate aggregated into the {t2cfg['grid_source']} 500 m grid under "
        f"`intersects` and scored against the frozen layer on "
        f"{len(frozen):,} cells"
        + (
            f" (of {len(cells):,}; restricted to the cells the candidate covers "
            "completely)"
            if t2cfg["full_coverage_only"] and len(frozen) < len(cells)
            else ""
        )
        + f". `within_band` = within {band:.2g} relative ({band:.2%}). "
        + f"Threshold: ≥ {thr:.0%} of cells within the band. "
        + (
            "**Tier 2 clears the threshold.**"
            if ok
            else "**Tier 2 misses the threshold.**"
        )
        + " Tier 2 does not decide the verdict."
    )
    return t2, pd.DataFrame(worst), note


def run_tier3(
    ctx: common.Context,
    cand: gpd.GeoDataFrame,
    pc_cols: dict[str, str],
    t3cfg: dict[str, Any],
    tol: dict[str, float],
    ref: gpd.GeoDataFrame,
    matched_ref: set[int],
) -> tuple[pd.DataFrame, str]:
    """Nearest-segment value at the frozen address points.

    The agreement band is `t3cfg["within_rel_tol"]`; `tol["exact"]` is the
    exact-match tolerance.
    """
    gcfg = ctx.cfg["grid"]
    band = float(t3cfg["within_rel_tol"])
    pt = gpd.read_file(ctx.path("frozen", "cents_disaggregated"))
    pos = gcfg["active_point_position"]
    positions = {pos: tuple(gcfg["point_positions"][pos])}
    n_all = len(pt)
    restricted = False
    if t3cfg["restrict_to_covered_points"] and len(matched_ref) < len(ref):
        # On a partial network every point outside the extent snaps to whatever
        # segment happens to be nearest the boundary, which is a statement about
        # the clip, not about the solver. Keep only the points whose nearest
        # reference segment is in the candidate.
        xc, yc = positions[pos]
        P = gpd.GeoDataFrame(
            pt[[]].copy(),
            geometry=gpd.points_from_xy(pt[xc], pt[yc]),
            crs=ref.crs,
        )
        nb = gpd.sjoin_nearest(P, ref[[SEG_IDX, "geometry"]], how="left")
        nb = nb[~nb.index.duplicated(keep="first")]
        keep = nb[SEG_IDX].isin(matched_ref).to_numpy()
        pt = pt[keep]
        restricted = True
        ctx.count(
            "tier 3: points whose nearest reference segment is in the candidate",
            len(pt),
            f"of {n_all}",
        )
    slim = cand[[SEG_IDX, "geometry"]].copy()
    for c in pc_cols.values():
        slim[c] = cand[c].astype(float)
    pm, _snap = lib.point_reproduction(ctx, pt, slim, positions, (tol["exact"], band))
    rows = []
    for meas, c in pc_cols.items():
        r = pm[(pm["measure"] == meas) & (pm["segment_column"] == c)]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append(
            {
                "measure": meas,
                "candidate_column": c,
                "n_points": int(r["n"]),
                "rel_tol": band,
                "within_band": float(r["within"]),
                "within_band_best_of_ties": float(r["within_best_of_ties"]),
                "fails": int(r["fails"]),
                "fails_at_tie": int(r["fails_at_tie"]),
                "median_rel_err": float(r["median_rel_err"]),
            }
        )
    t3 = pd.DataFrame(rows)
    thr = float(t3cfg["pass_share"])
    col = "within_band_best_of_ties" if t3cfg["allow_junction_ties"] else "within_band"
    ok = bool((t3[col] >= thr).all()) if len(t3) else False
    note = (
        "Nearest-segment value at "
        + (
            f"the {len(pt):,} of {n_all:,} frozen address points whose nearest "
            "*reference* segment is present in this partial candidate"
            if restricted
            else f"the {len(pt):,} frozen address points"
        )
        + f" (`{pos}` coordinates), scored on `{col}`"
        + (
            " — a point equidistant from several segments (a junction) may match "
            "any of them, because which one the frozen join picked is not "
            "recorded"
            if t3cfg["allow_junction_ties"]
            else ""
        )
        + f". `within_band` = within {band:.2g} relative ({band:.2%}). "
        + f"Threshold: ≥ {thr:.0%} of points within the band. "
        + (
            "**Tier 3 clears the threshold.**"
            if ok
            else "**Tier 3 misses the threshold.**"
        )
        + " Tier 3 does not decide the verdict."
    )
    return t3, note


if __name__ == "__main__":
    raise SystemExit(main())
