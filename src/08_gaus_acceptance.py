#!/usr/bin/env python
"""Acceptance test for a GAUS Lines run — CC / BC / FK / connectivity.

The submitted `GgAcc0` (CC), `GgBtw0` (BC), `GgCen0` (FK) and `GgCnc0`
(connectivity) columns are the output of the GAUS Lines v1.1 QGIS plugin
(Dalcin and Krafta 2021) and are frozen inputs. `src/13_gaus_headless.py` runs
the plugin headlessly so the same four measures can be recomputed at other
network extents for the boundary-sensitivity analysis. This test checks such a
run against the file of record:

    make gaus-accept FILE=<candidate>          # or
    python src/08_gaus_acceptance.py <candidate> [...]

Three tiers, all reported; the verdict is decided by tier 1 — the same design,
the same segment-matching machinery and the same partial-network handling as
`src/06_pc_acceptance.py`, whose functions this script imports rather than
restates.

  **Tier 1 — segment level** against the file of record
  (`inputs.segment_centralities`, `edges_combined.shp`). Per measure: exact
  share, share within 1e-9, within 0.1 %, within 1 %, max |rel diff|, Pearson
  and Spearman r.

  **Tier 2 — hexagon level** against the frozen 500 m layer's
  `GgAcc0_mean/_sum`, `GgBtw0_mean/_sum`, `GgCen0_mean/_sum` under the
  `intersects` rule of the submitted aggregation, within
  `gaus_acceptance.tier2.within_rel_tol`. Reported, does not decide.
  Connectivity has no frozen aggregate and is not scored here.

  **Tier 3 — point level** against the frozen 9,030-point layer's `CC` / `BC` /
  `FK`, within `gaus_acceptance.tier3.within_rel_tol`, junction ties allowed.
  Reported, does not decide.

**The tier-1 thresholds are tighter than the PC test's, deliberately**
(`gaus_acceptance.tier1` in `config.yaml`): PASS needs ≥ 99.99 % of common
segments within **1e-9** relative, not the PC test's 0.1 %. GAUS is
deterministic double arithmetic over integer path counts, so the only
legitimate source of disagreement on the reference extent is the reassociation
of ~65k float additions. A run that agrees only to 0.1 % has a *logic*
difference — a different adjacency rule, a different tie-break, a missing
halving — and a looser band would hide it.

**A clipped extent is expected to differ.** CC, BC and FK are global sums over
every reachable line, so removing lines changes them everywhere, not only near
the cut. The banner says so, every statistic is computed on the common segments,
and the verdict is only a pass/fail claim about a run that says it is the
*reference* extent. For a boundary variant (`--informational`) the tier-1
numbers are the *measurement* — how far the measures move — and the report
tabulates that split by whether a segment kept all of its neighbours.

Everything it reads is read-only. It estimates nothing.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import acceptance_lib as lib
import common


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The PC acceptance test owns segment matching, the per-column comparison and
# both reported tiers; they are imported so the two tests cannot drift apart.
pcacc = _load("pcacc", "06_pc_acceptance.py")
stage2 = pcacc.stage2
SEG_IDX = pcacc.SEG_IDX

# Which measures the frozen 500 m layer and the frozen point layer carry. The
# fourth measure, connectivity, exists only at the segment level.
TIER2_MEASURES = ("CC", "BC", "FK")
TIER3_MEASURES = ("CC", "BC", "FK")


# ---------------------------------------------------------------------------
# tier 1 — the numbers, at the GAUS tolerances
# ---------------------------------------------------------------------------


def compare_measure(
    cand: pd.Series, ref: pd.Series, tol: dict[str, float]
) -> dict[str, Any]:
    """`pc_acceptance.compare_series` plus a 1 % column.

    The PC test's three bands are (exact, tight, wide) = (1e-9, 0.1 %, 1 %). The
    GAUS test's are (1e-9, 1e-9, 0.1 %) — `tight` *is* the exact band here — so
    the conventional 1 % figure is computed alongside rather than read off
    `within_wide`.
    """
    rec = pcacc.compare_series(cand, ref, tol)
    one_pct = lib.shares(cand, ref, tol["exact"], 0.01)
    rec["within_1pct"] = float(one_pct["within"]) if one_pct["n"] else np.nan
    return rec


def worst_segments(
    m: pd.DataFrame, ref_col: str, cand_col: str, id_cols: list[str], n: int
) -> pd.DataFrame:
    """The `n` largest relative deviations, with their `(node1, node2)` key."""
    a = m["R__" + ref_col].astype(float)
    b = m["C__" + cand_col].astype(float)
    both_zero = (a == 0) & (b == 0)
    rel = ((b - a).abs() / a.abs().replace(0.0, np.nan)).where(~both_zero, 0.0)
    out = pd.DataFrame({"reference": a, "candidate": b, "abs_rel_diff": rel})
    for c in id_cols:
        if "R__" + c in m.columns:
            out[c] = m["R__" + c]
    return out.sort_values("abs_rel_diff", ascending=False).head(n)


def boundary_movement(
    m: pd.DataFrame,
    cols: dict[str, str],
    cand_cols: dict[str, str],
    cnc: str | None,
) -> tuple[pd.DataFrame, str]:
    """How far each measure moves on the common segments, interior vs boundary.

    "Boundary" is read off connectivity, not off geometry: a common segment whose
    candidate degree is **lower** than its reference degree lost at least one
    neighbour to the clip and is therefore adjacent to the cut. Everything else
    kept its whole local neighbourhood, so whatever it moves by is the *global*
    effect of the missing lines rather than a local rewiring.
    """
    if cnc is None or ("R__" + cols[cnc]) not in m.columns:
        return pd.DataFrame(), (
            "Boundary split not available: the candidate carries no connectivity "
            "column, so a segment that lost a neighbour cannot be told from one "
            "that did not."
        )
    dref = m["R__" + cols[cnc]].astype(float)
    dcand = m["C__" + cand_cols[cnc]].astype(float)
    lost = (dcand < dref).to_numpy()
    rows = []
    for meas, rc in cols.items():
        cc = cand_cols[meas]
        if ("R__" + rc) not in m.columns or ("C__" + cc) not in m.columns:
            continue
        a = m["R__" + rc].astype(float)
        b = m["C__" + cc].astype(float)
        both_zero = (a == 0) & (b == 0)
        rel = ((b - a).abs() / a.abs().replace(0.0, np.nan)).where(~both_zero, 0.0)
        for grp, mask in (
            ("all common", np.ones(len(m), dtype=bool)),
            ("kept every neighbour", ~lost),
            ("lost >= 1 neighbour", lost),
        ):
            r = rel[mask]
            rows.append(
                {
                    "measure": meas,
                    "segments": grp,
                    "n": int(mask.sum()),
                    "median_abs_rel_diff": float(np.nanmedian(r)) if len(r) else np.nan,
                    "p90_abs_rel_diff": float(np.nanquantile(r, 0.90))
                    if len(r)
                    else np.nan,
                    "max_abs_rel_diff": float(np.nanmax(r)) if len(r) else np.nan,
                    "median_ratio_cand_over_ref": float(
                        (b[mask] / a[mask].replace(0.0, np.nan)).median()
                    )
                    if len(r)
                    else np.nan,
                }
            )
    note = (
        f"{int(lost.sum()):,} of {len(m):,} common segments lost at least one "
        "neighbour to the clip; the rest kept their whole local neighbourhood, so "
        "their movement is the global effect of the missing lines. A segment "
        "where both sides are 0 counts as agreeing."
    )
    return pd.DataFrame(rows), note


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def write_report(ctx: common.Context, R: dict[str, Any], label: str) -> Path:
    L: list[str] = [
        f"# GAUS acceptance test — `{R['candidate_path']}`",
        "",
        (
            "*Generated by `src/08_gaus_acceptance.py` "
            f"(`make gaus-accept FILE={R['candidate_path']}`). "
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
        "Keys tried, in `gaus_acceptance.key_priority` order:",
        "",
    ]
    L += lib.md_table(R["keys_tried"])
    L += ["", "Per-measure agreement on the common segments:", ""]
    L += lib.md_table(R["tier1"])
    L += ["", R["tier1_note"], ""]
    if len(R["worst"]):
        L += [
            "Worst segments (largest relative deviation per measure):",
            "",
            *lib.md_table(R["worst"]),
            "",
        ]
    L += [
        "---",
        "",
        "## Boundary movement on the common segments",
        "",
        R["move_note"],
        "",
    ]
    if len(R["movement"]):
        L += lib.md_table(R["movement"])
        L += [""]
    L += [
        "---",
        "",
        "## Tier 2 — hexagon level, frozen 500 m layer",
        "",
        R["tier2_note"],
        "",
    ]
    if R["tier2"] is not None and len(R["tier2"]):
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
    if R["tier3"] is not None and len(R["tier3"]):
        L += lib.md_table(R["tier3"])
    L += [
        "",
        "---",
        "",
        "## Tables",
        "",
        f"- `outputs/tables/gaus_acceptance_{label}_summary.csv`",
        f"- `outputs/tables/gaus_acceptance_{label}_segments.csv`",
        f"- `outputs/tables/gaus_acceptance_{label}_boundary.csv`",
        f"- `outputs/tables/gaus_acceptance_{label}_hex.csv`",
        f"- `outputs/tables/gaus_acceptance_{label}_points.csv`",
        "",
        "The thresholds are `gaus_acceptance:` in `config.yaml`.",
        "",
    ]
    p = ctx.path("reports") / "gaus_acceptance" / f"{label}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n")
    ctx.log.info("wrote %s", p.relative_to(common.REPO))
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="08_gaus_acceptance.py",
        description=(
            "Acceptance test for a GAUS Lines run: does it reproduce the "
            "submitted CC / BC / FK / connectivity on the reference network?"
        ),
    )
    ap.add_argument("file", help="candidate run (.shp / .gpkg / .csv with WKT)")
    ap.add_argument("--layer", default=None, help="layer name (GeoPackage)")
    ap.add_argument("--cc-col", default=None, help="candidate column for CC")
    ap.add_argument("--bc-col", default=None, help="candidate column for BC")
    ap.add_argument("--fk-col", default=None, help="candidate column for FK")
    ap.add_argument("--cnc-col", default=None, help="candidate column for connectivity")
    ap.add_argument("--id-cols", nargs="*", default=None, help="segment id columns")
    ap.add_argument("--crs", type=int, default=None, help="EPSG code of the candidate")
    ap.add_argument("--wkt-column", default=None, help="WKT column name (CSV input)")
    ap.add_argument("--label", default=None, help="output stem (default: file stem)")
    ap.add_argument(
        "--informational",
        action="store_true",
        help=(
            "score a declared variant — a boundary extent, say: every tier-1/2/3 "
            "number is computed and reported as the size of the difference, the "
            "verdict decides nothing and the exit code is always 0 (the same "
            "mode as the PC test's)"
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "gaus_acceptance")
    acfg = ctx.cfg["gaus_acceptance"]
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

    ref_cols: dict[str, str] = dict(acfg["columns"])
    overrides = {
        "CC": args.cc_col,
        "BC": args.bc_col,
        "FK": args.fk_col,
        "CNC": args.cnc_col,
    }
    cand_cols = {m: (overrides.get(m) or c) for m, c in ref_cols.items()}

    # -- reference: the file of record ---------------------------------------
    ref = stage2.load_segment_centralities(ctx)
    ref_note = str(scfg["path"])
    ctx.count("reference segments", len(ref), ref_note)

    cand, declared_crs = pcacc.read_candidate(
        cand_path,
        crs,
        layer=args.layer,
        crs_override=args.crs,
        wkt_column=args.wkt_column,
    )
    ctx.count("candidate segments", len(cand), str(cand_path.name))

    missing = [m for m, c in cand_cols.items() if c not in cand.columns]
    absent_ref = [m for m, c in ref_cols.items() if c not in ref.columns]
    if absent_ref:
        raise SystemExit(
            "the reference layer carries no column for "
            + ", ".join(f"{m} (looked for `{ref_cols[m]}`)" for m in absent_ref)
        )

    # -- tier 1a: identity ----------------------------------------------------
    pairs, key = pcacc.match_segments(ctx, ref, cand, acfg, id_cols)
    ctx.count(
        "segments common to candidate and reference",
        key["n_matched"],
        f"key '{key['key_used']}'",
    )

    # -- tier 1b: the numbers -------------------------------------------------
    rows: list[dict[str, Any]] = []
    worst_rows: list[pd.DataFrame] = []
    movement, move_note = pd.DataFrame(), "Not computed: no common segments."
    m = pd.DataFrame()
    if key["n_matched"]:
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
        for meas, rc in ref_cols.items():
            cc = cand_cols[meas]
            rec = {
                "measure": meas,
                "reference_column": rc,
                "candidate_column": cc,
                "decides": meas in acfg["verdict_measures"],
            }
            if ("C__" + cc) not in m.columns:
                rec.update({"status": "COLUMN MISSING", "n_common": 0})
                rows.append(rec)
                continue
            rec["status"] = "compared"
            rec.update(
                compare_measure(
                    m["C__" + cc].astype(float), m["R__" + rc].astype(float), tol
                )
            )
            rows.append(rec)
            w = worst_segments(m, rc, cc, id_cols, int(t1cfg["worst_segments_listed"]))
            w.insert(0, "measure", meas)
            worst_rows.append(w)
        cnc_key = "CNC" if "CNC" in ref_cols and "CNC" not in missing else None
        movement, move_note = boundary_movement(m, ref_cols, cand_cols, cnc_key)
    tier1 = pd.DataFrame(rows)
    worst = pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame()

    # -- verdict (tier 1 only) ------------------------------------------------
    pass_tight = float(t1cfg["pass_share_tight"])
    warn_wide = float(t1cfg["warn_share_within"])
    decide = [m_ for m_ in acfg["verdict_measures"] if m_ in ref_cols]
    reasons: list[str] = []
    if missing:
        verdict = "FAIL"
        reasons.append(
            "the candidate has no column for "
            + ", ".join(f"{x} (looked for `{cand_cols[x]}`)" for x in missing)
        )
    elif not key["n_matched"]:
        verdict = "FAIL"
        reasons.append(
            "no candidate segment could be matched to a reference segment under "
            f"any key in {list(acfg['key_priority'])}"
        )
    else:
        core = tier1[tier1["measure"].isin(decide)]
        if (core["within_tight"] >= pass_tight).all():
            verdict = "PASS"
        elif (core["within_wide"] >= warn_wide).all():
            verdict = "PASS WITH WARNING"
            reasons.append(
                f"every scored measure clears {warn_wide:.2%} of common segments "
                f"within {tol['wide']:.1%}, but not {pass_tight:.2%} within "
                f"{tol['tight']:.0e} — that band is reserved for float "
                "reassociation, so something in the logic differs"
            )
        else:
            verdict = "FAIL"
            for _, r in core.iterrows():
                if r["status"] != "compared":
                    continue
                if r["within_tight"] >= pass_tight:
                    continue
                reasons.append(
                    f"{r['measure']} (`{r['candidate_column']}` vs "
                    f"`{r['reference_column']}`): "
                    f"{pcacc.fmt_pct(r['within_tight'])} of "
                    f"{int(r['n_common']):,} common segments within "
                    f"{tol['tight']:.0e}, {pcacc.fmt_pct(r['within_wide'])} within "
                    f"{tol['wide']:.1%}, max |rel diff| "
                    f"{r['max_abs_rel_diff']:.4g}, Pearson r {r['pearson_r']:.6f}"
                )

    # -- tiers 2 and 3 --------------------------------------------------------
    tier2, tier2_worst, tier2_note = None, None, ""
    tier3, tier3_note = None, ""
    if key["n_matched"] and not missing:
        t2_cols = {k: cand_cols[k] for k in TIER2_MEASURES if k in cand_cols}
        t2_stem = {k: ref_cols[k] for k in TIER2_MEASURES if k in ref_cols}
        tier2, tier2_worst, tier2_note = pcacc.run_tier2(
            ctx, cand, t2_cols, ref, t2cfg, tol, frozen_stem=t2_stem
        )
        tier2_note += (
            " Connectivity has no `*_mean` / `*_sum` twin in the frozen layer and "
            "is scored at the segment level only."
        )
        t3_cols = {k: cand_cols[k] for k in TIER3_MEASURES if k in cand_cols}
        tier3, tier3_note = pcacc.run_tier3(
            ctx, cand, t3_cols, t3cfg, tol, ref, set(pairs["ref_idx"].to_numpy())
        )
    else:
        tier2_note = tier3_note = (
            "Not run: tier 1 could not identify the candidate's columns."
        )

    # -- banner, notes, outputs ----------------------------------------------
    banner = ""
    if key["partial"]:
        banner = (
            f"**Partial network — statistics on {key['n_matched']:,} common "
            f"segments.** The candidate covers "
            f"{key['coverage_of_reference']:.2%} of the {len(ref):,} reference "
            "segments. A clipped or otherwise reduced extent is **expected** to "
            "differ, and by more than the cut: CC, BC and FK are global sums over "
            "every reachable line, so removing lines moves the values on segments "
            "far from the boundary as well. The verdict below is a pass/fail claim "
            "only for a run that says it is the reference extent; for a boundary "
            "variant the tier-1 numbers and the movement table are the "
            "*measurement*, not a failure."
        )
    exit_code = 1 if verdict == "FAIL" else 0
    tier1_verdict = verdict
    verdict_note = (
        (
            "CC, BC, FK and connectivity reproduce the reference"
            + (
                f" on the {key['n_matched']:,} segments this candidate shares with "
                "the reference network."
                if key["partial"]
                else " on the reference network, segment for segment."
            )
        )
        if verdict == "PASS"
        else ("**" + verdict + ".** " + " · ".join(reasons))
    )

    # -- informational mode -------------------------------------------------
    # A declared boundary variant is a different extent, and CC / BC / FK are
    # global sums over every reachable line: it is expected to move, and by more
    # than the cut. Scoring it against the submitted columns is still worth
    # doing -- the distance is the sensitivity result -- but a FAIL verdict on
    # a run that was never meant to reproduce would be a category error.
    if args.informational:
        exit_code = 0
        verdict_note = (
            "**INFORMATIONAL.** Scored as a declared variant, not as a "
            "reproduction: the tier-1 verdict below is "
            f"**{tier1_verdict}**, it decides nothing, and the numbers are the "
            "size of the difference the declared change makes. " + verdict_note
        )
        verdict = "INFORMATIONAL"
        banner = (
            "**Informational run.** This candidate declares an extent that "
            "differs from the reference network, so it is scored for the size of "
            "the difference, not for agreement. Nothing here can fail. "
        ) + banner
    tier1_note = (
        f"`exact` and `within_tight` are both within {tol['tight']:.0e}; "
        f"`within_wide` is within {tol['wide']:.1%}; `within_1pct` is within 1 %. "
        "A segment where both sides are 0 counts as agreeing. PASS needs "
        f"every scored measure ({', '.join(decide)}) at `within_tight` ≥ "
        f"{pass_tight:.2%}; PASS WITH WARNING needs `within_wide` ≥ "
        f"{warn_wide:.2%}. The tight band is {tol['tight']:.0e} rather than the PC "
        "test's 0.1 % because GAUS is deterministic double arithmetic over integer "
        "path counts and the only legitimate disagreement is float reassociation."
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
            {"item": "reference", "value": ref_note},
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
            {"item": "verdict_measures", "value": ";".join(decide)},
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
    summary.to_csv(ctx.table(f"gaus_acceptance_{label}_summary.csv"), index=False)
    tier1.to_csv(ctx.table(f"gaus_acceptance_{label}_segments.csv"), index=False)
    movement.to_csv(ctx.table(f"gaus_acceptance_{label}_boundary.csv"), index=False)
    (tier2 if tier2 is not None else pd.DataFrame()).to_csv(
        ctx.table(f"gaus_acceptance_{label}_hex.csv"), index=False
    )
    (tier3 if tier3 is not None else pd.DataFrame()).to_csv(
        ctx.table(f"gaus_acceptance_{label}_points.csv"), index=False
    )

    write_report(
        ctx,
        {
            "candidate_path": cand_rel,
            "reference_path": ref_note,
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
            "worst": worst,
            "movement": movement,
            "move_note": move_note,
            "tier2": tier2,
            "tier2_worst": tier2_worst,
            "tier2_note": tier2_note,
            "tier3": tier3,
            "tier3_note": tier3_note,
        },
        label,
    )
    ctx.write_counts(f"gaus_acceptance_{label}_counts.csv")
    ctx.log.info("VERDICT: %s (exit %d)", verdict, exit_code)
    print(f"\nGAUS ACCEPTANCE [{label}]: {verdict}")
    for r in reasons:
        print(f"  - {r}")
    ctx.finish()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
