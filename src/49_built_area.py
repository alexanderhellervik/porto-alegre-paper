#!/usr/bin/env python
"""Stage 4 — two extra controls per address: unit count and built area.

The referees asked for built area / unit count as controls (or for the
dependent variable to be renamed), and for control of development intensity.
Neither variable exists ready-made, so this script builds them once and writes
them beside the frozen layer's own `id`; `src/49_controls_types.R` estimates
with them.

It does not re-normalise addresses and it does not invent a join. The address
key is stage 1's (`dependent_variable.rebuilt_path` carries `id` and `addr_key`
for every frozen record), and the registry rows are selected with the same mask
stage 1 uses for the unit multiplier (`prep.unit_mask` plus the only-parking
fallback). The unit multiplier this script re-derives from those rows is
checked against stage 1's own `n_units` on every address; if they disagree the
two files are not describing the same units and the script stops.

The definitions (config `revision.controls`):
  U(a)  unit count  = stage 1's `n_units` -- the dependent variable's own
                      multiplier, parent-only collapse included.
  B(a)  built area  = sum of the registry's `Built Area` over exactly the rows
                      that produced U(a).
  determinate       = none of those rows carries the literal INDETERMINADO and
                      B(a) > 0. Only determinate addresses can enter a model
                      with ln B(a) in it.

Three alternates are written alongside for comparison, never substituted for
the headline:
  B_all   registry `Built Area` summed over all rows at the address (parking in)
  B_sold  mean `Built Area` of the ITBI transactions that set the price
  Area_Const  the frozen layer's own per-record built area (one unit, not the
              plot; zero on vacant plots)

Run:  python src/49_built_area.py     (called by `make controls-types`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import prep


def counted_unit_rows(ctx: common.Context, reg: pd.DataFrame) -> pd.Series:
    """Boolean mask: the registry rows the DV's unit multiplier counts.

    Mirrors `address_aggregates()` in src/10_dependent_variable.py exactly --
    the configured `multiplier_units` subset with `parent_rows_in_multiplier`,
    plus the fallback that re-admits every row at an address whose subset
    came out empty (the only-parking addresses).
    """
    dv = ctx.cfg["dependent_variable"]
    spec = dv["variants"]["headline"]
    um = prep.unit_mask(reg, spec["multiplier_units"], dv["parent_rows_in_multiplier"])

    mode = dv["only_parking_fallback"]
    if mode == "use_all_transactions":
        have = set(reg.addr_key[um].unique())
        empty = reg.addr_key[~reg.addr_key.isin(have)]
        n_empty = empty.nunique()
        um = um | reg.addr_key.isin(set(empty.unique()))
        ctx.log.info(
            "only-parking fallback: %d addresses had no non-parking unit row; "
            "all their rows are counted",
            n_empty,
        )
    elif mode != "leave_undefined":
        raise ValueError(f"unknown only_parking_fallback: {mode}")
    return um


def main() -> None:
    ctx = common.init(4, "built_area_controls")
    cc = ctx.cfg["revision"]["controls"]
    dv = ctx.cfg["dependent_variable"]
    codes = ctx.cfg["io"]["missing_codes"]
    indet = cc["built_area"]["indeterminate_code"]
    col = cc["built_area"]["column"]

    # ---- stage 1's own per-record table: id, addr_key, n_units, Type ---------
    s1_path = common.REPO / dv["rebuilt_path"]
    if not s1_path.exists():
        raise SystemExit(f"stage-1 output missing: {s1_path} -- run `make dv` first")
    s1 = pd.read_csv(s1_path)
    ctx.count("stage 1: frozen-layer records", len(s1), str(s1_path.name))

    # ---- the raw registry / ITBI, through stage 0b's own loaders -------------
    f = prep.load_frames(ctx)
    prep.purpose_mapping(ctx, f)
    prep.annotate_parking(ctx, f)

    reg = prep.dedupe(f["registry"], dv["registry_duplicates"], prep.REG_ROW_KEY)
    ctx.count("registry: rows after dedupe", len(reg), dv["registry_duplicates"])

    um = counted_unit_rows(ctx, reg)
    units = reg[um]
    ctx.count(
        "registry: rows the multiplier counts",
        len(units),
        f"multiplier_units={dv['variants']['headline']['multiplier_units']}, "
        f"parent_rows={dv['parent_rows_in_multiplier']}",
    )

    ba = prep.to_num(units[col], codes)
    is_indet = units[col].astype(str).str.strip() == indet
    g = units.groupby("addr_key")
    per = pd.DataFrame(
        {
            "n_rows_counted": g.size(),
            "n_parent_counted": g["parent"].sum(),
            "built_area_units": ba.groupby(units.addr_key).sum(min_count=1),
            "n_indeterminate": is_indet.groupby(units.addr_key).sum(),
            "n_built_area_numeric": ba.notna().groupby(units.addr_key).sum(),
        }
    )
    # stage 1's multiplier, re-derived from the same rows.
    per["n_units_derived"] = prep.parent_only_multiplier(
        per.n_rows_counted, per.n_parent_counted, dv["parent_only_address_counts_as"]
    )

    # ---- the three alternates ------------------------------------------------
    ba_all = prep.to_num(reg[col], codes)
    per["built_area_all_rows"] = ba_all.groupby(reg.addr_key).sum(min_count=1)
    per["n_indeterminate_all_rows"] = (
        (reg[col].astype(str).str.strip() == indet).groupby(reg.addr_key).sum()
    )

    itbi = prep.dedupe(f["itbi"], dv["itbi_duplicates"], prep.ITBI_ROW_KEY)
    pm = prep.price_mask(
        itbi, dv["variants"]["headline"]["price_transactions"], dv["blank_purpose"]
    )
    sold = itbi[pm]
    ba_sold = prep.to_num(sold[col], codes)
    ba_sold = ba_sold.where(ba_sold > 0)  # ITBI codes "missing" as a plain 0
    per["built_area_sold_mean"] = ba_sold.groupby(sold.addr_key).mean()
    per["n_sold_with_built_area"] = ba_sold.notna().groupby(sold.addr_key).sum()

    # ---- onto the frozen records, through stage 1's addr_key ----------------
    out = s1[["id", "addr_key", "Type", "Terreno", "n_units"]].copy()
    for c in per.columns:
        out[c] = out.addr_key.map(per[c])
    geo = f["geolayer"]
    if not geo["id"].is_unique:
        raise SystemExit("frozen point layer: `id` is not unique")
    area_const = pd.to_numeric(geo["Area_Const"], errors="coerce")
    out["area_const"] = out["id"].map(pd.Series(area_const.to_numpy(), index=geo["id"]))
    if out["area_const"].isna().sum() != area_const.isna().sum():
        raise SystemExit("stage-1 ids do not match the frozen layer's ids")

    # The check that makes the join auditable: the multiplier re-derived from
    # the rows we summed the built area over must equal stage 1's `n_units`,
    # address by address, or the two files disagree about which units they mean.
    both = out.n_units_derived.notna() & out.n_units.notna()
    bad = int((out.n_units_derived[both] != out.n_units[both]).sum())
    n_missing = int((~both).sum())
    if bad:
        raise SystemExit(
            f"the re-derived unit multiplier disagrees with stage 1's `n_units` "
            f"on {bad} records -- the unit mask has drifted from stage 1"
        )
    ctx.log.info(
        "unit-mask check: the multiplier re-derived from the rows we sum built "
        "area over is identical to stage 1's `n_units` on %d of %d records "
        "(%d are in no registry row at all)",
        int(both.sum()),
        len(out),
        n_missing,
    )

    # ---- determinacy ---------------------------------------------------------
    rule = cc["built_area"]["determinacy"]
    if rule == "all_counted_rows_numeric":
        det = (
            out.n_rows_counted.fillna(0).gt(0)
            & out.n_indeterminate.fillna(1).eq(0)
            & out.built_area_units.fillna(0).gt(0)
        )
    elif rule == "any_row_numeric":
        det = out.n_built_area_numeric.fillna(0).gt(0) & out.built_area_units.fillna(
            0
        ).gt(0)
    else:
        raise ValueError(f"unknown determinacy rule: {rule}")
    out["built_area_determinate"] = det.astype(int)
    # the lenient count, for the report
    lenient = out.n_built_area_numeric.fillna(0).gt(0) & out.built_area_units.fillna(
        0
    ).gt(0)
    out["built_area_determinate_lenient"] = lenient.astype(int)
    out["built_area_partial"] = (lenient & ~det).astype(int)

    ctx.count(
        "records with a determinate built area",
        int(det.sum()),
        f"rule={rule}; lenient rule would keep {int(lenient.sum())}",
    )

    # parent-only addresses: U(a) collapses to 1 but B(a) sums every parent row
    par_only = out.n_units.eq(1) & out.n_rows_counted.gt(1)
    out["parent_only_collapse"] = par_only.astype(int)
    ctx.count(
        "records where U(a) collapsed to 1 (parent-only)",
        int(par_only.sum()),
        "B(a) still sums every counted row -- reported as a caveat",
    )

    out["built_area_per_unit"] = out.built_area_units / out.n_units.replace(0, np.nan)

    p = ctx.interim(Path(ctx.cfg["revision"]["controls"]["path"]).name)
    out.to_csv(p, index=False)
    ctx.log.info(
        "wrote %s (%d rows, %d columns)",
        p.relative_to(common.REPO),
        len(out),
        out.shape[1],
    )
    ctx.write_counts("stage4_controls_built_area_counts.csv")
    common.capture_env(ctx)
    ctx.finish()


if __name__ == "__main__":
    main()
