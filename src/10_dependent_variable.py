#!/usr/bin/env python
"""Stage 1 — rebuild the dependent variable from the raw CSVs.

The frozen point layer ships `Preco` with the four-scenario unit multiplier
already applied (`Puni = Preco / Terreno`), and the script that built it is not
part of the submitted material. This stage therefore rebuilds the variable from
the ITBI transactions and the unit registry, and scores the rebuild against the
frozen column record by record.

The four scenarios, per the paper section 4.3:
  1. plot with one unit, sold          -> unit value / plot area
  2. plot with many units, one sale    -> replicate sold value across all units,
                                          sum, / plot area
  3. plot with many units, several sold-> mean of sold units assigned to all,
                                          sum, / plot area
  4. plot with no sales                -> excluded

Under the conventions in `dependent_variable:` those four branches collapse into
one expression (see the report's section 2), which is why a single formula
reproduces most of the frozen column.

Everything this script decides comes from `dependent_variable.*` and
`exclusions.*` in config.yaml; the loaders, address normalisation, purpose
mapping and parking/parent flags are shared with stage 0b via src/prep.py.

Run:  python src/10_dependent_variable.py    (or `make dv`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import prep
from prep import md_table, pct

# Columns carried from the frozen point layer into every interim output. The
# centralities are frozen inputs: attached, never recomputed.
GEO_CARRY = [
    "id",
    "addr_key",
    "Endereco",
    "Rua",
    "Numero",
    "Unidade",
    "Finalidade",
    "Type",
    "Terreno",
    "Preco",
    "Puni",
    "PC1",
    "PC2",
    "CC",
    "BC",
    "FK",
]
MEASURES = ["PC1", "PC2", "CC", "BC", "FK"]


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------


def variant_specs(ctx: common.Context) -> dict[str, dict[str, str]]:
    """The headline plus `revision.parking.alternates`, checked for drift.

    `dependent_variable.variants.headline` duplicates three top-level keys so
    the alternates can be written in the same shape. If one copy is edited and
    not the other, the rebuild would silently change convention, so this fails
    instead.
    """
    dv = ctx.cfg["dependent_variable"]
    variants = dict(dv["variants"])
    head = variants["headline"]
    for k in ("price_transactions", "multiplier_units", "price_statistic"):
        if head[k] != dv[k]:
            raise ValueError(
                f"config drift: dependent_variable.{k} = {dv[k]!r} but "
                f"variants.headline.{k} = {head[k]!r}"
            )
    for name in ctx.cfg["revision"]["parking"]["alternates"]:
        if name not in variants:
            raise KeyError(
                f"revision.parking.alternates names unknown variant {name!r}"
            )
    return variants


def plot_area(ctx: common.Context, f: dict[str, pd.DataFrame]) -> pd.Series:
    """Plot area per frozen-layer record, from `plot_area_source`.

    The geolayer's own `Terreno` is the configured source; the registry and
    ITBI alternatives are implemented so the choice can be re-run.
    """
    return plot_area_from(ctx, f, ctx.cfg["dependent_variable"]["plot_area_source"])


def plot_area_from(
    ctx: common.Context, f: dict[str, pd.DataFrame], src: str
) -> pd.Series:
    """Plot area per frozen-layer record from one named source."""
    codes = ctx.cfg["io"]["missing_codes"]
    geo = f["geolayer"]
    if src == "geolayer":
        return pd.to_numeric(geo["Terreno"], errors="coerce")
    if src == "registry":
        per = (
            prep.to_num(f["registry"]["Plot Area"], codes)
            .groupby(f["registry"].addr_key)
            .median()
        )
        return geo.addr_key.map(per)
    if src == "itbi":
        it = f["itbi"]
        area = prep.to_num(it["Plot Area"], codes)
        per = area[area > 0].groupby(it.addr_key).median()
        return geo.addr_key.map(per)
    raise ValueError(f"unknown plot_area_source: {src}")


def plot_area_agreement(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """How far the geolayer's plot area agrees with the registry's and ITBI's.

    Per frozen-layer record: the registry value is the median `Plot Area` of
    the address's registry rows, the ITBI value the median of its non-zero
    `Plot Area`s. Agreement is |geolayer - other| / other within
    `profiling.plot_area.conflict_rel_tol`, on the records where the other
    source has a value.
    """
    tol = float(ctx.cfg["profiling"]["plot_area"]["conflict_rel_tol"])
    geo = plot_area_from(ctx, f, "geolayer")
    rows = []
    for src in ("registry", "itbi"):
        other = plot_area_from(ctx, f, src)
        ok = other.notna() & (other != 0)
        rel = (geo[ok] - other[ok]).abs() / other[ok]
        n_agree = int((rel <= tol).sum())
        rows.append(
            {
                "comparison": f"geolayer vs {src}",
                "records": len(geo),
                "records_comparable": int(ok.sum()),
                "rel_tol": tol,
                "n_agree": n_agree,
                "n_conflict": int(ok.sum()) - n_agree,
                "share_agree": round(n_agree / int(ok.sum()), 6)
                if ok.sum()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the rebuild
# ---------------------------------------------------------------------------


def address_aggregates(
    ctx: common.Context, f: dict[str, pd.DataFrame], spec: dict[str, str]
) -> dict[str, pd.Series]:
    """Per-address price statistic, transaction count and unit multiplier."""
    dv = ctx.cfg["dependent_variable"]
    itbi = prep.dedupe(f["itbi"], dv["itbi_duplicates"], prep.ITBI_ROW_KEY)
    reg = prep.dedupe(f["registry"], dv["registry_duplicates"], prep.REG_ROW_KEY)

    pm = prep.price_mask(itbi, spec["price_transactions"], dv["blank_purpose"])
    um = prep.unit_mask(reg, spec["multiplier_units"], dv["parent_rows_in_multiplier"])
    sold, units = itbi[pm], reg[um]
    stat = spec["price_statistic"]
    price = sold.groupby("addr_key")["price"].agg(stat)
    n_tx = sold.groupby("addr_key").size()

    # Only-parking addresses: the non-parking subset is empty but the frozen
    # column still carries a price, so the original fell back to the full
    # transaction set. Never triggers for a variant that already uses all sales.
    n_rows = units.groupby("addr_key").size()
    n_parent_sel = units.groupby("addr_key")["parent"].sum()
    fallback = pd.Series(False, index=price.index, dtype=bool)
    unit_fallback = pd.Series(False, index=n_rows.index, dtype=bool)
    mode = dv["only_parking_fallback"]
    if mode == "use_all_transactions":
        every = itbi.groupby("addr_key")["price"].agg(stat)
        missing = every.index.difference(price.index)
        if len(missing):
            price = pd.concat([price, every.loc[missing]])
            n_tx = pd.concat([n_tx, itbi.groupby("addr_key").size().loc[missing]])
            fallback = pd.concat([fallback, pd.Series(True, index=missing, dtype=bool)])
        # symmetric on the other factor: a plot whose only registered unit is a
        # garage still has one unit, so an empty subset falls back to all rows.
        every_u = reg.groupby("addr_key").size()
        missing_u = every_u.index.difference(n_rows.index)
        if len(missing_u):
            n_rows = pd.concat([n_rows, every_u.loc[missing_u]])
            n_parent_sel = pd.concat(
                [n_parent_sel, reg.groupby("addr_key")["parent"].sum().loc[missing_u]]
            )
            unit_fallback = pd.concat(
                [unit_fallback, pd.Series(True, index=missing_u, dtype=bool)]
            )
    elif mode != "leave_undefined":
        raise ValueError(f"unknown only_parking_fallback: {mode}")

    # An address whose selected registry rows are all parent rows and that has
    # no numbered unit at all is one plot, not n_parents.
    n_units = prep.parent_only_multiplier(
        n_rows, n_parent_sel, dv["parent_only_address_counts_as"]
    )
    parent_only = (n_parent_sel.reindex(n_rows.index).fillna(0) == n_rows) & (
        n_rows > 1
    )

    return {
        "price": price,
        "n_tx": n_tx,
        "n_units": n_units,
        "n_units_rows": n_rows,
        "n_units_parent_selected": n_parent_sel,
        "parent_only_address": parent_only,
        "price_fallback": fallback,
        "unit_fallback": unit_fallback,
    }


def build_variant(
    ctx: common.Context,
    f: dict[str, pd.DataFrame],
    name: str,
    spec: dict[str, str],
    flags: pd.DataFrame,
    terreno: pd.Series,
) -> pd.DataFrame:
    """One row per frozen-layer record, with the rebuilt Preco/Puni attached."""
    geo = f["geolayer"]
    agg = address_aggregates(ctx, f, spec)

    out = geo[GEO_CARRY].copy()
    out = out.rename(columns={"Preco": "Preco_frozen", "Puni": "Puni_frozen"})
    for c in ("Preco_frozen", "Puni_frozen", *MEASURES):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Terreno"] = terreno.to_numpy()
    out["variant"] = name
    out["price_statistic"] = spec["price_statistic"]
    out["price_transactions"] = spec["price_transactions"]
    out["multiplier_units"] = spec["multiplier_units"]

    out["price_stat"] = out.addr_key.map(agg["price"])
    out["n_tx"] = out.addr_key.map(agg["n_tx"])
    out["n_units"] = out.addr_key.map(agg["n_units"])
    out["n_units_rows"] = out.addr_key.map(agg["n_units_rows"])
    out["n_units_parent_selected"] = out.addr_key.map(agg["n_units_parent_selected"])
    out["parent_only_address"] = out.addr_key.map(agg["parent_only_address"]).fillna(
        False
    )
    out["price_fallback"] = out.addr_key.map(agg["price_fallback"]).fillna(False)
    out["unit_fallback"] = out.addr_key.map(agg["unit_fallback"]).fillna(False)
    out["Preco"] = out.price_stat * out.n_units
    out["Puni"] = out.Preco / out.Terreno.replace(0, np.nan)

    # the paper's four scenarios, as they land in the data (report section 2)
    n_u, n_t = out.n_units, out.n_tx
    out["scenario"] = np.select(
        [
            n_t.isna() | (n_t == 0) | n_u.isna() | (n_u == 0),
            (n_u == 1),
            (n_u > 1) & (n_t == 1),
            (n_u > 1) & (n_t > 1),
        ],
        [
            "S4_no_sale_or_no_units",
            "S1_single_unit",
            "S2_multi_one_sale",
            "S3_multi_several_sales",
        ],
        default="unclassified",
    )
    return out.join(flags, on="addr_key")


def address_flags(ctx: common.Context, f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Variant-independent per-address descriptors used for the residual audit."""
    dv = ctx.cfg["dependent_variable"]
    itbi = prep.dedupe(f["itbi"], dv["itbi_duplicates"], prep.ITBI_ROW_KEY)
    reg = prep.dedupe(f["registry"], dv["registry_duplicates"], prep.REG_ROW_KEY)
    g = itbi.groupby("addr_key")
    r = reg.groupby("addr_key")
    flags = pd.DataFrame(
        {
            "n_tx_all": g.size(),
            "n_tx_parking": g["parking"].sum(),
            "n_tx_blank_purpose": g.apply(
                lambda d: int((d.Purpose == "").sum()), include_groups=False
            ),
            "price_min": g["price"].min(),
            "price_max": g["price"].max(),
        }
    )
    flags["n_units_all"] = r.size()
    flags["n_units_parking"] = r["parking"].sum()
    flags["n_units_parent"] = r["parent"].sum()
    flags = flags.fillna({"n_units_all": 0, "n_units_parking": 0, "n_units_parent": 0})
    flags["only_parking_address"] = flags.n_tx_parking == flags.n_tx_all
    flags["any_parking_tx"] = flags.n_tx_parking > 0
    flags["any_blank_purpose_tx"] = flags.n_tx_blank_purpose > 0
    flags["in_registry"] = flags.n_units_all > 0
    return flags


# ---------------------------------------------------------------------------
# validation against the frozen column
# ---------------------------------------------------------------------------


def validate(ctx: common.Context, df: pd.DataFrame) -> pd.DataFrame:
    """Classify every record: exact / within-tol / rounding / residual."""
    v = ctx.cfg["dependent_variable"]["validation"]
    est, frozen = df.Preco, df.Preco_frozen
    rel = (est - frozen).abs() / frozen.replace(0, np.nan)

    df = df.copy()
    df["rel_diff"] = rel
    df["exact"] = np.isclose(est, frozen, rtol=v["exact_rtol"], equal_nan=False)
    df["within_tol"] = rel <= v["within_rel_tol"]
    df["rounds_to_frozen"] = (est - frozen).abs() <= v["rounding_tol_abs"]
    df["undefined"] = est.isna()

    # If the frozen value is an exact integer multiple of our price statistic,
    # the disagreement is entirely in the unit multiplier, not in the price
    # average.
    implied = frozen / df.price_stat.replace(0, np.nan)
    df["implied_units"] = implied
    df["implied_units_int"] = implied.round()
    df["implied_units_is_integer"] = (implied - implied.round()).abs() <= v[
        "implied_units_tol"
    ]
    df["delta_units"] = df.implied_units_int - df.n_units

    df["status"] = np.select(
        [df.undefined, df.exact, df.within_tol],
        ["undefined", "exact", "within_tol"],
        default="residual",
    )
    return df


def subset_probe(
    ctx: common.Context, df: pd.DataFrame, f: dict[str, pd.DataFrame]
) -> dict[str, int]:
    """Is the frozen value our formula over a strict subset of the sales?

    For every residual address with at most
    `validation.residual_subset_probe_max_tx` transactions, try every subset:
    if `mean(subset) x our unit count` lands on the frozen value, the two
    constructions disagree about which transactions belong to the address, not
    about the arithmetic. Brute force, hence the cap.
    """
    import itertools

    v = ctx.cfg["dependent_variable"]["validation"]
    dv = ctx.cfg["dependent_variable"]
    cap = int(v["residual_subset_probe_max_tx"])
    itbi = prep.dedupe(f["itbi"], dv["itbi_duplicates"], prep.ITBI_ROW_KEY)
    sold = itbi[prep.price_mask(itbi, dv["price_transactions"], dv["blank_purpose"])]
    prices = sold.groupby("addr_key")["price"].apply(list)

    res = df[df.status == "residual"]
    n_probed = n_hit = 0
    for row in res.itertuples():
        pl = prices.get(row.addr_key, [])
        if not pl or len(pl) > cap or not np.isfinite(row.n_units):
            continue
        n_probed += 1
        target = row.Preco_frozen
        for r in range(1, len(pl)):  # strict subsets only
            if any(
                abs(float(np.mean(c)) * row.n_units - target) <= v["rounding_tol_abs"]
                for c in itertools.combinations(pl, r)
            ):
                n_hit += 1
                break
    ctx.count(
        "residual: reproduced by a strict subset of the sales",
        n_hit,
        f"probed {n_probed} of {len(res)} (<= {cap} transactions)",
    )
    return {"probed": n_probed, "hit": n_hit, "residual": len(res), "cap": cap}


def alternative_conventions(
    ctx: common.Context, f: dict[str, pd.DataFrame], df: pd.DataFrame
) -> pd.DataFrame:
    """Score rules that could explain the residual, on the full layer and on it.

    Stage 0b's grid asked which convention fits best overall; this asks the
    narrower question that matters now -- does any of them fit the addresses the
    headline misses? Two rules beyond the grid are tested because they are the
    obvious way a manual workflow would go wrong: averaging only the sales whose
    `Purpose` matches the record's own, and rounding the mean before multiplying.
    """
    dv = ctx.cfg["dependent_variable"]
    v = dv["validation"]
    itbi = prep.dedupe(f["itbi"], dv["itbi_duplicates"], prep.ITBI_ROW_KEY)
    reg = prep.dedupe(f["registry"], dv["registry_duplicates"], prep.REG_ROW_KEY)
    geo = f["geolayer"]
    frozen = df.Preco_frozen
    resid = (df.status == "residual").to_numpy()

    units = {
        "non_parking": reg[~reg.parking].groupby("addr_key").size(),
        "all": reg.groupby("addr_key").size(),
    }
    rows = []

    def score(label: str, est: pd.Series) -> None:
        ok = np.isclose(est.to_numpy(), frozen.to_numpy(), rtol=v["exact_rtol"])
        rows.append(
            {
                "convention": label,
                "exact on all records": pct(float(ok.mean())),
                "residual records rescued": int((ok & resid).sum()),
            }
        )

    for pname, pmask in (
        ("all", pd.Series(True, index=itbi.index)),
        ("non-parking", ~itbi.parking),
    ):
        for stat in ("mean", "median"):
            pr = itbi[pmask].groupby("addr_key")["price"].agg(stat)
            for uname, u in units.items():
                score(
                    f"{stat} of {pname} prices x {uname} units",
                    geo.addr_key.map(pr) * geo.addr_key.map(u),
                )
    # sales whose Purpose matches the record's own Finalidade / its category
    for label, ikey, gkey in (
        (
            "mean of same-`Purpose` sales x non_parking units",
            "purpose_norm",
            "purpose_norm",
        ),
        ("mean of same-category sales x non_parking units", "category", "category"),
    ):
        ik = itbi.addr_key + "||" + itbi[ikey].astype(str)
        gk = geo.addr_key + "||" + geo[gkey].astype(str)
        pr = itbi[~itbi.parking].groupby(ik[~itbi.parking])["price"].mean()
        score(label, gk.map(pr) * geo.addr_key.map(units["non_parking"]))
    pr = itbi[~itbi.parking].groupby("addr_key")["price"].mean()
    score(
        "round(mean of non-parking prices) x non_parking units",
        geo.addr_key.map(pr).round() * geo.addr_key.map(units["non_parking"]),
    )
    return pd.DataFrame(rows).sort_values(
        "residual records rescued", ascending=False, ignore_index=True
    )


def rule_refinement(
    ctx: common.Context, f: dict[str, pd.DataFrame], df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the parent-only refinement and the two diagnostic patterns.

    All four candidates are scored on the WHOLE layer against the frozen
    `Preco`, and against the pre-refinement rule (`n_parents`) so that
    "rescues" and "newly breaks" are countable rather than asserted. Only the
    parent-only refinement is a candidate for adoption: `parent_only_address_
    counts_as` in config decides which of the two it is. The other two are
    reported as diagnostics and not adopted.

      (i)  numbered non-parking units minus those whose `Purpose` category
           differs from the categories of the address's own sold units.
      (ii) parking transactions re-admitted to the price mean at addresses with
           at most `refinement_diagnostics.parking_in_price_max_sales` sales,
           under both readings of "sales" (non-parking, and all).
    """
    dv = ctx.cfg["dependent_variable"]
    v = dv["validation"]
    rcfg = dv["refinement_diagnostics"]
    cap = int(rcfg["parking_in_price_max_sales"])
    itbi = prep.dedupe(f["itbi"], dv["itbi_duplicates"], prep.ITBI_ROW_KEY)
    reg = prep.dedupe(f["registry"], dv["registry_duplicates"], prep.REG_ROW_KEY)
    frozen = df.Preco_frozen
    sold = itbi[prep.price_mask(itbi, dv["price_transactions"], dv["blank_purpose"])]

    def pct2(x: float) -> str:
        """Two decimals: the refinement moves the rate by ~0.04 pp, which the
        report's usual one-decimal `pct()` would round away entirely."""
        return f"{100 * x:.2f}%"

    def classify(est: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        exact = np.isclose(est, frozen, rtol=v["exact_rtol"], equal_nan=False)
        rel = (est - frozen).abs() / frozen.replace(0, np.nan)
        return exact, (rel <= v["within_rel_tol"]).fillna(False).to_numpy()

    # the two readings of the multiplier that the refinement chooses between
    u_n_parents = df.n_units_rows
    u_one = prep.parent_only_multiplier(
        df.n_units_rows, df.n_units_parent_selected, "one"
    )
    _, base_ok = classify(df.price_stat * u_n_parents)

    rows: list[dict[str, object]] = []

    def score(label: str, est: pd.Series, adopted: str, note: str) -> np.ndarray:
        exact, ok = classify(est)
        rows.append(
            {
                "rule": label,
                "exact": pct2(float(exact.mean())),
                f"within {v['within_rel_tol']:.0%}": pct2(float(ok.mean())),
                "residual records": int((~ok).sum()),
                "rescues (vs n_parents)": int((ok & ~base_ok).sum()),
                "newly breaks": int((~ok & base_ok).sum()),
                "adopted": adopted,
                "note": note,
            }
        )
        return ok

    score(
        "baseline — parent-only address counts as `n_parents`",
        df.price_stat * u_n_parents,
        "yes" if dv["parent_only_address_counts_as"] == "n_parents" else "no",
        "one unit per address-level registry row",
    )
    ok_one = score(
        "refinement — parent-only address counts as `one`",
        df.price_stat * u_one,
        "yes" if dv["parent_only_address_counts_as"] == "one" else "no",
        "an address with only parent rows and no numbered unit is one plot",
    )

    # (i) numbered non-parking units whose category matches the sold units'
    sold_cats = sold.groupby("addr_key")["category"].apply(lambda s: set(s.dropna()))
    numbered = reg[prep.unit_mask(reg, dv["multiplier_units"], "exclude")]
    match = [
        c in sold_cats.get(a, set())
        for a, c in zip(numbered.addr_key, numbered.category)
    ]
    u_i = numbered.assign(match=match).groupby("addr_key")["match"].sum()
    u_i = df.addr_key.map(u_i)
    score(
        "(i) numbered units whose category matches the sold units'",
        df.price_stat * u_i,
        "no — diagnostic only",
        f"{int((u_i.fillna(0) != u_one.fillna(0)).sum())} records get a different "
        "multiplier; addresses with no matching numbered unit lose their value",
    )
    score(
        "(i′) same, falling back to the refined count when it would be 0",
        df.price_stat * u_i.where(u_i.fillna(0) > 0, u_one),
        "no — diagnostic only",
        "the survivable form of (i)",
    )

    # (ii) parking re-admitted to the price mean at low-sale addresses
    every = itbi.groupby("addr_key")["price"].agg(dv["price_statistic"])
    p_all = df.addr_key.map(every)
    for basis, counts in (
        ("non-parking sales", df.addr_key.map(sold.groupby("addr_key").size())),
        ("all sales", df.addr_key.map(itbi.groupby("addr_key").size())),
    ):
        p_alt = df.price_stat.where(counts.fillna(0) > cap, p_all)
        score(
            f"(ii) parking in the price mean when {basis} ≤ {cap}",
            p_alt * u_one,
            "no — diagnostic only",
            f"{int((~np.isclose(p_alt.fillna(-1), df.price_stat.fillna(-1))).sum())} "
            "records get a different price average",
        )

    tbl = pd.DataFrame(rows)

    # the records the refinement moves, one row each
    moved = (u_one != u_n_parents).fillna(False)
    detail = pd.DataFrame(
        {
            "addr_key": df.addr_key[moved],
            "Endereco": df.Endereco[moved],
            "Preco_frozen": frozen[moved],
            "Preco_n_parents": (df.price_stat * u_n_parents)[moved],
            "Preco_one": (df.price_stat * u_one)[moved],
            "price_stat": df.price_stat[moved],
            "n_tx": df.n_tx[moved],
            "n_parent_rows": df.n_units_rows[moved],
            "was_within_tol": base_ok[moved.to_numpy()],
            "is_within_tol": ok_one[moved.to_numpy()],
        }
    ).sort_values("addr_key", ignore_index=True)

    ctx.count(
        "refinement: parent-only addresses whose multiplier collapses to 1",
        int(moved.sum()),
        f"rescued {int(tbl.loc[1, 'rescues (vs n_parents)'])}, "
        f"newly broken {int(tbl.loc[1, 'newly breaks'])}",
    )
    return tbl, detail


def residual_characterisation(
    ctx: common.Context, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rate of each descriptor among reproduced vs residual records."""
    res = df.status == "residual"
    ok = df.status.isin(["exact", "within_tol"])
    attrs = {
        "address absent from the registry": ~df.in_registry.fillna(False),
        "no usable transaction (undefined)": df.undefined,
        "multi-unit plot (units > 1)": df.n_units > 1,
        "many units (> 10)": df.n_units > 10,
        "several sales at the address (tx > 1)": df.n_tx > 1,
        "at least one parking transaction": df.any_parking_tx.fillna(False),
        "only-parking address": df.only_parking_address.fillna(False),
        "blank-`Purpose` transaction present": df.any_blank_purpose_tx.fillna(False),
        "duplicated address record": df.dup_addr_record,
        "price statistic is fractional": (df.price_stat % 1 != 0),
        "frozen / our mean is an integer": df.implied_units_is_integer.fillna(False),
        "our multiplier too LOW (delta > 0)": df.delta_units > 0,
        "our multiplier too HIGH (delta < 0)": df.delta_units < 0,
    }
    rows = []
    for label, mask in attrs.items():
        mask = mask.fillna(False)
        rows.append(
            {
                "attribute": label,
                "n in residual": int((mask & res).sum()),
                "% of residual": pct(float((mask & res).sum() / max(res.sum(), 1))),
                "% of reproduced": pct(float((mask & ok).sum() / max(ok.sum(), 1))),
                "lift": round(
                    float((mask & res).sum() / max(res.sum(), 1))
                    / max(float((mask & ok).sum() / max(ok.sum(), 1)), 1e-9),
                    2,
                ),
            }
        )
    tbl = pd.DataFrame(rows)

    detail_cols = [
        "addr_key",
        "Endereco",
        "Finalidade",
        "Preco_frozen",
        "Preco",
        "rel_diff",
        "price_stat",
        "n_tx",
        "n_units",
        "implied_units",
        "delta_units",
        "implied_units_is_integer",
        "n_units_all",
        "n_units_parking",
        "n_units_parent",
        "n_tx_all",
        "n_tx_parking",
        "n_tx_blank_purpose",
        "scenario",
        "status",
    ]
    detail = df.loc[df.status.isin(["residual", "undefined"]), detail_cols]
    return tbl, detail.sort_values("rel_diff", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# funnel
# ---------------------------------------------------------------------------


def funnel(
    ctx: common.Context, df: pd.DataFrame, tag: str, preco: str, puni: str
) -> tuple[pd.DataFrame, list[int]]:
    """Run the exclusion funnel of the submitted analysis on one version of
    the variable.

    Uses `common.apply_exclusions` (the Python twin of the R helper the gate
    uses), so the order -- positivity before the IQR fence -- cannot drift.
    """
    # built explicitly: `df` carries both the frozen and the rebuilt column, and
    # renaming one onto the other's name would leave duplicate labels.
    frame = pd.DataFrame(
        {
            "addr_key": df.addr_key,
            "Preco": df[preco],
            "Puni": df[puni],
            "Terreno": df.Terreno,
            **{m: df[m] for m in MEASURES},
        }
    )
    ctx.log.info("funnel [%s]", tag)
    mark = len(ctx.counts)
    kept = common.apply_exclusions(ctx, frame, "disaggregated")
    # apply_exclusions logs "<level>: loaded / after positivity / after IQR";
    # take the counts straight off the log rather than recomputing them.
    steps = [int(c["n"]) for c in ctx.counts[mark:]]
    ctx.count(f"funnel[{tag}]: final N", len(kept))
    return kept, steps


def rebuilt_steps_index(ctx: common.Context, head: pd.DataFrame) -> pd.Index:
    """Records of the rebuilt variable that pass the positivity filter."""
    cols = ctx.cfg["exclusions"]["positivity"]["disaggregated"]
    frame = head[["Preco", "Puni", "Terreno", *MEASURES]]
    keep = (frame[cols].notna() & (frame[cols] > 0)).all(axis=1)
    return head.index[keep.to_numpy()]


# ---------------------------------------------------------------------------


def main() -> int:
    ctx = common.init(1, "dependent_variable")
    drift = common.verify_manifest(ctx)
    dv = ctx.cfg["dependent_variable"]
    vcfg = dv["validation"]
    tgt = ctx.cfg["exclusions"]["targets"]
    iqr_mult = ctx.cfg["exclusions"]["iqr_outliers"]["multiplier"]
    pos_cols = ctx.cfg["exclusions"]["positivity"]["disaggregated"]

    f = prep.load_frames(ctx)
    mapping, mstats = prep.purpose_mapping(ctx, f)
    parking_pattern = prep.annotate_parking(ctx, f)

    # The purpose mapping is a tracked artefact; this consistency check confirms
    # that the rules in config still reproduce it.
    stored_path = common.REPO / dv["purpose_mapping"]
    mapping_matches = "not found"
    if stored_path.exists():
        stored = pd.read_csv(stored_path, keep_default_na=False)
        cols = ["file", "purpose_value", "count", "category"]
        mapping_matches = str(
            stored[cols]
            .astype(str)
            .reset_index(drop=True)
            .equals(mapping[cols].astype(str).reset_index(drop=True))
        )
    ctx.log.info("purpose mapping reproduces the stored CSV: %s", mapping_matches)

    geo, itbi = f["geolayer"], f["itbi"]
    flags = address_flags(ctx, f)
    terreno = plot_area(ctx, f)
    pa_agree = plot_area_agreement(ctx, f)
    pa_agree.to_csv(ctx.table("stage1_plot_area_agreement.csv"), index=False)
    for r in pa_agree.itertuples():
        ctx.log.info(
            "plot area, %s: %d of %d comparable records agree within %.0f%% (%.2f%%)",
            r.comparison,
            r.n_agree,
            r.records_comparable,
            100 * r.rel_tol,
            100 * r.share_agree,
        )
    dup_addr = geo.addr_key.duplicated(keep=False).to_numpy()

    variants = variant_specs(ctx)
    built: dict[str, pd.DataFrame] = {}
    for name, spec in variants.items():
        d = build_variant(ctx, f, name, spec, flags, terreno)
        d["dup_addr_record"] = dup_addr
        built[name] = d
        ctx.count(
            f"variant[{name}]: records with a rebuilt Preco",
            int(d.Preco.notna().sum()),
            f"price={spec['price_transactions']}, units={spec['multiplier_units']}",
        )

    # --- validation ---------------------------------------------------------
    head = validate(ctx, built["headline"])
    built["headline"] = head
    n = len(head)
    v_exact, v_within = float(head.exact.mean()), float(head.within_tol.mean())
    ctx.count("headline: exact matches vs frozen Preco", int(head.exact.sum()))
    ctx.count("headline: within-tolerance matches", int(head.within_tol.sum()))

    # same scoring on the distinct-address subset and on the analysis sample
    uniq = head.drop_duplicates("addr_key")
    val_rows = [
        {
            "sample": "all frozen-layer records",
            "n": n,
            "exact": pct(v_exact),
            f"within {vcfg['within_rel_tol']:.0%}": pct(v_within),
            "rounds to frozen": pct(float(head.rounds_to_frozen.mean())),
            "undefined": int(head.undefined.sum()),
        },
        {
            "sample": "distinct addresses",
            "n": len(uniq),
            "exact": pct(float(uniq.exact.mean())),
            f"within {vcfg['within_rel_tol']:.0%}": pct(float(uniq.within_tol.mean())),
            "rounds to frozen": pct(float(uniq.rounds_to_frozen.mean())),
            "undefined": int(uniq.undefined.sum()),
        },
    ]

    # --- funnels ------------------------------------------------------------
    frozen_kept, frozen_steps = funnel(
        ctx, head, "frozen", "Preco_frozen", "Puni_frozen"
    )
    rebuilt_kept, rebuilt_steps = funnel(ctx, head, "rebuilt", "Preco", "Puni")
    # the submitted funnel, as printed in the manuscript
    submitted = [
        tgt["disaggregated_records"],
        tgt["disaggregated_after_positivity"],
        tgt["disaggregated_n"],
    ]

    # the analysis sample is defined by records, not addresses: select by index
    pub_sample = head.loc[frozen_kept.index]
    val_rows.append(
        {
            "sample": f"analysis sample (frozen column, N = {len(pub_sample):,})",
            "n": len(pub_sample),
            "exact": pct(float(pub_sample.exact.mean())),
            f"within {vcfg['within_rel_tol']:.0%}": pct(
                float(pub_sample.within_tol.mean())
            ),
            "rounds to frozen": pct(float(pub_sample.rounds_to_frozen.mean())),
            "undefined": int(pub_sample.undefined.sum()),
        }
    )
    val_tbl = pd.DataFrame(val_rows)
    val_tbl.to_csv(ctx.table("stage1_validation_summary.csv"), index=False)

    funnel_tbl = pd.DataFrame(
        {
            "step": [
                "ITBI transaction rows",
                "... after de-duplication",
                "distinct ITBI addresses",
                "addresses reaching the frozen point layer",
                "records with a defined rebuilt Preco",
                f"after positivity ({'/'.join(pos_cols)} > 0)",
                f"after {iqr_mult}x IQR on the land value",
            ],
            "submitted": [
                tgt["itbi_rows"],
                "n/a",
                "n/a",
                submitted[0],
                submitted[0],
                submitted[1],
                submitted[2],
            ],
            "frozen column": [
                len(itbi),
                len(prep.dedupe(itbi, dv["itbi_duplicates"], prep.ITBI_ROW_KEY)),
                itbi.addr_key.nunique(),
                len(geo),
                int(head.Preco_frozen.notna().sum()),
                frozen_steps[1],
                frozen_steps[2],
            ],
            "rebuilt": [
                len(itbi),
                len(prep.dedupe(itbi, dv["itbi_duplicates"], prep.ITBI_ROW_KEY)),
                itbi.addr_key.nunique(),
                len(geo),
                int(head.Preco.notna().sum()),
                rebuilt_steps[1],
                rebuilt_steps[2],
            ],
        }
    )
    funnel_tbl["delta (rebuilt - submitted)"] = [
        (b - a) if isinstance(a, int) and isinstance(b, int) else "n/a"
        for a, b in zip(funnel_tbl["submitted"], funnel_tbl["rebuilt"], strict=True)
    ]
    funnel_tbl.to_csv(ctx.table("stage1_funnel.csv"), index=False)

    kept_idx = set(rebuilt_kept.index)
    frozen_idx = set(frozen_kept.index)
    overlap = len(kept_idx & frozen_idx)
    only_rebuilt = len(kept_idx - frozen_idx)
    only_frozen = len(frozen_idx - kept_idx)

    # --- residual characterisation -----------------------------------------
    res_tbl, res_detail = residual_characterisation(ctx, head)
    res_tbl.to_csv(ctx.table("stage1_residual_characterisation.csv"), index=False)
    res_detail.round(6).to_csv(ctx.table("stage1_residuals.csv"), index=False)
    ctx.count("headline: residual records (not within tolerance)", len(res_detail))

    residual = head[head.status == "residual"]
    reproduced = head[head.status.isin(["exact", "within_tol"])]
    int_share = float(residual.implied_units_is_integer.fillna(False).mean())
    over_share = float((residual.Preco > residual.Preco_frozen).mean())
    med_rel = float(residual.rel_diff.median())
    probe = subset_probe(ctx, head, f)
    alts = alternative_conventions(ctx, f, head)
    alts.to_csv(ctx.table("stage1_alternative_conventions.csv"), index=False)
    refine, refine_detail = rule_refinement(ctx, f, head)
    refine.to_csv(ctx.table("stage1_rule_refinement.csv"), index=False)
    refine_detail.round(6).to_csv(
        ctx.table("stage1_rule_refinement_records.csv"), index=False
    )
    delta_counts = (
        residual.loc[residual.implied_units_is_integer.fillna(False), "delta_units"]
        .value_counts()
        .head(8)
        .rename_axis("implied units - our units")
        .reset_index(name="records")
    )

    # --- scenarios ----------------------------------------------------------
    scen = (
        head.groupby("scenario")
        .agg(
            records=("scenario", "size"),
            exact=("exact", "sum"),
            within_tol=("within_tol", "sum"),
            median_units=("n_units", "median"),
            median_tx=("n_tx", "median"),
        )
        .reset_index()
    )
    scen["exact %"] = [pct(a / b) for a, b in zip(scen.exact, scen.records)]
    scen["in analysis sample"] = [
        int((pub_sample.scenario == s).sum()) for s in scen.scenario
    ]
    scen.to_csv(ctx.table("stage1_scenarios.csv"), index=False)
    s12 = scen[scen.scenario.isin(["S1_single_unit", "S2_multi_one_sale"])]
    rate_s12 = pct(float(s12.exact.sum() / s12.records.sum()))
    s3 = scen[scen.scenario == "S3_multi_several_sales"]
    rate_s3 = pct(float(s3.exact.sum() / s3.records.sum()))
    lift_tx = float(
        res_tbl.loc[
            res_tbl.attribute == "several sales at the address (tx > 1)", "lift"
        ].iloc[0]
    )

    # --- variant comparison --------------------------------------------------
    vrows = []
    for name, d in built.items():
        spec = variants[name]
        vs_head = np.isclose(
            d.Preco.fillna(-1), built["headline"].Preco.fillna(-1), rtol=1e-9
        )
        vrows.append(
            {
                "variant": name,
                "price transactions": spec["price_transactions"],
                "units in multiplier": spec["multiplier_units"],
                "statistic": spec["price_statistic"],
                "records with Preco": int(d.Preco.notna().sum()),
                "median Preco": float(d.Preco.median()),
                "median Puni": float(d.Puni.median()),
                "mean units": round(float(d.n_units.mean()), 2),
                "identical to headline": pct(float(vs_head.mean())),
                "exact vs frozen": pct(
                    float(
                        np.isclose(
                            d.Preco, d.Preco_frozen, rtol=vcfg["exact_rtol"]
                        ).mean()
                    )
                ),
            }
        )
    var_tbl = pd.DataFrame(vrows)
    var_tbl.to_csv(ctx.table("stage1_variant_comparison.csv"), index=False)

    # --- write the interim datasets -----------------------------------------
    out_cols = [
        "id",
        "addr_key",
        "Endereco",
        "Rua",
        "Numero",
        "Unidade",
        "Finalidade",
        "Type",
        "variant",
        "price_statistic",
        "price_transactions",
        "multiplier_units",
        "price_stat",
        "n_tx",
        "n_units",
        "n_tx_all",
        "n_tx_parking",
        "n_tx_blank_purpose",
        "n_units_all",
        "n_units_parking",
        "n_units_parent",
        "Preco",
        "Terreno",
        "Puni",
        "Preco_frozen",
        "Puni_frozen",
        "scenario",
        "only_parking_address",
        "any_parking_tx",
        "any_blank_purpose_tx",
        "price_fallback",
        "unit_fallback",
        "in_registry",
        "dup_addr_record",
        *MEASURES,
    ]
    written: dict[str, Path] = {}
    for name, d in built.items():
        cols = [c for c in out_cols if c in d.columns]
        # the headline is the rebuilt dependent variable stage 4 reads
        p = (
            common.REPO / dv["rebuilt_path"]
            if name == "headline"
            else ctx.interim(f"dependent_variable_{name}.csv")
        )
        d[cols].to_csv(p, index=False)
        written[name] = p
        ctx.log.info("wrote %s (%d rows)", p.relative_to(common.REPO), len(d))
    man = common.write_manifest(ctx, written, "stage1_manifest.json")
    digests = json.loads(man.read_text())

    # =======================================================================
    # report
    # =======================================================================
    tol_s = f"{vcfg['within_rel_tol']:.0%}"
    within_col = f"within {tol_s}"

    def pct_value(s: str) -> float:
        return float(str(s).rstrip("%")) / 100

    # the only-parking fallback: what the variable would look like without it
    fb = head.price_fallback.fillna(False).astype(bool)
    n_fb = int(fb.sum())
    n_ufb = int(head.unit_fallback.fillna(False).astype(bool).sum())
    fb_exact = int((fb & head.exact).sum())
    fb_after_pos = int(fb.loc[rebuilt_steps_index(ctx, head)].sum())
    exact_without_fb = float((head.exact & ~fb).mean())

    both_alt = "parking_in_both"
    alt_exact = var_tbl.set_index("variant")["exact vs frozen"]
    alt_ident = var_tbl.set_index("variant")["identical to headline"]

    ref_base, ref_new = refine.iloc[0], refine.iloc[1]
    adopted_one = dv["parent_only_address_counts_as"] == "one"
    diag = refine.iloc[2:]
    diag_gap = [
        pct_value(ref_new[within_col]) - pct_value(r[within_col])
        for _, r in diag.iterrows()
    ]
    alt_best_exact = max(pct_value(s) for s in alts["exact on all records"])
    lift_is_max = lift_tx == float(res_tbl.lift.max())
    gap_records = head.within_tol & ~head.exact
    rounding_share = (
        float((gap_records & head.rounds_to_frozen).sum() / gap_records.sum())
        if gap_records.sum()
        else float("nan")
    )
    n_resid_only_parking = int(residual.only_parking_address.fillna(False).sum())

    md: list[str] = []
    md += [
        "# Stage 1 — rebuilding the dependent variable",
        "",
        (
            f"Generated by `src/10_dependent_variable.py`. Seed "
            f"{ctx.cfg['repro']['seed']}. "
            + (
                "Inputs match `data/interim/input_manifest.json`."
                if not drift
                else f"**Input drift in: {', '.join(drift)}.**"
            )
        ),
        "",
        (
            "The dependent variable is rebuilt from the raw ITBI transactions and "
            "the unit registry under the conventions in `dependent_variable:`, "
            "then scored **record by record** against the frozen `Preco` column "
            "and put through the exclusion funnel of the submitted analysis. "
            f"`dependent_variable.source` is `{dv['source']}`: the validation gate "
            "reproduces the submitted numbers from the frozen column. The rebuilt "
            f"column is written to `{dv['rebuilt_path']}`, which the revision's "
            "analyses read."
        ),
        "",
        "---",
        "",
        "## 0. Summary",
        "",
        (
            f"1. **The rebuild reproduces the frozen `Preco` exactly on "
            f"{pct(v_exact)} of the {n:,} frozen-layer records and within {tol_s} on "
            f"{pct(v_within)}.**"
        ),
        (
            f"2. **The residual is concentrated in multi-sale plots.** Of the "
            f"{len(residual):,} records ({pct(len(residual) / n)}) outside the "
            f"{tol_s} band, {pct(float((residual.n_tx > 1).mean()))} have more than "
            f"one sale at the address, against "
            f"{pct(float((reproduced.n_tx > 1).mean()))} of the reproduced records. "
            f"Single-sale plots reproduce exactly at {rate_s12}; the best of "
            f"{len(alts)} alternative conventions rescues "
            f"{int(alts['residual records rescued'].max())} residual records "
            "(section 5)."
        ),
        (
            f"3. **The funnel:** {rebuilt_steps[0]:,} records → "
            f"{rebuilt_steps[1]:,} after positivity → {rebuilt_steps[2]:,} after the "
            f"IQR fence, against the submitted {submitted[1]:,} → {submitted[2]:,} "
            f"and the frozen column's {frozen_steps[1]:,} → {frozen_steps[2]:,}. "
            f"{overlap:,} records are in both final samples."
        ),
        (
            f"4. **The only-parking fallback.** {n_fb} records are at addresses "
            "where every ITBI transaction is parking, yet the frozen column carries "
            "a price for them, which can only come from the parking sales. "
            f"Without the fallback they would have no value ({fb_after_pos} of "
            "them pass the positivity filter with it), and exact reproduction "
            f"would be {pct(exact_without_fb)} instead of {pct(v_exact)} "
            f"(`only_parking_fallback: {dv['only_parking_fallback']}`). "
            f"The unit-count fallback applies to {n_ufb} records."
        ),
        (
            "5. **Neither parking alternate reproduces the frozen column as well "
            f"as the headline.** `{both_alt}` (parking in the price average and "
            f"in the multiplier) reproduces it exactly on {alt_exact[both_alt]} of "
            "records; see section 3."
        ),
        (
            "6. **The parent-only refinement of the multiplier.** An address whose "
            'registry rows are all address-level ("parent") rows with no numbered '
            f"unit is counted as {'one plot' if adopted_one else 'one unit per parent row'} "
            f"(`parent_only_address_counts_as: {dv['parent_only_address_counts_as']}`). "
            f"Counting it as one plot moves {len(refine_detail)} records, rescues "
            f"{int(ref_new['rescues (vs n_parents)'])} and newly breaks "
            f"{int(ref_new['newly breaks'])}: exact {ref_base['exact']} → "
            f"{ref_new['exact']}, {within_col} {ref_base[within_col]} → "
            f"{ref_new[within_col]} (section 5.1)."
        ),
        "",
        "---",
        "",
        "## 1. Conventions in force",
        "",
        *md_table(
            pd.DataFrame(
                [
                    {"key": f"dependent_variable.{k}", "value": str(dv[k])}
                    for k in (
                        "itbi_duplicates",
                        "registry_duplicates",
                        "blank_purpose",
                        "parent_rows_in_multiplier",
                        "parent_only_address_counts_as",
                        "only_parking_fallback",
                        "price_statistic",
                        "price_transactions",
                        "multiplier_units",
                        "plot_area_source",
                    )
                ]
            )
        ),
        "",
        (
            f"Parking is `{ctx.cfg['profiling']['purpose']['active_parking_definition']}`"
            f" = `{parking_pattern}`. The purpose→category mapping in "
            f"`{dv['purpose_mapping']}` reproduces from the config rules: "
            f"**{mapping_matches}** ({mstats['low_confidence']} rows come from a "
            "low-confidence rule)."
        ),
        "",
        "---",
        "",
        "## 2. The construction, as one formula",
        "",
        (
            "Everything the manuscript's four-scenario description (§4.3) covers is "
            "this one expression, evaluated address by address:"
        ),
        "",
        "```",
        f"                 {dv['price_statistic']}( P(a) ) x U(a)",
        "    Puni(a)  =  ---------------------",
        "                        A(a)",
        "```",
        "",
        "with five named conventions, all of them recovered from the frozen column:",
        "",
        (
            "1. **`P(a)`, the prices.** The `Price` field of every ITBI transaction "
            "recorded at address `a`, after byte-identical duplicate rows are "
            "dropped, **excluding** transactions whose purpose is a parking space or "
            "garage. Transactions with a **blank** purpose are kept and treated as "
            "non-parking. *Exception (the only-parking fallback):* where every "
            "transaction at the address is parking, the parking prices are used "
            "rather than dropping the address."
        ),
        (
            "2. **`U(a)`, the multiplier.** The number of rows for `a` in the "
            "municipal unit registry, after byte-identical duplicate rows are "
            "dropped, **excluding** parking and garage units. Address-level rows "
            'without a unit number ("parent") **are** counted as units. '
            "*Exception (the parent-only rule):* an address whose selected "
            "registry rows are **all** parent rows, with no numbered unit, is one "
            "plot — `U = 1`, not one unit per parent row. *Exception (the "
            "fallback again):* where the non-parking selection is empty, all "
            "registry rows are used."
        ),
        (
            "3. **`A(a)`, the plot area.** "
            f"`plot_area_source: {dv['plot_area_source']}` — `Terreno` on the "
            "georeferenced address layer."
        ),
        (
            f"4. **The statistic is the {dv['price_statistic']}**, taken over "
            "transactions — not over units, and not weighted by floor area."
        ),
        (
            "5. **An address with no usable transaction has no value** (`P(a)` "
            "empty) and leaves the sample at the positivity filter; the exclusions "
            f"then run positivity before the {iqr_mult}×IQR fence, with the fence "
            "computed on the surviving subsample."
        ),
        "",
        "The same thing in code order:",
        "",
        "```",
        "# --- inputs ------------------------------------------------------------",
        "ITBI      : one row per transaction  (Price, Purpose, Street, Number, ...)",
        "REGISTRY  : one row per registered unit (Purpose, Street, Number, Unity...)",
        "GEO       : frozen point layer, one record per georeferenced address,",
        "            carrying Terreno and the five frozen centralities",
        "",
        "# --- normalisation (identical in all three files) -----------------------",
        "addr_key(row) = upper(fold_accents(Street)) + '|' + trim(Number)",
        "                with '?' matched as a single-character wildcard",
        "",
        "# --- row selection ------------------------------------------------------",
        f"ITBI  <- drop byte-identical duplicate rows          # {dv['itbi_duplicates']}",
        f"REG   <- drop byte-identical duplicate rows          # {dv['registry_duplicates']}",
        "SOLD  <- ITBI where NOT parking                       # blank Purpose kept",
        "UNITS <- REG  where NOT parking                       # parent rows kept",
        "# if either subset is empty at an address, fall back to the full set",
        f"# there (only_parking_fallback: {dv['only_parking_fallback']})",
        "",
        "# --- per address --------------------------------------------------------",
        "for a in addresses(GEO):",
        f"    p[a] = {dv['price_statistic']}( SOLD.Price  where addr_key == a )",
        "    u[a] = count( UNITS      where addr_key == a )     # the multiplier",
        "    if every UNITS row at a is a parent row and u[a] > 1:",
        f"        u[a] = 1          # parent_only_address_counts_as: {dv['parent_only_address_counts_as']}",
        "    Preco[a] = p[a] * u[a]",
        "    Terreno[a] = GEO.Terreno[a]                        # plot_area_source",
        "    Puni[a]  = Preco[a] / Terreno[a]                   # the dependent variable",
        "",
        "# --- exclusions, in this order -----------------------------------------",
        f"keep rows with {', '.join(pos_cols)} all > 0",
        f"then drop {iqr_mult} x IQR outliers on Puni, computed on that subsample",
        "```",
        "",
        "### 2.1 The manuscript's four scenarios, mapped onto that expression",
        "",
        (
            "§4.3 branches on how many units a plot has and how many of them sold. "
            "Under this convention the branches are not separate code paths — they "
            "are cases of the same product `mean(sold prices) × units(plot)`:"
        ),
        "",
        "| scenario | what §4.3 says | this rebuild |",
        "|---|---|---|",
        (
            "| 1 · one unit, sold | unit value / plot area | `u = 1`, so "
            "`Preco = mean(price) × 1` = the sale price |"
        ),
        (
            "| 2 · many units, one sale | replicate the sold value across all units, "
            "sum | `n_tx = 1`, so `mean(price) = that sale`, times `u` |"
        ),
        (
            "| 3 · many units, several sales | mean of the sold units assigned to all, "
            "sum | exactly `mean(price) × u` |"
        ),
        (
            "| 4 · no sales | excluded | `n_tx = 0` ⇒ `Preco` undefined ⇒ the record "
            "drops at the positivity filter |"
        ),
        "",
        *md_table(
            scen[
                [
                    "scenario",
                    "records",
                    "exact %",
                    "median_units",
                    "median_tx",
                    "in analysis sample",
                ]
            ]
        ),
        "",
        "---",
        "",
        "## 3. Variants: the headline and the two parking alternates",
        "",
        (
            "`revision.parking.alternates` names two comparators. The headline is "
            "the parking-excluded specification, which is also the one that "
            "reproduces the frozen column; the alternates put parking back in."
        ),
        "",
        "| variant | price average | unit multiplier |",
        "|---|---|---|",
        "| `headline` | non-parking transactions | non-parking units |",
        (
            "| `parking_in_multiplier_only` | non-parking transactions | **all** "
            "units (parking spaces count as units but never contribute a price) |"
        ),
        f"| `{both_alt}` | **all** transactions | **all** units |",
        "",
        *md_table(var_tbl),
        "",
        (
            f"`parking_in_multiplier_only` differs from the headline on "
            f"{100 - 100 * pct_value(alt_ident['parking_in_multiplier_only']):.1f}% "
            f"of records, `{both_alt}` on "
            f"{100 - 100 * pct_value(alt_ident[both_alt]):.1f}%. Both are written "
            "to `data/interim/` so stage 4 estimates them without rebuilding "
            "anything."
        ),
        "",
        "---",
        "",
        "## 4. Validation against the frozen `Preco`, record by record",
        "",
        *md_table(val_tbl),
        "",
        (
            f"The {n_fb} only-parking records (`price_fallback`) are included "
            f"above; {fb_exact} of them reproduce the frozen value exactly under the "
            "fallback, and all of them acquire a value."
        ),
        "",
        (
            f"**{pct(float(head.rounds_to_frozen.mean()))} of records land within "
            f"{vcfg['rounding_tol_abs']} BRL of the frozen value** (the frozen "
            "column is stored rounded). Of the records within "
            f"{tol_s} but not exact, {pct(rounding_share)} are within that rounding "
            "tolerance."
        ),
        "",
        "---",
        "",
        "## 5. The residual: what distinguishes the records not reproduced",
        "",
        (
            f"**{len(residual):,} records ({pct(len(residual) / n)}) miss the {tol_s} "
            f"band**, plus {int(head.undefined.sum())} with no rebuilt value at all."
        ),
        "",
        *md_table(res_tbl),
        "",
        (
            f"- **Several sales at the address.** "
            f"{pct(float((residual.n_tx > 1).mean()))} of residual records have more "
            f"than one sale, against {pct(float((reproduced.n_tx > 1).mean()))} of "
            f"reproduced ones (lift {lift_tx}"
            + (", the largest in the table" if lift_is_max else "")
            + "). By scenario: S1 and S2 — a single sale, whether the plot has one "
            f"unit or many — reproduce exactly on {rate_s12}; S3, several sales on "
            f"a multi-unit plot, on {rate_s3} (section 2.1)."
        ),
        (
            f"- **Not only the unit count.** On {pct(int_share)} of residual "
            "records `frozen Preco / our mean price` is an integer to within "
            f"{vcfg['implied_units_tol']} — those addresses agree on the price "
            "average and disagree only about how many units the plot has. On the "
            f"other {pct(1 - int_share)} the price average differs too."
        ),
        "",
        "Where the multiplier alone differs, by how much (`implied − our units`):",
        "",
        *md_table(delta_counts),
        "",
        (
            f"- **Direction.** Our `Preco` exceeds the frozen one on "
            f"{pct(over_share)} of residual records; median relative difference "
            f"{med_rel:.1%}."
        ),
        (
            f"- **Strict subsets of the sales.** In {probe['hit']} of the "
            f"{probe['probed']} residual records small enough to test exhaustively "
            f"(≤ {probe['cap']} transactions), the frozen value is our formula "
            "computed over a strict subset of the address's sales, with our unit "
            "count unchanged."
        ),
        (
            f"- **Alternative conventions.** {len(alts)} candidate rules were "
            "re-scored on the residual records: the price × statistic × units "
            "combinations of the stage-0b grid, averaging only the sales whose "
            "`Purpose` (or category) matches the record's own, and rounding the "
            "mean before multiplying. The best recovers "
            f"**{int(alts['residual records rescued'].max())}** of "
            f"{len(residual):,}; the best exact rate among them on the full layer "
            f"is {pct(alt_best_exact)}, against the headline's {pct(v_exact)}:"
        ),
        "",
        *md_table(alts),
        "",
        (
            f"- **Blank `Purpose` and join ambiguity**: "
            f"{pct(float(residual.any_blank_purpose_tx.fillna(False).mean()))} of "
            "residual records involve a blank-`Purpose` transaction (vs "
            f"{pct(float(reproduced.any_blank_purpose_tx.fillna(False).mean()))} of "
            f"reproduced ones); {int((~residual.in_registry.fillna(False)).sum())} have "
            f"no registry row at all; {int(residual.dup_addr_record.sum())} share "
            f"their address with another record of the layer; "
            f"{n_resid_only_parking} are only-parking addresses."
        ),
        "",
        (
            "Every residual record, with both values, the implied unit count and all "
            "the counts behind it: `outputs/tables/stage1_residuals.csv` "
            f"({len(res_detail):,} rows)."
        ),
        "",
        "### 5.1 The parent-only refinement and two diagnostics",
        "",
        (
            "Three patterns visible in `stage1_residuals.csv` were turned into "
            "candidate rules and re-scored on the **whole** layer. Only the first "
            "is a candidate for adoption; the other two are diagnostics."
        ),
        "",
        *md_table(refine),
        "",
        (
            f"Counting a parent-only address as one plot changes exact reproduction "
            f"from {ref_base['exact']} to {ref_new['exact']} and the {within_col} "
            f"rate from {ref_base[within_col]} to {ref_new[within_col]}, rescuing "
            f"{int(ref_new['rescues (vs n_parents)'])} residual records and "
            f"breaking {int(ref_new['newly breaks'])}. It touches "
            f"{len(refine_detail)} records, with "
            f"{int(refine_detail.n_parent_rows.min()) if len(refine_detail) else 0}–"
            f"{int(refine_detail.n_parent_rows.max()) if len(refine_detail) else 0} "
            "parent rows each. The config uses "
            f"`{dv['parent_only_address_counts_as']}`."
        ),
        "",
        *md_table(refine_detail),
        "",
        (
            "The diagnostics are not adopted: on the full layer their "
            f"{within_col} rate is lower than the refinement's by "
            + ", ".join(f"{100 * g:.1f} pp" for g in diag_gap)
            + " respectively (rows 3 onwards of the table)."
        ),
        "",
        "---",
        "",
        "## 6. The funnel",
        "",
        *md_table(funnel_tbl),
        "",
        (
            "Positivity **before** the IQR fence, both runs, via "
            "`common.apply_exclusions` — the Python twin of the helper the R gate "
            "uses. Counts in and out of every filter: "
            "`outputs/tables/stage1_counts.csv`."
        ),
        "",
        (
            f"Positivity keeps {rebuilt_steps[1]:,} rebuilt records against "
            f"{submitted[1]:,} submitted and {frozen_steps[1]:,} for the frozen "
            f"column. The IQR step drops {rebuilt_steps[1] - rebuilt_steps[2]} "
            f"records against the submitted {submitted[1] - submitted[2]}, because "
            "the fence is computed on the rebuilt `Puni` and the records whose "
            "value differs move the quartiles. Net: "
            f"**{rebuilt_steps[2]:,}** against **{submitted[2]:,}**, {overlap:,} "
            f"records in both final samples, {only_rebuilt} only in the rebuilt one "
            f"and {only_frozen} only in the frozen one."
        ),
        "",
        (
            "The step the pipeline does not reproduce is the one from "
            f"{itbi.addr_key.nunique():,} distinct ITBI addresses to the "
            f"{len(geo):,} georeferenced records: that selection is the frozen "
            "layer's own, produced by the georeferencing, and no rule over the raw "
            "files reconstructs it."
        ),
        "",
        "---",
        "",
        "## 7. Outputs",
        "",
        *md_table(
            pd.DataFrame(
                [
                    {
                        "file": str(p.relative_to(common.REPO)),
                        "rows": len(built[name]),
                        "sha256": digests[name]["sha256"][:16] + "…",
                    }
                    for name, p in written.items()
                ]
            )
        ),
        "",
        (
            "Full digests: `data/interim/stage1_manifest.json`. Tables: "
            "`stage1_counts.csv`, `stage1_validation_summary.csv`, "
            "`stage1_residual_characterisation.csv`, `stage1_residuals.csv`, "
            "`stage1_alternative_conventions.csv`, `stage1_rule_refinement.csv`, "
            "`stage1_rule_refinement_records.csv`, `stage1_scenarios.csv`, "
            "`stage1_variant_comparison.csv`, `stage1_funnel.csv`, "
            "`stage1_plot_area_agreement.csv` (all under "
            "`outputs/tables/`)."
        ),
        "",
    ]
    out = ctx.report("stage1_dependent_variable.md")
    out.write_text("\n".join(md) + "\n")
    ctx.log.info("wrote %s", out.relative_to(common.REPO))

    ctx.write_counts("stage1_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
