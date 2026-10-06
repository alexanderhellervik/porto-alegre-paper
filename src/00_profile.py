#!/usr/bin/env python
"""Stage 0b — full data profiling. Produces reports/stage0_profile.md.

Makes no analytical decision: it measures the raw ITBI transactions, the unit
registry and the frozen georeferenced address layer, and lays the alternative
conventions side by side with the evidence for each. The conventions the
pipeline uses are set in `dependent_variable:` in config.yaml; everything this
script thresholds on comes from `profiling:`.

It reports:

  - distinct `Purpose` values with counts in both files (and in the frozen
    layer), and the purpose -> category mapping (parking / residential /
    commercial / special / missing) with uncertain rows flagged.
    `Purpose_EN` in raw.itbi_xlsx translates the ITBI purpose values; the
    registry has no such column, so its vocabulary is resolved onto the ITBI
    one by accent-folded, '?'-wildcarded matching and translated from there.
  - exact-duplicate ITBI row counts (config: dependent_variable.itbi_duplicates)
  - missingness incl. the literal `INDETERMINADO` code (io.missing_codes)
  - unit counts per address under each counting convention
    (all / non-parking / non-parking-non-parent)
  - plot-area distribution; mega-plot outliers; ITBI vs registry vs geolayer
    plot-area conflicts
  - ITBI -> registry -> geolayer join match rates after normalisation, with a
    sample of failures (rapidfuzz for diagnostics only: the score and the
    threshold are recorded and no fuzzy match is accepted into the data)
  - parking share of transactions and of registry units; price distribution
    parking vs non-parking
  - which candidate construction of the frozen `Preco` column the data supports

Figures quoted in the referee reports and the analysis-sample size are
checked explicitly as MATCH / MISMATCH against `validation.parking_reference`
and `exclusions.targets`.

Run:  python src/00_profile.py     (or `make profile`)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import prep
from prep import md_table, pct, to_num, verdict, wildcard_pattern

# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def section_purpose(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]

    # The mapping itself lives in src/prep.py so that stage 1 classifies the
    # same strings the same way.
    mapping, mstats = prep.purpose_mapping(ctx, f)
    out = ctx.table("stage0b_purpose_mapping.csv")
    mapping.to_csv(out, index=False)
    ctx.log.info("wrote %s (%d rows)", out.name, len(mapping))

    n_unresolved = mstats["unresolved"]
    n_lowconf = mstats["low_confidence"]
    ctx.count("purpose: distinct values (ITBI)", int(itbi.purpose_norm.nunique()))
    ctx.count("purpose: distinct values (registry)", int(reg.purpose_norm.nunique()))

    cats = (
        pd.DataFrame(
            {
                "ITBI transactions": itbi.category.value_counts(),
                "registry units": reg.category.value_counts(),
                "geolayer addresses": geo.category.value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    cats.insert(0, "category", cats.index)

    md = [
        "## 2. `Purpose` vocabularies and the category mapping",
        "",
        (
            f"ITBI has **{itbi.purpose_norm.nunique()}** distinct `Purpose` values "
            f"(including blank), the registry **{reg.purpose_norm.nunique()}**, the "
            f"frozen point layer **{geo.purpose_norm.nunique()}** (`Finalidade`)."
        ),
        "",
        (
            "Only ITBI carries a translation (`Purpose_EN` in the ITBI workbook). The "
            "other two vocabularies were mapped onto ITBI's by accent-folded, "
            "separator-free, `?`-wildcarded matching and translated from there. "
            f"**{n_unresolved}** registry/geolayer values have no ITBI counterpart "
            f"(and so no translation); **{n_lowconf}** mapping rows come from a "
            "low-confidence rule and are flagged `low_confidence_rule`."
        ),
        "",
        (
            "Full mapping: `outputs/tables/stage0b_purpose_mapping.csv` "
            "(purpose value, file, count, EN, category, confidence, flags)."
        ),
        "",
        "### 2.1 Category totals under the mapping",
        "",
        *md_table(cats),
        "",
        "### 2.2 Flagged mapping rows",
        "",
        *md_table(
            mapping[mapping["flags"] != ""][
                [
                    "file",
                    "purpose_value",
                    "count",
                    "purpose_en",
                    "category",
                    "confidence",
                    "flags",
                ]
            ].reset_index(drop=True)
        ),
        "",
    ]
    return md, mapping, {"unresolved": n_unresolved, "low_confidence": n_lowconf}


def section_parking(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, Any]]:
    pcfg = ctx.cfg["profiling"]["purpose"]
    ref = ctx.cfg["validation"]["parking_reference"]
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]

    defs = pcfg["parking_definitions"]
    active = pcfg["active_parking_definition"]
    rows = []
    for name, pattern in defs.items():
        it_mask = itbi.purpose_norm.str.contains(pattern, regex=True)
        rg_mask = reg.purpose_norm.str.contains(pattern, regex=True)
        g = itbi.assign(p=it_mask).groupby("addr_key")["p"].agg(["sum", "count"])
        only = int((g["sum"] == g["count"]).sum())
        rows.append(
            {
                "definition": name + (" (active)" if name == active else ""),
                "pattern": pattern,
                "ITBI parking rows": int(it_mask.sum()),
                "ITBI share": pct(it_mask.mean()),
                "registry parking units": int(rg_mask.sum()),
                "registry share": pct(rg_mask.mean()),
                "only-parking addresses": only,
            }
        )
    defs_tbl = pd.DataFrame(rows)

    # flags come from prep so stage 1 uses the identical parking definition
    pattern = prep.annotate_parking(ctx, f)
    price = itbi["price"]
    share = float(itbi.parking.mean())
    med_p = float(price[itbi.parking].median())
    med_o = float(price[~itbi.parking].median())
    reg_share = float(reg.parking.mean())
    g = itbi.groupby("addr_key")["parking"].agg(["sum", "count"])
    only_parking = int((g["sum"] == g["count"]).sum())
    any_parking = int((g["sum"] > 0).sum())
    coverage = float(((itbi.Purpose != "") & (itbi["Built Area"] != "")).mean())

    ctx.count(
        "itbi: parking transactions", int(itbi.parking.sum()), f"definition={active}"
    )
    ctx.count("itbi: non-parking transactions", int((~itbi.parking).sum()))
    ctx.count("registry: parking units", int(reg.parking.sum()))
    ctx.count("itbi: distinct addresses", int(itbi.addr_key.nunique()))
    ctx.count("itbi: only-parking addresses", only_parking)

    dist = pd.DataFrame(
        {
            "statistic": ["n", "mean", "median", "p25", "p75", "min", "max"],
            "parking": [
                int(itbi.parking.sum()),
                price[itbi.parking].mean(),
                med_p,
                price[itbi.parking].quantile(0.25),
                price[itbi.parking].quantile(0.75),
                price[itbi.parking].min(),
                price[itbi.parking].max(),
            ],
            "non-parking": [
                int((~itbi.parking).sum()),
                price[~itbi.parking].mean(),
                med_o,
                price[~itbi.parking].quantile(0.25),
                price[~itbi.parking].quantile(0.75),
                price[~itbi.parking].min(),
                price[~itbi.parking].max(),
            ],
        }
    )

    res = {
        "share": share,
        "median_parking": med_p,
        "median_other": med_o,
        "registry_share": reg_share,
        "only_parking": only_parking,
        "any_parking": any_parking,
        "coverage": coverage,
        "only_parking_in_geolayer": int(
            geo.addr_key.isin(set(g[g["sum"] == g["count"]].index)).sum()
        ),
    }

    md = [
        "## 3. Parking",
        "",
        f"Active definition: **`{active}`** = `{pattern}`. The other definitions "
        "are reported alongside it so the effect of the edge cases is visible. "
        + (
            "The `broad` definition (which adds the `BOX` / `VAGA` spellings) "
            "returns exactly the same ITBI rows as `standard`."
            if int(
                defs_tbl.loc[defs_tbl.definition == "broad", "ITBI parking rows"].iloc[
                    0
                ]
            )
            == int(
                defs_tbl.loc[
                    defs_tbl.definition.str.startswith("standard"), "ITBI parking rows"
                ].iloc[0]
            )
            else "The `broad` definition adds ITBI rows that `standard` does not match."
        ),
        "",
        *md_table(defs_tbl),
        "",
        "### 3.1 Price distribution, parking vs non-parking (ITBI, BRL)",
        "",
        *md_table(dist, floatfmt="{:,.0f}"),
        "",
        (
            f"Of {itbi.addr_key.nunique():,} distinct ITBI addresses, {any_parking:,} "
            f"have at least one parking transaction and **{only_parking}** have "
            f"nothing but parking. Only **{res['only_parking_in_geolayer']}** of those "
            f"only-parking addresses reached the frozen {len(geo):,}-point layer."
        ),
        "",
        (
            f"Figures quoted in the referee reports (`validation.parking_reference`): share "
            f"{ref['share']:.3f}, medians {ref['median_price_parking']:,} / "
            f"{ref['median_price_other']:,}, {ref['addresses_only_parking']} "
            f"only-parking addresses, registry share {ref['registry_parking_share']:.3f} "
            "— reconciled in section 9."
        ),
        "",
    ]
    return md, res


def section_duplicates_missing(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, Any]]:
    codes = ctx.cfg["io"]["missing_codes"]
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]
    raw_cols_itbi = prep.ITBI_ROW_KEY
    raw_cols_reg = prep.REG_ROW_KEY

    dup_itbi = int(itbi.duplicated(subset=raw_cols_itbi).sum())
    dup_reg = int(reg.duplicated(subset=raw_cols_reg).sum())
    dup_geo_key = int(len(geo) - geo.addr_key.nunique())
    n_is_one = int((pd.to_numeric(geo["n"], errors="coerce") == 1).sum())

    ctx.count(
        "itbi: rows after exact dedupe",
        len(itbi) - dup_itbi,
        f"{dup_itbi} exact duplicates",
    )
    ctx.count(
        "registry: rows after exact dedupe",
        len(reg) - dup_reg,
        f"{dup_reg} exact duplicates",
    )
    ctx.count(
        "geolayer: distinct address keys",
        int(geo.addr_key.nunique()),
        f"{dup_geo_key} duplicated address rows",
    )

    miss_rows = []
    for name, df, cols in (
        ("itbi", itbi, raw_cols_itbi),
        ("registry", reg, raw_cols_reg),
    ):
        for c in cols:
            s = df[c]
            miss_rows.append(
                {
                    "file": name,
                    "column": c,
                    "n": len(df),
                    "blank": int((s == "").sum()),
                    "INDETERMINADO": int((s == "INDETERMINADO").sum()),
                    "zero": int((s == "0").sum()),
                    "missing_any": int(s.isin(codes).sum()),
                    "missing_pct": round(100 * float(s.isin(codes).mean()), 2),
                }
            )
    miss = pd.DataFrame(miss_rows)
    miss.to_csv(ctx.table("stage0b_missingness.csv"), index=False)

    dupdf = pd.DataFrame(
        {
            "file": ["ITBI", "registry", "frozen geolayer"],
            "rows": [len(itbi), len(reg), len(geo)],
            "exact duplicate rows": [dup_itbi, dup_reg, "n/a (all columns differ)"],
            "distinct address keys": [
                itbi.addr_key.nunique(),
                reg.addr_key.nunique(),
                geo.addr_key.nunique(),
            ],
            "duplicated address rows": [
                "n/a (repeat sales)",
                "n/a (unit rows)",
                dup_geo_key,
            ],
        }
    )

    blank_purpose = int((itbi.Purpose == "").sum())
    md = [
        "## 4. Duplicates and missingness",
        "",
        *md_table(dupdf),
        "",
        (
            f"**ITBI: {dup_itbi} exactly duplicated rows** (same price, address, unit, "
            "purpose). They are indistinguishable from a genuine repeat sale of the "
            "same unit at the same price in the same year, so this is a convention, "
            "not a defect (`dependent_variable.itbi_duplicates`, currently `dedupe`)."
        ),
        "",
        (
            f"**Registry: {dup_reg:,} exactly duplicated rows** "
            f"({pct(dup_reg / len(reg))}). These matter directly: they inflate the "
            "unit multiplier. Deduplicating drops the registry from "
            f"{len(reg):,} to {len(reg) - dup_reg:,} unit rows."
        ),
        "",
        (
            f"**Frozen point layer: {dup_geo_key} rows duplicate an address already "
            "present** (identical `Rua`/`Numero`/`Unidade`/`Preco`/`Puni`), so the "
            f"{len(geo):,} points cover only {geo.addr_key.nunique():,} distinct "
            "addresses. The layer's `n` column is a duplicate counter, not a unit "
            f"multiplier (it is 1 on {n_is_one:,} rows)."
        ),
        "",
        "### 4.1 Missingness, including the literal `INDETERMINADO` code",
        "",
        *md_table(miss),
        "",
        (
            f"`Purpose` is blank on **{blank_purpose:,} ITBI rows "
            f"({pct(blank_purpose / len(itbi))})**. Those rows "
            "cannot be classified parking / non-parking at all, which is why "
            "`dependent_variable.blank_purpose` exists. Section 8 shows what each "
            "choice does to the reconstructed dependent variable."
        ),
        "",
        (
            f"`Built Area` is `INDETERMINADO` on {int((reg['Built Area'] == 'INDETERMINADO').sum()):,} "
            f"registry rows and `Plot Area` on only "
            f"{int((reg['Plot Area'] == 'INDETERMINADO').sum())}. In ITBI the "
            "equivalent code is a plain zero, not a string — see section 6."
        ),
        "",
    ]
    return md, {
        "dup_itbi": dup_itbi,
        "dup_reg": dup_reg,
        "dup_geo_key": dup_geo_key,
        "blank_purpose": blank_purpose,
    }


def section_units(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, Any]]:
    reg, geo = f["registry"], f["geolayer"]
    convs = ctx.cfg["profiling"]["unit_count_conventions"]

    masks = {
        "all": pd.Series(True, index=reg.index),
        "non_parking": ~reg.parking,
        "non_parking_non_parent": ~reg.parking & ~reg.parent,
    }
    rows = []
    per_addr: dict[str, pd.Series] = {}
    geo_keys = set(geo.addr_key)
    for name in convs:
        m = masks[name]
        counts = reg[m].groupby("addr_key").size()
        per_addr[name] = counts
        on_geo = pd.Series(geo.addr_key.map(counts).to_numpy(), index=geo.index)
        rows.append(
            {
                "convention": name,
                "registry rows used": int(m.sum()),
                "addresses with >=1 unit": len(counts),
                "mean units/address": round(float(counts.mean()), 2),
                "median": int(counts.median()),
                "max": int(counts.max()),
                "addresses dropped to 0": int(len(reg.addr_key.unique()) - len(counts)),
                f"mean units, {len(geo):,} analysis addresses": round(
                    float(on_geo.mean()), 2
                ),
                "analysis addresses with 0/NA": int(on_geo.isna().sum()),
            }
        )
        ctx.count(f"registry: unit rows [{name}]", int(m.sum()))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(ctx.table("stage0b_unit_counts.csv"), index=False)

    n_parent = int(reg.parent.sum())
    md = [
        "## 5. Unit counts per address, by counting convention",
        "",
        (
            "The unit multiplier is the whole of the dependent variable's numerator, "
            "so the convention is not cosmetic. Counts below are on the **as-supplied** "
            f"registry (deduplicating it first removes {int(reg.duplicated().sum()):,} "
            "rows and lowers every count — see section 4)."
        ),
        "",
        *md_table(tbl),
        "",
        (
            f"`Unity` is blank on **{n_parent:,} registry rows ({pct(n_parent / len(reg))})** "
            "— the address-level 'parent' rows. "
            "Excluding them is far more destructive than it looks: it takes "
            f"{int(len(reg.addr_key.unique()) - len(per_addr['non_parking_non_parent'])):,} "
            f"of {reg.addr_key.nunique():,} registry addresses to **zero** units, "
            "because a detached house is represented by a single parent row with no "
            f"numbered unit under it. On the {len(geo):,} analysis addresses the same "
            f"convention leaves {int(geo.addr_key.map(per_addr['non_parking_non_parent']).isna().sum()):,} "
            "addresses with no units at all."
        ),
        "",
        (
            f"Under the `all` convention, {len(geo_keys - set(reg.addr_key))} of the "
            f"{len(geo):,} analysis addresses have no registry row."
        ),
        "",
    ]
    return md, {"per_addr": per_addr, "parent_rows": n_parent}


def section_plot_area(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, Any]]:
    cfg = ctx.cfg["profiling"]["plot_area"]
    codes = ctx.cfg["io"]["missing_codes"]
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]

    reg["plot_area"] = to_num(reg["Plot Area"], codes)
    itbi["plot_area"] = to_num(itbi["Plot Area"], codes)
    geo["plot_area"] = pd.to_numeric(geo["Terreno"], errors="coerce")

    def area_col(s: pd.Series) -> list[str]:
        return [
            f"{int(s.notna().sum()):,}",
            f"{int((s == 0).sum()):,}",
            f"{s.mean():,.1f}",
            f"{s.median():,.1f}",
            f"{s.quantile(0.95):,.1f}",
            f"{s.max():,.0f}",
        ]

    dist = pd.DataFrame(
        {
            "statistic": ["n non-missing", "zero", "mean", "median", "p95", "max"],
            "ITBI (m2)": area_col(itbi.plot_area),
            "registry (m2)": area_col(reg.plot_area),
            "geolayer (m2)": area_col(geo.plot_area),
        }
    )

    cap = cfg["mega_plot_flag_m2"]
    tol = cfg["conflict_rel_tol"]
    reg_addr = reg.groupby("addr_key")["plot_area"].median()
    itbi_addr = itbi[itbi.plot_area > 0].groupby("addr_key")["plot_area"].median()
    cmp = geo[["addr_key", "Rua", "Numero", "Finalidade", "plot_area"]].copy()
    cmp["registry"] = cmp.addr_key.map(reg_addr)
    cmp["itbi"] = cmp.addr_key.map(itbi_addr)
    for src in ("registry", "itbi"):
        cmp[f"rel_diff_{src}"] = (cmp.plot_area - cmp[src]).abs() / cmp[src].replace(
            0, np.nan
        )
    conflict_reg = cmp[cmp.rel_diff_registry > tol]
    conflict_itbi = cmp[cmp.rel_diff_itbi > tol]
    conflicts = cmp[
        (cmp.rel_diff_registry > tol) | (cmp.rel_diff_itbi > tol)
    ].sort_values("rel_diff_registry", ascending=False)
    conflicts.round(4).to_csv(ctx.table("stage0b_plot_area_conflicts.csv"), index=False)
    ctx.count("geolayer: plot-area conflicts vs another source", len(conflicts))

    agree = pd.DataFrame(
        {
            "comparison": ["geolayer vs registry", "geolayer vs ITBI (non-zero only)"],
            "addresses comparable": [
                int(cmp.registry.notna().sum()),
                int(cmp.itbi.notna().sum()),
            ],
            f"coverage of {len(geo):,}": [
                pct(float(cmp.registry.notna().mean())),
                pct(float(cmp.itbi.notna().mean())),
            ],
            f"agree within {tol:.0%}": [
                int((cmp.rel_diff_registry <= tol).sum()),
                int((cmp.rel_diff_itbi <= tol).sum()),
            ],
            "conflicts": [len(conflict_reg), len(conflict_itbi)],
        }
    )

    mega_reg = reg[reg.plot_area > cap]
    mega_geo = geo[geo.plot_area > cap]
    especial_geo = int((geo.purpose_norm == "IMOVEL ESPECIAL").sum())

    # The analysis sample of the submitted disaggregated tables, through the
    # same exclusion code and config as the rest of the pipeline.
    final = common.apply_exclusions(ctx, geo, "disaggregated")
    especial_final = int((final.purpose_norm == "IMOVEL ESPECIAL").sum())
    mega_final = int((final.Terreno > cap).sum())

    reg_max = reg.loc[reg.plot_area.idxmax()]
    geo_max = float(geo.plot_area.max())
    final_max = float(final.Terreno.max())
    top = final.nlargest(8, "Terreno")[
        ["Rua", "Numero", "Finalidade", "Terreno", "Preco", "Puni"]
    ].reset_index(drop=True)

    md = [
        "## 6. Plot area: distribution, sources and conflicts",
        "",
        *md_table(dist, floatfmt="{:,.1f}"),
        "",
        (
            f"**ITBI `Plot Area` is unusable as the denominator.** It is 0 on "
            f"{int((itbi.plot_area == 0).sum()):,} of {len(itbi):,} rows "
            f"({pct(float((itbi.plot_area == 0).mean()))}), apartment units among "
            "them."
        ),
        "",
        f"### 6.1 Agreement between sources, on the {len(geo):,} analysis addresses",
        "",
        *md_table(agree),
        "",
        (
            f"The geolayer's `Terreno` equals the registry's plot area on "
            f"{int((cmp.rel_diff_registry <= tol).sum()):,} of "
            f"{int(cmp.registry.notna().sum()):,} addresses "
            f"({pct(float((cmp.rel_diff_registry <= tol).sum() / cmp.registry.notna().sum()))}), "
            f"leaving {len(conflict_reg)} conflicts. Where ITBI reports a non-zero "
            f"plot area it disagrees with the geolayer on {len(conflict_itbi)} of "
            f"{int(cmp.itbi.notna().sum()):,} addresses. The geolayer's `Terreno` "
            "(`plot_area_source: geolayer`) is therefore, in practice, nearly "
            "identical to the registry's plot area."
        ),
        "",
        (
            f"Every disagreeing address is listed in "
            f"`outputs/tables/stage0b_plot_area_conflicts.csv` ({len(conflicts)} "
            "rows, with the value from each source and the relative difference)."
        ),
        "",
        f"### 6.2 Mega-plots (flag threshold {cap:,} m², report-only)",
        "",
        (
            f"- registry unit rows above the flag: **{len(mega_reg):,}** on "
            f"{mega_reg.addr_key.nunique():,} addresses; largest "
            f"{reg.plot_area.max():,.0f} m² ({reg_max['Street']}, "
            f"`{reg_max['Purpose']}`)."
        ),
        (
            f"- frozen {len(geo):,}-point layer: **{len(mega_geo)}** points above the "
            f"flag, max {geo_max:,.0f} m²."
        ),
        (
            f"- **analysis sample, N = {len(final):,}: {mega_final} points above the "
            f"flag survive, and {especial_final} of the {especial_geo} "
            "`IMOVEL ESPECIAL` records survive.** "
            + (
                f"The largest frozen plot ({geo_max:,.0f} m²) does not survive the "
                f"exclusions; the largest that does is {final_max:,.0f} m²."
                if final_max < geo_max
                else f"The largest plot in the sample is {final_max:,.0f} m²."
            )
        ),
        "",
        "Largest plots inside the analysis sample:",
        "",
        *md_table(top, floatfmt="{:,.0f}"),
        "",
    ]
    return md, {
        "conflict_reg": len(conflict_reg),
        "conflict_itbi": len(conflict_itbi),
        "mega_final": mega_final,
        "especial_final": especial_final,
        "n_final": len(final),
        "final": final,
    }


def section_joins(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, Any]]:
    from rapidfuzz import fuzz, process

    fcfg = ctx.cfg["profiling"]["fuzzy"]
    acfg = ctx.cfg["profiling"]["address"]
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]

    reg_keys = set(reg.addr_key)
    geo_keys = set(geo.addr_key)
    itbi_keys = set(itbi.addr_key)

    # '?' wildcard hop: registry keys carrying a corrupted street name are
    # matched against the clean ITBI/geolayer spellings. Set
    # profiling.address.question_mark to `literal` to see what is lost.
    wildcard_on = acfg["question_mark"] == "wildcard"
    q_streets = (
        sorted({s for s in reg.street_norm.unique() if "?" in s}) if wildcard_on else []
    )
    clean_streets = sorted(itbi.street_norm.unique()) + sorted(geo.street_norm.unique())
    wildcard_rows = []
    recovered_keys: set[str] = set()
    recovered_rows = 0
    for q in q_streets:
        pat = wildcard_pattern(q)
        hits = sorted({c for c in clean_streets if pat.match(c)})
        n_itbi = int(itbi.street_norm.isin(hits).sum())
        recovered = set()
        if hits:
            nums = set(reg.loc[reg.street_norm == q, "number_norm"])
            for h in hits:
                recovered |= {
                    f"{h}|{n}"
                    for n in itbi.loc[itbi.street_norm == h, "number_norm"]
                    if n in nums
                }
        recovered_keys |= recovered
        recovered_rows += int(itbi.addr_key.isin(recovered).sum())
        wildcard_rows.append(
            {
                "corrupted registry street": q,
                "registry rows": int((reg.street_norm == q).sum()),
                "resolves to": "; ".join(hits) or "(no counterpart)",
                "ITBI rows on that street": n_itbi,
                "ITBI address keys recovered": len(recovered),
            }
        )
    wildcard = pd.DataFrame(wildcard_rows)

    reg_keys_wild = reg_keys | recovered_keys

    hops = []

    def hop(name: str, left: pd.Series, right: set[str], unit: str) -> float:
        rate = float(left.isin(right).mean())
        hops.append(
            {
                "hop": name,
                "unit": unit,
                "n": len(left),
                "matched": int(left.isin(right).sum()),
                "match_rate": round(rate, 4),
                "match_pct": pct(rate),
            }
        )
        return rate

    itbi_addr = pd.Series(sorted(itbi_keys))
    r_rows = hop("ITBI -> registry", itbi.addr_key, reg_keys_wild, "transaction rows")
    r_addr = hop("ITBI -> registry", itbi_addr, reg_keys_wild, "distinct addresses")
    hop("ITBI -> geolayer", itbi.addr_key, geo_keys, "transaction rows")
    g_addr = hop("ITBI -> geolayer", itbi_addr, geo_keys, "distinct addresses")
    hop("geolayer -> registry", geo.addr_key, reg_keys_wild, "address points")
    hop("geolayer -> ITBI", geo.addr_key, itbi_keys, "address points")
    hops_df = pd.DataFrame(hops)
    hops_df.to_csv(ctx.table("stage0b_join_rates.csv"), index=False)
    ctx.count(
        "itbi: rows matched to registry", int(itbi.addr_key.isin(reg_keys_wild).sum())
    )
    ctx.count("itbi: rows matched to geolayer", int(itbi.addr_key.isin(geo_keys).sum()))

    # --- failure diagnostics; rapidfuzz scores are reported, never applied ------
    fails = itbi_addr[~itbi_addr.isin(reg_keys_wild)]
    reg_streets = sorted(reg.street_norm.unique())
    scorer = getattr(fuzz, fcfg["scorer"])
    rows = []
    for k in fails:
        street, number = k.rsplit("|", 1)
        best = process.extractOne(street, reg_streets, scorer=scorer)
        rows.append(
            {
                "itbi_street": street,
                "itbi_number": number,
                "nearest_registry_street": best[0],
                "fuzzy_score": round(float(best[1]), 1),
                "scorer": fcfg["scorer"],
                "report_threshold": fcfg["report_threshold"],
                "street_matches_exactly": best[1] == 100.0,
                "number_exists_on_that_street": f"{best[0]}|{number}" in reg_keys_wild,
                "accepted_into_data": False,
                "itbi_rows_affected": int((itbi.addr_key == k).sum()),
            }
        )
    fails_df = pd.DataFrame(rows)
    fails_df.to_csv(ctx.table("stage0b_join_failures.csv"), index=False)
    ctx.count("itbi: addresses failing the registry join", len(fails_df))

    exact_street = int(fails_df.street_matches_exactly.sum()) if len(fails_df) else 0
    recoverable = (
        int(
            (
                fails_df.number_exists_on_that_street & ~fails_df.street_matches_exactly
            ).sum()
        )
        if len(fails_df)
        else 0
    )
    above = (
        int((fails_df.fuzzy_score >= fcfg["report_threshold"]).sum())
        if len(fails_df)
        else 0
    )
    rows_lost = int(fails_df.itbi_rows_affected.sum()) if len(fails_df) else 0

    sample_cols = [
        "itbi_street",
        "itbi_number",
        "nearest_registry_street",
        "fuzzy_score",
        "street_matches_exactly",
        "number_exists_on_that_street",
        "itbi_rows_affected",
    ]
    sample = (
        fails_df.sort_values("fuzzy_score", ascending=False)
        .head(fcfg["failure_sample_n"])[sample_cols]
        .reset_index(drop=True)
        if len(fails_df)
        else pd.DataFrame(columns=sample_cols)
    )

    md = [
        "## 7. Joins: ITBI → registry → geolayer",
        "",
        (
            "Key = normalised `Street` + `Number` (ITBI has no complete-address "
            "column). Normalisation: accent fold, upper-case, non-alphanumerics "
            "collapsed to single spaces, `?` treated as a single-character wildcard. "
            "No street-type expansion (e.g. `DR` → `DOUTOR`) is applied."
        ),
        "",
        "### 7.1 Match rate per hop",
        "",
        *md_table(hops_df),
        "",
        "### 7.2 What the `?` wildcard actually buys",
        "",
        *md_table(wildcard),
        "",
        (
            f"The {int((reg.street_norm.isin(q_streets)).sum())} corrupted registry "
            f"street rows carry {len(q_streets)} distinct names; "
            f"{int((wildcard['resolves to'] != '(no counterpart)').sum()) if len(wildcard) else 0} "
            "of them have a counterpart in the transaction or address data. "
            f"Treating `?` literally would drop **{len(recovered_keys)}** ITBI "
            f"address key(s) covering {recovered_rows} transactions, without any "
            "error."
        ),
        "",
        "### 7.3 Failures (rapidfuzz diagnostics — nothing is accepted)",
        "",
        (
            f"**{len(fails_df)}** ITBI addresses ({pct(1 - r_addr)} of "
            f"{len(itbi_addr):,}) have no registry row, covering **{rows_lost:,}** "
            f"transactions ({pct(1 - r_rows)}). Scored with "
            f"`rapidfuzz.fuzz.{fcfg['scorer']}`, report threshold "
            f"{fcfg['report_threshold']}. The scores are diagnostics only: **no fuzzy "
            "match is written into any dataset**."
        ),
        "",
        (
            f"- {exact_street} of {len(fails_df)} failures have a street name that "
            "matches the registry **exactly** (score 100). They fail on the *number*, "
            "not on spelling — fuzzy street matching cannot rescue them."
        ),
        f"- {above} failures score ≥ {fcfg['report_threshold']}.",
        (
            f"- {recoverable} failures would be rescued by accepting the best fuzzy "
            "street and keeping the number. That is the entire upside of fuzzy "
            f"matching here: {pct(recoverable / max(len(fails_df), 1))} of the "
            "failures, and it is not taken."
        ),
        "",
        (
            f"Sample of failures (full list: `outputs/tables/stage0b_join_failures.csv`, "
            f"{len(fails_df)} rows, every score recorded):"
        ),
        "",
        *md_table(sample),
        "",
    ]
    return md, {
        "row_rate": r_rows,
        "addr_rate": r_addr,
        "geo_addr_rate": g_addr,
        "n_fail": len(fails_df),
        "rows_lost": rows_lost,
        "wildcard_recovered": len(recovered_keys),
    }


def section_preco(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[list[str], dict[str, Any]]:
    cfg = ctx.cfg["profiling"]["preco_reverse_engineering"]
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]
    if not cfg["enabled"]:
        return ["## 8. Reconstructing `Preco` — disabled in config.", ""], {}

    tol = cfg["match_rel_tol"]
    preco = pd.to_numeric(geo["Preco"], errors="coerce")
    price_masks = {
        "all": pd.Series(True, index=itbi.index),
        "non_parking": ~itbi.parking,
    }
    unit_masks = {
        "all": pd.Series(True, index=reg.index),
        "non_parking": ~reg.parking,
        "non_parking_non_parent": ~reg.parking & ~reg.parent,
    }
    blank_masks = {
        "kept_as_nonparking": pd.Series(True, index=itbi.index),
        "dropped": itbi.Purpose != "",
    }

    rows = []
    for pdedupe in cfg["dedupe_grid"]:
        base_i = (
            itbi.drop_duplicates(subset=prep.ITBI_ROW_KEY)
            if pdedupe == "dedupe"
            else itbi
        )
        for rdedupe in cfg["dedupe_grid"]:
            base_r = reg.drop_duplicates() if rdedupe == "dedupe" else reg
            for psub in cfg["price_subsets"]:
                for pstat in cfg["price_stats"]:
                    pm = base_i[price_masks[psub].reindex(base_i.index)]
                    price = pm.groupby("addr_key")["price"].agg(pstat)
                    for usub in cfg["unit_subsets"]:
                        um = base_r[unit_masks[usub].reindex(base_r.index)]
                        units = um.groupby("addr_key").size()
                        est = geo.addr_key.map(price) * geo.addr_key.map(units)
                        rel = (est - preco).abs() / preco.replace(0, np.nan)
                        rows.append(
                            {
                                "itbi_dedupe": pdedupe,
                                "registry_dedupe": rdedupe,
                                "price_subset": psub,
                                "price_stat": pstat,
                                "unit_subset": usub,
                                "exact": round(
                                    float(np.isclose(est, preco, rtol=1e-6).mean()), 4
                                ),
                                f"within_{tol:.0%}": round(
                                    float((rel <= tol).mean()), 4
                                ),
                                "undefined": int(est.isna().sum()),
                            }
                        )
    grid = pd.DataFrame(rows).sort_values("exact", ascending=False, ignore_index=True)
    grid.to_csv(ctx.table("stage0b_preco_conventions.csv"), index=False)

    # Blank-Purpose sensitivity: the best row's price statistic, non-parking
    # prices from the ITBI rows as supplied, non-parking units from the
    # deduplicated registry. Only the treatment of blank purposes varies.
    best = grid.iloc[0]
    blank_rows = []
    for bname, bmask in blank_masks.items():
        pm = itbi[(~itbi.parking) & bmask]
        price = pm.groupby("addr_key")["price"].agg(best.price_stat)
        um = reg.drop_duplicates()
        um = um[(~um.parking).reindex(um.index)]
        units = um.groupby("addr_key").size()
        est = geo.addr_key.map(price) * geo.addr_key.map(units)
        rel = (est - preco).abs() / preco.replace(0, np.nan)
        blank_rows.append(
            {
                "blank Purpose treated as": bname,
                "ITBI rows used": len(pm),
                "exact reproduction of frozen Preco": round(
                    float(np.isclose(est, preco, rtol=1e-6).mean()), 4
                ),
                f"within {tol:.0%}": round(float((rel <= tol).mean()), 4),
                "addresses left undefined": int(est.isna().sum()),
            }
        )
    blank_tbl = pd.DataFrame(blank_rows)
    blank_n = int((itbi.Purpose == "").sum())
    exact_col = "exact reproduction of frozen Preco"
    kept_exact = float(blank_tbl[exact_col].iloc[0])
    dropped_exact = float(blank_tbl[exact_col].iloc[1])
    kept_better = kept_exact > dropped_exact
    by_stat = grid.groupby("price_stat")["exact"].max().sort_values(ascending=False)
    stat_sentence = (
        "By price statistic, the best exact agreement is "
        + ", ".join(f"{s} {pct(float(v))}" for s, v in by_stat.items())
        + "."
    )

    md = [
        "## 8. What the frozen `Preco` says about the original convention",
        "",
        (
            "The manuscript's section 4.3 describes four scenarios but does not say "
            "whether parking spaces count as units. Every candidate convention was "
            f"scored against the frozen column on all {len(geo):,} points. Full grid "
            f"({len(grid)} combinations): `outputs/tables/stage0b_preco_conventions.csv`."
        ),
        "",
        "Top of the grid:",
        "",
        *md_table(grid.head(8)),
        "",
        (
            f"The best convention — the {best.price_stat} of the "
            f"`{best.price_subset}` ITBI prices at the address × the count of "
            f"`{best.unit_subset}` registry units at the address (ITBI "
            f"`{best.itbi_dedupe}`, registry `{best.registry_dedupe}`) — "
            f"reproduces the frozen `Preco` exactly on {pct(float(best['exact']))} of "
            f"the {len(geo):,} points and within {tol:.0%} on "
            f"{pct(float(best[f'within_{tol:.0%}']))}. Counting *all* units with all "
            "prices gives at most "
            f"{pct(float(grid[(grid.unit_subset == 'all') & (grid.price_subset == 'all')].exact.max()))} "
            "exact agreement; excluding parent rows as well gives at most "
            f"{pct(float(grid[grid.unit_subset == 'non_parking_non_parent'].exact.max()))}. "
            + stat_sentence
        ),
        "",
        f"### 8.1 Blank `Purpose` ({blank_n:,} rows): kept vs dropped, side by side",
        "",
        *md_table(blank_tbl),
        "",
        (
            "Keeping blank-`Purpose` rows as non-parking reproduces the frozen column "
            + ("better" if kept_better else "no better")
            + " than dropping them (exact agreement "
            + f"{pct(kept_exact)} vs {pct(dropped_exact)})."
        ),
        "",
    ]
    return md, {"grid": grid, "best": best.to_dict(), "blank_tbl": blank_tbl}


def section_reconcile(
    ctx: common.Context, res: dict[str, Any]
) -> tuple[list[str], list[str]]:
    ref = ctx.cfg["validation"]["parking_reference"]
    tgt = ctx.cfg["exclusions"]["targets"]
    park, plot = res["parking"], res["plot"]

    source_ref = "validation.parking_reference (referee reports)"
    checks = [
        (
            "analysis sample N after exclusions",
            plot["n_final"],
            tgt["disaggregated_n"],
            0.0,
            "exclusions.targets.disaggregated_n",
        ),
        (
            "parking share of ITBI transactions",
            park["share"],
            ref["share"],
            0.01,
            source_ref,
        ),
        (
            "median price, parking",
            park["median_parking"],
            ref["median_price_parking"],
            0.001,
            source_ref,
        ),
        (
            "median price, non-parking",
            park["median_other"],
            ref["median_price_other"],
            0.001,
            source_ref,
        ),
        (
            "only-parking addresses",
            park["only_parking"],
            ref["addresses_only_parking"],
            0.0,
            source_ref,
        ),
        (
            "registry parking share",
            park["registry_share"],
            ref["registry_parking_share"],
            0.01,
            source_ref,
        ),
        (
            "Purpose AND Built Area present",
            park["coverage"],
            ref["purpose_and_built_area_present"],
            0.01,
            source_ref,
        ),
    ]
    rows, escalations = [], []
    for label, obs, target, tol, source in checks:
        v = verdict(float(obs), float(target), tol)
        target_s = f"{target:,}" if abs(target) >= 1 else f"{target:.3f}"
        obs_s = f"{obs:,.6g}" if abs(obs) >= 1 else f"{obs:.4f}"
        rows.append(
            {
                "figure": label,
                "target": target_s,
                "observed": obs_s,
                "verdict": v,
                "source": source,
            }
        )
        if v == "MISMATCH":
            escalations.append(
                f"{label}: target {target_s}, observed {obs_s} ({source})"
            )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(ctx.table("stage0b_reconciliation.csv"), index=False)

    md = [
        "## 9. Reconciliation against known targets",
        "",
        (
            "The analysis-sample size and the parking figures quoted in the "
            "referee reports, checked against this profile "
            "(`outputs/tables/stage0b_reconciliation.csv`)."
        ),
        "",
        *md_table(tbl),
        "",
    ]
    if escalations:
        md += ["**Mismatches:**", "", *[f"- {e}" for e in escalations], ""]
    else:
        md += [f"All {len(tbl)} figures match.", ""]
    return md, escalations


# ---------------------------------------------------------------------------


def md_rec_rows(ctx: common.Context) -> pd.DataFrame:
    return pd.read_csv(ctx.table("stage0b_reconciliation.csv"))


def decimal_separator_check(f: dict[str, pd.DataFrame]) -> int:
    """Count numeric-column values that contain a comma (a comma decimal)."""
    cols = {
        "itbi": ["Price", "Built Area", "Plot Area"],
        "registry": ["Built Area", "Plot Area"],
    }
    return int(
        sum(
            f[k][c].astype(str).str.contains(",").sum()
            for k, cs in cols.items()
            for c in cs
        )
    )


def main() -> int:
    ctx = common.init(0, "profile")
    drift = common.verify_manifest(ctx)

    f = prep.load_frames(ctx)
    n_comma = decimal_separator_check(f)
    md_purpose, _mapping, purpose_res = section_purpose(ctx, f)
    md_parking, parking_res = section_parking(ctx, f)
    md_dup, dup_res = section_duplicates_missing(ctx, f)
    md_units, units_res = section_units(ctx, f)
    md_plot, plot_res = section_plot_area(ctx, f)
    md_join, join_res = section_joins(ctx, f)
    md_preco, preco_res = section_preco(ctx, f)

    res = {
        "n_itbi": len(f["itbi"]),
        "n_reg": len(f["registry"]),
        "n_geo": len(f["geolayer"]),
        "purpose": purpose_res,
        "parking": parking_res,
        "dup": dup_res,
        "units": units_res,
        "plot": plot_res,
        "joins": join_res,
        "preco": preco_res,
    }
    md_rec, escalations = section_reconcile(ctx, res)

    cfg = ctx.cfg
    dv = cfg["dependent_variable"]
    n_geo = len(f["geolayer"])
    n_final = plot_res["n_final"]
    best = preco_res["best"]
    grid = preco_res["grid"]
    zero_units = int(
        f["registry"].addr_key.nunique()
        - len(units_res["per_addr"]["non_parking_non_parent"])
    )
    flags = [
        (
            f"**{dup_res['dup_geo_key']} of the {n_geo:,} frozen points are duplicate "
            f"address records** (identical street, number, unit, price and `Puni`); "
            f"{int(plot_res['final'].addr_key.duplicated().sum())} of them survive into "
            f"the analysis sample of N = {n_final:,}, so that N counts "
            f"{n_final - plot_res['final'].addr_key.nunique()} addresses more than once."
        ),
        (
            f"**The frozen `Preco` is reproduced best by the {best['price_stat']} of "
            f"`{best['price_subset']}` prices × the count of `{best['unit_subset']}` "
            f"units** (section 8): {pct(float(best['exact']))} exact."
        ),
        (
            f"**{plot_res['especial_final']} `IMOVEL ESPECIAL` records and "
            f"{plot_res['mega_final']} plots above "
            f"{cfg['profiling']['plot_area']['mega_plot_flag_m2']:,} m² are in the "
            f"analysis sample of N = {n_final:,}.**"
        ),
        (
            "**Excluding parent rows from the unit count leaves many addresses with "
            f"no units**: it takes {zero_units:,} of "
            f"{f['registry'].addr_key.nunique():,} registry addresses to zero "
            "(section 5), because a detached house is a single parent row."
        ),
        (
            f"**ITBI `Plot Area` is zero on {pct(float((f['itbi'].plot_area == 0).mean()))} "
            "of rows** and conflicts with the geolayer on "
            f"{plot_res['conflict_itbi']} of the addresses where it is non-zero "
            "(section 6)."
        ),
    ]

    head = [
        "# Stage 0b — full data profile",
        "",
        (
            f"Generated by `src/00_profile.py`. Seed {cfg['repro']['seed']}. "
            + (
                "Inputs match `data/interim/input_manifest.json`."
                if not drift
                else "**Input drift against `data/interim/input_manifest.json`: "
                + ", ".join(drift)
                + ".**"
            )
        ),
        "",
        (
            "A descriptive profile of the raw inputs. It decides nothing: the "
            "conventions the pipeline uses are set in `dependent_variable:` in "
            "config.yaml, and section 10 lists each with the evidence from this "
            "profile."
        ),
        "",
        "---",
        "",
        "## 0. Summary",
        "",
        (
            f"All {len(md_rec_rows(ctx))} reference figures in section 9 match."
            if not escalations
            else "**Reference figures that do not match (section 9):**"
        ),
        "",
    ]
    if escalations:
        head += [f"- {e}" for e in escalations] + [""]
    head += [f"{i + 1}. {t}" for i, t in enumerate(flags)] + [""]

    inputs = pd.DataFrame(
        {
            "input": ["ITBI transactions", "unit registry", "frozen point layer"],
            "path": [
                cfg["paths"]["raw"]["itbi_csv"],
                cfg["paths"]["raw"]["registry_csv"],
                cfg["paths"]["frozen"]["cents_disaggregated"],
            ],
            "encoding": [
                cfg["io"]["csv_encoding_detected"]["itbi_csv"],
                cfg["io"]["csv_encoding_detected"]["registry_csv"],
                f"GPKG, EPSG:{cfg['repro']['crs_epsg']}",
            ],
            "rows": [len(f["itbi"]), len(f["registry"]), len(f["geolayer"])],
            "distinct addresses": [
                f["itbi"].addr_key.nunique(),
                f["registry"].addr_key.nunique(),
                f["geolayer"].addr_key.nunique(),
            ],
        }
    )
    md_inputs = [
        "---",
        "",
        "## 1. Inputs as profiled",
        "",
        *md_table(inputs),
        "",
        (
            "Encodings are as detected in stage 0a and fixed in "
            "`io.csv_encoding_detected`: the registry is **CP850**, not Latin-1. "
            + (
                "No value in the numeric columns (`Price`, `Built Area`, "
                "`Plot Area`) contains a comma, so the decimal separator is `.`. "
                if n_comma == 0
                else f"**{n_comma:,} values in the numeric columns contain a comma.** "
            )
            + "Counts in and out of every filter are in "
            "`outputs/tables/stage0b_counts.csv`."
        ),
        "",
    ]

    # -- section 10: the conventions in config.yaml, with this profile's evidence
    blank = preco_res.get("blank_tbl")
    dd = grid.pivot_table(
        index=["registry_dedupe", "price_subset", "price_stat", "unit_subset"],
        columns="itbi_dedupe",
        values="exact",
    )
    dedupe_spread = (
        float((dd["dedupe"] - dd["keep"]).abs().max())
        if {"dedupe", "keep"} <= set(dd.columns)
        else float("nan")
    )
    conv = [
        (
            "`profiling.purpose.active_parking_definition`",
            cfg["profiling"]["purpose"]["active_parking_definition"],
            (
                f"{parking_res['only_parking']} only-parking addresses under the active "
                "definition (section 3)."
            ),
        ),
        (
            "`dependent_variable.price_transactions` / `multiplier_units`",
            f"{dv['price_transactions']} / {dv['multiplier_units']}",
            (
                f"best reconstruction of the frozen `Preco`: `{best['price_subset']}` "
                f"prices, `{best['unit_subset']}` units, {pct(float(best['exact']))} "
                "exact (section 8)."
            ),
        ),
        (
            "`dependent_variable.price_statistic`",
            dv["price_statistic"],
            f"best statistic in the section-8 grid: `{best['price_stat']}`.",
        ),
        (
            "`dependent_variable.blank_purpose`",
            dv["blank_purpose"],
            (
                f"{dup_res['blank_purpose']:,} blank rows; exact agreement kept vs "
                f"dropped: {pct(float(blank.iloc[0, 2]))} vs "
                f"{pct(float(blank.iloc[1, 2]))} (section 8.1)."
                if blank is not None
                else "section 8 disabled."
            ),
        ),
        (
            "`dependent_variable.itbi_duplicates`",
            dv["itbi_duplicates"],
            (
                f"{dup_res['dup_itbi']} byte-identical ITBI rows; deduplicating moves "
                f"exact agreement in the section-8 grid by at most "
                f"{100 * dedupe_spread:.1f} percentage points."
            ),
        ),
        (
            "`dependent_variable.parent_rows_in_multiplier`",
            dv["parent_rows_in_multiplier"],
            (
                f"{units_res['parent_rows']:,} parent rows; excluding them zeroes "
                f"{zero_units:,} registry addresses; best exact agreement without them "
                f"{pct(float(grid[grid.unit_subset == 'non_parking_non_parent'].exact.max()))} "
                f"vs {pct(float(best['exact']))} with them."
            ),
        ),
        (
            "`dependent_variable.plot_area_source`",
            dv["plot_area_source"],
            (
                f"geolayer and registry agree on "
                f"{pct(1 - plot_res['conflict_reg'] / res['n_geo'])} of the {n_geo:,} "
                f"addresses; ITBI is zero on "
                f"{pct(float((f['itbi'].plot_area == 0).mean()))} of rows (section 6)."
            ),
        ),
        (
            "mega-plots and `IMOVEL ESPECIAL`",
            "kept (no exclusion)",
            (
                f"{plot_res['especial_final']} `IMOVEL ESPECIAL` records and "
                f"{plot_res['mega_final']} plots above the flag are in the analysis "
                "sample (section 6.2)."
            ),
        ),
        (
            "duplicate address records in the frozen layer",
            "kept and disclosed",
            (
                f"{dup_res['dup_geo_key']} repeated addresses in the layer; "
                f"{n_final - plot_res['final'].addr_key.nunique()} in the analysis "
                "sample (section 4)."
            ),
        ),
    ]
    md_conventions = [
        "## 10. Conventions and the evidence for each",
        "",
        (
            "The value each convention takes in config.yaml, with the figure from "
            "this profile that bears on it."
        ),
        "",
        *md_table(pd.DataFrame(conv, columns=["convention", "value", "evidence"])),
        "",
    ]

    report = (
        head
        + md_inputs
        + md_purpose
        + md_parking
        + md_dup
        + md_units
        + md_plot
        + md_join
        + md_preco
        + md_rec
        + md_conventions
    )
    out = ctx.report("stage0_profile.md")
    out.write_text("\n".join(report) + "\n")
    ctx.log.info("wrote %s", out.relative_to(common.REPO))

    ctx.write_counts("stage0b_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
