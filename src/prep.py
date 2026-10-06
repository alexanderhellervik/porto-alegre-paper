"""Shared data preparation for the raw ITBI / registry / frozen-layer triple.

Shared by the data profile (`src/00_profile.py`, stage 0b) and the
dependent-variable rebuild (`src/10_dependent_variable.py`, stage 1), so that
both use exactly the same loaders, address normalisation, purpose mapping and
parking/parent flags. If the two stages normalised addresses differently, the
rebuild's comparison with the frozen `Preco` would mean nothing.

Nothing here makes an analytical decision: every threshold, pattern and
convention comes from `config.yaml` (`profiling.*`, `dependent_variable.*`).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

import common

# Column sets that define a "byte-identical" row in each raw file. Used for the
# duplicate counts in stage 0b and for `dependent_variable.itbi_duplicates` /
# `registry_duplicates` in stage 1.
ITBI_ROW_KEY = [
    "Price",
    "Purpose",
    "Street",
    "Number",
    "Unity",
    "Neighbourhood",
    "Built Area",
    "Plot Area",
]
REG_ROW_KEY = [
    "Purpose",
    "Street",
    "Number",
    "Unity",
    "Neighbourhood",
    "Built Area",
    "Plot Area",
    "Complete Adress",
]


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def fold(text: str, *, fold_accents: bool, upper: bool) -> str:
    """Accent-fold and upper-case a string (the registry is CP850-encoded)."""
    s = str(text)
    if fold_accents:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
    return s.upper() if upper else s


def normalise(text: str, acfg: dict[str, Any]) -> str:
    """Config-driven token normalisation shared by every join hop."""
    s = fold(text, fold_accents=acfg["fold_accents"], upper=acfg["upper_case"])
    if acfg["strip_punctuation"]:
        keep = "?" if acfg["question_mark"] == "wildcard" else ""
        s = re.sub(rf"[^A-Z0-9{keep} ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def squeeze(text: str) -> str:
    """Separator-free form, for matching vocabularies across files."""
    return re.sub(r"[^A-Z0-9?]+", "", text)


def wildcard_pattern(corrupt: str) -> re.Pattern[str]:
    """'?' stands for exactly one character (a character lost in the source)."""
    return re.compile("^" + re.escape(corrupt).replace(r"\?", ".") + "$")


def resolve_vocabulary(
    corrupt_values: list[str], canonical: list[str]
) -> dict[str, str | None]:
    """Map each (possibly '?'-corrupted) value onto a canonical spelling.

    Matching is done on the separator-free form so that punctuation differences
    between the files -- `APART-HOTEL(FLAT)` vs `APART-HOTELFLAT`,
    `DEP?SITO / ARMAZÉM` vs `DEPOSITO  ARMAZEM` -- do not defeat it.
    Values with no canonical counterpart resolve to None and are flagged.
    """
    canon_sq = {squeeze(c): c for c in canonical}
    out: dict[str, str | None] = {}
    for v in corrupt_values:
        vs = squeeze(v)
        if vs in canon_sq:
            out[v] = canon_sq[vs]
            continue
        if "?" in vs:
            pat = wildcard_pattern(vs)
            hits = [c for sq, c in canon_sq.items() if pat.match(sq)]
            out[v] = hits[0] if len(hits) == 1 else None
        else:
            out[v] = None
    return out


def classify(norm_purpose: str, rules: list[dict[str, str]]) -> dict[str, str]:
    """First matching rule wins (config: profiling.purpose.rules)."""
    for rule in rules:
        if re.search(rule["pattern"], norm_purpose):
            return {
                "category": rule["category"],
                "subcategory": rule["subcategory"],
                "confidence": rule["confidence"],
                "rule": rule["pattern"],
            }
    return {
        "category": "unclassified",
        "subcategory": "unclassified",
        "confidence": "low",
        "rule": "",
    }


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_num(s: pd.Series, missing_codes: list[str]) -> pd.Series:
    return pd.to_numeric(s.replace(list(missing_codes), np.nan), errors="coerce")


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.4g}") -> list[str]:
    def cell(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, float) and np.isfinite(v):
            out = floatfmt.format(v)
            return out.removesuffix(".0")
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        # a literal '|' would break the markdown row
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(False)]
    return [head, rule, *rows]


def verdict(observed: float, target: float, rel_tol: float = 0.01) -> str:
    if target == 0:
        return "MATCH" if abs(observed) <= rel_tol else "MISMATCH"
    return "MATCH" if abs(observed - target) / abs(target) <= rel_tol else "MISMATCH"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_frames(ctx: common.Context) -> dict[str, pd.DataFrame]:
    """ITBI transactions, unit registry and the frozen point layer, keyed.

    `addr_key` = normalised street + '|' + trimmed house number, built the same
    way in all three files (ITBI has no complete-address column).
    """
    import geopandas as gpd

    cfg = ctx.cfg
    sep = cfg["io"]["csv_sep"]
    enc = cfg["io"]["csv_encoding_detected"]
    acfg = cfg["profiling"]["address"]

    itbi = pd.read_csv(
        ctx.path("raw", "itbi_csv"),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        encoding=enc["itbi_csv"],
    )
    reg = pd.read_csv(
        ctx.path("raw", "registry_csv"),
        sep=sep,
        dtype=str,
        keep_default_na=False,
        encoding=enc["registry_csv"],
    )
    geo = pd.DataFrame(
        gpd.read_file(ctx.path("frozen", "cents_disaggregated")).drop(
            columns="geometry"
        )
    )

    for df, street, number in (
        (itbi, "Street", "Number"),
        (reg, "Street", "Number"),
        (geo, "Rua", "Numero"),
    ):
        df["street_norm"] = df[street].map(lambda s: normalise(s, acfg))
        df["number_norm"] = df[number].astype(str).str.strip()
        df["addr_key"] = df.street_norm + "|" + df.number_norm

    ctx.count("itbi: raw transaction rows", len(itbi))
    ctx.count("registry: raw unit rows", len(reg))
    ctx.count("geolayer: frozen address points", len(geo))
    return {"itbi": itbi, "registry": reg, "geolayer": geo}


# ---------------------------------------------------------------------------
# purpose vocabulary -> category mapping
# ---------------------------------------------------------------------------


def purpose_mapping(
    ctx: common.Context, f: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the purpose->category mapping and attach it to all three frames.

    Adds `purpose_norm`, `category` and `subcategory` in place. Returns the
    mapping table (one row per file x purpose value) and two counts worth
    reviewing: values with no ITBI counterpart, and low-confidence rules.
    """
    cfg = ctx.cfg["profiling"]["purpose"]
    acfg = ctx.cfg["profiling"]["address"]
    itbi, reg, geo = f["itbi"], f["registry"], f["geolayer"]

    en = pd.read_excel(
        ctx.path("raw", "itbi_xlsx"), usecols=["Purpose", cfg["en_column"]]
    )
    en["norm"] = en.Purpose.fillna("").map(lambda s: normalise(s, acfg))
    en_map = (
        en.dropna(subset=[cfg["en_column"]])
        .drop_duplicates("norm")
        .set_index("norm")[cfg["en_column"]]
        .to_dict()
    )

    for df, col in ((itbi, "Purpose"), (reg, "Purpose"), (geo, "Finalidade")):
        df["purpose_norm"] = df[col].fillna("").map(lambda s: normalise(s, acfg))

    canonical = sorted(itbi.purpose_norm.unique())
    resolved: dict[str, dict[str, str | None]] = {}
    for name, df in (("registry", reg), ("geolayer", geo)):
        resolved[name] = resolve_vocabulary(sorted(df.purpose_norm.unique()), canonical)
    resolved["itbi"] = {v: v for v in canonical}

    rows = []
    for name, df in (("itbi", itbi), ("registry", reg), ("geolayer", geo)):
        vc = df.purpose_norm.value_counts()
        for value, n in vc.items():
            canon = resolved[name].get(value) or value
            cls = classify(canon, cfg["rules"])
            unresolved = resolved[name].get(value) is None and name != "itbi"
            flags = []
            if unresolved:
                flags.append("no_itbi_counterpart")
            if "?" in value:
                flags.append("corrupted_source_spelling")
            if cls["confidence"] == "low":
                flags.append("low_confidence_rule")
            if cls["category"] == "unclassified":
                flags.append("unclassified")
            rows.append(
                {
                    "file": name,
                    "purpose_value": value if value else "(blank)",
                    "canonical_value": canon if canon else "(blank)",
                    "count": int(n),
                    "purpose_en": en_map.get(canon, ""),
                    "category": cls["category"],
                    "subcategory": cls["subcategory"],
                    "confidence": cls["confidence"],
                    "rule": cls["rule"],
                    "flags": ";".join(flags),
                }
            )
    mapping = pd.DataFrame(rows).sort_values(
        ["file", "count"], ascending=[True, False], ignore_index=True
    )

    for name, df in (("itbi", itbi), ("registry", reg), ("geolayer", geo)):
        m = mapping[mapping.file == name].set_index("purpose_value")
        key = df.purpose_norm.replace("", "(blank)")
        df["category"] = key.map(m.category).fillna("unclassified")
        df["subcategory"] = key.map(m.subcategory)

    stats = {
        "unresolved": int(mapping["flags"].str.contains("no_itbi_counterpart").sum()),
        "low_confidence": int((mapping.confidence == "low").sum()),
    }
    return mapping, stats


# ---------------------------------------------------------------------------
# parking / parent flags and the numeric price
# ---------------------------------------------------------------------------


def annotate_parking(ctx: common.Context, f: dict[str, pd.DataFrame]) -> str:
    """Attach `parking` (all three frames), `parent` and `price` in place.

    Returns the active parking regex. `parent` = registry rows with a blank
    `Unity`, i.e. address-level ("parent") rows; whether they count toward the
    unit multiplier is `dependent_variable.parent_rows_in_multiplier`.
    """
    pcfg = ctx.cfg["profiling"]["purpose"]
    pattern = pcfg["parking_definitions"][pcfg["active_parking_definition"]]
    for key in ("itbi", "registry", "geolayer"):
        f[key]["parking"] = f[key].purpose_norm.str.contains(pattern, regex=True)
    f["registry"]["parent"] = f["registry"]["Unity"].str.strip() == ""
    f["itbi"]["price"] = pd.to_numeric(f["itbi"]["Price"], errors="coerce")
    return pattern


# ---------------------------------------------------------------------------
# the dependent-variable primitives (stage 1)
# ---------------------------------------------------------------------------


def dedupe(df: pd.DataFrame, mode: str, subset: list[str]) -> pd.DataFrame:
    """`dedupe` drops byte-identical rows; anything else keeps them."""
    return df.drop_duplicates(subset=subset) if mode == "dedupe" else df


def price_mask(itbi: pd.DataFrame, subset: str, blank_purpose: str) -> pd.Series:
    """Which ITBI rows enter the price average.

    `subset`: all | non_parking. `blank_purpose`: keep_as_nonparking | drop
    (the frozen column is reproduced when blank purposes are kept).
    """
    if subset == "all":
        m = pd.Series(True, index=itbi.index)
    elif subset == "non_parking":
        m = ~itbi.parking
    else:
        raise ValueError(f"unknown price_transactions subset: {subset}")
    if blank_purpose == "drop":
        m = m & (itbi.Purpose != "")
    elif blank_purpose != "keep_as_nonparking":
        raise ValueError(f"unknown blank_purpose: {blank_purpose}")
    return m


def parent_only_multiplier(
    n_units: pd.Series, n_parent: pd.Series, mode: str
) -> pd.Series:
    """Collapse the multiplier at addresses whose selected rows are ALL parents.

    A registry "parent" row is an address-level row with a blank `Unity`.
    Where an address contributes parent rows and *no numbered unit*,
    `n_parents` counts one unit per parent row; `one` counts the address as a
    single plot, which is the convention the frozen column was built under.

    Addresses with exactly one selected row are unaffected either way, so the
    two modes differ only where `n_parent == n_units > 1`.
    """
    if mode == "n_parents":
        return n_units
    if mode != "one":
        raise ValueError(f"unknown parent_only_address_counts_as: {mode}")
    par = n_parent.reindex(n_units.index).fillna(0)
    return n_units.mask((par == n_units) & (n_units > 1), 1)


def unit_mask(reg: pd.DataFrame, subset: str, parent_rows: str) -> pd.Series:
    """Which registry rows count as units in the multiplier.

    `subset`: all | non_parking | non_parking_non_parent (the third is a lower
    bound, reported for comparison). `parent_rows`: include | exclude.
    """
    if subset == "all":
        m = pd.Series(True, index=reg.index)
    elif subset == "non_parking":
        m = ~reg.parking
    elif subset == "non_parking_non_parent":
        m = ~reg.parking & ~reg.parent
    else:
        raise ValueError(f"unknown multiplier_units subset: {subset}")
    if parent_rows == "exclude":
        m = m & ~reg.parent
    elif parent_rows != "include":
        raise ValueError(f"unknown parent_rows_in_multiplier: {parent_rows}")
    return m
