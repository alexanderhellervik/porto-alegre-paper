"""Table 3 "Hierarchy" column, recomputed under the definition §4.4 states.

The submitted column (`validation.hierarchy_submitted`) is not produced by the
formula the manuscript states, nor by the log, share, Gini or saturating
variants tried. This script computes the Pareto ratio exactly as Faria et al.
(2025) define it -- the sum of the top 20 % of values divided by the sum of the
bottom 80 % -- on the analysis sample (positivity, then the IQR fence on the
frozen `Puni`), for the five measures and for both dependent-variable columns
(frozen, and the rebuilt headline on the same records).

The ratio is unbounded (Faria et al.'s own values run from 0.47 to 4.2), so it
is not a 0-1 index.

Output: outputs/tables/table3_hierarchy_faria.csv

Run:  python src/51_table3_hierarchy.py     (or `make table3-hierarchy`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

MEASURES = ["PC1", "PC2", "CC", "BC", "FK"]


def split_top_bottom(x: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = int(np.ceil(share * len(x)))
    return x[:k], x[k:]


def pareto_ratio(x: np.ndarray, share: float) -> float:
    """Faria et al. (2025): sum(top share) / sum(the rest)."""
    top, bottom = split_top_bottom(x, share)
    return float(top.sum() / bottom.sum())


def top_share(x: np.ndarray, share: float) -> float:
    top, bottom = split_top_bottom(x, share)
    return float(top.sum() / (top.sum() + bottom.sum()))


def main() -> int:
    ctx = common.init(4, "table3_hierarchy")
    cfg = ctx.cfg
    share = float(cfg["validation"]["hierarchy_top_share"])
    submitted = cfg["validation"]["hierarchy_submitted"]["disaggregated"]
    target_n = int(cfg["exclusions"]["targets"]["disaggregated_n"])

    g = gpd.read_file(ctx.path("frozen", "cents_disaggregated"))
    d = common.apply_exclusions(ctx, g, "disaggregated")
    if len(d) != target_n:
        raise SystemExit(f"analysis sample is {len(d)}, expected {target_n}")

    rebuilt_path = common.REPO / cfg["dependent_variable"]["rebuilt_path"]
    rebuilt = pd.read_csv(rebuilt_path, usecols=["id", "Puni"]).rename(
        columns={"Puni": "Puni_rebuilt"}
    )
    d = d.merge(rebuilt, on="id", how="left", validate="one_to_one")
    if d["Puni_rebuilt"].isna().any() or (d["Puni_rebuilt"] <= 0).any():
        raise SystemExit("rebuilt Puni missing or non-positive on the analysis sample")

    series = {m: d[m].to_numpy() for m in MEASURES}
    series["land_value"] = d["Puni"].to_numpy()
    series["land_value_rebuilt"] = d["Puni_rebuilt"].to_numpy()

    rows = []
    for name, x in series.items():
        top, _ = split_top_bottom(x, share)
        rows.append(
            {
                "variable": name,
                "n": len(x),
                "n_top20": len(top),
                "pareto_ratio_faria": round(pareto_ratio(x, share), 4),
                "top20_share": round(top_share(x, share), 4),
                "submitted": submitted.get(name.replace("_rebuilt", ""), np.nan),
                "definition": (
                    f"sum(top {100 * share:.0f} %) / sum(bottom "
                    f"{100 * (1 - share):.0f} %), raw values, ceil({share} n) top records"
                ),
            }
        )
    out = pd.DataFrame(rows)
    path = ctx.table("table3_hierarchy_faria.csv")
    out.to_csv(path, index=False)
    ctx.log.info("wrote %s", path.relative_to(common.REPO))
    ctx.log.info("\n%s", out.to_string(index=False))
    ctx.write_counts("table3_hierarchy_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
