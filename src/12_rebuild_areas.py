#!/usr/bin/env python
"""Stage 0 — rebuild the static weight R from the network, and score it.

    make rebuild-areas
    make rebuild-areas ARGS="--no-reversed"      # skip the order-dependence pass
    make rebuild-areas ARGS="--bbox 470000,6670000,485000,6685000"

The `area` column of `inputs.base_network` is the static weight R: **buildable
area in a 30 m buffer**. The submitted run's own R ships with the network, so the
base reproduction needs no rebuild. A *boundary-extent variant* does: clipping the
network changes which segments compete for space, so R has to be regenerated on
the clipped network or the variant is not the model the paper describes. This
script checks that the regeneration reproduces the shipped column on the full
network.

`rebuild_segment_areas.py` in the reconstruction bundle is that regeneration, and
this script runs it here and scores it: eligibility by `highway` class (12 values,
speed <= 61, bridges and tunnels excluded), non-overlap by one global Voronoi
diagram over points every 5 m along every eligible segment, each segment claiming
its own cells clipped to its own 30 m buffer, the road's own surface subtracted so
what remains is *buildable*, and a 10 m² cutoff below which a zone is deleted.

It reads `inputs.segment_centralities` and not `inputs.base_network`, because
eligibility needs `highway`, `bridge` and `tunnel` — attributes the base network
deliberately does not carry (it is a model input, not an attribute table).

The order-dependence pass is an essential part of the script.
`create_non_overlapping_buffers` keeps the first sample point at each whole-metre
coordinate, so which of two segments wins a contested point depends on the order
the edges were enumerated in. R is therefore not a pure function of the
geometry. The second pass rebuilds on the reversed segment order and reports how
far the two disagree; that is why every boundary-extent variant records its
segment enumeration order in its manifest.

Nothing here touches the base network. The rebuilt column goes into a comparison
table beside the shipped one and stays there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
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


def max_rss_gib() -> float:
    """This process's own high-water RSS.

    The cgroup counter measures the whole container, so it reads anything else
    running alongside as this script's memory. `ru_maxrss` is per process and is
    the relevant figure when other processes run alongside.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def score(rebuilt: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Agreement between a rebuilt R and a reference R, on the zones they share."""
    zr, zp = rebuilt > 0, reference > 0
    both = zr & zp
    rel = np.abs(rebuilt[both] - reference[both]) / reference[both]
    return {
        "n_rows": int(rebuilt.size),
        "n_zones_rebuilt": int(zr.sum()),
        "n_zones_reference": int(zp.sum()),
        "zone_set_agreement": float((zr == zp).mean()),
        "n_zone_only_rebuilt": int((zr & ~zp).sum()),
        "n_zone_only_reference": int((~zr & zp).sum()),
        "n_both_positive": int(both.sum()),
        "within_0.1pct": float((rel < 1e-3).mean()),
        "within_1pct": float((rel < 1e-2).mean()),
        "median_rel": float(np.median(rel)),
        "p90_rel": float(np.percentile(rel, 90)),
        "p99_rel": float(np.percentile(rel, 99)),
        "max_rel": float(rel.max()),
        "n_above_1pct": int((rel >= 1e-2).sum()),
        "share_above_1pct": float((rel >= 1e-2).mean()),
        "total_area_rebuilt": float(rebuilt.sum()),
        "total_area_reference": float(reference.sum()),
        "total_area_rel_diff": float(
            (rebuilt.sum() - reference.sum()) / reference.sum()
        ),
        "area_share_in_segments_above_1pct": float(
            reference[both][rel >= 1e-2].sum() / reference.sum()
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="12_rebuild_areas.py",
        description="Rebuild R from the network and score it against the shipped `area`.",
    )
    ap.add_argument(
        "--bbox",
        default=None,
        metavar="MINX,MINY,MAXX,MAXY",
        help="restrict the run to this window, in the source CRS (a partial run, "
        "not the full network)",
    )
    ap.add_argument(
        "--no-reversed",
        action="store_true",
        help="skip the reversed-order pass that measures the order dependence",
    )
    ap.add_argument(
        "--no-impediment",
        action="store_true",
        help="skip the road-surface subtraction (omitting it inflates R by about "
        "half; diagnostic only)",
    )
    ap.add_argument("--label", default="edges_combined")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "rebuild_areas")
    cfg = ctx.cfg

    bundle = common.REPO / cfg["inputs"]["pc_solver"]["reconstruction_bundle"]
    script = bundle / "rebuild_segment_areas.py"
    if not script.exists():
        ctx.log.error("rebuild script missing from the bundle: %s", script)
        return 2
    rebuild = _load_module("rebuild_segment_areas", script)

    seg_cfg = cfg["inputs"]["segment_centralities"]
    src = common.REPO / seg_cfg["path"]
    if not src.exists():
        ctx.log.error("inputs.segment_centralities.path does not exist: %s", src)
        return 2

    gdf = gpd.read_file(src)
    ctx.count("segments read", len(gdf), seg_cfg["path"])
    needed = ["highway", "speed", "area", "tunnel", "bridge"]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        ctx.log.error(
            "the eligibility rule needs %s; this layer has no %s. The base "
            "GeoPackage deliberately carries none of them — run this against "
            "`inputs.segment_centralities` (edges_combined.shp).",
            needed,
            missing,
        )
        return 1
    if args.bbox:
        minx, miny, maxx, maxy = (float(v) for v in args.bbox.split(","))
        gdf = gdf.cx[minx:maxx, miny:maxy].reset_index(drop=True)
        ctx.count("segments after bbox window", len(gdf), args.bbox)

    reference = gdf["area"].astype(float).to_numpy()
    ctx.count("reference zones (area > 0)", int((reference > 0).sum()))
    ctx.log.info("shipped sum R = %.5f", float(reference.sum()))

    results: dict[str, dict[str, Any]] = {}
    areas: dict[str, np.ndarray] = {}

    # -- pass 1: the shapefile's own row order -------------------------------
    # This is the export's `G.edges` order, which is why the forward rebuild is
    # expected to land very close to the shipped column.
    with common.MemoryWatch() as mem:
        t0 = time.time()
        forward = rebuild.rebuild_areas(gdf, use_impediment=not args.no_impediment)
        forward_seconds = time.time() - t0
        forward_peak = mem.peak_gib
        forward_rss = max_rss_gib()
    areas["forward"] = forward
    results["forward"] = score(forward, reference) | {
        "order": "shapefile row order (= export `G.edges` order)",
        "wall_seconds": round(forward_seconds, 1),
        "process_max_rss_gib": round(forward_rss, 2),
        "cgroup_peak_gib": round(forward_peak, 2),
        "use_impediment": not args.no_impediment,
    }
    r = results["forward"]
    ctx.count("rebuilt zones (forward order)", r["n_zones_rebuilt"])
    ctx.log.info(
        "forward: %d zones in %.1fs, zone-set agreement %.4f%%, within 0.1%% %.4f%%, "
        "median rel %.3g, total area %+.3e relative, own RSS %.2f GiB "
        "(cgroup %.2f GiB, container-wide)",
        r["n_zones_rebuilt"],
        r["wall_seconds"],
        100 * r["zone_set_agreement"],
        100 * r["within_0.1pct"],
        r["median_rel"],
        r["total_area_rel_diff"],
        r["process_max_rss_gib"],
        r["cgroup_peak_gib"],
    )

    # -- pass 2: the reversed order, which measures the order dependence ------
    if not args.no_reversed:
        rev_gdf = gdf.iloc[::-1].reset_index(drop=True)
        with common.MemoryWatch() as mem:
            t0 = time.time()
            rev = rebuild.rebuild_areas(rev_gdf, use_impediment=not args.no_impediment)
            rev_seconds = time.time() - t0
            rev_peak = mem.peak_gib
            rev_rss = max_rss_gib()
        reversed_area = rev[::-1].copy()  # back onto the original row order
        areas["reversed"] = reversed_area
        results["reversed_vs_reference"] = score(reversed_area, reference) | {
            "order": "reversed segment order",
            "wall_seconds": round(rev_seconds, 1),
            "process_max_rss_gib": round(rev_rss, 2),
            "cgroup_peak_gib": round(rev_peak, 2),
            "use_impediment": not args.no_impediment,
        }
        results["reversed_vs_forward"] = score(reversed_area, forward) | {
            "order": "reversed vs forward — the order dependence itself",
            "wall_seconds": round(rev_seconds, 1),
            "process_max_rss_gib": round(rev_rss, 2),
            "cgroup_peak_gib": round(rev_peak, 2),
            "use_impediment": not args.no_impediment,
        }
        o = results["reversed_vs_forward"]
        moved = 1.0 - o["within_0.1pct"]
        ctx.log.info(
            "order dependence: reversing the segment order and changing nothing "
            "else moves %.2f%% of areas by more than 0.1%%, %.2f%% by more than "
            "1%%, max %.1f%%; the zone set shifts by %d segments",
            100 * moved,
            100 * o["share_above_1pct"],
            100 * o["max_rel"],
            o["n_zone_only_rebuilt"] + o["n_zone_only_reference"],
        )
        ctx.count(
            "segments whose R moves > 1 % on reversed order",
            o["n_above_1pct"],
            f"{o['share_above_1pct']:.4%} of the common zones",
        )

    # -- the comparison table. The rebuilt column is written here and not onto
    # the base network, whose `area` stays the shipped one.
    id_cols = list(seg_cfg["columns"]["segment_id"])
    comp = pd.DataFrame({c: gdf[c].to_numpy() for c in id_cols})
    comp["area_reference"] = reference
    comp["area_rebuilt"] = forward
    with np.errstate(divide="ignore", invalid="ignore"):
        comp["rel_diff_rebuilt"] = np.where(
            reference > 0, np.abs(forward - reference) / reference, np.nan
        )
    if "reversed" in areas:
        comp["area_rebuilt_reversed_order"] = areas["reversed"]
        with np.errstate(divide="ignore", invalid="ignore"):
            comp["rel_diff_order"] = np.where(
                forward > 0, np.abs(areas["reversed"] - forward) / forward, np.nan
            )
    comp_path = ctx.interim(f"rebuild_areas_{args.label}_comparison.csv")
    comp.to_csv(comp_path, index=False)
    ctx.log.info("wrote %s (%d rows)", comp_path.relative_to(common.REPO), len(comp))
    np.savez(
        ctx.interim(f"rebuild_areas_{args.label}.npz"), **areas, reference=reference
    )

    summary = pd.DataFrame([{"comparison": k, **v} for k, v in results.items()])
    summary.to_csv(ctx.table(f"stage0_rebuild_areas_{args.label}.csv"), index=False)
    ctx.log.info(
        "wrote %s",
        ctx.table(f"stage0_rebuild_areas_{args.label}.csv").relative_to(common.REPO),
    )

    manifest = {
        "produced_by": "src/12_rebuild_areas.py",
        "what": (
            "the static weight R rebuilt from the network alone, scored against "
            "the shipped `area` column, plus the order-dependence measurement"
        ),
        "source": {
            "path": seg_cfg["path"],
            "sha256": common.checksum(src),
            "n_features": len(gdf),
            "bbox": args.bbox,
        },
        "rebuild_script": {
            "path": str(script.relative_to(common.REPO)),
            "sha256": common.checksum(script),
            "eligibility": {
                "allowed_highway": list(rebuild.ALLOWED_HIGHWAY),
                "max_speed": rebuild.MAX_SPEED,
                "forbidden_attributes": list(rebuild.FORBIDDEN_ATTRS),
            },
            "buffer_m": rebuild.BUFFER,
            "sample_spacing_m": rebuild.MIN_SPACING,
            "min_zone_area_m2": rebuild.MIN_AREA,
            "impediment_default_m": rebuild.IMPEDIMENT_DEFAULT,
            "use_impediment": not args.no_impediment,
        },
        "results": results,
        "order_dependence": (
            "R is not a pure function of the geometry: create_non_overlapping_"
            "buffers keeps the first sample point at each whole-metre coordinate, "
            "so a contested point goes to whichever segment was enumerated first. "
            "The base reproduction uses the shipped `area`; a boundary-extent "
            "variant rebuilds R on the clipped network and records the segment "
            "enumeration order in its run manifest."
        ),
        "outputs": {
            "comparison_csv": str(comp_path.relative_to(common.REPO)),
            "summary_csv": str(
                ctx.table(f"stage0_rebuild_areas_{args.label}.csv").relative_to(
                    common.REPO
                )
            ),
        },
        "python": sys.version.split()[0],
        "geopandas": gpd.__version__,
        "seed": int(cfg["repro"]["seed"]),
    }
    man = ctx.interim(f"rebuild_areas_{args.label}_manifest.json")
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    ctx.log.info("wrote %s", man.relative_to(common.REPO))
    ctx.write_counts(f"stage0_rebuild_areas_{args.label}_counts.csv")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
