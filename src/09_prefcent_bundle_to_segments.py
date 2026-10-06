#!/usr/bin/env python
"""Stage 0 — turn the prefcent reconstruction bundle into a scoreable segment layer.

    python src/09_prefcent_bundle_to_segments.py
    make pc-accept FILE=data/interim/prefcent_paper_run.gpkg

The `prefcent` 0.1.0 reconstruction bundle (`inputs.pc_solver.reconstruction_bundle`,
Hellervik 2026) ships `paper_run.npz`: the solver's own reproduction of the two
submitted preferential-centrality runs on the 65,357-segment network. The bundle
reports its own agreement with the submitted columns; this script does not take
that report on trust. It attaches the bundle's arrays to the geometry of the file
of record and writes a GeoPackage with the columns `make pc-accept` expects
(`node1`, `node2`, the `cd*` / `ca*` pairs below, `area`), so that
`src/06_pc_acceptance.py` scores it exactly as it would score any other
candidate.

The arrays are row-aligned to `edges_combined.shp`, not keyed. That alignment is
an assumption, and a wrong one would silently permute 65,357 rows into a
plausible-looking layer, so it is **checked before anything is written**: the
bundle carries the `area` column it was extracted with, and every one of the
65,357 values must be bitwise equal to the reference's own `area`. Anything else
aborts. (`area` is the static weight R, it varies over four orders of magnitude,
and it is not a function of position in the file — a permutation cannot survive
it.)

Output columns follow the batch driver's naming convention,
`cd{run}g{gamma}b{beta}k{100*Dp}`:

    cd1g1b2k0 / ca1g1b2k0    PC1 density / mass   (gamma = 1)
    cd4g0b2k0 / ca4g0b2k0    PC2 density / mass   (gamma = 0, no agglomeration)

Reads everything read-only and estimates nothing; it is a verification step
whose output no other target reads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import common


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# `write_layer()` stamps repro.gpkg_last_change and VACUUMs, so the .gpkg bytes
# are a function of the data alone. Reuse it rather than restating it.
stage2 = _load("stage2", "20_join_aggregate.py")

# npz array -> exported column. The run index (1, 4) is the batch position the
# submitted columns carry, not a parameter.
CASES = (
    ("PC1", "PC1_density", "cd1g1b2k0", "PC1_mass", "ca1g1b2k0"),
    ("PC2", "PC2_density", "cd4g0b2k0", "PC2_mass", "ca4g0b2k0"),
)


def git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(common.REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git dir
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="09_prefcent_bundle_to_segments.py",
        description="Attach a prefcent reconstruction run to the segment geometry.",
    )
    ap.add_argument(
        "--run",
        default="paper_run",
        help="bundle run stem, i.e. <bundle>/<run>.npz and .json "
        "(the bundle ships `paper_run`)",
    )
    ap.add_argument("--bundle", default=None, help="bundle directory (default: config)")
    ap.add_argument(
        "--label", default=None, help="output stem (default: prefcent_<run>)"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "prefcent_bundle")
    cfg = ctx.cfg

    solver = cfg["inputs"]["pc_solver"]
    bundle = (
        Path(args.bundle)
        if args.bundle
        else common.REPO / solver["reconstruction_bundle"]
    )
    npz_path = bundle / f"{args.run}.npz"
    json_path = bundle / f"{args.run}.json"
    for p in (npz_path, json_path):
        if not p.exists():
            ctx.log.error("bundle file missing: %s", p)
            return 2
    label = args.label or f"prefcent_{args.run}"

    # -- the reference: the file of record, whose rows the bundle is aligned to --
    seg = cfg["inputs"]["segment_centralities"]
    ref_path = common.REPO / seg["path"]
    ref = gpd.read_file(ref_path)
    ctx.count("reference segments", len(ref), str(ref_path.name))
    ctx.log.info("reference CRS %s (declared EPSG:%s)", ref.crs, seg["source_crs_epsg"])

    run = json.loads(json_path.read_text())
    npz = np.load(npz_path)
    area_bundle = npz["area"]
    area_ref = ref["area"].to_numpy(dtype=np.float64)

    # -- row alignment: abort rather than repair ------------------------------
    if area_bundle.shape[0] != len(ref):
        ctx.log.error(
            "row count mismatch: bundle %d, reference %d — the bundle is not a "
            "run on this network",
            area_bundle.shape[0],
            len(ref),
        )
        return 1
    same = area_bundle == area_ref
    n_same = int(same.sum())
    ctx.log.info(
        "row alignment: `area` bitwise equal on %d / %d rows", n_same, len(ref)
    )
    if n_same != len(ref):
        bad = np.flatnonzero(~same)[:5]
        for i in bad:
            ctx.log.error(
                "  row %d: bundle %r, reference %r", int(i), area_bundle[i], area_ref[i]
            )
        ctx.log.error(
            "aborting: the bundle's arrays are row-aligned to the reference by "
            "assumption, and %d rows disagree. Writing them onto this geometry "
            "would permute the layer silently.",
            len(ref) - n_same,
        )
        return 1
    ctx.count("rows with bitwise-equal `area`", n_same, "alignment check PASSED")

    # Independent restatement of the bundle's own counts, from the arrays.
    zone_rows = npz["zone_rows"]
    zones_here = np.flatnonzero(area_ref > 0.0)
    if not np.array_equal(zone_rows, zones_here):
        ctx.log.error("zone_rows disagrees with `area > 0` on the reference")
        return 1
    ctx.count("zones (area > 0)", int(zones_here.size))
    ctx.count("zero-capacity segments", int(len(ref) - zones_here.size))
    ctx.log.info("sum R = %.5f", float(area_ref.sum()))

    # -- build the export ------------------------------------------------------
    id_cols = seg["columns"]["segment_id"]
    out = pd.DataFrame({c: ref[c].to_numpy() for c in id_cols})
    for meas, dens_key, dens_col, mass_key, mass_col in CASES:
        out[dens_col] = npz[dens_key]
        out[mass_col] = npz[mass_key]
        zero_rows = npz[dens_key][area_ref == 0.0]
        ctx.log.info(
            "%s -> %s / %s: %d zero-capacity rows carry %s",
            meas,
            dens_col,
            mass_col,
            zero_rows.size,
            "0" if np.all(zero_rows == 0.0) else "non-zero values",
        )
    # R echoed back, so the acceptance test can check the static weights too.
    out["area"] = area_ref

    gdf = gpd.GeoDataFrame(
        out, geometry=ref.geometry.reset_index(drop=True), crs=ref.crs
    )
    gpkg = stage2.write_layer(ctx, gdf, f"{label}.gpkg")

    # -- manifest --------------------------------------------------------------
    manifest = {
        "label": label,
        "produced_by": "src/09_prefcent_bundle_to_segments.py",
        "what": (
            "the prefcent reconstruction of the submitted Porto Alegre PC run, "
            "attached to the geometry of the file of record for independent "
            "scoring with `make pc-accept`"
        ),
        "bundle": {
            "dir": str(bundle.relative_to(common.REPO))
            if bundle.is_relative_to(common.REPO)
            else str(bundle),
            "run": args.run,
            "npz_sha256": common.checksum(npz_path),
            "json_sha256": common.checksum(json_path),
            # the bundle records the absolute path of the file it was run on;
            # keep only its name, and identify it by checksum
            "source": Path(run["source"]["source"]).name,
            "source_sha256": run["source"]["source_sha256"],
            "iterations": run["iterations"],
        },
        "solver": {
            "package": solver["package"],
            "version": solver["version"],
            "installed_version": _installed_version(),
            "upstream_commit": solver["upstream_commit"],
            "upstream_tag": solver["upstream_tag"],
            "wheel": solver["wheel"],
            "wheel_sha256": solver["wheel_sha256"],
        },
        "parameters": solver["submitted_run"],
        "graph": solver["graph"],
        "reference": {
            "path": str(ref_path.relative_to(common.REPO)),
            "layer": seg["layer"],
            "sha256": common.checksum(ref_path),
            "crs": str(ref.crs),
            "n_features": len(ref),
        },
        "row_alignment": {
            "checked_on": "area",
            "rule": "bitwise equality on every row",
            "rows": len(ref),
            "rows_equal": n_same,
            "result": "PASS",
        },
        "counts": {
            "segments": len(ref),
            "zones": int(zones_here.size),
            "zero_capacity_segments": int(len(ref) - zones_here.size),
            "sum_R": float(area_ref.sum()),
        },
        # The solver's own per-run fingerprints, carried through so a column in
        # this file can be traced back to the run that produced it.
        "runs": [
            {
                "label": c["label"],
                "gamma": c["gamma"],
                "beta": c["beta"],
                "d0": c["d0"],
                "status": c["status"],
                "final_step_norm": c["final_step_norm"],
                "manifest_sha256": c["manifest_sha256"],
            }
            for c in run["cases"]
        ],
        "columns_written": [c for _l, _dk, d, _mk, m in CASES for c in (d, m)]
        + ["area"],
        "scored_by": "src/06_pc_acceptance.py (this repository's test, not the "
        "bundle's self-report)",
        "git_hash": git_hash(),
        "seed": int(cfg["repro"]["seed"]),
        "python": sys.version.split()[0],
        "geopandas": gpd.__version__,
        "outputs": {"gpkg": str(gpkg.relative_to(common.REPO))},
    }
    man_path = ctx.interim(f"{label}_manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    ctx.log.info("wrote %s", man_path.relative_to(common.REPO))
    for c in run["cases"]:
        ctx.log.info(
            "  %s: gamma=%s %s, final step norm %.3e, manifest %s",
            c["label"],
            c["gamma"],
            c["status"],
            c["final_step_norm"],
            c["manifest_sha256"][:12],
        )
    ctx.write_counts(f"{label}_counts.csv")
    ctx.log.info("next: make pc-accept FILE=%s", gpkg.relative_to(common.REPO))
    ctx.finish()
    return 0


def _installed_version() -> str:
    try:
        import prefcent

        return str(prefcent.__version__)
    except ImportError:  # pragma: no cover - solver not installed
        return "not installed"


if __name__ == "__main__":
    raise SystemExit(main())
