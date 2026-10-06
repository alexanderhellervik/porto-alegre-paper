#!/usr/bin/env python
"""Stage 0 driver: run the GAUS "Lines" v1.1 plugin headless under QGIS.

    make gaus-native                                     # the reference extent
    make gaus-native FILE=<layer> LAYER=<name> LABEL=<label>
    python src/13_gaus_headless.py [--input <layer>] [--clip-bbox ...] [...]

This runs the plugin itself -- GAUS Lines v1.1, the tool the paper cites
(Dalcin and Krafta 2021) -- under QGIS 3.34, and is the source of CC / BC / FK
at every extent in this package.

The plugin is not vendored: its upstream repository carries no licence file.
`make gaus-fetch` clones it into `tools/GAUS/`, which is where
`inputs.gaus_headless.script_source` points; `make gaus-native` refuses if it
is absent.

Two interpreters, kept apart on purpose
---------------------------------------
`qgis.core` lives in the system Python that QGIS is built against. The analysis
virtual environment has geopandas/shapely and no qgis, and is kept that way. So
this driver runs in the analysis environment and does every read, write, join
and check there, and shells out to `qgis_process`
(`inputs.gaus_headless.qgis_process`) for the GAUS step alone, with the virtual
environment stripped from the child's `PATH` (`inputs.gaus_headless.path_for_qgis`).
That last part matters: `qgis_process` embeds a Python whose `sys.prefix` follows
`PATH`, and with a virtual environment first it cannot import `qgis`, disables
Python support, and then reports "unknown algorithm script:GAUS_l11".

The plugin mutates its input layer
----------------------------------
The plugin (lines 243-323 of `GAUS Lines v1.1.py`) appends its result columns to
the layer it was given, via `dataProvider().addAttributes()` — an immediate write
to the `.dbf`, no edit buffer, no undo. It is therefore never pointed at
`data/frozen/` or at the fetched clone: the layer is copied into
`inputs.gaus_headless.work_dir` first, as an ESRI Shapefile in its own CRS (no
reprojection: GAUS measures planar lengths in the layer's projected CRS,
EPSG:32722 for the network of record), carrying only the id columns and a
`gaus_rid` row-order check. Dropping
every pre-existing `Gg*` column is what makes the plugin's name counter start at
zero, so the results come back as `GgAcc0/GgBtw0/GgCen0/GgCnc0` and not
`...Acc1`. Nothing else is read from the attribute table: no impedance, load,
supply or demand field was selected in the submitted run.

Writes
------
  * `data/interim/gaus_headless_<label>.gpkg` — the four columns on the
    geometry, keyed by `(node1, node2)`; what `make gaus-accept FILE=...` eats;
  * `outputs/tables/gaus_headless_<label>.csv` — the same table, no geometry;
  * `data/interim/gaus_headless_<label>_manifest.json` — QGIS version, script
    SHA-256, every parameter, the input SHA-256, wall time, the exact argv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


stage2 = _load("stage2", "20_join_aggregate.py")


def git_hash() -> str:
    """Commit of the package this run was made from, for the manifest."""
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


# Any column the plugin itself could have written on an earlier run:
# [T|G][g|radius][Acc|Btw|Cen|Opp|Cvg|Pol|Rea|Cnc][0-9]  (GAUS Lines v1.1,
# lines 242-294).
_METRIC_TAGS = ("Acc", "Btw", "Cen", "Opp", "Cvg", "Pol", "Rea", "Cnc")


def is_gaus_column(name: str) -> bool:
    """True for a name the plugin's own recipe could have produced."""
    if len(name) < 6 or name[0] not in "TG" or not name[-1].isdigit():
        return False
    if name[-4:-1] not in _METRIC_TAGS:
        return False
    mid = name[1:-4]
    return mid == "g" or (mid.isdigit() and 1 <= len(mid) <= 5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="13_gaus_headless.py",
        description="Run GAUS Lines v1.1 headless under QGIS and collect its columns.",
    )
    ap.add_argument("--input", default=None, help="line layer (default: config)")
    ap.add_argument("--layer", default=None, help="layer name (GeoPackage input)")
    ap.add_argument("--label", default=None, help="output stem")
    ap.add_argument(
        "--clip-bbox",
        default=None,
        metavar="MINX,MINY,MAXX,MAXY",
        help="clip the input to this bbox, in the input layer's own CRS",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds before the qgis_process child is killed (default: none)",
    )
    ap.add_argument(
        "--rm-work",
        action="store_true",
        help="delete the scratch shapefile GAUS mutated once its columns have "
        "been read back (default: keep it as evidence of the run)",
    )
    return ap.parse_args(argv)


def qgis_env(hcfg: dict) -> dict[str, str]:
    """Child environment for qgis_process: no venv on PATH, no X server."""
    env = dict(os.environ)
    env["PATH"] = hcfg["path_for_qgis"]
    env["QT_QPA_PLATFORM"] = hcfg["qt_qpa_platform"]
    # sys.prefix of the embedded interpreter follows PATH; VIRTUAL_ENV and
    # PYTHONHOME would drag it back into the virtual environment.
    for k in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    # Qt wants a runtime dir; use the caller's if set, else one under the
    # platform's temporary directory.
    env.setdefault("XDG_RUNTIME_DIR", str(Path(tempfile.gettempdir()) / "runtime-gaus"))
    Path(env["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def qgis_version(hcfg: dict) -> str:
    try:
        r = subprocess.run(
            [hcfg["qgis_process"], "--version"],
            capture_output=True,
            text=True,
            env=qgis_env(hcfg),
            check=False,
        )
        for line in r.stdout.splitlines():
            if line.startswith("QGIS "):
                return line.strip()
    except OSError:  # pragma: no cover - qgis not installed
        pass
    return "unknown"


def install_script(ctx: common.Context, hcfg: dict) -> tuple[Path, str]:
    """Copy the plugin into the QGIS profile's scripts dir; return path + sha."""
    src = common.REPO / hcfg["script_source"]
    if not src.exists():
        raise SystemExit(f"GAUS script not found: {src}")
    dest_dir = Path(os.path.expanduser(hcfg["profile_scripts_dir"]))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / hcfg["script_installed_name"]
    shutil.copyfile(src, dest)
    sha = common.checksum(src)
    ctx.log.info("registered %s as %s (sha256 %s)", hcfg["script_source"], dest, sha)
    return dest, sha


def build_argv(hcfg: dict, shp: Path) -> list[str]:
    """`qgis_process run script:GAUS_l11 -- name=value ...`.

    A multi-value enum is one comma-joined argument; repeating `metrics=` is
    rejected outright ("Incorrect parameter value for metrics"). Optional
    parameters left null are simply omitted — passing an empty `dest` would make
    the plugin write a second shapefile and then delete the result columns from
    the layer we are about to read (GAUS Lines v1.1, lines 315-323).
    """
    p = hcfg["parameters"]
    argv = [hcfg["qgis_process"], "run", hcfg["algorithm_id"], "--"]
    argv.append(f"inpLines={shp}")
    argv.append(f"analysis={int(p['analysis'])}")
    argv.append("metrics=" + ",".join(str(int(m)) for m in p["metrics"]))
    argv.append(f"geomrule={int(p['geomrule'])}")
    argv.append(f"radius={float(p['radius'])}")
    for name in ("impedance", "load", "supply", "demand", "dest"):
        if p[name]:
            argv.append(f"{name}={p[name]}")
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = common.init(0, "gaus_headless")
    hcfg = ctx.cfg["inputs"]["gaus_headless"]

    src = Path(args.input or hcfg["input"])
    if not src.is_absolute():
        src = common.REPO / src
    if not src.exists():
        raise SystemExit(f"input layer not found: {src}")
    label = args.label or src.stem

    # -- read, and check the CRS ---------------------------------------------
    # The configured layer name belongs to the configured input; it is not
    # applied to an --input the caller named (a shapefile has one layer, and
    # asking for `edges_combined` inside `edges.shp` is a hard error).
    layer = args.layer
    if layer is None and args.input is None:
        layer = hcfg["layer"]
    gdf = gpd.read_file(src, layer=layer)
    if gdf.crs is None:
        raise SystemExit(
            f"{src.name} declares no CRS. GAUS measures planar length in the "
            "layer's own coordinates, so an unknown projection is an unknown "
            "cost matrix. Set the layer's CRS first."
        )
    epsg = gdf.crs.to_epsg()
    if not gdf.crs.is_projected:
        raise SystemExit(
            f"{src.name} is in EPSG:{epsg}, a geographic CRS. GAUS's default "
            "`QgsDistanceArea` is planar, so lengths would be in degrees. "
            "Project to a metric CRS first (EPSG:31982 or EPSG:32722 here)."
        )
    ctx.log.info(
        "input CRS EPSG:%s (%s) — copied and measured as-is, not reprojected",
        epsg,
        gdf.crs.name,
    )
    ctx.count("input line features", len(gdf), src.name)

    if args.clip_bbox:
        minx, miny, maxx, maxy = (float(v) for v in args.clip_bbox.split(","))
        gdf = gdf.cx[minx:maxx, miny:maxy].reset_index(drop=True)
        ctx.count("after --clip-bbox", len(gdf), args.clip_bbox)
    gdf = gdf.reset_index(drop=True)
    if not len(gdf):
        raise SystemExit("no features left to run on")
    bad = gdf.geom_type[gdf.geom_type != "LineString"]
    if len(bad):
        raise SystemExit(
            f"{len(bad)} feature(s) are not LineStrings ({sorted(set(bad))}). "
            "GAUS reads one line per feature; explode multiparts first."
        )

    # -- the scratch copy GAUS is allowed to mutate ---------------------------
    id_cols = [c for c in hcfg["id_columns"] if c in gdf.columns]
    dropped = sorted(c for c in gdf.columns if is_gaus_column(c))
    work_dir = common.REPO / hcfg["work_dir"] / label
    if not work_dir.resolve().is_relative_to(
        (common.REPO / "data" / "interim").resolve()
    ):
        raise SystemExit(f"refusing to work outside data/interim: {work_dir}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    shp = work_dir / f"{label[:40]}.shp"

    work = gdf[id_cols].copy()
    work.insert(0, "gaus_rid", np.arange(len(gdf), dtype=np.int64))
    work = gpd.GeoDataFrame(work, geometry=gdf.geometry, crs=gdf.crs)
    work.to_file(shp, driver="ESRI Shapefile")
    ctx.log.info(
        "scratch copy %s: %d features, columns %s (dropped %d pre-existing "
        "GAUS column(s): %s)",
        shp.relative_to(common.REPO),
        len(work),
        ["gaus_rid", *id_cols],
        len(dropped),
        ", ".join(dropped) or "none",
    )

    # -- run the plugin -------------------------------------------------------
    _, script_sha = install_script(ctx, hcfg)
    qgis = qgis_version(hcfg)
    cmd = build_argv(hcfg, shp)
    timeout = args.timeout if args.timeout is not None else hcfg["timeout_seconds"]
    ctx.log.info("QGIS: %s", qgis)
    ctx.log.info("exec: %s", " ".join(cmd))
    runlog = common.REPO / ctx.cfg["paths"]["logs"] / f"gaus_headless_{label}.qgis.log"
    runlog.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with runlog.open("w") as fh:
        proc = subprocess.run(
            cmd,
            env=qgis_env(hcfg),
            stdout=fh,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    wall = time.perf_counter() - t0
    ctx.log.info(
        "qgis_process exit %d after %.1f s — log %s",
        proc.returncode,
        wall,
        runlog.relative_to(common.REPO),
    )
    tail = runlog.read_text(errors="replace").splitlines()[-12:]
    if proc.returncode != 0:
        for line in tail:
            ctx.log.error("qgis| %s", line)
        raise SystemExit(f"qgis_process failed (exit {proc.returncode}); see {runlog}")

    # -- read the result columns back, in the venv ---------------------------
    out = gpd.read_file(shp)
    want = list(hcfg["expected_columns"])
    missing = [c for c in want if c not in out.columns]
    if missing:
        raise SystemExit(
            f"GAUS wrote no {missing} column. Present: {sorted(out.columns)}. "
            "A suffix other than 0 means the scratch copy still carried an "
            "earlier run's columns."
        )
    if len(out) != len(gdf):
        raise SystemExit(f"feature count changed: {len(gdf)} in, {len(out)} out")
    order = out["gaus_rid"].to_numpy()
    if not np.array_equal(order, np.arange(len(out))):
        # GAUS keys everything on feat.id(); a reordered read would silently
        # transpose the results onto the wrong geometries.
        ctx.log.warning("shapefile came back reordered — realigning on gaus_rid")
        out = out.sort_values("gaus_rid").reset_index(drop=True)
    res = out[want].astype(float)
    ctx.count("features with GAUS columns", int(res.notna().all(axis=1).sum()), "")
    ctx.log.info(
        "connectivity: mean %.4f, max %d, %d connection(s)",
        res[want[3]].mean(),
        int(res[want[3]].max()),
        int(res[want[3]].sum() // 2),
    )

    tab = res.copy()
    if id_cols:
        for c in reversed(id_cols):
            tab.insert(0, c, gdf[c].to_numpy())
        key_note = "+".join(id_cols)
    else:
        tab.insert(0, "row_index", np.arange(len(gdf)))
        key_note = "row_index"
    ctx.log.info("output keyed by %s", key_note)
    csv_path = ctx.table(f"gaus_headless_{label}.csv")
    tab.to_csv(csv_path, index=False)

    out_gdf = gpd.GeoDataFrame(
        pd.concat(
            [gdf[id_cols].reset_index(drop=True), res.reset_index(drop=True)], axis=1
        ),
        geometry=gdf.geometry.reset_index(drop=True),
        crs=gdf.crs,
    )
    gpkg = stage2.write_layer(ctx, out_gdf, f"gaus_headless_{label}.gpkg")

    manifest = {
        "label": label,
        "runner": "qgis_process",
        "qgis_version": qgis,
        "algorithm_id": hcfg["algorithm_id"],
        "script_source": hcfg["script_source"],
        "script_sha256": script_sha,
        "argv": cmd,
        "parameters": hcfg["parameters"],
        "input": str(
            src.relative_to(common.REPO) if src.is_relative_to(common.REPO) else src
        ),
        "input_layer": layer,
        "input_sha256": common.checksum(src),
        "input_crs": f"EPSG:{epsg}",
        "input_crs_reprojected": False,
        "n_features_read": len(gdf),
        "clip_bbox": args.clip_bbox,
        "work_shapefile": str(shp.relative_to(common.REPO)),
        "work_columns": ["gaus_rid", *id_cols],
        "dropped_existing_gaus_columns": dropped,
        "columns_written": want,
        "key": key_note,
        "wall_seconds": round(wall, 3),
        "qgis_log": str(runlog.relative_to(common.REPO)),
        "git_hash": git_hash(),
        "python": sys.version.split()[0],
        "geopandas": gpd.__version__,
        "outputs": {
            "csv": str(csv_path.relative_to(common.REPO)),
            "gpkg": str(gpkg.relative_to(common.REPO)),
        },
    }
    mpath = ctx.interim(f"gaus_headless_{label}_manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ctx.log.info("wrote %s", mpath.relative_to(common.REPO))

    if args.rm_work:
        shutil.rmtree(work_dir)
        ctx.log.info("removed scratch %s", work_dir.relative_to(common.REPO))

    ctx.write_counts(f"gaus_headless_{label}_counts.csv")
    print(
        f"\nGAUS Lines v1.1 [{label}]: {len(gdf):,} lines, "
        f"{int(res[want[3]].sum() // 2):,} connections, {wall:.1f} s ({qgis})"
    )
    print(f"  {csv_path.relative_to(common.REPO)}")
    print(f"  {gpkg.relative_to(common.REPO)}   <- make gaus-accept FILE=this")
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
