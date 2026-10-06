#!/usr/bin/env python
"""Stage 0a — inventory data/raw and data/frozen before anything else.

Makes no analytical decision. Lists every declared input, probes the CSV
encodings, counts rows, checks each spatial layer's CRS, and records a
checksummed manifest (`data/interim/input_manifest.json`) so later stages can
detect input drift.

Run:  python src/00_inventory.py                 # inventory + manifest
      python src/00_inventory.py --verify        # check data/SHA256SUMS
      python src/00_inventory.py --write-sums    # regenerate data/SHA256SUMS

`data/SHA256SUMS` is tracked: it holds the checksums of the inputs as
distributed, so `--verify` detects a damaged or altered download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

# C1 control range. Real text never contains these; their presence means the
# candidate encoding decoded the bytes without error but produced garbage.
C1_RANGE = range(0x80, 0xA0)


def probe_encoding(path: Path, candidates: list[str]) -> tuple[str | None, int]:
    """First candidate that decodes the whole file *plausibly*.

    Decoding without raising is not sufficient: latin-1 maps every one of the
    256 byte values, so it "succeeds" on any input, including files it silently
    mangles (the registry is CP850, and read as Latin-1 it loses `Ç` and `É`
    without an error). A candidate is therefore rejected if it yields C1 control
    characters.
    """
    raw = path.read_bytes()
    if not any(b > 0x7F for b in raw):
        return "ascii", raw.count(b"\n")

    for enc in candidates:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if any(ord(ch) in C1_RANGE for ch in text):
            continue  # decoded, but into control characters -> wrong codepage
        return enc, text.count("\n")
    return None, -1


def verify_sums() -> int:
    if not common.SUMS_FILE.exists():
        print(f"{common.SUMS_FILE.relative_to(common.REPO)} not found")
        return 1
    n = sum(1 for ln in common.SUMS_FILE.read_text().splitlines() if ln.strip())
    bad = common.verify_sums()
    for b in bad:
        print(f"FAILED  {b}")
    print(f"{n - len(bad)} of {n} files match data/SHA256SUMS")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="check data/SHA256SUMS")
    mode.add_argument(
        "--write-sums", action="store_true", help="regenerate data/SHA256SUMS"
    )
    args = ap.parse_args()
    if args.verify:
        return verify_sums()
    if args.write_sums:
        n = common.write_sums()
        print(f"wrote data/SHA256SUMS ({n} files)")
        return 0

    ctx = common.init(0, "inventory")
    cfg = ctx.cfg
    inputs = common.input_paths(cfg)

    ctx.log.info("--- file inventory ---")
    rows = []
    for key, p in inputs.items():
        exists = p.exists()
        target = p.resolve() if exists else None
        if target is not None and target.is_dir():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        else:
            size = target.stat().st_size if target is not None else 0
        ctx.log.info(
            "%-32s %-6s %12s  %s",
            key,
            "OK" if exists else "MISSING",
            f"{size:,}" if size else "-",
            target.name if target else "(absent)",
        )
        rows.append(
            {
                "key": key,
                "exists": exists,
                "bytes": size,
                "path": common.repo_relative(p),
            }
        )

    ctx.log.info("--- csv probe ---")
    sep = cfg["io"]["csv_sep"]
    cand = cfg["io"]["csv_encoding_candidates"]
    csv_info = []
    for key in ("raw.itbi_csv", "raw.registry_csv"):
        p = inputs[key]
        if not p.exists():
            continue
        enc, nlines = probe_encoding(p, cand)
        if enc is None:
            raise ValueError(f"{key}: no candidate encoding decodes the file")
        with p.open(encoding=enc) as fh:
            header = fh.readline().rstrip("\n")
        cols = header.split(sep)
        ctx.log.info(
            "%-20s encoding=%-8s lines=%-8d cols=%d",
            key,
            enc,
            nlines,
            len(cols),
        )
        ctx.log.info("%-20s columns: %s", "", cols)
        csv_info.append(
            {
                "key": key,
                "encoding": enc,
                "lines": nlines,
                "data_rows": nlines - 1,
                "n_cols": len(cols),
                "columns": "|".join(cols),
            }
        )
        ctx.count(f"{key}: data rows", nlines - 1)

    ctx.log.info("--- spatial layers ---")
    import geopandas as gpd
    import pyogrio

    # Each layer is checked against the CRS it is declared to ship in.
    declared_crs = {
        "frozen.cents_disaggregated": cfg["repro"]["crs_epsg"],
        "frozen.cents_aggregated": cfg["grid"]["crs_epsg"],
        "frozen.segment_edges_dir": cfg["inputs"]["segment_centralities"][
            "source_crs_epsg"
        ],
    }
    layer_info = []
    for key, epsg in declared_crs.items():
        p = inputs[key]
        if not p.exists():
            continue
        # a shapefile bundle is a directory; open the .shp inside it
        if p.resolve().is_dir():
            shp = sorted(p.resolve().glob("*.shp"))
            if not shp:
                continue
            p = shp[0]
        layers = [lyr[0] for lyr in pyogrio.list_layers(p)]
        g = gpd.read_file(p)
        ctx.log.info(
            "%-28s layer=%-24s rows=%-6d crs=%s geom=%s",
            key,
            layers[0],
            len(g),
            g.crs,
            g.geom_type.iloc[0],
        )
        if g.crs is None or g.crs.to_epsg() != int(epsg):
            raise ValueError(f"{key}: CRS {g.crs} is not the declared EPSG:{epsg}")
        layer_info.append(
            {
                "key": key,
                "layer": layers[0],
                "rows": len(g),
                "crs": str(g.crs),
                "geom": g.geom_type.iloc[0],
                "columns": "|".join(g.columns),
            }
        )
        ctx.count(f"{key}: rows", len(g))

    common.write_manifest(ctx, inputs, common.INPUT_MANIFEST)

    import csv as _csv

    with ctx.table("stage0_inventory.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["key", "exists", "bytes", "path"])
        w.writeheader()
        w.writerows(rows)
    ctx.write_counts("stage0_counts.csv")

    lines = [
        "# Stage 0a — input inventory",
        "",
        f"Generated by `src/00_inventory.py`. Seed {cfg['repro']['seed']}.",
        "",
        "## Files on disk",
        "",
        "| key | status | bytes |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['key']}` | {'ok' if r['exists'] else 'MISSING'} | {r['bytes']:,} |"
        )
    lines += [
        "",
        "## CSV probe",
        "",
        "| file | encoding | data rows | cols |",
        "|---|---|---|---|",
    ]
    for c in csv_info:
        lines.append(
            f"| `{c['key']}` | {c['encoding']} | {c['data_rows']:,} | {c['n_cols']} |"
        )
    lines += [
        "",
        "## Spatial layers",
        "",
        "| file | layer | rows | crs | geometry |",
        "|---|---|---|---|---|",
    ]
    for lyr in layer_info:
        lines.append(
            f"| `{lyr['key']}` | {lyr['layer']} | {lyr['rows']:,} "
            f"| {lyr['crs']} | {lyr['geom']} |"
        )
    ctx.report("stage0a_inventory.md").write_text("\n".join(lines) + "\n")
    ctx.log.info("wrote reports/stage0a_inventory.md")

    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
