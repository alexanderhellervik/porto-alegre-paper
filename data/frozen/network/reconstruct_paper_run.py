"""Rerun the submitted Porto Alegre preferential-centrality run and score it.

Builds the routing graph from the segment file of record with
`extract_network_from_shapefile.extract()` (see that module for the graph
rule), runs prefcent with the submitted parameters for gamma = 1 (PC1) and
gamma = 0 (PC2), and compares density and mass with the submitted columns.

With the default input (`data/frozen/segments/edges_combined.shp`) the graph is
built from the very rows being compared, so no key matching is involved and
coverage is 100 % by construction. With `--shapefile` pointing at the base
network GeoPackage, which has no result columns, pass the shapefile as
`--reference`; the two are then joined on (node1, node2).

    python reconstruct_paper_run.py [--iterations 200] [--out <stem>]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import prefcent as pc

from extract_network_from_shapefile import REPO, extract, repo_relative

# The segment file of record.
PAPER = os.path.join(REPO, "data", "frozen", "segments", "edges_combined.shp")
CASES = [("PC1", 1.0, "cd1g1b2k0", "ca1g1b2k0"), ("PC2", 0.0, "cd4g0b2k0", "ca4g0b2k0")]


def tiers(got: np.ndarray, ref: np.ndarray) -> dict:
    """Tier-1 comparison: shares within a band, on rows not zero on both sides."""
    both_zero = (got == 0) & (ref == 0)
    live = ~both_zero
    rel = np.full(got.shape, np.nan)
    nz = live & (ref != 0)
    rel[nz] = np.abs(got[nz] - ref[nz]) / np.abs(ref[nz])
    r = rel[nz]
    return {
        "rows": int(got.size),
        "both_zero": int(both_zero.sum()),
        "compared": int(nz.sum()),
        "within_0.1pct": float(np.mean(r < 1e-3)),
        "within_1pct": float(np.mean(r < 1e-2)),
        "median_rel": float(np.median(r)),
        "p90_rel": float(np.percentile(r, 90)),
        "p99_rel": float(np.percentile(r, 99)),
        "max_rel": float(r.max()),
        "pearson": float(np.corrcoef(got[nz], ref[nz])[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--shapefile", default=PAPER,
                    help="network input; may be the base GeoPackage (no result columns)")
    ap.add_argument("--reference", default=None,
                    help="file of record to score against, when the input carries no "
                         "result columns; joined on (node1, node2)")
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "data", "interim", "prefcent_reconstruction",
                             "paper_run"),
        help="output stem; <stem>.npz and <stem>.json are written",
    )
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ref_cols = tuple(c for _l, _g, c, m in CASES for c in (c, m))
    t0 = time.time()
    if args.reference is None:
        net, meta = extract(args.shapefile, reference_cols=ref_cols)
    else:
        # The base network carries no results by design: score against the file of
        # record, aligned on the key of record rather than on row order.
        import pyogrio
        net, meta = extract(args.shapefile)
        ref = pyogrio.read_dataframe(args.reference, read_geometry=False,
                                     columns=["node1", "node2", *ref_cols])
        pos = {(str(a), str(b)): i for i, (a, b) in
               enumerate(zip(ref["node1"], ref["node2"], strict=True))}
        order = np.array([pos[(str(a), str(b))] for a, b in
                          zip(net["segment_node1"], net["segment_node2"], strict=True)])
        if len(set(order.tolist())) != len(order):
            raise ValueError("reference join is not one-to-one on (node1, node2)")
        for c in ref_cols:
            net[f"ref_{c}"] = ref[c].astype(np.float64).to_numpy()[order]
        meta["reference"] = repo_relative(args.reference)
        print(f"joined {len(order)} rows to the reference on (node1, node2)", flush=True)
    print(f"extracted in {time.time()-t0:.1f}s: {meta['n_segments']} segments, "
          f"{meta['graph_nodes']} graph nodes, {meta['n_zones']} zones", flush=True)

    t1 = time.time()
    kernel = pc.kernels.from_graph(
        net["graph_edges"], net["graph_costs"], meta["graph_nodes"],
        sources=net["zone_graph_index"], directed=False, n_jobs=args.n_jobs,
        beta=2.0, d0=5000.0, cost_units=meta["cost_units"], self_interaction=False)
    print(f"kernel built in {time.time()-t1:.1f}s: {kernel.shape} {kernel.dtype}, "
          f"is_symmetric={kernel.is_symmetric}", flush=True)

    area = np.zeros(meta["n_segments"], dtype=np.float64)
    area[net["zone_rows"]] = net["R"]
    landscape = pc.Landscape(net["R"], labels=net["zone_ids"])
    report = {"source": meta, "iterations": args.iterations, "cases": []}
    saved: dict[str, np.ndarray] = {"area": area, "zone_rows": net["zone_rows"]}

    for label, gamma, dens_col, mass_col in CASES:
        t2 = time.time()
        model = pc.Model(kernel, landscape, gamma=gamma)
        res = model.evolve(max_iter=args.iterations)
        wall = time.time() - t2
        mass = np.zeros(meta["n_segments"], dtype=np.float64)
        mass[net["zone_rows"]] = res.mass
        dens = np.zeros_like(mass)
        nz = area > 0
        dens[nz] = mass[nz] / area[nz]
        saved[f"{label}_mass"] = mass
        saved[f"{label}_density"] = dens
        case = {
            "label": label, "gamma": gamma, "beta": 2.0, "d0": 5000.0,
            "status": res.status.name, "final_step_norm": float(res.final_step_norm),
            "wall_seconds": round(wall, 1), "sum_mass": float(mass.sum()),
            "sum_R": float(area.sum()),
            "density_vs_" + dens_col: tiers(dens, net[f"ref_{dens_col}"]),
            "mass_vs_" + mass_col: tiers(mass, net[f"ref_{mass_col}"]),
            "manifest_sha256": res.manifest.fingerprint_sha256,
        }
        report["cases"].append(case)
        d = case["density_vs_" + dens_col]
        print(f"\n{label} (gamma={gamma}) {res.status.name} in {wall:.1f}s, "
              f"final step norm {res.final_step_norm:.3e}", flush=True)
        print(f"   vs {dens_col}: within 0.1% = {d['within_0.1pct']:.2%}, "
              f"within 1% = {d['within_1pct']:.2%}", flush=True)
        print(f"   median rel {d['median_rel']:.4g}  p90 {d['p90_rel']:.4g}  "
              f"max {d['max_rel']:.4g}  pearson {d['pearson']:.6f}", flush=True)

    np.savez(args.out + ".npz", **saved)
    with open(args.out + ".json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}.npz / .json  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
