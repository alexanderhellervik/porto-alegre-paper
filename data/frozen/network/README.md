# Porto Alegre base network — input for the preferential-centrality runs

`poa_base_network.gpkg` (layer `segments`, EPSG:32722) holds the network and
capacities of the **submitted** Porto Alegre preferential-centrality run, and
nothing else. It has no `ca*`, `cd*` or `Gg*` columns: it is an input, and it
cannot be scored against its own outputs by accident.

| column | role |
|---|---|
| `node1`, `node2` | segment key; unique over all 65,357 rows |
| `dist` | segment length, metres |
| `speed` | km/h |
| `area` | static weight R (buildable area in a 30 m buffer); `area == 0` marks a segment that carries routes but is not a zone |
| geometry | one LineString per segment |

65,357 segments · 56,162 zones · 9,195 zero-capacity segments · Σ R = 243,801,730.

## Where it comes from

The street network (segment geometry and the `highway`, `road_type`, `tunnel`,
`bridge`, `drive` and speed attributes) derives from OpenStreetMap data
(© OpenStreetMap contributors) and is made available under the Open Database
License (ODbL) 1.0; see `data/README.md`.

The model object of the submitted run is not part of the package. Its output
shapefile, `data/frozen/segments/edges_combined.shp`, was written from the
model's own segment graph and carries the complete network (`node1`, `node2`,
`dist`, `speed`, `area`) next to the result columns. This GeoPackage is that
network with the result columns removed. The sha256 of the source shapefile is
recorded in `poa_base_network_manifest.json` and in `config.yaml`
(`inputs.base_network.derived_from_sha256`). The shapefile is the solver's
export with the GAUS Lines columns (`Gg*`) appended; compared with the export
before GAUS ran, no network attribute, key or value differs.

## Rebuilding the model from it

The routing graph rests on three properties of the model that built the
submitted run:

1. a zone sits at the **midpoint** of its own segment;
2. its connector to the road has **`dist` exactly 0**;
3. splitting a segment preserves its length exactly.

So the routing graph is the segment network subdivided at midpoints:

- nodes = segment endpoints ∪ one midpoint per segment;
- each segment contributes two half-edges of `cost/2`, where
  `cost = dist / (speed / 60)` — metre-equivalents at 60 km/h, so 1000 = one
  minute;
- zones are the midpoints of the `area > 0` segments; zero-capacity segments
  stay in the graph and carry routes;
- kernel `f(c) = (c + d0)^(−β)` with a zero diagonal (`self_interaction=False`).

**Self-loop segments** (`node1 == node2`, 53 rows, 47 with zones) contribute two
identical undirected half-edges. Duplicates are resolved by the **minimum**;
summing them would double the access cost of those 47 zones.
`prefcent.kernels.from_graph` does this.

`extract_network_from_shapefile.py` builds this graph from any segment file
with those five columns.

## The submitted run's parameters

γ ∈ {1, 0} · β = 2 · d0 = 5000 (a 5-minute start/end penalty) · κ = 0 ·
a₀ = R · direct substitution (ω = 1) · L1 renormalisation to Σ R at each step ·
**exactly 200 iterations, no tolerance** (`stop="budget_only"`).

The iteration budget is part of the model definition: PC1 ends near but not at
a fixed point (final step norm 2.16e-05) and matches the submitted column to
about 1e-14 because the submitted run also stopped at 200 iterations. Run to
convergence, it would not match. γ = 1 gives PC1 (preferential centrality with
agglomeration), γ = 0 gives PC2 (preferential centrality without
agglomeration).

## Verification

`paper_run.json` / `paper_run.npz` are the output of `reconstruct_paper_run.py`
on the shipped shapefile. The run reproduces all four submitted columns
(density `cd*` and mass `ca*` for PC1 and PC2) on all 56,162 zones; the 9,195
zero-capacity segments are zero on both sides.

| | PC1 `cd1g1b2k0` | PC2 `cd4g0b2k0` |
|---|---|---|
| within 0.1 % | **100.0000 %** | **100.0000 %** |
| median relative error | 2.2e-14 | 8.9e-15 |
| max relative error | 1.6e-12 | 4.3e-13 |

Only `node1`, `node2`, `dist`, `speed` and `area` enter the computation, so this
is a prediction of the submitted columns from the network and capacities alone,
not a fit.

## Running it

From this directory, with `prefcent==0.1.0` installed:

```python
import prefcent as pc
from extract_network_from_shapefile import extract

net, meta = extract("poa_base_network.gpkg")
kernel = pc.kernels.from_graph(
    net["graph_edges"], net["graph_costs"], meta["graph_nodes"],
    sources=net["zone_graph_index"], directed=False,
    beta=2.0, d0=5000.0, cost_units=meta["cost_units"], self_interaction=False)
model = pc.Model(kernel, pc.Landscape(net["R"], labels=net["zone_ids"]), gamma=1.0)
result = model.evolve(max_iter=200)          # fixed budget, as submitted
```

`reconstruct_paper_run.py` does this end to end, for both γ, and scores the
result against the submitted columns (joining on `node1`, `node2` when the
input is the GeoPackage and `--reference` names the shapefile). On the machine
of record it takes about five minutes with 64 workers: 107 s for the kernel
(Dijkstra over 111,030 nodes) and about 95 s per 200-iteration solve. The dense
kernel needs 23.5 GiB of RAM. The pipeline's own runs go through
`make pc-runs`, which uses the same functions.

**Variants.** A β variant changes only the decay: same graph, kernel rebuilt
with another `beta`. A γ variant changes only the attraction `W = γa + R`. A
boundary variant drops segment rows before extraction; keep `node1`/`node2`
stable and the acceptance test's partial-network handling does the rest.

## The three scripts

| script | what it does |
|---|---|
| `extract_network_from_shapefile.py` | builds the routing graph (edges, costs, zones, R) from a segment file |
| `reconstruct_paper_run.py` | runs the submitted parameters on it and scores against the submitted columns |
| `rebuild_segment_areas.py` | recomputes the static weight R (the `area` column) from the segment geometries; needs the `highway`, `speed`, `road_type`, `tunnel`, `bridge` and `drive` columns of `edges_combined.shp` |
