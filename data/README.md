# `data/` — the inputs, and the terms each one comes under

Nothing in `data/raw/` or `data/frozen/` is produced by the pipeline; they are
its inputs. `data/interim/` holds intermediate files and per-run manifests;
everything in it is regenerable by `make` and it is not tracked.

`data/SHA256SUMS` lists the SHA-256 checksum of every file in `data/raw/`,
`data/frozen/` and `env/wheels/`, as distributed. `make verify` checks the
files against it and lists anything missing or altered. Each run additionally
records the checksums of the inputs it read in its manifest under
`data/interim/`.

---

## `data/raw/` — the records the dependent variable is rebuilt from

| file | rows | source | terms |
|---|---|---|---|
| `itbi_transactions.csv` | 36,579 | ITBI (property-transfer tax) transactions, Finance Department of the Porto Alegre municipality; supplied with the first submission as supplementary material | redistributed here with the article so that its results can be reproduced; any other use is subject to the municipality's terms |
| `unit_registry.csv` | 844,739 | Real-estate unit registry, Porto Alegre municipality, same supplementary material | as above |
| `itbi_transactions_purpose_en.xlsx` | 36,579 | the same ITBI records with a `Purpose_EN` column, an English translation of the purpose added by the authors | as above |

`itbi_transactions.csv` is `;`-separated and pure ASCII (accents already
stripped at the source). `unit_registry.csv` is `;`-separated and **CP850**,
the Brazilian DOS codepage, not Latin-1: read as Latin-1 it silently mangles
`Ç` and `É` without raising an error, which is why `src/00_inventory.py`
rejects any candidate encoding that produces C1 control characters. The
registry also contains 13,971 literal `?` characters where accented characters
were lost before the file was supplied; 86 registry rows (4 distinct street
names) carry one in `Street`, which is part of the join key, so address
normalisation folds accents *and* treats `?` as a single-character wildcard.

The `.xlsx` is read for its `Purpose_EN` column alone, to label the
purpose → category mapping table. The categories themselves come from
`profiling.purpose.rules` in `config.yaml`, never from the translation.

## `data/frozen/` — the layers the submitted analysis was run on

| path | contents | terms |
|---|---|---|
| `cents_disaggregated.gpkg` | 9,030 address points, EPSG:31982; the address fields, `Terreno` (plot area), `Preco`, `Puni`, `Type` and the property-type indicators, and the five centralities `CC`, `BC`, `FK`, `PC1`, `PC2` | the authors; the centralities are computed on the OpenStreetMap-derived network (© OpenStreetMap contributors) |
| `cents_aggregated.gpkg` | the 500 m hexagon grid, 821 cells, EPSG:31982; `*_sum` / `*_mean` / `*_median` per measure and `Puni_mean` (BRL per m²; `US$` is the same at 5.3633 BRL per US$) | the authors; the centralities are computed on the OpenStreetMap-derived network (© OpenStreetMap contributors) |
| `segments/edges_combined.*` | 65,357 street segments, EPSG:32722; the eight PC variants plus the GAUS `Gg*` columns — the layer every centrality in the paper was aggregated from | derived from OpenStreetMap data (© OpenStreetMap contributors, <https://www.openstreetmap.org/copyright>) and made available under the Open Database License (ODbL) 1.0, <https://opendatacommons.org/licenses/odbl/1-0/>; the centrality and capacity columns are the authors' |
| `network/` | `poa_base_network.gpkg` (the same network with every result column removed: it is an *input*), its manifest, the two scripts that build the routing graph and rebuild the static weight `R`, the recipe (`README.md`) and the `paper_run` verification run | derived from OpenStreetMap data (© OpenStreetMap contributors, <https://www.openstreetmap.org/copyright>) and made available under the Open Database License (ODbL) 1.0, <https://opendatacommons.org/licenses/odbl/1-0/>; the capacity column are the authors' |
| `boundary/RS_Municipios_2025.*` | IBGE *Malha Municipal Digital* 2025, 499 municipalities of Rio Grande do Sul, EPSG:4674. Porto Alegre is `CD_MUN = 4314902` | IBGE's own terms: see `boundary/LEIA-ME.txt`, which asks users of the mesh to read IBGE's methodological note at <https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais.html> |

`cents_disaggregated.gpkg` is the authors' address-point layer with the
leftover fields of its KML export removed (`Name`, `descriptio`, `timestamp`,
`begin`, `end`, `altitudeMo`, `tessellate`, `extrude`, `visibility`,
`drawOrder`, `icon`, `layer`, `path`). Those fields were empty or described
the authors' file layout, and no step of the analysis read them; every other
field and every value is unchanged.

CRS, explicitly: the two GeoPackages are **EPSG:31982** (SIRGAS 2000 / UTM 22S,
metric, so kNN distances are metres); the segment layers are **EPSG:32722**
(WGS 84 / UTM 22S). The two differ only in datum, by less than 1e-4 m here,
but every script transforms explicitly rather than assuming. The IBGE mesh is
geographic (EPSG:4674) and is projected before anything is measured on it.

CC, BC and FK on the frozen layers were computed with **GAUS Lines v1.1**
(Dalcin and Krafta 2021), and PC1/PC2 with the preferential-centrality solver
published as `prefcent` 0.1.0 (Hellervik 2026; PyPI `prefcent==0.1.0`; source
<https://github.com/prefcent/prefcent>, tag `v0.1.0`; archived at
<https://doi.org/10.5281/zenodo.22643614>; the wheel is in `env/wheels/`). The
frozen columns are never recomputed by the pipeline; `make gaus-native` and
`make pc-runs` recompute the measures from the network when a sensitivity
analysis needs them at another extent or parameter value.

## `data/interim/` — regenerable

Not tracked. The run manifests here (`*_manifest.json`) record what produced
each file: parameters, input checksums, wall time and
the exact command line, with paths relative to the package root.
