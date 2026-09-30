# AGENTS.md

## Repository Charter

This repository is the **remote-sensing data platform / MCP**, not the formal scientific experiment repository.

Its primary responsibility is to make remote-sensing data reliably available to agents and downstream experiments:

- ELITE FY-4A/AGRI LST acquisition and standardized extraction.
- MODIS, Landsat and ERA5/ERA5-Land acquisition.
- Static/scaling-factor acquisition.
- QA/QC, spatial clipping, reprojection, grid alignment and provenance.
- Sensor observation-time parsing and exact-time matching utilities.
- MCP/API interfaces that expose those data capabilities.

The formal downscaling research repository is:

`yuyan3616/jiangchidu4kmto100m`

Formal methods, ablations, paper experiments and paper-facing results belong there.

## Mandatory Repository Boundary

Before adding or changing code, classify the task.

### Belongs in this repository

Put work here when its reusable output is **data or a data-access capability**, for example:

- download/fetch/export a sensor product;
- decode ELITE source files;
- apply product-native QA;
- convert MODIS local solar view time to UTC;
- exact-time interpolate ELITE observations;
- align a product to a requested grid;
- expose data through MCP/API;
- produce standardized snapshots plus provenance.

### Does NOT belong here

Do not make this repository the source of truth for:

- 4 km -> 1 km or 1 km -> 100 m downscaling methods;
- v1/v2/v3/v4 model implementations;
- paper baselines and ablations;
- scientific model selection;
- Landsat/station performance claims;
- final figures/tables used by the paper;
- long-lived method-specific training pipelines.

Those belong in `yuyan3616/jiangchidu4kmto100m`.

## Temporary Experiment Exception

A short-lived script under `experiments/` may run here when it is needed to validate a **data-interface or mechanism hypothesis** and this repository already provides the required cloud credentials/data path.

This is an exception, not a change of ownership.

When using the exception:

1. Keep the script isolated under `experiments/`.
2. Name temporary workflows clearly, preferably `temp-*.yml`.
3. Do not describe the temporary implementation as the canonical paper method.
4. Store large results in GitHub Actions artifacts rather than Git.
5. Once the mechanism is accepted as a method candidate, migrate the canonical implementation and paper-facing documentation to `yuyan3616/jiangchidu4kmto100m`.
6. Reduce temporary push-trigger workflows back to `workflow_dispatch` after debugging.
7. Prefer extracting reusable acquisition/time-matching/QC code into data utilities here and keeping model logic in the experiment repository.

Current example: `experiments/run_parent_bias_correction_v4_pilot.py` is a mechanism pilot using the MCP data environment. If promoted, the formal v4 implementation must live in the downscaling repository.

## Cross-Repository Contract

The intended dependency direction is:

```text
remote-sensing-mcp
  data acquisition / QC / timing / alignment / provenance
                     |
                     v
yuyan3616/jiangchidu4kmto100m
  methods / training / ablation / validation / paper results
```

Avoid the reverse dependency. The data platform must not depend on a paper method in order to provide a sensor product.

When the scientific repository needs a new input, prefer adding a reusable data capability here, then consume its standardized output from the scientific repository.

## Result Ownership and Compute Boundary

This repository is the **Data Plane**, not the long-term result archive and not the compute scheduler.

For downstream experiments:

- standardized sensor inputs, QA/QC outputs and provenance may be published as GitHub Actions artifacts or external/Kaggle Datasets;
- large rasters, matchup tables, feature matrices and model checkpoints should not be committed to this repository;
- formal paper metrics, ablation summaries and scientific conclusions must be migrated to `yuyan3616/jiangchidu4kmto100m`;
- Kaggle execution, staged checkpoints and parallel notebook scheduling belong to `yuyan3616/kaggle-mcp`.

The intended three-plane collaboration is:

```text
remote-sensing-mcp
  Data Plane
  acquisition / QC / timing / alignment / provenance
                    |
                    v
jiangchidu4kmto100m
  Science Plane
  methods / configs / experiment definitions / paper results
                    |
                    v
kaggle-mcp + Kaggle
  Compute Plane
  staged execution / parallel runs / checkpoint datasets
```

A temporary pilot executed here may leave its large artifacts in Actions, but once the result affects method selection, the small paper-facing outputs and the scientific interpretation must be copied into the scientific repository.

## Current Data Conventions

- Formal coarse source: ELITE FY-4A/AGRI hourly 4 km LST.
- Project time standard: UTC. ELITE filename HHMM is treated as UTC for the project.
- MODIS time matching should use the product view-time bands and per-pixel longitude conversion rather than fixed nominal overpass time.
- MODIS LST QC used by current experiments: bits 0-1 <= 1, bits 2-3 == 0, bits 6-7 <= 2.
- Landsat remains an independent high-resolution validation source in the formal paper workflow; do not silently use Landsat LST as a training target for a formal method.
- Preserve source/product identifiers and relevant processing metadata whenever producing experiment snapshots.

## Working Rules for Agents

- Read this file before substantial edits.
- Preserve existing data interfaces unless the task explicitly changes them.
- Do not hard-code credentials or tokens. Use repository/environment secrets.
- Do not commit large raw sensor archives or generated rasters unless explicitly required.
- Prefer reproducible scripts and GitHub Actions artifacts for diagnostics.
- Distinguish a data-quality diagnostic from a scientific model claim.
- If a task starts as data plumbing but evolves into model design, move the model work to the scientific repository instead of expanding this repository indefinitely.

## Validation Before Finishing

For data-platform changes, check the smallest relevant combination of:

- Python syntax/import checks;
- acquisition/API success;
- QA/QC pixel counts;
- grid dimensions, CRS and transform;
- time standard / observation-time metadata;
- provenance fields;
- MCP endpoint or worker behavior when affected.

For temporary experiment harnesses, also record the run ID and artifact name so the formal scientific repository can cite or migrate the result.
