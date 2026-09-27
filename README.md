# Remote Sensing MCP

Remote MCP gateway for the LST downscaling workflow.

## Architecture

- **Vercel**: lightweight MCP/API gateway.
- **GitHub Actions**: heavy ELITE FY-4A processing.
- **GitHub repository `data/`**: persistent China-area ELITE hourly cache.
- **Google Earth Engine**: optional authenticated discovery/export.

## ELITE repository cache

Large Zenodo monthly ZIP files are **not** committed to GitHub. They exist only on the temporary Actions runner when a requested hour is missing.

Persistent cache layout:

```text
data/
├── elite/
│   └── china/
│       └── YYYY/
│           └── MM/
│               └── DD/
│                   ├── ELITE_FY4A_LST_YYYYMMDD_0000_CHINA_K.tif
│                   ├── ELITE_FY4A_LST_YYYYMMDD_0100_CHINA_K.tif
│                   └── ...
└── metadata/
    └── elite-index.json
```

China cache extent:

```text
longitude: 73E .. 135E
latitude:  18N .. 54N
CRS:       EPSG:4326
spacing:   0.035932611365 degrees (~4 km)
```

The repository cache uses `uint16` with `scale_factor=0.01` Kelvin to reduce file size while retaining ELITE's temperature precision.

### Cache behavior

For each requested hour:

1. Check `data/elite/china/...`.
2. If present, crop the requested ROI directly from the repository cache.
3. If absent, temporarily download the required monthly Zenodo archive.
4. Extract only the missing HDF hours.
5. Read HDF4/HDF5, geolocate FY-4A/AGRI, and create the China-area cache GeoTIFF.
6. Commit all newly created China cache files back to the repository in one commit.
7. Delete the monthly ZIP and HDF temporary files.
8. Upload only the requested ROI GeoTIFFs and `result.json` as a short-lived GitHub Actions Artifact.

The workflow is serialized with an `elite-china-cache` concurrency group to avoid simultaneous cache writers.

## MCP tools

Read-only tools:

- `service_status`
- `list_supported_datasets`
- `elite_fy4a_lst_catalog`
- `plan_elite_fy4a_lst_download`
- `elite_storage_layout`

Privileged ELITE tools:

- `submit_elite_fy4a_lst_job`
- `elite_job_status`

`submit_elite_fy4a_lst_job` accepts:

- `start_date`
- `end_date`
- `bbox`
- `output_unit`
- optional `region_name`, e.g. `zhangye` or `beijing`

The requested bbox must be inside the configured China cache extent.

## Vercel environment variables

Remote ELITE dispatch:

- `GITHUB_WORKFLOW_TOKEN`
- `REMOTE_MCP_TOKEN`
- `GITHUB_WORKFLOW_REPOSITORY=ygangxian-cpu/remote-sensing-mcp`
- `GITHUB_WORKFLOW_ID=remote-sensing-elite.yml`
- `GITHUB_WORKFLOW_REF=main`

Earth Engine:

- `EE_PROJECT`
- `EE_SERVICE_ACCOUNT_JSON` or `EE_SERVICE_ACCOUNT_JSON_BASE64`
- optional `GEE_GCS_BUCKET` only for Earth Engine exports
- `REMOTE_MCP_TOKEN`

No GCS bucket is required for ELITE repository caching.

## Security

When privileged credentials are configured, MCP requests must send:

```http
Authorization: Bearer <REMOTE_MCP_TOKEN>
```

## Notes on GitHub storage

The cache intentionally stores only China-area hourly GeoTIFFs, not monthly ZIP files or full FY-4A disk products. Each file should remain well below GitHub's normal per-file limit, but a long multi-year archive can still make Git history large. If the cache grows substantially, move old years to an external archive while keeping the same index/layout.
