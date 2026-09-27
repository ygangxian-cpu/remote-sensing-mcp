# Remote Sensing MCP

Remote MCP gateway for the LST downscaling workflow.

## Architecture

- **Vercel**: lightweight MCP/API gateway.
- **GitHub Actions**: heavy ELITE FY-4A processing.
- **Google Cloud Storage (optional but recommended)**: persistent Raw -> Cache -> Derived data lake.
- **Google Earth Engine**: optional authenticated dataset discovery/export.

### ELITE persistent storage

The worker now uses three storage layers:

```text
raw/elite/YYYY/MM/YYYYMM.zip
    Original Zenodo monthly archive. Downloaded only when the raw layer misses.

cache/elite/YYYY/MM/DD/
    ELITE_FY4A_LST_YYYYMMDD_HHMM_FULLDISK_K.tif
    Reusable full-disk 4 km cache in the native FY-4A/AGRI geostationary grid.
    Stored in canonical Kelvin and written as Cloud Optimized GeoTIFF when supported.

derived/elite/<region>-<bbox_hash>/YYYY/MM/DD/
    ROI GeoTIFFs in EPSG:4326 and the requested Celsius/Kelvin unit.
```

A later Zhangye, Beijing, Dunhuang, or other ROI request reuses the same hourly full-disk cache. Repeating the same ROI can reuse the derived object directly.

Without a GCS bucket the worker still works, but Raw and Cache are ephemeral because GitHub Actions runners are temporary.

## GitHub Actions secrets for persistent ELITE storage

Configure these repository Actions secrets:

- `REMOTE_DATA_BUCKET`: GCS bucket name. You may use the same bucket as Earth Engine exports.
- `GCP_SERVICE_ACCOUNT_JSON_BASE64`: base64-encoded Google service-account JSON with object read/write access to the bucket.

The worker also accepts `GEE_GCS_BUCKET` as a fallback bucket name.

Recommended bucket layout:

```text
remote-sensing-data/
├── raw/elite/
├── cache/elite/
└── derived/elite/
```

For least privilege, grant the worker service account object read/write permissions only on the selected bucket. Workload Identity Federation/OIDC is preferable to a long-lived JSON key for a production setup; JSON credentials are supported here to keep the initial setup simple.

## MCP tools

Public/read-only tools:

- `service_status`
- `list_supported_datasets`
- `elite_fy4a_lst_catalog`
- `plan_elite_fy4a_lst_download`
- `elite_storage_layout`

Privileged ELITE tools:

- `submit_elite_fy4a_lst_job`
- `elite_job_status`

`submit_elite_fy4a_lst_job` accepts an optional `region_name`, for example `zhangye` or `beijing`. A bbox hash is appended internally so changing the ROI cannot silently reuse the wrong derived output.

## Vercel environment variables

Remote ELITE dispatch:

- `GITHUB_WORKFLOW_TOKEN`
- `REMOTE_MCP_TOKEN`
- `GITHUB_WORKFLOW_REPOSITORY=ygangxian-cpu/remote-sensing-mcp`
- `GITHUB_WORKFLOW_ID=remote-sensing-elite.yml`
- `GITHUB_WORKFLOW_REF=main`
- optional `REMOTE_DATA_BUCKET` for status/layout reporting

Earth Engine:

- `EE_PROJECT`
- `EE_SERVICE_ACCOUNT_JSON` or `EE_SERVICE_ACCOUNT_JSON_BASE64`
- `GEE_GCS_BUCKET` for GEE exports
- `REMOTE_MCP_TOKEN`

Endpoints:

- `/health`
- `/mcp`

Security rule: when privileged credentials are configured, MCP requests must send:

```http
Authorization: Bearer <REMOTE_MCP_TOKEN>
```

## ELITE worker behavior

The worker supports both HDF5 and the HDF4 format used by ELITE. It parses ELITE's `YYYYDDDHHMM` day-of-year timestamps, applies the 0.01 LST scale, preserves a full-disk canonical cache, and generates EPSG:4326 ROI GeoTIFFs. GitHub Artifacts contain only the requested derived outputs and `result.json`; long-lived Raw/Cache/Derived objects live in GCS when configured.
