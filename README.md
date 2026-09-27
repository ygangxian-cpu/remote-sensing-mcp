# Remote Sensing MCP

Remote MCP gateway for the LST downscaling workflow.

## Architecture

- **Vercel**: lightweight MCP/API gateway.
- **GitHub Actions**: heavy ELITE FY-4A processing.
- **GitHub repository `data/`**: persistent China-area ELITE hourly cache.
- **Google Earth Engine**: authenticated ERA5-Land, MODIS and Landsat acquisition plus optional dataset discovery/export.

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


## ERA5-Land hourly ROI cache

ERA5-Land is acquired from Google Earth Engine by a dedicated GitHub Actions worker.

Canonical bands:

| Output band | GEE source band | Unit |
| --- | --- | --- |
| `T2_C` | `temperature_2m` | °C |
| `TD2_C` | `dewpoint_temperature_2m` | °C |
| `U10_MPS` | `u_component_of_wind_10m` | m/s |
| `V10_MPS` | `v_component_of_wind_10m` | m/s |
| `PSFC_PA` | `surface_pressure` | Pa |
| `SWDOWN_WM2` | `surface_solar_radiation_downwards_hourly` | W/m² |
| `GLW_WM2` | `surface_thermal_radiation_downwards_hourly` | W/m² |

Temperature is converted from Kelvin to Celsius. Hourly accumulated shortwave and longwave radiation are divided by 3600 to obtain W/m².

Persistent cache layout:

```text
data/era5_land/v1/
└── <region>-<bbox_hash>/
    └── YYYY/MM/DD/
        └── ERA5LAND_YYYYMMDD_HHMM_UTC.tif
```

The cache is ROI-specific rather than China-wide because ERA5-Land is coarse (~0.1° / 11 km) and repeated ROI downloads are small. Repeating the same ROI and hour reuses the repository file directly.

GitHub Actions uses the existing Earth Engine OAuth credential:

- repository secret `EE_USER_CREDENTIALS_JSON_BASE64`
- repository variable `EE_PROJECT=ee-ygangxian`

A service account remains an optional future fallback. No GCS bucket is required.

MCP tools:

- `era5_land_schema`
- `era5_storage_layout`
- `submit_era5_land_job`
- `era5_job_status`

One job may request up to 384 hours (16 days).


## MODIS Terra/Aqua daily LST

Datasets:

- `MODIS/061/MOD11A1` (Terra)
- `MODIS/061/MYD11A1` (Aqua)

Repository cache:

```text
data/modis_lst/v2/<region>-<bbox_hash>/YYYY/MM/DD/
├── MOD11A1_YYYYMMDD_LST_QA.tif
└── MYD11A1_YYYYMMDD_LST_QA.tif
```

The v2 downloader is **QA-preserving**: it keeps the product's native LST availability and does not apply the research QC mask during acquisition.

Each file contains:

- `LST_DAY_C`
- `LST_NIGHT_C`
- `DAY_VIEW_TIME_LOCAL_H`
- `NIGHT_VIEW_TIME_LOCAL_H`
- `QC_DAY`
- `QC_NIGHT`

LST conversion: `DN * 0.02 - 273.15`.

The previous strict research rule is retained as a **diagnostic/recommended downstream rule**, not a destructive download mask:

```text
bits 0-1 <= 1
bits 2-3 == 0
bits 6-7 <= 2
```

For each day/platform, `result.json` reports native LST coverage, strict-QC coverage, strict retention of native pixels, mandatory-QA class counts, data-quality class counts and LST-error classes. This makes it possible to distinguish true product gaps/cloud contamination from pixels removed only by a later research QA choice.

MCP tools:

- `modis_lst_schema`
- `submit_modis_lst_job`
- `modis_job_status`

## Landsat 8/9 Collection 2 Level 2

Datasets:

- `LANDSAT/LC08/C02/T1_L2`
- `LANDSAT/LC09/C02/T1_L2`

Repository cache:

```text
data/landsat_c2_l2/v2/<region>-<bbox_hash>/YYYY/MM/DD/
└── <LANDSAT_PRODUCT_ID>_L2_RAW_QA.tif
```

The v2 downloader preserves native product availability and QA instead of permanently masking research-quality pixels at download time.

Each cached scene contains:

- `LST_C`
- `ST_QA_K`
- `SR_B2` .. `SR_B7`
- `QA_PIXEL`
- `QA_RADSAT`

Conversions:

```text
LST_C = ST_B10 * 0.00341802 + 149.0 - 273.15
SR    = SR_Bx * 0.0000275 - 0.2
```

No additional QA mask is applied during download. The quality report in `result.json` separately records native LST coverage, clear-LST coverage, QA_PIXEL bit counts, water fraction, radiometric saturation and clear/unsaturated SR coverage.

Recommended clear-pixel logic for downstream experiments is:

```text
QA_PIXEL bits 0,1,2,3,4,5 == 0
```

Water (bit 7) is preserved. `QA_RADSAT` is preserved and reported separately rather than being used as a blanket mask that can erase an otherwise valid LST pixel because a reflective band is saturated.

MCP tools:

- `landsat_schema`
- `submit_landsat_job`
- `landsat_job_status`

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

No GCS bucket is required for ELITE, ERA5-Land, MODIS or Landsat repository caching.

## Security

When privileged credentials are configured, MCP requests must send:

```http
Authorization: Bearer <REMOTE_MCP_TOKEN>
```

## Notes on GitHub storage

The cache intentionally stores only China-area hourly GeoTIFFs, not monthly ZIP files or full FY-4A disk products. Each file should remain well below GitHub's normal per-file limit, but a long multi-year archive can still make Git history large. If the cache grows substantially, move old years to an external archive while keeping the same index/layout.
