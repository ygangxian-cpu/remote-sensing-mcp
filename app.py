from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import ee
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

ZENODO_API = "https://zenodo.org/api"
ELITE_TITLE = "ELITE land surface temperature: FY-4A/AGRI hourly 4km seamless LST"
KNOWN_ELITE_RECORDS = {2019: 10672052, 2021: 8378354}
DATASETS = {
    "era5_land_hourly": "ECMWF/ERA5_LAND/HOURLY",
    "modis_terra_lst": "MODIS/061/MOD11A1",
    "modis_aqua_lst": "MODIS/061/MYD11A1",
    "landsat8_c2_l2": "LANDSAT/LC08/C02/T1_L2",
    "landsat9_c2_l2": "LANDSAT/LC09/C02/T1_L2",
    "srtm_dem": "USGS/SRTMGL1_003",
    "nasadem": "NASA/NASADEM_HGT/001",
    "worldcover_2021": "ESA/WorldCover/v200/2021",
    "modis_albedo": "MODIS/061/MCD43A3",
}

mcp = MCPServer(
    "Remote Sensing MCP",
    instructions=(
        "Remote-sensing data gateway for public ELITE FY-4A catalog/planning and authenticated "
        "Google Earth Engine discovery/export. Large ELITE downloads are delegated to GitHub Actions."
    ),
)

_EE_READY = False


def _privileged_credentials_present() -> bool:
    return bool(
        os.getenv("EE_SERVICE_ACCOUNT_JSON")
        or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        or os.getenv("GITHUB_WORKFLOW_TOKEN")
    )


def _initialize_ee(project: str | None = None) -> str:
    global _EE_READY
    effective_project = project or os.getenv("EE_PROJECT")
    if _EE_READY:
        return effective_project or ""
    raw = os.getenv("EE_SERVICE_ACCOUNT_JSON")
    raw_b64 = os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
    if not raw and raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        raise RuntimeError(
            "Earth Engine is not configured. Add EE_PROJECT and EE_SERVICE_ACCOUNT_JSON "
            "(or EE_SERVICE_ACCOUNT_JSON_BASE64) to Vercel environment variables."
        )
    info = json.loads(raw)
    effective_project = effective_project or info.get("project_id")
    if not effective_project:
        raise RuntimeError("EE_PROJECT is required.")
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(credentials, project=effective_project)
    _EE_READY = True
    return effective_project


def _bbox_region(bbox: list[float]):
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin, ymin, xmax, ymax]")
    xmin, ymin, xmax, ymax = map(float, bbox)
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326", geodesic=False)


def _resolve_dataset(dataset: str) -> str:
    return DATASETS.get(dataset, dataset)


def _zenodo_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def _elite_record(year: int) -> dict[str, Any]:
    record_id = KNOWN_ELITE_RECORDS.get(year)
    if record_id:
        payload = _zenodo_json(f"{ZENODO_API}/records/{record_id}")
    else:
        search = _zenodo_json(
            f"{ZENODO_API}/records",
            {"q": f"FY-4A AGRI seamless LST {year}", "size": 50, "sort": "mostrecent"},
        )
        hits = ((search.get("hits") or {}).get("hits") or [])
        payload = next(
            (
                hit
                for hit in hits
                if "fy-4a/agri" in str((hit.get("metadata") or {}).get("title", "")).lower()
                and "seamless lst" in str((hit.get("metadata") or {}).get("title", "")).lower()
                and str(year) in str((hit.get("metadata") or {}).get("title", ""))
            ),
            None,
        )
        if payload is None:
            raise LookupError(f"No ELITE FY-4A/AGRI seamless LST Zenodo record found for {year}")
    metadata = payload.get("metadata") or {}
    files_obj = payload.get("files") or {}
    entries = files_obj.get("entries", files_obj) if isinstance(files_obj, dict) else files_obj
    items = list(entries.values()) if isinstance(entries, dict) else list(entries or [])
    files = []
    for item in items:
        key = str(item.get("key") or item.get("filename") or "")
        links = item.get("links") or {}
        files.append(
            {
                "key": key,
                "size": int(item.get("size") or 0),
                "checksum": item.get("checksum"),
                "download_url": links.get("content")
                or links.get("self")
                or f"https://zenodo.org/records/{payload['id']}/files/{key}?download=1",
            }
        )
    return {
        "record_id": int(payload["id"]),
        "title": metadata.get("title"),
        "doi": metadata.get("doi") or payload.get("doi"),
        "html_url": (payload.get("links") or {}).get("html"),
        "files": files,
    }


def _months(start_date: str, end_date: str) -> list[str]:
    start = datetime.fromisoformat(start_date[:10])
    end = datetime.fromisoformat(end_date[:10])
    if end <= start:
        raise ValueError("end_date must be after start_date")
    out = []
    y, m = start.year, start.month
    while True:
        month_start = datetime(y, m, 1)
        if month_start >= end and (y, m) != (start.year, start.month):
            break
        out.append(f"{y:04d}{m:02d}")
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def _github_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv("GITHUB_WORKFLOW_ID", "remote-sensing-elite.yml")
    ref = os.getenv("GITHUB_WORKFLOW_REF", "main")
    token = os.getenv("GITHUB_WORKFLOW_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GitHub Actions remote dispatch is not configured. Add GITHUB_WORKFLOW_TOKEN "
            "to Vercel, or run the ELITE workflow manually in GitHub Actions."
        )
    return repo, workflow, ref, token


@mcp.tool()
def service_status() -> dict[str, Any]:
    """Show which online capabilities are configured."""
    return {
        "service": "remote-sensing-mcp",
        "elite_catalog": True,
        "elite_plan": True,
        "earth_engine_configured": bool(
            os.getenv("EE_SERVICE_ACCOUNT_JSON") or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        ),
        "elite_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "elite_persistent_storage_configured": bool(
            os.getenv("REMOTE_DATA_BUCKET") or os.getenv("GEE_GCS_BUCKET")
        ),
        "gcs_bucket_configured": bool(os.getenv("GEE_GCS_BUCKET")),
    }


@mcp.tool()
def list_supported_datasets() -> dict[str, str]:
    """List built-in GEE aliases."""
    return DATASETS


@mcp.tool()
def elite_fy4a_lst_catalog(year: int) -> dict[str, Any]:
    """Discover ELITE FY-4A/AGRI hourly 4 km seamless LST monthly archives on Zenodo."""
    return _elite_record(year)


@mcp.tool()
def plan_elite_fy4a_lst_download(start_date: str, end_date: str) -> dict[str, Any]:
    """Plan ELITE downloads without transferring multi-GB archives."""
    archives = []
    for yyyymm in _months(start_date, end_date):
        record = _elite_record(int(yyyymm[:4]))
        filename = f"{yyyymm}.zip"
        item = next((f for f in record["files"] if f["key"] == filename), None)
        if not item:
            raise LookupError(f"{filename} not found in Zenodo record {record['record_id']}")
        archives.append({**item, "record_id": record["record_id"], "year_month": yyyymm})
    total = sum(x["size"] for x in archives)
    return {
        "product": ELITE_TITLE,
        "start_date": start_date,
        "end_date": end_date,
        "spatial_resolution": "4 km",
        "temporal_resolution": "1 hour",
        "scale": 0.01,
        "archives": archives,
        "total_size_bytes": total,
        "total_size_gib": round(total / (1024**3), 3),
    }


@mcp.tool()
def elite_storage_layout() -> dict[str, Any]:
    """Describe the Raw -> Cache -> Derived object-storage layout used by ELITE jobs."""
    bucket = os.getenv("REMOTE_DATA_BUCKET") or os.getenv("GEE_GCS_BUCKET")
    return {
        "architecture": "raw -> cache -> derived",
        "persistent_storage_configured": bool(bucket),
        "bucket": bucket,
        "raw": "raw/elite/YYYY/MM/YYYYMM.zip",
        "cache": "cache/elite/YYYY/MM/DD/ELITE_FY4A_LST_YYYYMMDD_HHMM_FULLDISK_K.tif",
        "derived": "derived/elite/<region>-<bbox_hash>/YYYY/MM/DD/*.tif",
        "cache_grid": "FY-4A/AGRI native geostationary 4 km",
        "cache_unit": "kelvin",
        "derived_crs": "EPSG:4326",
        "note": (
            "Raw monthly archives and full-disk hourly cache are reused across ROIs when "
            "GitHub Actions has REMOTE_DATA_BUCKET and GCP_SERVICE_ACCOUNT_JSON_BASE64 secrets."
        ),
    }


@mcp.tool()
def submit_elite_fy4a_lst_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    output_unit: str = "celsius",
    region_name: str = "",
) -> dict[str, Any]:
    """Submit a heavy ELITE download + HDF geolocation + ROI crop to GitHub Actions."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    if output_unit not in {"celsius", "kelvin"}:
        raise ValueError("output_unit must be celsius or kelvin")
    repo, workflow, ref, token = _github_config()
    job_key = f"elite-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    response = requests.post(
        url,
        timeout=30,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "ref": ref,
            "inputs": {
                "job_key": job_key,
                "start_date": start_date,
                "end_date": end_date,
                "bbox": ",".join(str(float(x)) for x in bbox),
                "region_name": region_name,
                "output_unit": output_unit,
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(f"GitHub dispatch failed: {response.status_code} {response.text[:300]}")
    return {
        "submitted": True,
        "job_key": job_key,
        "repository": repo,
        "workflow": workflow,
        "status_tool": "elite_job_status",
        "region_name": region_name or None,
        "storage_architecture": "raw -> cache -> derived",
    }


@mcp.tool()
def elite_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted ELITE GitHub Actions job."""
    repo, workflow, _, token = _github_config()
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs",
        params={"event": "workflow_dispatch", "per_page": 50},
        timeout=30,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    response.raise_for_status()
    runs = response.json().get("workflow_runs", [])
    for run in runs:
        title = str(run.get("display_title") or run.get("name") or "")
        if job_key in title:
            return {
                "found": True,
                "job_key": job_key,
                "run_id": run.get("id"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "html_url": run.get("html_url"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
            }
    return {"found": False, "job_key": job_key, "checked_runs": len(runs)}


@mcp.tool()
def gee_auth_status(project: str | None = None) -> dict[str, Any]:
    """Validate the Vercel Earth Engine service-account configuration."""
    try:
        used_project = _initialize_ee(project)
        ee.Number(1).getInfo()
        return {"ok": True, "project": used_project}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def check_collection_availability(
    dataset: str,
    start_date: str,
    end_date: str,
    bbox: list[float],
    project: str | None = None,
) -> dict[str, Any]:
    """Count GEE images and return matching timestamps for a WGS84 bbox."""
    _initialize_ee(project)
    region = _bbox_region(bbox)
    dataset_id = _resolve_dataset(dataset)
    collection = ee.ImageCollection(dataset_id).filterDate(start_date, end_date).filterBounds(region)
    count = int(collection.size().getInfo())
    timestamps = (
        ee.List(collection.aggregate_array("system:time_start"))
        .map(lambda t: ee.Date(t).format("YYYY-MM-dd HH:mm"))
        .getInfo()
    )
    return {
        "dataset_id": dataset_id,
        "count": count,
        "timestamps": timestamps[:500],
        "truncated": count > 500,
    }


def _start_gcs_export(
    image,
    region,
    description: str,
    scale: float,
    crs: str = "EPSG:4326",
) -> dict[str, Any]:
    bucket = os.getenv("GEE_GCS_BUCKET")
    if not bucket:
        raise RuntimeError("Set GEE_GCS_BUCKET in Vercel before starting GEE exports.")
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        bucket=bucket,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        fileFormat="GeoTIFF",
        maxPixels=1e13,
        formatOptions={"cloudOptimized": True, "noData": -9999},
    )
    task.start()
    status = task.status()
    return {
        "task_id": status.get("id") or getattr(task, "id", None),
        "description": description,
        "state": status.get("state"),
        "bucket": bucket,
        "prefix": description,
    }


@mcp.tool()
def export_era5_land_to_gcs(
    start_date: str,
    end_date: str,
    bands: list[str],
    bbox: list[float],
    scale: float = 1000,
    convert_units: bool = True,
    project: str | None = None,
) -> dict[str, Any]:
    """Start hourly ERA5-Land GeoTIFF exports to GCS. Intended for short batches per call."""
    _initialize_ee(project)
    region = _bbox_region(bbox)
    collection = (
        ee.ImageCollection(DATASETS["era5_land_hourly"])
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .sort("system:time_start")
    )
    size = int(collection.size().getInfo())
    if size > 48:
        raise ValueError("For the online gateway, export at most 48 ERA5 hours per call.")
    tasks = []
    for i in range(size):
        image = ee.Image(collection.toList(size).get(i))
        ts = ee.Date(image.get("system:time_start")).format("YYYYMMdd_HHmm").getInfo()
        output = []
        for band in bands:
            b = image.select(band)
            name = band
            if convert_units and band in {"temperature_2m", "dewpoint_temperature_2m"}:
                b = b.subtract(273.15)
                name = "T2_C" if band == "temperature_2m" else "TD2_C"
            elif convert_units and band in {
                "surface_solar_radiation_downwards_hourly",
                "surface_thermal_radiation_downwards_hourly",
            }:
                b = b.divide(3600.0)
                name = "SWDOWN_Wm2" if "solar" in band else "GLW_Wm2"
            output.append(b.rename(name))
        prepared = ee.Image.cat(output)
        tasks.append(_start_gcs_export(prepared, region, f"ERA5LAND_{ts}", scale))
    return {"count": len(tasks), "tasks": tasks}


@mcp.tool()
def list_export_tasks(limit: int = 100, project: str | None = None) -> list[dict[str, Any]]:
    """List recent GEE batch tasks."""
    _initialize_ee(project)
    return [t.status() for t in ee.batch.Task.list()[: max(1, min(limit, 500))]]


@mcp.tool()
def cancel_export_task(task_id: str, project: str | None = None) -> dict[str, Any]:
    """Cancel a GEE batch task."""
    _initialize_ee(project)
    task = ee.batch.Task(task_id)
    task.cancel()
    return {"task_id": task_id, "cancel_requested": True}


REMOTE_TOKEN = os.getenv("REMOTE_MCP_TOKEN", "")
if _privileged_credentials_present() and not REMOTE_TOKEN:
    raise RuntimeError(
        "REMOTE_MCP_TOKEN must be set whenever Earth Engine or GitHub workflow credentials are configured."
    )

mcp_app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="Remote Sensing MCP",
    description="GEE + ELITE FY-4A remote-sensing MCP gateway",
    version="0.4.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def bearer_guard(request: Request, call_next):
    if REMOTE_TOKEN and request.url.path.startswith("/mcp"):
        if request.headers.get("authorization", "") != f"Bearer {REMOTE_TOKEN}":
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/")
def home():
    return {
        "service": "Remote Sensing MCP",
        "status": "ok",
        "mcp_endpoint": "/mcp",
        "health_endpoint": "/health",
        "mode": "privileged" if _privileged_credentials_present() else "public-readonly",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "remote-sensing-mcp",
        "vercel": bool(os.getenv("VERCEL")),
        "ee_configured": bool(
            os.getenv("EE_SERVICE_ACCOUNT_JSON") or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        ),
        "github_actions_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "gcs_bucket_configured": bool(os.getenv("GEE_GCS_BUCKET")),
        "remote_data_bucket_configured": bool(
            os.getenv("REMOTE_DATA_BUCKET") or os.getenv("GEE_GCS_BUCKET")
        ),
        "auth_enabled": bool(REMOTE_TOKEN),
    }


app.mount("/", mcp_app)
