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
ERA5_LAND_BANDS = {
    "T2_C": {"source": "temperature_2m", "unit": "degC"},
    "TD2_C": {"source": "dewpoint_temperature_2m", "unit": "degC"},
    "U10_MPS": {"source": "u_component_of_wind_10m", "unit": "m s-1"},
    "V10_MPS": {"source": "v_component_of_wind_10m", "unit": "m s-1"},
    "PSFC_PA": {"source": "surface_pressure", "unit": "Pa"},
    "SWDOWN_WM2": {"source": "surface_solar_radiation_downwards_hourly", "unit": "W m-2"},
    "GLW_WM2": {"source": "surface_thermal_radiation_downwards_hourly", "unit": "W m-2"},
}
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
        "Remote-sensing data gateway for ELITE FY-4A, ERA5-Land, MODIS, Landsat and scaling factors. "
        "Heavy downloads are delegated to GitHub Actions. When the user asks to download data, do not "
        "stop after returning a job id or completed status: after the job succeeds, call get_job_result "
        "to obtain a short-lived artifact URL, then use the client environment to save the ZIP to the "
        "user's requested local directory. The remote MCP server itself cannot write to the client's filesystem."
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


def _github_era5_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv("GITHUB_ERA5_WORKFLOW_ID", "remote-sensing-era5.yml")
    ref = os.getenv("GITHUB_WORKFLOW_REF", "main")
    token = os.getenv("GITHUB_WORKFLOW_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GitHub Actions remote dispatch is not configured. Add GITHUB_WORKFLOW_TOKEN "
            "to Vercel, or run the ERA5-Land workflow manually in GitHub Actions."
        )
    return repo, workflow, ref, token


def _github_modis_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv("GITHUB_MODIS_WORKFLOW_ID", "remote-sensing-modis.yml")
    ref = os.getenv("GITHUB_WORKFLOW_REF", "main")
    token = os.getenv("GITHUB_WORKFLOW_TOKEN", "")
    if not token:
        raise RuntimeError("GitHub Actions remote dispatch is not configured.")
    return repo, workflow, ref, token


def _github_landsat_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv("GITHUB_LANDSAT_WORKFLOW_ID", "remote-sensing-landsat.yml")
    ref = os.getenv("GITHUB_WORKFLOW_REF", "main")
    token = os.getenv("GITHUB_WORKFLOW_TOKEN", "")
    if not token:
        raise RuntimeError("GitHub Actions remote dispatch is not configured.")
    return repo, workflow, ref, token


def _github_scaling_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv("GITHUB_SCALING_WORKFLOW_ID", "remote-sensing-scaling-factors.yml")
    ref = os.getenv("GITHUB_WORKFLOW_REF", "main")
    token = os.getenv("GITHUB_WORKFLOW_TOKEN", "")
    if not token:
        raise RuntimeError("GitHub Actions remote dispatch is not configured.")
    return repo, workflow, ref, token


def _github_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _find_job_run(repo: str, token: str, job_key: str) -> dict[str, Any] | None:
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/runs",
        params={"event": "workflow_dispatch", "per_page": 100},
        timeout=30,
        headers=_github_headers(token),
    )
    response.raise_for_status()
    for run in response.json().get("workflow_runs", []):
        title = str(run.get("display_title") or run.get("name") or "")
        if job_key in title:
            return run
    return None


def _artifact_for_run(
    repo: str,
    token: str,
    run_id: int,
    job_key: str,
) -> dict[str, Any] | None:
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts",
        params={"per_page": 100},
        timeout=30,
        headers=_github_headers(token),
    )
    response.raise_for_status()
    artifacts = response.json().get("artifacts", [])
    exact = next((a for a in artifacts if a.get("name") == job_key), None)
    if exact:
        return exact
    if len(artifacts) == 1:
        return artifacts[0]
    return None


def _artifact_signed_download_url(repo: str, token: str, artifact_id: int) -> str:
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip",
        timeout=30,
        headers=_github_headers(token),
        allow_redirects=False,
    )
    if response.status_code in {301, 302, 303, 307, 308}:
        location = response.headers.get("Location")
        if location:
            return location
    if response.status_code == 410:
        raise RuntimeError("The GitHub Actions artifact has expired.")
    response.raise_for_status()
    raise RuntimeError("GitHub did not return an artifact download redirect.")


@mcp.tool()
def get_job_result(job_key: str) -> dict[str, Any]:
    """Return a short-lived direct download URL for a completed job artifact.

    The URL points directly to GitHub's artifact object storage and does not
    expose the repository token. The client should download it immediately.
    """
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    token = os.getenv("GITHUB_WORKFLOW_TOKEN", "")
    if not token:
        raise RuntimeError("GitHub Actions result retrieval is not configured.")

    run = _find_job_run(repo, token, job_key)
    if run is None:
        return {
            "found": False,
            "job_key": job_key,
            "ready": False,
            "message": "No matching workflow_dispatch run was found.",
        }

    status = run.get("status")
    conclusion = run.get("conclusion")
    result: dict[str, Any] = {
        "found": True,
        "job_key": job_key,
        "run_id": run.get("id"),
        "status": status,
        "conclusion": conclusion,
        "workflow_url": run.get("html_url"),
        "ready": False,
    }
    if status != "completed":
        result["message"] = "The job is still running. Check again later."
        return result
    if conclusion != "success":
        result["message"] = "The job completed but did not succeed."
        return result

    artifact = _artifact_for_run(repo, token, int(run["id"]), job_key)
    if artifact is None:
        result["message"] = "The job succeeded but no matching result artifact was found."
        return result
    if artifact.get("expired"):
        result["message"] = "The result artifact has expired."
        result["artifact_id"] = artifact.get("id")
        result["artifact_expires_at"] = artifact.get("expires_at")
        return result

    signed_url = _artifact_signed_download_url(
        repo,
        token,
        int(artifact["id"]),
    )
    result.update(
        {
            "ready": True,
            "artifact_id": artifact.get("id"),
            "artifact_name": artifact.get("name"),
            "artifact_size_bytes": artifact.get("size_in_bytes"),
            "artifact_expires_at": artifact.get("expires_at"),
            "download_url": signed_url,
            "suggested_filename": f"{job_key}.zip",
            "download_url_note": (
                "This GitHub-signed URL is short-lived. Download it immediately; "
                "the long-lived result remains in the repository cache where applicable."
            ),
            "client_next_step": (
                "Save download_url to the user's requested local directory. "
                "On Windows, Codex can use curl.exe -L <download_url> -o <path>."
            ),
        }
    )
    return result


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
        "era5_land_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "modis_lst_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "landsat_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "scaling_factors_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "era5_land_cache_backend": "github_repository_roi",
        "modis_lst_cache_backend": "github_repository_roi",
        "landsat_cache_backend": "github_repository_roi",
        "scaling_factors_cache_backend": "github_repository_roi",
        "job_result_download_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "elite_persistent_storage_configured": True,
        "elite_cache_backend": "github_repository",
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
    """Describe the repository-backed China cache used by ELITE jobs."""
    return {
        "architecture": "Zenodo temporary archive -> GitHub China cache -> ROI artifact",
        "persistent_storage_configured": True,
        "backend": "github_repository",
        "raw_archive": "temporary only; deleted after missing hours are extracted",
        "cache": "data/elite/china/YYYY/MM/DD/ELITE_FY4A_LST_YYYYMMDD_HHMM_CHINA_K.tif",
        "metadata": "data/metadata/elite-index.json",
        "china_bbox": [73.0, 18.0, 135.0, 54.0],
        "cache_crs": "EPSG:4326",
        "cache_resolution_degrees": 0.035932611365,
        "cache_dtype": "uint16",
        "cache_scale_factor": 0.01,
        "cache_unit": "kelvin",
        "derived": "GitHub Actions artifact only",
        "note": (
            "If a requested hour already exists in data/elite/china, the worker skips "
            "the Zenodo monthly archive and crops the ROI directly from the repository cache."
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
        "result_tool": "get_job_result",
        "region_name": region_name or None,
        "storage_architecture": "Zenodo temporary archive -> GitHub China cache -> ROI artifact",
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
def era5_land_schema() -> dict[str, Any]:
    """Describe the canonical ERA5-Land hourly variables and conversions."""
    return {
        "dataset": DATASETS["era5_land_hourly"],
        "temporal_resolution": "1 hour",
        "native_spatial_resolution": "about 0.1 degree / 11 km",
        "timezone": "UTC",
        "bands": ERA5_LAND_BANDS,
        "conversions": {
            "T2_C": "temperature_2m - 273.15",
            "TD2_C": "dewpoint_temperature_2m - 273.15",
            "SWDOWN_WM2": "surface_solar_radiation_downwards_hourly / 3600",
            "GLW_WM2": "surface_thermal_radiation_downwards_hourly / 3600",
        },
    }


@mcp.tool()
def era5_storage_layout() -> dict[str, Any]:
    """Describe the repository-backed ERA5-Land ROI cache."""
    return {
        "architecture": "GEE -> GitHub Actions -> ROI cache -> Artifact",
        "backend": "github_repository",
        "cache": (
            "data/era5_land/v1/<region>-<bbox_hash>/YYYY/MM/DD/"
            "ERA5LAND_YYYYMMDD_HHMM_UTC.tif"
        ),
        "metadata": "data/metadata/era5-land-index.json",
        "cache_scope": "ROI-specific",
        "cache_crs": "EPSG:4326",
        "cache_bands": list(ERA5_LAND_BANDS.keys()),
        "note": (
            "ERA5-Land is small enough that ROI-level caching is preferred over a China-wide cache. "
            "Repeated requests for the same ROI and hour reuse the repository GeoTIFF."
        ),
    }


@mcp.tool()
def submit_era5_land_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    region_name: str = "",
) -> dict[str, Any]:
    """Submit ERA5-Land hourly ROI acquisition to GitHub Actions."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    hours = int((end - start).total_seconds() // 3600)
    if hours < 1:
        raise ValueError("end_date must be after start_date")
    if hours > 384:
        raise ValueError("One ERA5-Land job is limited to 384 hours (16 days)")

    repo, workflow, ref, token = _github_era5_config()
    job_key = f"era5-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
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
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(
            f"GitHub ERA5-Land dispatch failed: {response.status_code} {response.text[:300]}"
        )
    return {
        "submitted": True,
        "job_key": job_key,
        "repository": repo,
        "workflow": workflow,
        "region_name": region_name or None,
        "hours": hours,
        "bands": list(ERA5_LAND_BANDS.keys()),
        "status_tool": "era5_job_status",
        "result_tool": "get_job_result",
    }


@mcp.tool()
def era5_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted ERA5-Land GitHub Actions job."""
    repo, workflow, _, token = _github_era5_config()
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
def modis_lst_schema() -> dict[str, Any]:
    """Describe MODIS Terra/Aqua daily LST processing and QC."""
    return {
        "datasets": {
            "terra": DATASETS["modis_terra_lst"],
            "aqua": DATASETS["modis_aqua_lst"],
        },
        "spatial_resolution": "1 km",
        "temporal_resolution": "daily product with day/night observations",
        "bands": ["LST_DAY_C", "LST_NIGHT_C", "DAY_VIEW_TIME_LOCAL_H", "NIGHT_VIEW_TIME_LOCAL_H"],
        "qc_rule": "bits 0-1 <= 1; bits 2-3 == 0; bits 6-7 <= 2",
        "lst_conversion": "DN * 0.02 - 273.15",
        "view_time_conversion": "DN * 0.1 hours local solar time",
        "cache": "data/modis_lst/v1/<region>-<bbox_hash>/YYYY/MM/DD/{MOD11A1|MYD11A1}_YYYYMMDD_QC.tif",
    }


@mcp.tool()
def submit_modis_lst_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    region_name: str = "",
    platforms: str = "terra,aqua",
) -> dict[str, Any]:
    """Submit Terra/Aqua MODIS daily LST acquisition and QC processing."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    requested = [x.strip().lower() for x in platforms.split(",") if x.strip()]
    if not requested or any(x not in {"terra", "aqua"} for x in requested):
        raise ValueError("platforms must contain terra and/or aqua")

    repo, workflow, ref, token = _github_modis_config()
    job_key = f"modis-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
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
                "platforms": ",".join(requested),
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(f"GitHub MODIS dispatch failed: {response.status_code} {response.text[:300]}")
    return {
        "submitted": True,
        "job_key": job_key,
        "workflow": workflow,
        "platforms": requested,
        "status_tool": "modis_job_status",
        "result_tool": "get_job_result",
    }


@mcp.tool()
def modis_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted MODIS GitHub Actions job."""
    repo, workflow, _, token = _github_modis_config()
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
    for run in response.json().get("workflow_runs", []):
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
    return {"found": False, "job_key": job_key}


@mcp.tool()
def landsat_schema() -> dict[str, Any]:
    """Describe Landsat 8/9 Collection 2 Level 2 preprocessing."""
    return {
        "datasets": {
            "L8": DATASETS["landsat8_c2_l2"],
            "L9": DATASETS["landsat9_c2_l2"],
        },
        "spatial_resolution": "30 m",
        "bands": ["LST_C", "ST_QA_K", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "lst_conversion": "ST_B10 * 0.00341802 + 149.0 - 273.15",
        "surface_reflectance_conversion": "SR_Bx * 0.0000275 - 0.2",
        "qa_mask": "QA_PIXEL bits 0,1,2,3,4,5,7 == 0 and QA_RADSAT == 0",
        "processing_level": "L2SP",
        "cache": "data/landsat_c2_l2/v1/<region>-<bbox_hash>/YYYY/MM/DD/<LANDSAT_PRODUCT_ID>_L2_QC.tif",
    }


@mcp.tool()
def submit_landsat_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    region_name: str = "",
    satellites: str = "L8,L9",
    cloud_cover_max: float = 80.0,
) -> dict[str, Any]:
    """Submit Landsat 8/9 C2 L2 LST + surface-reflectance acquisition."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    requested = [x.strip().upper() for x in satellites.split(",") if x.strip()]
    if not requested or any(x not in {"L8", "L9"} for x in requested):
        raise ValueError("satellites must contain L8 and/or L9")
    if not 0 <= cloud_cover_max <= 100:
        raise ValueError("cloud_cover_max must be between 0 and 100")

    repo, workflow, ref, token = _github_landsat_config()
    job_key = f"landsat-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
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
                "satellites": ",".join(requested),
                "cloud_cover_max": str(float(cloud_cover_max)),
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(f"GitHub Landsat dispatch failed: {response.status_code} {response.text[:300]}")
    return {
        "submitted": True,
        "job_key": job_key,
        "workflow": workflow,
        "satellites": requested,
        "cloud_cover_max": cloud_cover_max,
        "status_tool": "landsat_job_status",
        "result_tool": "get_job_result",
    }


@mcp.tool()
def landsat_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted Landsat GitHub Actions job."""
    repo, workflow, _, token = _github_landsat_config()
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
    for run in response.json().get("workflow_runs", []):
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
    return {"found": False, "job_key": job_key}


@mcp.tool()
def scaling_factors_schema() -> dict[str, Any]:
    """Describe the LST downscaling scaling-factor library."""
    return {
        "terrain": {
            "dataset": "USGS/SRTMGL1_003",
            "bands": ["DEM_M", "SLOPE_DEG", "ASPECT_DEG"],
            "target_scale_m": 100,
        },
        "landcover": {
            "dataset": "ESA/WorldCover/v200",
            "band": "LANDCOVER",
            "native_scale_m": 10,
            "target_scale_m": 100,
            "reference_year": 2021,
            "note": "quasi-static factor; reference year differs from the 2019 experiment",
        },
        "surface": {
            "datasets": [
                "LANDSAT/LC08/C02/T1_L2",
                "LANDSAT/LC09/C02/T1_L2",
            ],
            "native_scale_m": 30,
            "target_scale_m": 100,
            "resampling": "bilinear",
            "surface_reflectance_conversion": "SR_Bx*0.0000275-0.2",
            "qa_mask": "QA_PIXEL bits 0,1,2,3,4,5 == 0; QA_RADSAT == 0; water retained",
            "composite": "median over requested period",
            "bands": [
                "BLUE",
                "GREEN",
                "RED",
                "NIR",
                "SWIR1",
                "SWIR2",
                "NDVI",
                "EVI",
                "FVC",
                "MNDWI",
                "NDBI",
                "BSI",
                "NDMI",
            ],
        },
        "albedo": {
            "dataset": "MODIS/061/MCD43A3",
            "target_scale_m": 500,
            "bands": ["BSA_SHORTWAVE", "WSA_SHORTWAVE", "ALBEDO_QA"],
            "quality_rule": "BRDF_Albedo_Band_Mandatory_Quality_shortwave <= 1",
            "blue_sky_albedo": "not computed here; combine BSA/WSA with diffuse fraction later",
        },
        "storage": "data/scaling_factors/v1/<region>-<bbox_hash>/...",
    }


@mcp.tool()
def submit_scaling_factors_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    region_name: str = "",
) -> dict[str, Any]:
    """Build or reuse terrain, land-cover, spectral-index and albedo factors."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    start = datetime.fromisoformat(start_date[:10])
    end = datetime.fromisoformat(end_date[:10])
    days = (end - start).days
    if days < 1:
        raise ValueError("end_date must be after start_date")
    if days > 62:
        raise ValueError("One scaling-factor job is limited to 62 days")

    repo, workflow, ref, token = _github_scaling_config()
    job_key = f"scaling-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
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
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(
            f"GitHub scaling-factor dispatch failed: {response.status_code} {response.text[:300]}"
        )
    return {
        "submitted": True,
        "job_key": job_key,
        "workflow": workflow,
        "region_name": region_name or None,
        "days": days,
        "status_tool": "scaling_factors_job_status",
        "result_tool": "get_job_result",
    }


@mcp.tool()
def scaling_factors_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted scaling-factor GitHub Actions job."""
    repo, workflow, _, token = _github_scaling_config()
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
    for run in response.json().get("workflow_runs", []):
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
    return {"found": False, "job_key": job_key}


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
    description="ELITE FY-4A + ERA5-Land + MODIS + Landsat + scaling factors MCP gateway",
    version="0.9.0",
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
        "version": "0.9.0",
        "vercel": bool(os.getenv("VERCEL")),
        "ee_configured": bool(
            os.getenv("EE_SERVICE_ACCOUNT_JSON") or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        ),
        "github_actions_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "gcs_bucket_configured": bool(os.getenv("GEE_GCS_BUCKET")),
        "elite_repo_cache_enabled": True,
        "era5_land_repo_cache_enabled": True,
        "modis_lst_repo_cache_enabled": True,
        "landsat_repo_cache_enabled": True,
        "scaling_factors_repo_cache_enabled": True,
        "job_result_download_enabled": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "auth_enabled": bool(REMOTE_TOKEN),
    }


app.mount("/", mcp_app)
