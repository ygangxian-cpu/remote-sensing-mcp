from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
TPDC_ANCFDS_DATASET_ID = "4adbc070-afb3-4e9c-85a0-2ce68d1388ad"
TPDC_ANCFDS_DOI = "10.11888/RemoteSen.tpdc.303249"
TPDC_ANCFDS_PAGE = f"https://data.tpdc.ac.cn/en/data/{TPDC_ANCFDS_DATASET_ID}"
ERA5_LAND_BANDS = {
    "T2_C": {"source": "temperature_2m", "unit": "degC"},
    "TD2_C": {"source": "dewpoint_temperature_2m", "unit": "degC"},
    "U10_MPS": {"source": "u_component_of_wind_10m", "unit": "m s-1"},
    "V10_MPS": {"source": "v_component_of_wind_10m", "unit": "m s-1"},
    "PSFC_PA": {"source": "surface_pressure", "unit": "Pa"},
    "SWDOWN_WM2": {"source": "surface_solar_radiation_downwards_hourly", "unit": "W m-2"},
    "GLW_WM2": {"source": "surface_thermal_radiation_downwards_hourly", "unit": "W m-2"},
}
MODIS_L2_SWATH_PRODUCTS = {"terra": "MOD11_L2", "aqua": "MYD11_L2"}
ASTER_AST08_SHORT_NAME = "AST_08"
ASTER_AST08_VERSION = "004"
ASTER_CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
ASTER_EARTHDATA_SEARCH_URL = "https://search.earthdata.nasa.gov/search?q=AST_08"

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
        "Remote-sensing data gateway for ELITE FY-4A, TPDC ANCFDS-LST, ERA5-Land, MODIS, Landsat, ASTER AST_08 and scaling factors. "
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


def _bbox_values(bbox: list[float]) -> list[float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin, ymin, xmax, ymax]")
    xmin, ymin, xmax, ymax = map(float, bbox)
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    return [xmin, ymin, xmax, ymax]


def _cmr_utc(value: str) -> str:
    raw = str(value).strip()
    if re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", raw):
        datetime.fromisoformat(raw)
        return f"{raw}T00:00:00Z"
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _aster_ast08_search_payload(
    start_date: str,
    end_date: str,
    bbox: list[float],
    page_size: int = 100,
) -> dict[str, Any]:
    if not 1 <= int(page_size) <= 2000:
        raise ValueError("page_size must be between 1 and 2000")
    xmin, ymin, xmax, ymax = _bbox_values(bbox)
    start_utc = _cmr_utc(start_date)
    end_utc = _cmr_utc(end_date)
    if datetime.fromisoformat(end_utc.replace("Z", "+00:00")) <= datetime.fromisoformat(
        start_utc.replace("Z", "+00:00")
    ):
        raise ValueError("end_date must be after start_date")

    response = requests.get(
        ASTER_CMR_GRANULES_URL,
        params={
            "short_name": ASTER_AST08_SHORT_NAME,
            "version": ASTER_AST08_VERSION,
            "provider": "LPCLOUD",
            "bounding_box": f"{xmin},{ymin},{xmax},{ymax}",
            "temporal": f"{start_utc},{end_utc}",
            "page_size": int(page_size),
        },
        headers={
            "Accept": "application/json",
            "User-Agent": "remote-sensing-mcp/aster-ast08",
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    entries = ((payload.get("feed") or {}).get("entry") or [])
    scenes: list[dict[str, Any]] = []
    for entry in entries:
        links = []
        direct_download_links = []
        browse_links = []
        for item in entry.get("links") or []:
            href = item.get("href")
            if not href:
                continue
            rel = str(item.get("rel") or "")
            title = str(item.get("title") or "")
            link = {
                "href": href,
                "title": title or None,
                "rel": rel or None,
                "type": item.get("type"),
            }
            links.append(link)
            rel_lower = rel.lower()
            title_lower = title.lower()
            if "data#" in rel_lower or "get data" in title_lower or "download" in title_lower:
                direct_download_links.append(href)
            if "browse#" in rel_lower or "browse" in title_lower or "visualization" in rel_lower:
                browse_links.append(href)
        scenes.append(
            {
                "concept_id": entry.get("id"),
                "granule_ur": entry.get("title"),
                "producer_granule_id": entry.get("producer_granule_id"),
                "start_time": entry.get("time_start"),
                "end_time": entry.get("time_end"),
                "day_night_flag": entry.get("day_night_flag"),
                "cloud_cover": entry.get("cloud_cover"),
                "boxes": entry.get("boxes") or [],
                "polygons": entry.get("polygons") or [],
                "direct_download_links": direct_download_links,
                "browse_links": browse_links,
                "links": links,
            }
        )
    try:
        total_hits = int(response.headers.get("CMR-Hits", len(scenes)))
    except (TypeError, ValueError):
        total_hits = len(scenes)
    return {
        "product": "ASTER L2 Surface Kinetic Temperature",
        "short_name": ASTER_AST08_SHORT_NAME,
        "version": ASTER_AST08_VERSION,
        "provider": "LPCLOUD",
        "start_date": start_utc,
        "end_date": end_utc,
        "bbox_wgs84": [xmin, ymin, xmax, ymax],
        "total_hits": total_hits,
        "returned": len(scenes),
        "scenes": scenes,
        "production_mode": "on-demand",
        "availability_semantics": (
            "A CMR match means an ASTER observation intersects the requested space/time and can "
            "be selected for AST_08 on-demand processing. It does not guarantee that a ready-to-"
            "download AST_08 file has already been materialized."
        ),
        "earthdata_search_url": ASTER_EARTHDATA_SEARCH_URL,
    }


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


def _github_modis_l2_swath_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv(
        "GITHUB_MODIS_L2_SWATH_WORKFLOW_ID",
        "remote-sensing-modis-l2-swath.yml",
    )
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


def _github_tpdc_ancfds_config() -> tuple[str, str, str, str]:
    repo = os.getenv("GITHUB_WORKFLOW_REPOSITORY", "ygangxian-cpu/remote-sensing-mcp")
    workflow = os.getenv("GITHUB_TPDC_ANCFDS_WORKFLOW_ID", "remote-sensing-tpdc-ancfds.yml")
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
    artifact = _artifact_for_run(repo, token, int(run["id"]), job_key)
    if artifact is None:
        if conclusion != "success":
            result["message"] = (
                "The workflow did not succeed and no matching result artifact was found."
            )
        else:
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
            "workflow_succeeded": conclusion == "success",
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
    if conclusion != "success":
        result["warning"] = (
            "The workflow conclusion is not success, but a non-expired result artifact exists. "
            "The artifact is still downloadable; inspect result.json for cache-persistence warnings."
        )
    return result


@mcp.tool()
def aster_ast08_schema() -> dict[str, Any]:
    """Describe official ASTER AST_08 Level-2 surface kinetic temperature access."""
    return {
        "product": "ASTER L2 Surface Kinetic Temperature",
        "short_name": ASTER_AST08_SHORT_NAME,
        "version": ASTER_AST08_VERSION,
        "provider": "NASA LP DAAC / LPCLOUD",
        "native_spatial_resolution": "90 m",
        "unit": "kelvin",
        "product_level": 2,
        "production_mode": "on-demand",
        "catalog": "NASA EOSDIS Common Metadata Repository (CMR)",
        "catalog_search_requires_auth": False,
        "order_requires_earthdata_login": True,
        "order_interface": "NASA Earthdata Search",
        "earthdata_search_url": ASTER_EARTHDATA_SEARCH_URL,
        "note": (
            "The MCP can discover orderable AST_08 scenes directly through public CMR. "
            "NASA documents Earthdata Search as the ordering interface for on-demand higher-level "
            "ASTER products, so this integration does not pretend that catalog hits are already "
            "materialized downloads."
        ),
    }


@mcp.tool()
def search_aster_ast08_scenes(
    start_date: str,
    end_date: str,
    bbox: list[float],
    page_size: int = 100,
) -> dict[str, Any]:
    """Search ASTER AST_08 orderable scenes by UTC interval and WGS84 bbox using NASA CMR.

    Use an exclusive end time/date for a clean daily query, for example
    2019-09-24 to 2019-09-25 for the UTC day 2019-09-24.
    """
    return _aster_ast08_search_payload(start_date, end_date, bbox, page_size)


@mcp.tool()
def plan_aster_ast08_order(
    start_date: str,
    end_date: str,
    bbox: list[float],
    page_size: int = 100,
) -> dict[str, Any]:
    """Prepare an AST_08 Earthdata Search order plan for matching CMR scenes."""
    result = _aster_ast08_search_payload(start_date, end_date, bbox, page_size)
    return {
        **result,
        "automated_order_submitted": False,
        "order_reason": (
            "AST_08 is an on-demand higher-level ASTER product. NASA's supported ordering "
            "workflow is Earthdata Search with an authenticated Earthdata Login session."
        ),
        "order_next_step": (
            "Open earthdata_search_url, sign in to Earthdata, select AST_08, reapply the returned "
            "space/time filter, select the matching granules, and submit the on-demand order."
        ),
        "post_order_note": (
            "When NASA finishes processing, the order notification provides download links. "
            "Those generated files can then be cropped/reprojected to the experiment ROI."
        ),
    }


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
        "vercel_direct_earth_engine_configured": bool(
            os.getenv("EE_SERVICE_ACCOUNT_JSON") or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        ),
        "vercel_direct_earth_engine_note": (
            "This only describes direct Earth Engine access from Vercel. "
            "GitHub Actions workers authenticate separately with repository secrets."
        ),
        "elite_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "era5_land_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "modis_lst_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "landsat_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "tpdc_ancfds_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "scaling_factors_worker_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "aster_ast08_cmr_search": True,
        "aster_ast08_order_mode": "Earthdata Search on-demand",
        "era5_land_cache_backend": "github_repository_roi",
        "modis_lst_cache_backend": "github_repository_roi",
        "modis_lst_cache_version": "v2",
        "modis_quality_reporting": True,
        "landsat_cache_backend": "github_repository_roi",
        "landsat_cache_version": "v3",
        "landsat_cache_scope": "validation_lst_qa",
        "landsat_quality_reporting": True,
        "tpdc_ancfds_public_api": True,
        "tpdc_ancfds_cache_backend": "github_repository_roi",
        "tpdc_ancfds_cache_version": "v1",
        "qa_preserving_downloads": True,
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
def modis_l2_swath_schema() -> dict[str, Any]:
    """Describe exact-time MODIS Terra/Aqua 5-minute L2 swath LST acquisition."""
    return {
        "products": MODIS_L2_SWATH_PRODUCTS,
        "collection": "061",
        "source": "NASA LAADS DAAC",
        "source_type": "5-minute Level-2 swath",
        "native_spatial_resolution": "1 km",
        "time_standard": {
            "granule_filename": "UTC start time encoded as HHMM",
            "View_time": "per-pixel local solar time, DN*0.1 hour",
            "derived_output": "VIEW_TIME_UTC_H = local solar time - longitude/15, modulo 24",
        },
        "source_sds": [
            "LST",
            "QC",
            "Error_LST",
            "Emis_31",
            "Emis_32",
            "View_angle",
            "View_time",
            "Latitude",
            "Longitude",
        ],
        "output_bands": [
            "LST_C or LST_K",
            "QC",
            "ERROR_LST_K",
            "EMIS_31",
            "EMIS_32",
            "VIEW_ZENITH_DEG",
            "VIEW_TIME_LOCAL_H",
            "VIEW_TIME_UTC_H",
            "SOURCE_LAT",
            "SOURCE_LON",
        ],
        "conversions": {
            "LST": "DN*0.02 K; Celsius subtracts 273.15",
            "Error_LST": "DN*0.04 K",
            "Emis_31_32": "DN*0.002+0.49",
            "View_angle": "DN*0.5 degree",
            "View_time": "DN*0.1 local-solar hour",
        },
        "geolocation": (
            "Latitude/Longitude are stored every 5 scan lines/samples. "
            "The worker honors MOD11_L2 offset=2, increment=5 and then "
            "nearest-regrids the swath to the requested regular WGS84 ROI grid."
        ),
        "recommended_strict_qc": "bits 0-1 <= 1; bits 2-3 == 0; bits 6-7 <= 2",
        "authentication": (
            "Historical LAADS HDF download requires a NASA Earthdata/LAADS bearer token "
            "stored in the GitHub Actions repository secret LAADS_TOKEN "
            "(EARTHDATA_TOKEN/NASA_EARTHDATA_TOKEN/EDL_TOKEN are accepted fallbacks)."
        ),
        "cache_version": "v1",
        "cache": (
            "data/modis_l2_swath/v1/<region>-<bbox_hash>/YYYY/MM/DD/"
            "<MOD11_L2|MYD11_L2>_YYYYMMDD_HHMM_UTC_<C|K>.tif"
        ),
        "raw_hdf_policy": "temporary for cache; included in short-lived job Artifact for provenance",
    }


@mcp.tool()
def submit_modis_l2_swath_job(
    date: str,
    bbox: list[float],
    target_time_utc: str,
    region_name: str = "",
    platform: str = "terra",
    time_window_minutes: int = 15,
    output_unit: str = "celsius",
) -> dict[str, Any]:
    """Submit exact-time MOD11_L2/MYD11_L2 swath acquisition from NASA LAADS."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    platform = platform.strip().lower()
    if platform not in MODIS_L2_SWATH_PRODUCTS:
        raise ValueError("platform must be terra or aqua")
    if output_unit not in {"celsius", "kelvin"}:
        raise ValueError("output_unit must be celsius or kelvin")
    if not 0 <= int(time_window_minutes) <= 180:
        raise ValueError("time_window_minutes must be between 0 and 180")
    datetime.fromisoformat(date[:10])
    try:
        hh, mm = [int(x) for x in target_time_utc.split(":")]
    except Exception as exc:
        raise ValueError("target_time_utc must be HH:MM") from exc
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("target_time_utc must be HH:MM")

    repo, workflow, ref, token = _github_modis_l2_swath_config()
    job_key = f"modis-l2-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        timeout=30,
        headers=_github_headers(token),
        json={
            "ref": ref,
            "inputs": {
                "job_key": job_key,
                "date": date[:10],
                "bbox": ",".join(str(float(x)) for x in bbox),
                "region_name": region_name,
                "platform": platform,
                "target_time_utc": f"{hh:02d}:{mm:02d}",
                "time_window_minutes": str(int(time_window_minutes)),
                "output_unit": output_unit,
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(
            f"GitHub MODIS L2 swath dispatch failed: {response.status_code} {response.text[:300]}"
        )
    return {
        "submitted": True,
        "job_key": job_key,
        "repository": repo,
        "workflow": workflow,
        "product": MODIS_L2_SWATH_PRODUCTS[platform],
        "platform": platform,
        "date": date[:10],
        "target_time_utc": f"{hh:02d}:{mm:02d}",
        "time_window_minutes": int(time_window_minutes),
        "region_name": region_name or None,
        "status_tool": "modis_l2_swath_job_status",
        "result_tool": "get_job_result",
        "source": "NASA LAADS DAAC",
    }


@mcp.tool()
def modis_l2_swath_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted MODIS L2 swath GitHub Actions job."""
    repo, workflow, _, token = _github_modis_l2_swath_config()
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs",
        params={"event": "workflow_dispatch", "per_page": 50},
        timeout=30,
        headers=_github_headers(token),
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
def modis_lst_schema() -> dict[str, Any]:
    """Describe QA-preserving MODIS Terra/Aqua daily LST acquisition."""
    return {
        "datasets": {
            "terra": DATASETS["modis_terra_lst"],
            "aqua": DATASETS["modis_aqua_lst"],
        },
        "spatial_resolution": "1 km",
        "temporal_resolution": "daily product with day/night observations",
        "bands": [
            "LST_DAY_C",
            "LST_NIGHT_C",
            "DAY_VIEW_TIME_LOCAL_H",
            "NIGHT_VIEW_TIME_LOCAL_H",
            "QC_DAY",
            "QC_NIGHT",
        ],
        "download_policy": "preserve native product availability; no additional research QA mask",
        "qa_preserved": True,
        "recommended_strict_qc": "bits 0-1 <= 1; bits 2-3 == 0; bits 6-7 <= 2",
        "quality_report": (
            "result.json reports native LST coverage, strict-QC coverage, retention ratio, "
            "mandatory-QA counts, data-quality counts and LST-error classes for day/night"
        ),
        "lst_conversion": "DN * 0.02 - 273.15",
        "view_time_conversion": "DN * 0.1 hours local solar time",
        "cache_version": "v2",
        "cache": "data/modis_lst/v2/<region>-<bbox_hash>/YYYY/MM/DD/{MOD11A1|MYD11A1}_YYYYMMDD_LST_QA.tif",
    }


@mcp.tool()
def submit_modis_lst_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    region_name: str = "",
    platforms: str = "terra,aqua",
) -> dict[str, Any]:
    """Submit QA-preserving Terra/Aqua MODIS daily LST acquisition."""
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
    """Describe QA-preserving Landsat 8/9 Collection 2 Level 2 acquisition."""
    return {
        "datasets": {
            "L8": DATASETS["landsat8_c2_l2"],
            "L9": DATASETS["landsat9_c2_l2"],
        },
        "spatial_resolution": "30 m",
        "bands": [
            "LST_C",
            "ST_QA_K",
            "QA_PIXEL",
            "QA_RADSAT",
        ],
        "lst_conversion": "ST_B10 * 0.00341802 + 149.0 - 273.15",
        "download_policy": (
            "validation-oriented LST/QA cache; preserve native product availability "
            "with no additional QA mask"
        ),
        "spectral_factors": (
            "Reflective SR bands are intentionally excluded from the persistent Landsat cache. "
            "Use scaling_factors_schema / submit_scaling_factors_job for SR-derived predictors."
        ),
        "qa_preserved": True,
        "water_preserved": True,
        "recommended_clear_mask": "QA_PIXEL bits 0,1,2,3,4,5 == 0; water bit 7 is retained",
        "qa_radsat_policy": "preserved and reported separately; not used to erase LST",
        "quality_report": (
            "result.json reports native LST coverage, clear-LST coverage, individual QA bit counts, "
            "water fraction and radiometric saturation"
        ),
        "processing_level": "L2SP",
        "cache_version": "v3",
        "cache_scope": "validation_lst_qa",
        "cache": "data/landsat_c2_l2/v3/<region>-<bbox_hash>/YYYY/MM/DD/<LANDSAT_PRODUCT_ID>_L2_LST_QA.tif",
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
    """Submit Landsat 8/9 C2 L2 validation LST + QA acquisition."""
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
def tpdc_ancfds_lst_schema() -> dict[str, Any]:
    """Describe the public TPDC ANCFDS-LST hourly 0.01-degree product."""
    return {
        "dataset": "ANCFDS-LST",
        "doi": TPDC_ANCFDS_DOI,
        "dataset_id": TPDC_ANCFDS_DATASET_ID,
        "dataset_page": TPDC_ANCFDS_PAGE,
        "coverage": "2018-2023",
        "temporal_resolution": "1 hour",
        "time_standard": "UTC",
        "spatial_resolution": "0.01 degree",
        "crs": "EPSG:4326",
        "source_format": "GeoTIFF",
        "source_dtype": "uint16",
        "source_nodata": 0,
        "source_scale_factor": 0.1,
        "source_unit": "kelvin",
        "bands": {
            "1": "T_dir: directional FY-4A/AGRI-view LST",
            "2": "T_nadir: angular-normalized nadir LST",
            "3": "T_hemi: hemispherical-equivalent LST",
        },
        "access": "TPDC public file API; file listing and file download work without TPDC login",
        "mcp_job_limit": "24 source hours per job",
        "output_units": ["celsius", "kelvin"],
    }


@mcp.tool()
def tpdc_ancfds_storage_layout() -> dict[str, Any]:
    """Describe TPDC ANCFDS-LST download, crop and repository-cache behavior."""
    return {
        "architecture": "TPDC public full-domain GeoTIFF -> temporary Actions download -> ROI crop -> GitHub ROI cache -> artifact",
        "raw_source_storage": "temporary only; full-domain source GeoTIFF is deleted after ROI crop",
        "source_size": "typically about 100-140 MB per hourly file",
        "cache": (
            "data/tpdc_ancfds/v1/<region>-<bbox_hash>/YYYY/MM/DD/"
            "ANCFDS_FY4A_YYYYMMDD_HH00_<C|K>.tif"
        ),
        "cache_backend": "github_repository_roi",
        "artifact": "requested ROI GeoTIFFs plus result.json",
        "source_dataset_id": TPDC_ANCFDS_DATASET_ID,
        "source_doi": TPDC_ANCFDS_DOI,
    }


@mcp.tool()
def submit_tpdc_ancfds_lst_job(
    start_date: str,
    end_date: str,
    bbox: list[float],
    region_name: str = "",
    hours: list[int] | None = None,
    output_unit: str = "celsius",
) -> dict[str, Any]:
    """Submit TPDC ANCFDS-LST hourly downloads and WGS84 ROI crops to GitHub Actions."""
    if len(bbox) != 4:
        raise ValueError("bbox must be [xmin,ymin,xmax,ymax]")
    xmin, ymin, xmax, ymax = map(float, bbox)
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    if output_unit not in {"celsius", "kelvin"}:
        raise ValueError("output_unit must be celsius or kelvin")

    start = datetime.fromisoformat(start_date[:10])
    end = datetime.fromisoformat(end_date[:10])
    if end <= start:
        raise ValueError("end_date must be after start_date")
    day_count = (end - start).days
    hour_values = list(range(24)) if hours is None else sorted(set(int(x) for x in hours))
    if not hour_values or any(hour < 0 or hour > 23 for hour in hour_values):
        raise ValueError("hours must contain integers from 0 to 23")
    source_hours = day_count * len(hour_values)
    if source_hours > 24:
        raise ValueError(
            "One ANCFDS-LST job is limited to 24 source hours. "
            "Use a shorter date range or select fewer hours."
        )

    repo, workflow, ref, token = _github_tpdc_ancfds_config()
    job_key = f"ancfds-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    response = requests.post(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches",
        timeout=30,
        headers=_github_headers(token),
        json={
            "ref": ref,
            "inputs": {
                "job_key": job_key,
                "start_date": start_date,
                "end_date": end_date,
                "bbox": ",".join(str(float(x)) for x in bbox),
                "region_name": region_name,
                "hours": "all" if hours is None else ",".join(str(x) for x in hour_values),
                "output_unit": output_unit,
            },
        },
    )
    if response.status_code != 204:
        raise RuntimeError(
            f"GitHub TPDC ANCFDS dispatch failed: {response.status_code} {response.text[:300]}"
        )
    return {
        "submitted": True,
        "job_key": job_key,
        "repository": repo,
        "workflow": workflow,
        "dataset": "ANCFDS-LST",
        "doi": TPDC_ANCFDS_DOI,
        "region_name": region_name or None,
        "source_hours": source_hours,
        "hours_utc": hour_values,
        "output_unit": output_unit,
        "status_tool": "tpdc_ancfds_job_status",
        "result_tool": "get_job_result",
        "storage_architecture": (
            "TPDC full-domain temporary download -> GitHub ROI cache -> Actions artifact"
        ),
    }


@mcp.tool()
def tpdc_ancfds_job_status(job_key: str) -> dict[str, Any]:
    """Look up a submitted TPDC ANCFDS-LST GitHub Actions job."""
    repo, workflow, _, token = _github_tpdc_ancfds_config()
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs",
        params={"event": "workflow_dispatch", "per_page": 50},
        timeout=30,
        headers=_github_headers(token),
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
    """Validate direct Earth Engine access from Vercel only.

    This is not a readiness check for MODIS/Landsat/ERA5 GitHub Actions workers,
    which authenticate independently with repository secrets.
    """
    note = (
        "Scope is Vercel-direct Earth Engine only. A false result does not imply "
        "that GitHub Actions MODIS/Landsat/ERA5 workers are unavailable."
    )
    try:
        used_project = _initialize_ee(project)
        ee.Number(1).getInfo()
        return {
            "ok": True,
            "scope": "vercel_direct_earth_engine",
            "project": used_project,
            "github_actions_workers_checked": False,
            "note": note,
        }
    except Exception as exc:
        return {
            "ok": False,
            "scope": "vercel_direct_earth_engine",
            "github_actions_workers_checked": False,
            "error": str(exc),
            "note": note,
        }


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
    description="ELITE FY-4A + TPDC ANCFDS-LST + ERA5-Land + MODIS + Landsat + scaling factors MCP gateway",
    version="0.11.1",
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
        "version": "0.11.1",
        "vercel": bool(os.getenv("VERCEL")),
        "ee_configured": bool(
            os.getenv("EE_SERVICE_ACCOUNT_JSON") or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        ),
        "vercel_direct_ee_configured": bool(
            os.getenv("EE_SERVICE_ACCOUNT_JSON") or os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64")
        ),
        "vercel_direct_ee_scope_note": (
            "GitHub Actions Earth Engine workers authenticate separately; "
            "this field does not represent worker readiness."
        ),
        "github_actions_dispatch_configured": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "gcs_bucket_configured": bool(os.getenv("GEE_GCS_BUCKET")),
        "elite_repo_cache_enabled": True,
        "era5_land_repo_cache_enabled": True,
        "modis_lst_repo_cache_enabled": True,
        "landsat_repo_cache_enabled": True,
        "landsat_cache_version": "v3",
        "landsat_cache_scope": "validation_lst_qa",
        "tpdc_ancfds_repo_cache_enabled": True,
        "tpdc_ancfds_public_api_enabled": True,
        "scaling_factors_repo_cache_enabled": True,
        "job_result_download_enabled": bool(os.getenv("GITHUB_WORKFLOW_TOKEN")),
        "auth_enabled": bool(REMOTE_TOKEN),
    }


app.mount("/", mcp_app)
