from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import ee
import numpy as np
import rasterio
import requests
from google.oauth2 import service_account

DATASET = "ECMWF/ERA5_LAND/HOURLY"
CACHE_VERSION = "v1"
NODATA = -9999.0
DEFAULT_PROJECT = "ee-yangxian"

BANDS = [
    ("T2_C", "temperature_2m", "degC"),
    ("TD2_C", "dewpoint_temperature_2m", "degC"),
    ("U10_MPS", "u_component_of_wind_10m", "m s-1"),
    ("V10_MPS", "v_component_of_wind_10m", "m s-1"),
    ("PSFC_PA", "surface_pressure", "Pa"),
    ("SWDOWN_WM2", "surface_solar_radiation_downwards_hourly", "W m-2"),
    ("GLW_WM2", "surface_thermal_radiation_downwards_hourly", "W m-2"),
]


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def hourly_range(start: datetime, end: datetime) -> list[datetime]:
    if end <= start:
        raise ValueError("end_date must be after start_date")
    if any((start.minute, start.second, start.microsecond, end.minute, end.second, end.microsecond)):
        raise ValueError("ERA5-Land requests must use exact hourly boundaries")
    hours: list[datetime] = []
    current = start
    while current < end:
        hours.append(current)
        current += timedelta(hours=1)
    if len(hours) > 384:
        raise ValueError("One ERA5-Land job is limited to 384 hours (16 days)")
    return hours


def parse_bbox(value: str) -> list[float]:
    bbox = [float(x) for x in value.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = bbox
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    return bbox


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()
    return value[:64] or "roi"


def bbox_hash(bbox: list[float]) -> str:
    canonical = ",".join(f"{x:.6f}" for x in bbox)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


def region_id(region_name: str, bbox: list[float]) -> str:
    return f"{slugify(region_name or 'roi')}-{bbox_hash(bbox)}"


def cache_path(rid: str, ts: datetime) -> Path:
    return (
        Path("data")
        / "era5_land"
        / CACHE_VERSION
        / rid
        / f"{ts:%Y}"
        / f"{ts:%m}"
        / f"{ts:%d}"
        / f"ERA5LAND_{ts:%Y%m%d_%H%M}_UTC.tif"
    )


def init_ee() -> str:
    raw = os.getenv("EE_SERVICE_ACCOUNT_JSON", "").strip()
    raw_b64 = os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64", "").strip()
    if not raw and raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        raise RuntimeError(
            "Earth Engine credentials are missing. Add the GitHub Actions secret "
            "EE_SERVICE_ACCOUNT_JSON_BASE64."
        )

    info = json.loads(raw)
    project = os.getenv("EE_PROJECT", "").strip() or info.get("project_id") or DEFAULT_PROJECT
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(credentials, project=project)
    return project


def prepared_image(ts: datetime):
    start = ts.strftime("%Y-%m-%dT%H:%M:%S")
    end = (ts + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    collection = ee.ImageCollection(DATASET).filterDate(start, end)
    count = int(collection.size().getInfo())
    if count < 1:
        raise RuntimeError(f"No ERA5-Land image found for {ts.isoformat()} UTC")

    src = ee.Image(collection.first())
    t2 = src.select("temperature_2m").subtract(273.15).rename("T2_C")
    td2 = src.select("dewpoint_temperature_2m").subtract(273.15).rename("TD2_C")
    u10 = src.select("u_component_of_wind_10m").rename("U10_MPS")
    v10 = src.select("v_component_of_wind_10m").rename("V10_MPS")
    psfc = src.select("surface_pressure").rename("PSFC_PA")
    swdown = (
        src.select("surface_solar_radiation_downwards_hourly")
        .divide(3600.0)
        .rename("SWDOWN_WM2")
    )
    glw = (
        src.select("surface_thermal_radiation_downwards_hourly")
        .divide(3600.0)
        .rename("GLW_WM2")
    )
    return ee.Image.cat([t2, td2, u10, v10, psfc, swdown, glw]).toFloat()


def download_hour(ts: datetime, bbox: list[float], out: Path) -> dict[str, Any]:
    xmin, ymin, xmax, ymax = bbox
    region = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj="EPSG:4326", geodesic=False)
    image = prepared_image(ts)

    # ERA5-Land is about 0.1 degree / 11 km. Keep a near-native sampling;
    # downstream downscaling decides how to resample it onto FY-4A / 1 km grids.
    url = image.getDownloadURL(
        {
            "name": f"ERA5LAND_{ts:%Y%m%d_%H%M}_UTC",
            "region": region,
            "scale": 11132,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )

    tmp_dir = Path("output") / "work" / "era5"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_dir / f"ERA5LAND_{ts:%Y%m%d_%H%M}_raw.tif"

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with raw_path.open("wb") as dst:
            for chunk in response.iter_content(2 * 1024 * 1024):
                if chunk:
                    dst.write(chunk)

    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raw_path) as src:
        data = src.read(masked=True).astype("float32")
        filled = data.filled(NODATA)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=len(BANDS),
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=False,
        )

        with rasterio.open(out, "w", **profile) as dst:
            dst.write(filled)
            for i, (name, source_band, unit) in enumerate(BANDS, start=1):
                dst.set_band_description(i, name)
                dst.update_tags(
                    i,
                    variable=name,
                    source_band=source_band,
                    unit=unit,
                )
            dst.update_tags(
                dataset=DATASET,
                timestamp_utc=ts.isoformat(),
                bbox=",".join(str(x) for x in bbox),
                cache_version=CACHE_VERSION,
                radiation_conversion="hourly_accumulation_Jm2 / 3600 = Wm2",
                temperature_conversion="K - 273.15 = degC",
            )

        stats = {}
        for idx, (name, _, unit) in enumerate(BANDS):
            arr = data[idx].compressed()
            stats[name] = {
                "unit": unit,
                "valid_pixels": int(arr.size),
                "min": float(arr.min()) if arr.size else None,
                "max": float(arr.max()) if arr.size else None,
                "mean": float(arr.mean()) if arr.size else None,
            }

        result = {
            "path": str(out),
            "size_bytes": out.stat().st_size,
            "width": int(src.width),
            "height": int(src.height),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "stats": stats,
        }

    raw_path.unlink(missing_ok=True)
    return result


def load_index(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "product": "ERA5-Land hourly",
        "dataset": DATASET,
        "cache_version": CACHE_VERSION,
        "bands": [
            {"name": name, "source": source, "unit": unit}
            for name, source, unit in BANDS
        ],
        "regions": {},
    }


def save_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--region-name", default="")
    args = parser.parse_args()

    start = parse_dt(args.start_date)
    end = parse_dt(args.end_date)
    hours = hourly_range(start, end)
    bbox = parse_bbox(args.bbox)
    rid = region_id(args.region_name, bbox)

    project = init_ee()

    index_path = Path("data") / "metadata" / "era5-land-index.json"
    index = load_index(index_path)
    region_meta = index["regions"].setdefault(
        rid,
        {
            "region_name": args.region_name or None,
            "bbox": bbox,
            "hours": {},
        },
    )

    output_dir = Path("output") / "era5_land" / rid
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_hits = 0
    cache_created = 0
    outputs: list[dict[str, Any]] = []

    for ts in hours:
        cache = cache_path(rid, ts)
        if cache.exists():
            cache_hits += 1
            created = False
            cache_stats = None
        else:
            cache_stats = download_hour(ts, bbox, cache)
            cache_created += 1
            created = True
            region_meta["hours"][ts.isoformat()] = str(cache)

        artifact_path = output_dir / cache.name
        shutil.copy2(cache, artifact_path)

        outputs.append(
            {
                "timestamp_utc": ts.isoformat(),
                "cache_path": str(cache),
                "artifact_path": str(artifact_path),
                "cache_created": created,
                "details": cache_stats,
            }
        )

    if cache_created:
        index["updated_at"] = datetime.utcnow().isoformat() + "Z"
        save_index(index_path, index)
    elif not index_path.exists():
        save_index(index_path, index)

    result = {
        "product": "ERA5-Land hourly",
        "dataset": DATASET,
        "project": project,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "timezone": "UTC",
        "region_name": args.region_name or None,
        "region_id": rid,
        "bbox": bbox,
        "hours_requested": len(hours),
        "cache_hits": cache_hits,
        "cache_created": cache_created,
        "cache_root": f"data/era5_land/{CACHE_VERSION}/{rid}",
        "bands": [
            {"name": name, "source": source, "unit": unit}
            for name, source, unit in BANDS
        ],
        "files": outputs,
    }
    Path("output").mkdir(exist_ok=True)
    (Path("output") / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shutil.rmtree(Path("output") / "work", ignore_errors=True)


if __name__ == "__main__":
    main()
