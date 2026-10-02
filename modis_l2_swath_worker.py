from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from pyhdf.SD import SD, SDC
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

LAADS_ROOT = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61"
CACHE_VERSION = "v1"
NODATA = -9999.0
PRODUCTS = {"terra": "MOD11_L2", "aqua": "MYD11_L2"}


def parse_bbox(value: str) -> list[float]:
    bbox = [float(x) for x in value.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = bbox
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    return bbox


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-_").lower()[:64] or "roi"


def bbox_hash(bbox: list[float]) -> str:
    canonical = ",".join(f"{x:.6f}" for x in bbox)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]


def region_id(region_name: str, bbox: list[float]) -> str:
    return f"{slugify(region_name or 'roi')}-{bbox_hash(bbox)}"


def parse_hhmm(value: str) -> int:
    m = re.fullmatch(r"(\d{2}):(\d{2})", value.strip())
    if not m:
        raise ValueError("target_time_utc must be HH:MM")
    h, minute = map(int, m.groups())
    if h > 23 or minute > 59:
        raise ValueError("Invalid target_time_utc")
    return h * 60 + minute


def circular_minute_distance(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 1440 - d)


def laads_token() -> str:
    return (
        os.getenv("LAADS_TOKEN", "").strip()
        or os.getenv("EARTHDATA_TOKEN", "").strip()
        or os.getenv("NASA_EARTHDATA_TOKEN", "").strip()
        or os.getenv("EDL_TOKEN", "").strip()
    )


def list_granules(product: str, day: datetime) -> list[dict[str, Any]]:
    doy = day.timetuple().tm_yday
    url = f"{LAADS_ROOT}/{product}/{day:%Y}/{doy:03d}/"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    pattern = re.compile(
        rf'({re.escape(product)}\.A{day:%Y}{doy:03d}\.(\d{{4}})\.061\.[A-Za-z0-9.]+\.hdf)'
    )
    found = {}
    for filename, hhmm in pattern.findall(response.text):
        found[filename] = {
            "filename": filename,
            "hhmm": hhmm,
            "minute_of_day": int(hhmm[:2]) * 60 + int(hhmm[2:]),
            "url": f"{url}{filename}",
        }
    return sorted(found.values(), key=lambda x: x["minute_of_day"])


def candidate_granules(
    granules: list[dict[str, Any]],
    target_minute: int,
    window_minutes: int,
) -> list[dict[str, Any]]:
    out = []
    for g in granules:
        delta = circular_minute_distance(g["minute_of_day"], target_minute)
        if delta <= window_minutes:
            out.append({**g, "target_offset_minutes": delta})
    return sorted(out, key=lambda x: (x["target_offset_minutes"], x["minute_of_day"]))


def download_granule(url: str, out: Path, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=240, allow_redirects=True) as response:
        if response.status_code in {401, 403}:
            raise PermissionError(
                "LAADS download requires an Earthdata/LAADS bearer token. "
                "Configure repository secret LAADS_TOKEN (or EARTHDATA_TOKEN)."
            )
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            raise PermissionError(
                "LAADS returned HTML instead of HDF; authentication is likely missing or invalid."
            )
        with out.open("wb") as dst:
            for chunk in response.iter_content(2 * 1024 * 1024):
                if chunk:
                    dst.write(chunk)
    if out.stat().st_size < 50_000:
        raise RuntimeError(f"Downloaded granule is unexpectedly small: {out.stat().st_size} bytes")


def sds_array(hdf: SD, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    sds = hdf.select(name)
    arr = np.asarray(sds.get())
    attrs = sds.attributes()
    sds.endaccess()
    return arr, attrs


def scaled(
    raw: np.ndarray,
    scale: float,
    offset: float = 0.0,
    fill: int | float | None = 0,
    valid_min: float | None = None,
    valid_max: float | None = None,
) -> np.ndarray:
    r = raw.astype(np.float32)
    valid = np.isfinite(r)
    if fill is not None:
        valid &= r != float(fill)
    if valid_min is not None:
        valid &= r >= valid_min
    if valid_max is not None:
        valid &= r <= valid_max
    out = np.full(r.shape, np.nan, dtype=np.float32)
    out[valid] = r[valid] * scale + offset
    return out


def interpolate_geolocation(lat5: np.ndarray, lon5: np.ndarray, shape1: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map 5-km geolocation SDS to the 1-km SDS grid.

    MOD11_L2 V6.1 specifies offset=2 and increment=5: geolocation element (0,0)
    corresponds to LST pixel (2,2).
    """
    h, w = shape1
    lat = np.empty((h, w), dtype=np.float32)
    lon = np.empty((h, w), dtype=np.float32)
    cols = (np.arange(w, dtype=np.float32) - 2.0) / 5.0
    chunk = 128
    for r0 in range(0, h, chunk):
        r1 = min(h, r0 + chunk)
        rows = (np.arange(r0, r1, dtype=np.float32) - 2.0) / 5.0
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        coords = np.vstack([rr.ravel(), cc.ravel()])
        lat[r0:r1] = map_coordinates(lat5, coords, order=1, mode="nearest").reshape(r1-r0, w)
        lon[r0:r1] = map_coordinates(lon5, coords, order=1, mode="nearest").reshape(r1-r0, w)
    return lat, lon


def read_hdf(path: Path, output_unit: str) -> dict[str, np.ndarray]:
    hdf = SD(str(path), SDC.READ)
    try:
        lst_raw, _ = sds_array(hdf, "LST")
        qc_raw, _ = sds_array(hdf, "QC")
        err_raw, _ = sds_array(hdf, "Error_LST")
        e31_raw, _ = sds_array(hdf, "Emis_31")
        e32_raw, _ = sds_array(hdf, "Emis_32")
        va_raw, _ = sds_array(hdf, "View_angle")
        vt_raw, _ = sds_array(hdf, "View_time")
        lat5, _ = sds_array(hdf, "Latitude")
        lon5, _ = sds_array(hdf, "Longitude")
    finally:
        hdf.end()

    lst_k = scaled(lst_raw, 0.02, fill=0, valid_min=7500, valid_max=65535)
    lst = lst_k - 273.15 if output_unit == "celsius" else lst_k
    err = scaled(err_raw, 0.04, fill=0, valid_min=1, valid_max=255)
    e31 = scaled(e31_raw, 0.002, 0.49, fill=0, valid_min=1, valid_max=255)
    e32 = scaled(e32_raw, 0.002, 0.49, fill=0, valid_min=1, valid_max=255)
    va = scaled(va_raw, 0.5, fill=255, valid_min=0, valid_max=180)
    vt = scaled(vt_raw, 0.1, fill=255, valid_min=0, valid_max=240)
    lat, lon = interpolate_geolocation(
        np.asarray(lat5, dtype=np.float32),
        np.asarray(lon5, dtype=np.float32),
        lst_raw.shape,
    )
    utc_h = np.mod(vt - lon / 15.0, 24.0).astype(np.float32)
    qc = qc_raw.astype(np.float32)
    qc[~np.isfinite(lst)] = np.nan

    return {
        "LST": lst.astype(np.float32),
        "QC": qc,
        "ERROR_LST_K": err,
        "EMIS_31": e31,
        "EMIS_32": e32,
        "VIEW_ZENITH_DEG": va,
        "VIEW_TIME_LOCAL_H": vt,
        "VIEW_TIME_UTC_H": utc_h,
        "LAT": lat,
        "LON": lon,
    }


def strict_qc_mask(qc: np.ndarray, lst: np.ndarray) -> np.ndarray:
    q = np.where(np.isfinite(qc), qc, 0).astype(np.uint16)
    return (
        np.isfinite(lst)
        & np.isfinite(qc)
        & ((q & 3) <= 1)
        & (((q >> 2) & 3) == 0)
        & (((q >> 6) & 3) <= 2)
    )


def swath_intersects(data: dict[str, np.ndarray], bbox: list[float], margin: float = 0.03) -> bool:
    xmin, ymin, xmax, ymax = bbox
    lat, lon = data["LAT"], data["LON"]
    mask = (
        np.isfinite(data["LST"])
        & np.isfinite(lat) & np.isfinite(lon)
        & (lon >= xmin-margin) & (lon <= xmax+margin)
        & (lat >= ymin-margin) & (lat <= ymax+margin)
    )
    return bool(mask.any())


def regularize_roi(
    data: dict[str, np.ndarray],
    bbox: list[float],
    resolution_deg: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    xmin, ymin, xmax, ymax = bbox
    width = max(1, int(math.ceil((xmax - xmin) / resolution_deg)))
    height = max(1, int(math.ceil((ymax - ymin) / resolution_deg)))
    transform = rasterio.transform.from_origin(xmin, ymax, resolution_deg, resolution_deg)

    lat, lon = data["LAT"], data["LON"]
    margin = max(0.04, resolution_deg * 4)
    src_mask = (
        np.isfinite(data["LST"]) & np.isfinite(lat) & np.isfinite(lon)
        & (lon >= xmin-margin) & (lon <= xmax+margin)
        & (lat >= ymin-margin) & (lat <= ymax+margin)
    )
    if int(src_mask.sum()) < 4:
        raise RuntimeError("Granule does not contain enough valid LST pixels near requested ROI")

    mean_lat = (ymin + ymax) / 2.0
    coslat = max(0.2, math.cos(math.radians(mean_lat)))
    src_xy = np.column_stack([lon[src_mask] * coslat, lat[src_mask]])
    tree = cKDTree(src_xy)

    rows, cols = np.indices((height, width))
    tx = xmin + (cols + 0.5) * resolution_deg
    ty = ymax - (rows + 0.5) * resolution_deg
    target_xy = np.column_stack([(tx.ravel() * coslat), ty.ravel()])
    dist, idx = tree.query(target_xy, k=1)

    # 0.025 degrees is ~2.8 km in latitude. Keep a conservative cutoff so
    # off-swath cells remain nodata instead of being filled from distant pixels.
    max_dist = max(0.025, resolution_deg * 2.5)
    ok = dist <= max_dist
    src_flat_idx = np.flatnonzero(src_mask)
    chosen = src_flat_idx[idx]

    bands = {}
    for name, arr in data.items():
        if name in {"LAT", "LON"}:
            continue
        out = np.full(height * width, np.nan, dtype=np.float32)
        vals = arr.ravel()[chosen]
        out[ok] = vals[ok].astype(np.float32)
        bands[name] = out.reshape(height, width)

    bands["SOURCE_LAT"] = np.full((height, width), np.nan, dtype=np.float32)
    bands["SOURCE_LON"] = np.full((height, width), np.nan, dtype=np.float32)
    flat_lat = bands["SOURCE_LAT"].ravel()
    flat_lon = bands["SOURCE_LON"].ravel()
    src_lat = lat.ravel()[chosen]
    src_lon = lon.ravel()[chosen]
    flat_lat[ok] = src_lat[ok]
    flat_lon[ok] = src_lon[ok]

    profile = {
        "crs": "EPSG:4326",
        "transform": transform,
        "height": height,
        "width": width,
        "resolution_deg": resolution_deg,
    }
    return bands, profile


def write_tif(
    out: Path,
    bands: dict[str, np.ndarray],
    profile: dict[str, Any],
    tags: dict[str, str],
) -> dict[str, Any]:
    names = [
        "LST_C" if tags["output_unit"] == "celsius" else "LST_K",
        "QC",
        "ERROR_LST_K",
        "EMIS_31",
        "EMIS_32",
        "VIEW_ZENITH_DEG",
        "VIEW_TIME_LOCAL_H",
        "VIEW_TIME_UTC_H",
        "SOURCE_LAT",
        "SOURCE_LON",
    ]
    source_keys = ["LST","QC","ERROR_LST_K","EMIS_31","EMIS_32","VIEW_ZENITH_DEG",
                   "VIEW_TIME_LOCAL_H","VIEW_TIME_UTC_H","SOURCE_LAT","SOURCE_LON"]
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": len(names),
        "dtype": "float32",
        "crs": profile["crs"],
        "transform": profile["transform"],
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
    }
    with rasterio.open(out, "w", **meta) as dst:
        for i, (name, key) in enumerate(zip(names, source_keys), 1):
            arr = bands[key]
            dst.write(np.where(np.isfinite(arr), arr, NODATA).astype(np.float32), i)
            dst.set_band_description(i, name)
        dst.update_tags(**tags)

    lst = bands["LST"]
    qc = bands["QC"]
    strict = strict_qc_mask(qc, lst)
    native = np.isfinite(lst)
    utc_vals = bands["VIEW_TIME_UTC_H"][native]
    local_vals = bands["VIEW_TIME_LOCAL_H"][native]
    return {
        "path": str(out),
        "size_bytes": out.stat().st_size,
        "width": profile["width"],
        "height": profile["height"],
        "crs": profile["crs"],
        "resolution_deg": profile["resolution_deg"],
        "native_lst_valid_pixels": int(native.sum()),
        "native_lst_valid_ratio": float(native.mean()),
        "strict_qc_pixels": int(strict.sum()),
        "strict_qc_ratio": float(strict.mean()),
        "strict_retention_of_native": float(strict.sum()/native.sum()) if native.sum() else 0.0,
        "view_time_local_h_mean": float(np.nanmean(local_vals)) if local_vals.size else None,
        "view_time_utc_h_mean": float(np.nanmean(utc_vals)) if utc_vals.size else None,
        "lst_mean": float(np.nanmean(lst)) if native.any() else None,
        "lst_std": float(np.nanstd(lst)) if native.any() else None,
    }


def cache_path(rid: str, product: str, day: datetime, hhmm: str, unit: str) -> Path:
    suffix = "C" if unit == "celsius" else "K"
    return (
        Path("data") / "modis_l2_swath" / CACHE_VERSION / rid
        / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
        / f"{product}_{day:%Y%m%d}_{hhmm}_UTC_{suffix}.tif"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--region-name", default="")
    p.add_argument("--platform", choices=["terra","aqua"], default="terra")
    p.add_argument("--target-time-utc", required=True)
    p.add_argument("--time-window-minutes", type=int, default=15)
    p.add_argument("--output-unit", choices=["celsius","kelvin"], default="celsius")
    p.add_argument("--resolution-deg", type=float, default=0.01)
    args = p.parse_args()

    day = datetime.fromisoformat(args.date[:10]).replace(tzinfo=timezone.utc)
    bbox = parse_bbox(args.bbox)
    if not 0 <= args.time_window_minutes <= 180:
        raise ValueError("time-window-minutes must be 0..180")
    if not 0.001 <= args.resolution_deg <= 0.05:
        raise ValueError("resolution-deg must be 0.001..0.05")

    product = PRODUCTS[args.platform]
    target_minute = parse_hhmm(args.target_time_utc)
    all_granules = list_granules(product, day)
    candidates = candidate_granules(all_granules, target_minute, args.time_window_minutes)

    result: dict[str, Any] = {
        "product": product,
        "collection": "061",
        "source": "NASA LAADS DAAC",
        "source_root": LAADS_ROOT,
        "date": args.date,
        "platform": args.platform,
        "bbox": bbox,
        "region_name": args.region_name or None,
        "target_time_utc": args.target_time_utc,
        "time_window_minutes": args.time_window_minutes,
        "output_unit": args.output_unit,
        "resolution_deg": args.resolution_deg,
        "archive_listing_count": len(all_granules),
        "candidate_granules": [
            {k:v for k,v in g.items() if k != "url"} for g in candidates
        ],
        "auth_token_present": bool(laads_token()),
        "status": "started",
    }
    Path("output").mkdir(exist_ok=True)

    if not candidates:
        result["status"] = "no_candidate_granule"
        (Path("output")/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        raise RuntimeError("No LAADS granule found in requested time window")

    token = laads_token()
    rid = region_id(args.region_name, bbox)
    work = Path("output/work/modis_l2_swath")
    work.mkdir(parents=True, exist_ok=True)
    attempts = []

    try:
        for g in candidates:
            raw = work / g["filename"]
            attempt = {k:v for k,v in g.items() if k != "url"}
            try:
                download_granule(g["url"], raw, token)
                attempt["downloaded_bytes"] = raw.stat().st_size
                data = read_hdf(raw, args.output_unit)
                attempt["swath_shape"] = list(data["LST"].shape)
                attempt["lat_range"] = [float(np.nanmin(data["LAT"])), float(np.nanmax(data["LAT"]))]
                attempt["lon_range"] = [float(np.nanmin(data["LON"])), float(np.nanmax(data["LON"]))]
                if not swath_intersects(data, bbox):
                    attempt["covers_roi"] = False
                    attempts.append(attempt)
                    raw.unlink(missing_ok=True)
                    continue
                attempt["covers_roi"] = True

                bands, profile = regularize_roi(data, bbox, args.resolution_deg)
                cache = cache_path(rid, product, day, g["hhmm"], args.output_unit)
                tags = {
                    "dataset": product,
                    "collection": "061",
                    "source_granule": g["filename"],
                    "granule_time_utc": g["hhmm"],
                    "source_archive": "NASA LAADS DAAC",
                    "output_unit": args.output_unit,
                    "regridding": "nearest swath pixel onto regular EPSG:4326 ROI grid",
                    "geolocation_mapping": "Latitude/Longitude 5km SDS interpolated to 1km using offset=2 increment=5",
                    "lst_conversion": "DN*0.02 K; Celsius subtracts 273.15",
                    "error_lst_conversion": "DN*0.04 K",
                    "emissivity_conversion": "DN*0.002+0.49",
                    "view_angle_conversion": "DN*0.5 deg",
                    "view_time_conversion": "DN*0.1 local solar hour",
                    "recommended_strict_qc": "bits0-1<=1; bits2-3==0; bits6-7<=2",
                    "cache_version": CACHE_VERSION,
                }
                details = write_tif(cache, bands, profile, tags)

                artifact_dir = Path("output/modis_l2_swath") / rid
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact = artifact_dir / cache.name
                shutil.copy2(cache, artifact)
                raw_artifact = artifact_dir / g["filename"]
                shutil.copy2(raw, raw_artifact)

                attempts.append(attempt)
                result.update({
                    "status": "success",
                    "selected_granule": {k:v for k,v in g.items() if k != "url"},
                    "cache_path": str(cache),
                    "artifact_path": str(artifact),
                    "raw_hdf_artifact": str(raw_artifact),
                    "details": details,
                    "attempts": attempts,
                })
                break
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                attempts.append(attempt)
                if isinstance(exc, PermissionError):
                    result.update({
                        "status": "authentication_required",
                        "error": str(exc),
                        "attempts": attempts,
                    })
                    break
            finally:
                raw.unlink(missing_ok=True)

        if result["status"] == "started":
            result.update({
                "status": "no_covering_granule",
                "attempts": attempts,
                "error": "Candidates were downloaded but none covered the requested ROI.",
            })
    except Exception as exc:
        result.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": attempts,
        })

    (Path("output")/"result.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.rmtree(work, ignore_errors=True)

    print(json.dumps(result,indent=2,ensure_ascii=False))
    if result["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
