from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
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

PROJECT_DEFAULT = "ee-ygangxian"
NODATA = -9999.0
CACHE_VERSION = "v1"

SRTM = "USGS/SRTMGL1_003"
WORLDCOVER = "ESA/WorldCover/v200"
S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
MCD43A3 = "MODIS/061/MCD43A3"

TARGET_SCALE_M = 100
ALBEDO_SCALE_M = 500


def init_ee() -> str:
    project = os.getenv("EE_PROJECT", "").strip() or PROJECT_DEFAULT
    cred_file = Path.home() / ".config" / "earthengine" / "credentials"
    if cred_file.exists():
        ee.Initialize(project=project)
        return project

    raw = os.getenv("EE_SERVICE_ACCOUNT_JSON", "").strip()
    raw_b64 = os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64", "").strip()
    if not raw and raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if raw:
        info = json.loads(raw)
        project = os.getenv("EE_PROJECT", "").strip() or info.get("project_id") or PROJECT_DEFAULT
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=[
                "https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform",
            ],
        )
        ee.Initialize(credentials, project=project)
        return project

    raise RuntimeError("Earth Engine credentials are missing.")


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value[:10])


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
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


def region_id(region_name: str, bbox: list[float]) -> str:
    return f"{slugify(region_name or 'roi')}-{bbox_hash(bbox)}"


def repo_root(rid: str) -> Path:
    return Path("data") / "scaling_factors" / CACHE_VERSION / rid


def region_geom(bbox: list[float]):
    return ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)


def get_download(image, bbox: list[float], scale: float, name: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    region = region_geom(bbox)
    url = ee.Image(image).getDownloadURL(
        {
            "name": name,
            "region": region,
            "scale": scale,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )
    with requests.get(url, stream=True, timeout=240) as response:
        response.raise_for_status()
        with out.open("wb") as dst:
            for chunk in response.iter_content(2 * 1024 * 1024):
                if chunk:
                    dst.write(chunk)


def normalize_float_tif(
    src_path: Path,
    out_path: Path,
    band_names: list[str],
    tags: dict[str, str],
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        data = src.read(masked=True).astype("float32")
        if data.shape[0] != len(band_names):
            raise RuntimeError(
                f"Band-count mismatch for {src_path.name}: got {data.shape[0]}, expected {len(band_names)}"
            )
        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            count=len(band_names),
            nodata=NODATA,
            compress="deflate",
            predictor=3,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.filled(NODATA))
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
            dst.update_tags(**tags)

        stats = {}
        for i, name in enumerate(band_names):
            vals = data[i].compressed()
            stats[name] = {
                "valid_pixels": int(vals.size),
                "min": float(vals.min()) if vals.size else None,
                "max": float(vals.max()) if vals.size else None,
                "mean": float(vals.mean()) if vals.size else None,
            }

        return {
            "path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "stats": stats,
        }


def normalize_class_tif(src_path: Path, out_path: Path, tags: dict[str, str]) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        arr = src.read(1, masked=True)
        # WorldCover classes fit safely in uint8. 255 is reserved as nodata.
        data = np.full(arr.shape, 255, dtype="uint8")
        valid = ~np.ma.getmaskarray(arr)
        data[valid] = np.asarray(arr.data[valid], dtype="uint8")

        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            count=1,
            nodata=255,
            compress="deflate",
            predictor=2,
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, "LANDCOVER")
            dst.update_tags(**tags)

        unique, counts = np.unique(data[data != 255], return_counts=True)
        class_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
        return {
            "path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "class_counts": class_counts,
        }


def build_terrain(bbox: list[float], out: Path) -> dict[str, Any]:
    work = Path("output/work/scaling")
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "terrain_raw.tif"

    # Resample elevation to the target grid first, then derive slope/aspect at
    # the same effective scale to avoid mixing 30 m and 100 m terrain geometry.
    dem = ee.Image(SRTM).select("elevation").resample("bilinear")
    dem100 = dem.reproject(crs="EPSG:4326", scale=TARGET_SCALE_M)
    terrain = ee.Terrain.products(dem100)
    image = ee.Image.cat(
        [
            dem100.rename("DEM_M"),
            terrain.select("slope").rename("SLOPE_DEG"),
            terrain.select("aspect").rename("ASPECT_DEG"),
        ]
    ).toFloat()

    get_download(image, bbox, TARGET_SCALE_M, "terrain_100m", raw)
    result = normalize_float_tif(
        raw,
        out,
        ["DEM_M", "SLOPE_DEG", "ASPECT_DEG"],
        {
            "dataset": SRTM,
            "target_scale_m": str(TARGET_SCALE_M),
            "terrain_method": "SRTM resampled to target grid, then ee.Terrain.products",
        },
    )
    raw.unlink(missing_ok=True)
    return result


def build_worldcover(bbox: list[float], out: Path) -> dict[str, Any]:
    work = Path("output/work/scaling")
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "worldcover_raw.tif"

    image = ee.ImageCollection(WORLDCOVER).first().select("Map").rename("LANDCOVER")
    # Categorical factor: preserve class labels with nearest-neighbour sampling.
    get_download(image, bbox, TARGET_SCALE_M, "worldcover_100m", raw)
    result = normalize_class_tif(
        raw,
        out,
        {
            "dataset": WORLDCOVER,
            "reference_year": "2021",
            "target_scale_m": str(TARGET_SCALE_M),
            "temporal_note": "quasi-static land-cover factor; reference year differs from 2019 experiment",
        },
    )
    raw.unlink(missing_ok=True)
    return result


def mask_s2(image):
    scl = image.select("SCL")
    # Keep dark-area, vegetation, bare-soil and water classes. Remove
    # no-data/defective, cloud shadow, uncertain/cloud, cirrus and snow/ice.
    valid = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(valid)


def safe_div(num, den):
    return num.divide(den.where(den.abs().lt(1e-6), 1e-6))


def build_surface_factors(
    start: datetime,
    end: datetime,
    bbox: list[float],
    out: Path,
) -> dict[str, Any]:
    geom = region_geom(bbox)
    collection = (
        ee.ImageCollection(S2_SR)
        .filterDate(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .filterBounds(geom)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_s2)
    )
    scene_count = int(collection.size().getInfo())
    if scene_count < 1:
        raise RuntimeError("No Sentinel-2 SR scenes found for requested period")

    composite = collection.median().multiply(0.0001)

    blue = composite.select("B2").rename("BLUE")
    green = composite.select("B3").rename("GREEN")
    red = composite.select("B4").rename("RED")
    nir = composite.select("B8").rename("NIR")
    swir1 = composite.select("B11").rename("SWIR1")
    swir2 = composite.select("B12").rename("SWIR2")

    ndvi = safe_div(nir.subtract(red), nir.add(red)).rename("NDVI")
    evi = (
        nir.subtract(red)
        .multiply(2.5)
        .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
        .rename("EVI")
    )
    mndwi = safe_div(green.subtract(swir1), green.add(swir1)).rename("MNDWI")
    ndbi = safe_div(swir1.subtract(nir), swir1.add(nir)).rename("NDBI")
    ndmi = safe_div(nir.subtract(swir1), nir.add(swir1)).rename("NDMI")
    bsi_num = swir1.add(red).subtract(nir.add(blue))
    bsi_den = swir1.add(red).add(nir).add(blue)
    bsi = safe_div(bsi_num, bsi_den).rename("BSI")

    # Adaptive dimidiate-pixel FVC: use robust NDVI percentiles in this ROI.
    ndvi_stats = ndvi.reduceRegion(
        reducer=ee.Reducer.percentile([5, 95]),
        geometry=geom,
        scale=TARGET_SCALE_M,
        bestEffort=True,
        maxPixels=100_000_000,
    ).getInfo()
    ndvi_soil = ndvi_stats.get("NDVI_p5")
    ndvi_veg = ndvi_stats.get("NDVI_p95")
    if ndvi_soil is None or ndvi_veg is None or float(ndvi_veg) - float(ndvi_soil) < 0.05:
        ndvi_soil, ndvi_veg = 0.2, 0.86
    ndvi_soil = float(ndvi_soil)
    ndvi_veg = float(ndvi_veg)
    fvc = ndvi.subtract(ndvi_soil).divide(ndvi_veg - ndvi_soil).clamp(0, 1).rename("FVC")

    factors = ee.Image.cat(
        [blue, green, red, nir, swir1, swir2, ndvi, evi, fvc, mndwi, ndbi, bsi, ndmi]
    ).toFloat()

    work = Path("output/work/scaling")
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "surface_factors_raw.tif"
    get_download(factors, bbox, TARGET_SCALE_M, "surface_factors_100m", raw)

    bands = [
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
    ]
    result = normalize_float_tif(
        raw,
        out,
        bands,
        {
            "dataset": S2_SR,
            "composite": "median",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "scene_count": str(scene_count),
            "reflectance_scale_applied": "0.0001",
            "cloud_mask": "SCL excludes 0,1,3,7,8,9,10,11",
            "target_scale_m": str(TARGET_SCALE_M),
            "fvc_method": "linear dimidiate-pixel NDVI scaling, clamped 0-1",
            "fvc_ndvi_soil": str(ndvi_soil),
            "fvc_ndvi_veg": str(ndvi_veg),
        },
    )
    result["scene_count"] = scene_count
    result["fvc_ndvi_soil"] = ndvi_soil
    result["fvc_ndvi_veg"] = ndvi_veg
    raw.unlink(missing_ok=True)
    return result


def daterange(start: datetime, end: datetime):
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)


def albedo_path(root: Path, day: datetime) -> Path:
    return (
        root
        / "albedo"
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{day:%d}"
        / f"MCD43A3_{day:%Y%m%d}_BSA_WSA_QC.tif"
    )


def build_albedo_day(day: datetime, bbox: list[float], out: Path) -> dict[str, Any] | None:
    start = day.strftime("%Y-%m-%d")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    collection = ee.ImageCollection(MCD43A3).filterDate(start, end).filterBounds(region_geom(bbox))
    count = int(collection.size().getInfo())
    if count < 1:
        return None

    src = ee.Image(collection.first())
    qa = src.select("BRDF_Albedo_Band_Mandatory_Quality_shortwave")
    mask = qa.lte(1)

    bsa = src.select("Albedo_BSA_shortwave").multiply(0.001).updateMask(mask).rename("BSA_SHORTWAVE")
    wsa = src.select("Albedo_WSA_shortwave").multiply(0.001).updateMask(mask).rename("WSA_SHORTWAVE")
    qa_out = qa.updateMask(mask).toFloat().rename("ALBEDO_QA")
    image = ee.Image.cat([bsa, wsa, qa_out]).toFloat()

    work = Path("output/work/scaling")
    work.mkdir(parents=True, exist_ok=True)
    raw = work / f"albedo_{day:%Y%m%d}_raw.tif"
    get_download(image, bbox, ALBEDO_SCALE_M, f"mcd43a3_{day:%Y%m%d}", raw)
    result = normalize_float_tif(
        raw,
        out,
        ["BSA_SHORTWAVE", "WSA_SHORTWAVE", "ALBEDO_QA"],
        {
            "dataset": MCD43A3,
            "date": start,
            "albedo_scale_applied": "0.001",
            "mandatory_quality_rule": "shortwave QA <= 1",
            "target_scale_m": str(ALBEDO_SCALE_M),
            "blue_sky_note": "Blue-sky albedo is not computed here; combine BSA/WSA with diffuse fraction later.",
        },
    )
    raw.unlink(missing_ok=True)
    return result


def load_index(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "product": "LST downscaling scaling-factor library",
        "cache_version": CACHE_VERSION,
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

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")
    if (end - start).days > 62:
        raise ValueError("One scaling-factor job is limited to 62 days")

    bbox = parse_bbox(args.bbox)
    rid = region_id(args.region_name, bbox)
    project = init_ee()

    root = repo_root(rid)
    static_dir = root / "static"
    period_dir = root / "surface" / f"{start:%Y%m%d}_{(end - timedelta(days=1)):%Y%m%d}"

    terrain_file = static_dir / "SRTM_TERRAIN_100M.tif"
    landcover_file = static_dir / "WORLDCOVER_2021_100M.tif"
    surface_file = period_dir / "S2_SCALING_FACTORS_100M.tif"

    index_path = Path("data/metadata/scaling-factors-index.json")
    index = load_index(index_path)
    region_meta = index["regions"].setdefault(
        rid,
        {
            "region_name": args.region_name or None,
            "bbox": bbox,
            "static": {},
            "surface_periods": {},
            "albedo_days": {},
        },
    )

    hits = 0
    created = 0
    details: dict[str, Any] = {}

    if terrain_file.exists():
        hits += 1
    else:
        details["terrain"] = build_terrain(bbox, terrain_file)
        region_meta["static"]["terrain"] = str(terrain_file)
        created += 1

    if landcover_file.exists():
        hits += 1
    else:
        details["worldcover"] = build_worldcover(bbox, landcover_file)
        region_meta["static"]["worldcover_2021"] = str(landcover_file)
        created += 1

    period_key = f"{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    if surface_file.exists():
        hits += 1
    else:
        details["surface"] = build_surface_factors(start, end, bbox, surface_file)
        region_meta["surface_periods"][period_key] = str(surface_file)
        created += 1

    albedo_files = []
    missing_albedo = []
    for day in daterange(start, end):
        out = albedo_path(root, day)
        key = day.strftime("%Y-%m-%d")
        if out.exists():
            hits += 1
            albedo_files.append(str(out))
            continue
        info = build_albedo_day(day, bbox, out)
        if info is None:
            missing_albedo.append(key)
            continue
        created += 1
        details.setdefault("albedo", {})[key] = info
        region_meta["albedo_days"][key] = str(out)
        albedo_files.append(str(out))

    if created or not index_path.exists():
        index["updated_at"] = datetime.utcnow().isoformat() + "Z"
        save_index(index_path, index)

    artifact_dir = Path("output/scaling_factors") / rid
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for src in [terrain_file, landcover_file, surface_file]:
        if src.exists():
            shutil.copy2(src, artifact_dir / src.name)

    albedo_artifact = artifact_dir / "albedo"
    for src_str in albedo_files:
        src = Path(src_str)
        dst = albedo_artifact / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    result = {
        "product": "LST downscaling scaling-factor library",
        "project": project,
        "region_name": args.region_name or None,
        "region_id": rid,
        "bbox": bbox,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "target_surface_scale_m": TARGET_SCALE_M,
        "albedo_scale_m": ALBEDO_SCALE_M,
        "cache_hits": hits,
        "cache_created": created,
        "terrain": str(terrain_file),
        "landcover": {
            "path": str(landcover_file),
            "reference_year": 2021,
            "note": "quasi-static factor; not temporally synchronous with 2019 experiment",
        },
        "surface_factors": {
            "path": str(surface_file),
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
            "source": S2_SR,
            "composite": "median over requested period",
        },
        "albedo": {
            "dataset": MCD43A3,
            "bands": ["BSA_SHORTWAVE", "WSA_SHORTWAVE", "ALBEDO_QA"],
            "files": albedo_files,
            "missing_dates": missing_albedo,
            "blue_sky_albedo_computed": False,
        },
        "details": details,
    }

    Path("output").mkdir(exist_ok=True)
    (Path("output") / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.rmtree("output/work", ignore_errors=True)


if __name__ == "__main__":
    main()
