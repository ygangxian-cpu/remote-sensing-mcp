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

PROJECT_DEFAULT = "ee-ygangxian"
NODATA = -9999.0
CACHE_VERSION = "v1"
DATASETS = {
    "terra": "MODIS/061/MOD11A1",
    "aqua": "MODIS/061/MYD11A1",
}


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


def date_range(start: datetime, end: datetime) -> list[datetime]:
    if end <= start:
        raise ValueError("end_date must be after start_date")
    out = []
    cur = start
    while cur < end:
        out.append(cur)
        cur += timedelta(days=1)
    if len(out) > 62:
        raise ValueError("One MODIS job is limited to 62 days")
    return out


def parse_bbox(value: str) -> list[float]:
    bbox = [float(x) for x in value.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = bbox
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    return bbox


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()[:64] or "roi"


def bbox_hash(bbox: list[float]) -> str:
    canonical = ",".join(f"{x:.6f}" for x in bbox)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]


def region_id(region_name: str, bbox: list[float]) -> str:
    return f"{slugify(region_name or 'roi')}-{bbox_hash(bbox)}"


def qc_mask(qc):
    mandatory = qc.bitwiseAnd(3).lte(1)
    data_quality = qc.rightShift(2).bitwiseAnd(3).eq(0)
    lst_error = qc.rightShift(6).bitwiseAnd(3).lte(2)
    return mandatory.And(data_quality).And(lst_error)


def prepared_image(image):
    qd = image.select("QC_Day")
    qn = image.select("QC_Night")
    md = qc_mask(qd)
    mn = qc_mask(qn)

    day = (
        image.select("LST_Day_1km")
        .multiply(0.02)
        .subtract(273.15)
        .updateMask(md)
        .rename("LST_DAY_C")
    )
    night = (
        image.select("LST_Night_1km")
        .multiply(0.02)
        .subtract(273.15)
        .updateMask(mn)
        .rename("LST_NIGHT_C")
    )
    day_time = (
        image.select("Day_view_time")
        .multiply(0.1)
        .updateMask(md)
        .rename("DAY_VIEW_TIME_LOCAL_H")
    )
    night_time = (
        image.select("Night_view_time")
        .multiply(0.1)
        .updateMask(mn)
        .rename("NIGHT_VIEW_TIME_LOCAL_H")
    )
    return ee.Image.cat([day, night, day_time, night_time]).toFloat()


def cache_path(rid: str, platform: str, day: datetime) -> Path:
    product = "MOD11A1" if platform == "terra" else "MYD11A1"
    return (
        Path("data") / "modis_lst" / CACHE_VERSION / rid
        / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
        / f"{product}_{day:%Y%m%d}_QC.tif"
    )


def download_one(platform: str, day: datetime, bbox: list[float], out: Path) -> dict[str, Any] | None:
    start = day.strftime("%Y-%m-%d")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    collection = (
        ee.ImageCollection(DATASETS[platform])
        .filterDate(start, end)
        .filterBounds(ee.Geometry.Rectangle(bbox))
    )
    count = int(collection.size().getInfo())
    if count < 1:
        return None

    image = prepared_image(ee.Image(collection.first()))
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    url = image.getDownloadURL({
        "name": f"{platform}_{day:%Y%m%d}",
        "region": region,
        "scale": 1000,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })

    tmp = Path("output/work") / f"{platform}_{day:%Y%m%d}_raw.tif"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    out.parent.mkdir(parents=True, exist_ok=True)
    names = ["LST_DAY_C", "LST_NIGHT_C", "DAY_VIEW_TIME_LOCAL_H", "NIGHT_VIEW_TIME_LOCAL_H"]
    units = ["degC", "degC", "hour_local_solar_time", "hour_local_solar_time"]

    with rasterio.open(tmp) as src:
        data = src.read(masked=True).astype("float32")
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=NODATA, compress="deflate", predictor=3)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data.filled(NODATA))
            for i, (name, unit) in enumerate(zip(names, units), start=1):
                dst.set_band_description(i, name)
                dst.update_tags(i, unit=unit)
            dst.update_tags(
                dataset=DATASETS[platform],
                platform=platform,
                date=day.strftime("%Y-%m-%d"),
                qc_rule="bits0-1<=1; bits2-3==0; bits6-7<=2",
                lst_conversion="DN*0.02-273.15",
            )

        stats = {}
        for i, name in enumerate(names):
            vals = data[i].compressed()
            stats[name] = {
                "valid_pixels": int(vals.size),
                "min": float(vals.min()) if vals.size else None,
                "max": float(vals.max()) if vals.size else None,
                "mean": float(vals.mean()) if vals.size else None,
            }
        result = {
            "path": str(out),
            "size_bytes": out.stat().st_size,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "stats": stats,
        }

    tmp.unlink(missing_ok=True)
    return result


def load_index(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"product": "MODIS Terra/Aqua daily LST V6.1", "cache_version": CACHE_VERSION, "regions": {}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--region-name", default="")
    p.add_argument("--platforms", default="terra,aqua")
    args = p.parse_args()

    start, end = parse_date(args.start_date), parse_date(args.end_date)
    days = date_range(start, end)
    bbox = parse_bbox(args.bbox)
    platforms = [x.strip().lower() for x in args.platforms.split(",") if x.strip()]
    if any(x not in DATASETS for x in platforms):
        raise ValueError("platforms must contain terra and/or aqua")

    rid = region_id(args.region_name, bbox)
    project = init_ee()
    index_path = Path("data/metadata/modis-lst-index.json")
    index = load_index(index_path)
    region_meta = index["regions"].setdefault(rid, {"region_name": args.region_name or None, "bbox": bbox, "files": {}})

    out_root = Path("output/modis_lst") / rid
    out_root.mkdir(parents=True, exist_ok=True)
    created = 0
    hits = 0
    missing = []
    files = []

    for day in days:
        for platform in platforms:
            cache = cache_path(rid, platform, day)
            key = f"{day:%Y-%m-%d}:{platform}"
            details = None
            if cache.exists():
                hits += 1
            else:
                details = download_one(platform, day, bbox, cache)
                if details is None:
                    missing.append(key)
                    continue
                created += 1
                region_meta["files"][key] = str(cache)

            artifact = out_root / cache.name
            shutil.copy2(cache, artifact)
            files.append({
                "date": day.strftime("%Y-%m-%d"),
                "platform": platform,
                "cache_path": str(cache),
                "artifact_path": str(artifact),
                "created": details is not None,
                "details": details,
            })

    if created:
        index["updated_at"] = datetime.utcnow().isoformat() + "Z"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    result = {
        "product": "MODIS Terra/Aqua Daily LST V6.1",
        "project": project,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bbox": bbox,
        "region_name": args.region_name or None,
        "region_id": rid,
        "platforms": platforms,
        "qc_rule": "bits0-1<=1; bits2-3==0; bits6-7<=2",
        "cache_hits": hits,
        "cache_created": created,
        "missing": missing,
        "files": files,
    }
    Path("output").mkdir(exist_ok=True)
    (Path("output") / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree("output/work", ignore_errors=True)


if __name__ == "__main__":
    main()
