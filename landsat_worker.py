from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
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
COLLECTIONS = {
    "L8": "LANDSAT/LC08/C02/T1_L2",
    "L9": "LANDSAT/LC09/C02/T1_L2",
}
BANDS = [
    ("LST_C", "ST_B10", "degC"),
    ("ST_QA_K", "ST_QA", "K"),
    ("SR_B2", "SR_B2", "reflectance"),
    ("SR_B3", "SR_B3", "reflectance"),
    ("SR_B4", "SR_B4", "reflectance"),
    ("SR_B5", "SR_B5", "reflectance"),
    ("SR_B6", "SR_B6", "reflectance"),
    ("SR_B7", "SR_B7", "reflectance"),
]


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


def parse_bbox(value: str) -> list[float]:
    bbox = [float(x) for x in value.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    return bbox


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()[:64] or "roi"


def bbox_hash(bbox: list[float]) -> str:
    canonical = ",".join(f"{x:.6f}" for x in bbox)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]


def region_id(region_name: str, bbox: list[float]) -> str:
    return f"{slugify(region_name or 'roi')}-{bbox_hash(bbox)}"


def mask_and_scale(image):
    qa = image.select("QA_PIXEL")
    mask = ee.Image(1)
    for bit in [0, 1, 2, 3, 4, 5, 7]:
        mask = mask.And(qa.bitwiseAnd(1 << bit).eq(0))
    mask = mask.And(image.select("QA_RADSAT").eq(0))

    lst = image.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST_C")
    stqa = image.select("ST_QA").multiply(0.01).rename("ST_QA_K")
    sr = image.select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]).multiply(0.0000275).add(-0.2)

    prepared = ee.Image.cat([lst, stqa, sr]).updateMask(mask).toFloat()
    return ee.Image(
        prepared.copyProperties(
            image,
            ["system:time_start","LANDSAT_PRODUCT_ID","LANDSAT_SCENE_ID","SPACECRAFT_ID","CLOUD_COVER","WRS_PATH","WRS_ROW"],
        )
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def cache_path(rid: str, ts: datetime, product_id: str) -> Path:
    return (
        Path("data") / "landsat_c2_l2" / CACHE_VERSION / rid
        / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}"
        / f"{safe_name(product_id)}_L2_QC.tif"
    )


def download_band(image, band: str, bbox: list[float], tmp: Path):
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    url = image.select(band).getDownloadURL({
        "name": band,
        "region": region,
        "scale": 30,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })
    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_scene(image, bbox: list[float], out: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    work = Path("output/work/landsat") / safe_name(metadata["product_id"])
    work.mkdir(parents=True, exist_ok=True)

    arrays = []
    profile = None
    stats = {}
    for name, _, unit in BANDS:
        p = work / f"{name}.tif"
        download_band(image, name, bbox, p)
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True).astype("float32")
            arrays.append(arr)
            if profile is None:
                profile = src.profile.copy()
            vals = arr.compressed()
            stats[name] = {
                "unit": unit,
                "valid_pixels": int(vals.size),
                "min": float(vals.min()) if vals.size else None,
                "max": float(vals.max()) if vals.size else None,
                "mean": float(vals.mean()) if vals.size else None,
            }

    assert profile is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(BANDS), dtype="float32", nodata=NODATA, compress="deflate", predictor=3)
    with rasterio.open(out, "w", **profile) as dst:
        for i, ((name, source, unit), arr) in enumerate(zip(BANDS, arrays), start=1):
            dst.write(arr.filled(NODATA), i)
            dst.set_band_description(i, name)
            dst.update_tags(i, source_band=source, unit=unit)
        dst.update_tags(
            product_id=metadata["product_id"],
            spacecraft=metadata["spacecraft"],
            acquired_utc=metadata["acquired_utc"],
            cloud_cover=str(metadata["cloud_cover"]),
            lst_conversion="ST_B10*0.00341802+149-273.15",
            sr_conversion="SR_Bx*0.0000275-0.2",
            qa_mask="QA_PIXEL bits 0,1,2,3,4,5,7 == 0 and QA_RADSAT == 0",
        )

    result = {
        "path": str(out),
        "size_bytes": out.stat().st_size,
        "width": profile["width"],
        "height": profile["height"],
        "crs": str(profile["crs"]),
        "stats": stats,
    }
    shutil.rmtree(work, ignore_errors=True)
    return result


def load_index(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"product": "Landsat 8/9 C2 L2", "cache_version": CACHE_VERSION, "regions": {}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--region-name", default="")
    p.add_argument("--satellites", default="L8,L9")
    p.add_argument("--cloud-cover-max", type=float, default=80.0)
    args = p.parse_args()

    bbox = parse_bbox(args.bbox)
    rid = region_id(args.region_name, bbox)
    sats = [x.strip().upper() for x in args.satellites.split(",") if x.strip()]
    if any(s not in COLLECTIONS for s in sats):
        raise ValueError("satellites must contain L8 and/or L9")

    project = init_ee()
    region = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)

    collections = []
    for sat in sats:
        col = (
            ee.ImageCollection(COLLECTIONS[sat])
            .filterDate(args.start_date, args.end_date)
            .filterBounds(region)
            .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
            .filter(ee.Filter.lte("CLOUD_COVER", args.cloud_cover_max))
        )
        collections.append(col)

    merged = collections[0]
    for col in collections[1:]:
        merged = merged.merge(col)
    merged = merged.sort("system:time_start")

    count = int(merged.size().getInfo())
    if count > 20:
        raise ValueError("One Landsat job is limited to 20 scenes")

    lst = merged.toList(count)
    index_path = Path("data/metadata/landsat-index.json")
    index = load_index(index_path)
    region_meta = index["regions"].setdefault(rid, {"region_name": args.region_name or None, "bbox": bbox, "scenes": {}})
    out_root = Path("output/landsat") / rid
    out_root.mkdir(parents=True, exist_ok=True)

    hits = 0
    created = 0
    files = []

    for i in range(count):
        source = ee.Image(lst.get(i))
        props = ee.Dictionary({
            "product_id": source.get("LANDSAT_PRODUCT_ID"),
            "scene_id": source.get("LANDSAT_SCENE_ID"),
            "spacecraft": source.get("SPACECRAFT_ID"),
            "cloud_cover": source.get("CLOUD_COVER"),
            "time": source.get("system:time_start"),
        }).getInfo()
        product_id = str(props.get("product_id") or props.get("scene_id") or f"scene_{i}")
        ts = datetime.utcfromtimestamp(float(props["time"]) / 1000.0)
        metadata = {
            "product_id": product_id,
            "scene_id": props.get("scene_id"),
            "spacecraft": props.get("spacecraft"),
            "cloud_cover": float(props.get("cloud_cover") or 0),
            "acquired_utc": ts.isoformat(),
        }

        cache = cache_path(rid, ts, product_id)
        details = None
        if cache.exists():
            hits += 1
        else:
            details = download_scene(mask_and_scale(source), bbox, cache, metadata)
            created += 1
            region_meta["scenes"][product_id] = {
                "path": str(cache),
                **metadata,
            }

        artifact = out_root / cache.name
        shutil.copy2(cache, artifact)
        files.append({
            **metadata,
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
        "product": "Landsat 8/9 Collection 2 Level 2",
        "project": project,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bbox": bbox,
        "region_name": args.region_name or None,
        "region_id": rid,
        "satellites": sats,
        "cloud_cover_max": args.cloud_cover_max,
        "scene_count": count,
        "cache_hits": hits,
        "cache_created": created,
        "bands": [{"name": a, "source": b, "unit": c} for a,b,c in BANDS],
        "files": files,
    }
    Path("output").mkdir(exist_ok=True)
    (Path("output") / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree("output/work", ignore_errors=True)


if __name__ == "__main__":
    main()
