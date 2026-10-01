from __future__ import annotations

import json
from pathlib import Path

import requests
import rasterio
from rasterio.windows import from_bounds

DATASET_ID = "4adbc070-afb3-4e9c-85a0-2ce68d1388ad"
BASE = "https://data.tpdc.ac.cn/file/file"
DATE = "20190924"
HOUR = "040000"
BBOX = [99.86, 38.67, 100.5, 39.43]

S = requests.Session()
S.headers["User-Agent"] = "remote-sensing-mcp/0.11 TPDC-ANCFDS"
OUT = Path("output/tpdc_download_test")
OUT.mkdir(parents=True, exist_ok=True)

def api_get(path: str, **params):
    r = S.get(BASE + path, params=params, timeout=90)
    r.raise_for_status()
    obj = r.json()
    if str(obj.get("code")) != "200":
        raise RuntimeError(obj)
    return obj.get("data") or []

root = api_get("/getRootFileDataList", metadataId=DATASET_ID)
lst = next(x for x in root if x.get("name") == "LST")
years = api_get("/getFileDataList", parentId=lst["id"])
year = next(x for x in years if x.get("name") == DATE[:4])
days = api_get("/getFileDataList", parentId=year["id"])
day = next(x for x in days if x.get("name") == DATE)
files = api_get("/getFileDataList", parentId=day["id"])
target = next(x for x in files if f"_{DATE}_{HOUR}_" in x.get("name", ""))

print("TARGET", json.dumps(target, ensure_ascii=False), flush=True)
url = BASE + "/batchDownloadByFileId"
with S.post(url, params={"fileId": target["id"]}, stream=True, timeout=(60, 600)) as r:
    print("DOWNLOAD_STATUS", r.status_code, flush=True)
    print("CONTENT_TYPE", r.headers.get("content-type"), flush=True)
    print("CONTENT_LENGTH", r.headers.get("content-length"), flush=True)
    print("CONTENT_DISPOSITION", r.headers.get("content-disposition"), flush=True)
    r.raise_for_status()
    raw = OUT / target["name"]
    total = 0
    with raw.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                if total % (32 * 1024 * 1024) < 1024 * 1024:
                    print("DOWNLOADED_BYTES", total, flush=True)

print("DOWNLOADED", raw, raw.stat().st_size, flush=True)

with rasterio.open(raw) as src:
    info = {
        "driver": src.driver,
        "crs": str(src.crs),
        "bounds": list(src.bounds),
        "width": src.width,
        "height": src.height,
        "count": src.count,
        "dtypes": list(src.dtypes),
        "nodata": src.nodata,
        "scales": list(src.scales),
        "offsets": list(src.offsets),
        "descriptions": list(src.descriptions),
        "tags": src.tags(),
    }
    print("RASTER_INFO", json.dumps(info, ensure_ascii=False, default=str), flush=True)
    if src.crs and src.crs.to_epsg() == 4326:
        win = from_bounds(*BBOX, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(window=win)
        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update(
            width=arr.shape[2],
            height=arr.shape[1],
            transform=transform,
            compress="deflate",
        )
        crop = OUT / f"ANCFDS_LST_{DATE}_{HOUR[:2]}00_ZHANGYE.tif"
        with rasterio.open(crop, "w", **profile) as dst:
            dst.write(arr)
            dst.update_tags(
                source_doi="10.11888/RemoteSen.tpdc.303249",
                source_file=target["name"],
                source_file_id=target["id"],
                requested_bbox=",".join(map(str, BBOX)),
            )
        print("CROP", crop, crop.stat().st_size, arr.shape, flush=True)
    else:
        print("CROP_SKIPPED_NON_4326", src.crs, flush=True)

manifest = {
    "dataset_id": DATASET_ID,
    "doi": "10.11888/RemoteSen.tpdc.303249",
    "date": DATE,
    "hour": HOUR,
    "bbox": BBOX,
    "source": target,
    "downloaded_bytes": raw.stat().st_size,
    "raster": info,
}
(OUT / "result.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
