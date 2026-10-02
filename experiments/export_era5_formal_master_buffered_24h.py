from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import rasterio

DATE = "2019-09-24"
REGION_NAME = "zhangye-formal-buffered"
# Formal master footprint is [99.8652818242, 38.6791139735, 100.4761362174, 39.3977662008].
# Keep >= one ERA5-Land native 0.1-degree cell of buffer on every side.
REQUEST_BBOX = [99.70, 38.50, 100.70, 39.60]
FORMAL_BBOX = [99.86528182419725, 38.67911397351706, 100.47613621740223, 39.39776620081703]
INDEX = Path("data/metadata/era5-land-index.json")
OUT = Path("data/metadata/era5-formal-master-buffered-20190924-24h.json.gz.b64")

EXPECTED_BANDS = [
    "T2_C",
    "TD2_C",
    "U10_MPS",
    "V10_MPS",
    "PSFC_PA",
    "SWDOWN_WM2",
    "GLW_WM2",
]


def find_region_id() -> str:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    matches = []
    for rid, item in index.get("regions", {}).items():
        if item.get("region_name") != REGION_NAME:
            continue
        bbox = [float(x) for x in item.get("bbox", [])]
        if len(bbox) == 4 and all(abs(a - b) < 1e-9 for a, b in zip(bbox, REQUEST_BBOX)):
            matches.append(rid)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one buffered ERA5 region, got {matches}")
    return matches[0]


def read_hour(root: Path, hour: int) -> dict:
    path = root / f"ERA5LAND_20190924_{hour:02d}00_UTC.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        desc = list(src.descriptions)
        if desc != EXPECTED_BANDS:
            raise RuntimeError(f"unexpected bands in {path}: {desc}")
        left, bottom, right, top = src.bounds
        fw, fs, fe, fn = FORMAL_BBOX
        if not (left <= fw and bottom <= fs and right >= fe and top >= fn):
            raise RuntimeError(
                f"ERA5 buffered raster does not contain formal master footprint: "
                f"src={tuple(src.bounds)}, formal={FORMAL_BBOX}"
            )
        data = src.read(masked=True).astype("float32")
        bands = {name: data[i].filled(np.nan).tolist() for i, name in enumerate(EXPECTED_BANDS)}
        return {
            "hour_utc": hour,
            "source_path": str(path),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "height": int(src.height),
            "width": int(src.width),
            "bounds": [float(left), float(bottom), float(right), float(top)],
            "bands": bands,
            "dataset": src.tags().get("dataset", "ECMWF/ERA5_LAND/HOURLY"),
            "timestamp_utc": src.tags().get("timestamp_utc", f"2019-09-24T{hour:02d}:00:00"),
        }


def main() -> None:
    rid = find_region_id()
    root = Path("data") / "era5_land" / "v1" / rid / "2019" / "09" / "24"
    hours = [read_hour(root, h) for h in range(24)]
    keys = ("crs", "transform", "height", "width", "bounds")
    base = {k: hours[0][k] for k in keys}
    for item in hours[1:]:
        for key in keys:
            if item[key] != base[key]:
                raise RuntimeError(f"ERA5 grid changes at hour {item['hour_utc']}: {key}")

    payload = {
        "schema": "era5-land-formal-master-buffered-24h-v2",
        "date": DATE,
        "region_id": rid,
        "region_name": REGION_NAME,
        "request_bbox": REQUEST_BBOX,
        "formal_master_bbox": FORMAL_BBOX,
        "buffer_note": "At least one 0.1-degree ERA5-Land native-cell buffer around the formal 4 km master footprint.",
        "time_standard": "UTC",
        "bands": EXPECTED_BANDS,
        "grid": base,
        "hours": hours,
        "source_note": (
            "ERA5-Land cache downloaded specifically for the unified 4 km master footprint; "
            "radiation bands converted from hourly J m-2 accumulation to W m-2 by division by 3600."
        ),
    }
    raw = json.dumps(payload, allow_nan=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded + "\n", encoding="ascii")
    print(json.dumps({
        "output": str(OUT),
        "region_id": rid,
        "request_bbox": REQUEST_BBOX,
        "formal_master_bbox": FORMAL_BBOX,
        "grid": base,
        "hours": len(hours),
        "encoded_bytes": OUT.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
