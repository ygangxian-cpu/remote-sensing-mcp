from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import rasterio

DATE = "2019-09-24"
RID = "zhangye-c6c86510ae"
ROOT = Path("data") / "era5_land" / "v1" / RID / "2019" / "09" / "24"
OUT = Path("data") / "metadata" / "era5-formal-roi-20190924-24h.json.gz.b64"

EXPECTED_BANDS = [
    "T2_C",
    "TD2_C",
    "U10_MPS",
    "V10_MPS",
    "PSFC_PA",
    "SWDOWN_WM2",
    "GLW_WM2",
]


def read_hour(hour: int) -> dict:
    path = ROOT / f"ERA5LAND_20190924_{hour:02d}00_UTC.tif"
    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        desc = list(src.descriptions)
        if desc != EXPECTED_BANDS:
            raise RuntimeError(f"unexpected bands in {path}: {desc}")
        data = src.read(masked=True).astype("float32")
        bands = {
            name: data[i].filled(np.nan).tolist()
            for i, name in enumerate(EXPECTED_BANDS)
        }
        return {
            "hour_utc": hour,
            "source_path": str(path),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "height": int(src.height),
            "width": int(src.width),
            "bands": bands,
            "dataset": src.tags().get("dataset", "ECMWF/ERA5_LAND/HOURLY"),
            "timestamp_utc": src.tags().get("timestamp_utc", f"2019-09-24T{hour:02d}:00:00"),
        }


def main() -> None:
    hours = [read_hour(h) for h in range(24)]
    grid_keys = ("crs", "transform", "height", "width")
    base = {k: hours[0][k] for k in grid_keys}
    for item in hours[1:]:
        for key in grid_keys:
            if item[key] != base[key]:
                raise RuntimeError(f"ERA5 grid changes at hour {item['hour_utc']}: {key}")

    payload = {
        "schema": "era5-land-formal-roi-24h-v1",
        "date": DATE,
        "region_id": RID,
        "roi": [99.86, 38.67, 100.5, 39.43],
        "time_standard": "UTC",
        "bands": EXPECTED_BANDS,
        "grid": base,
        "hours": hours,
        "source_note": (
            "Existing MCP ERA5-Land v1 cache; radiation bands already converted from "
            "hourly J m-2 accumulation to W m-2 by division by 3600."
        ),
    }

    raw = json.dumps(payload, allow_nan=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded + "\n", encoding="ascii")

    print(json.dumps({
        "output": str(OUT),
        "encoded_bytes": OUT.stat().st_size,
        "hours": len(hours),
        "grid": base,
        "bands": EXPECTED_BANDS,
    }, indent=2))


if __name__ == "__main__":
    main()
