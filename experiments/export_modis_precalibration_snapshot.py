from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import rasterio

DATE = "2019-09-24"
ROOT = Path("data/modis_lst/v2")
OUT = Path("data/metadata/modis-precalibration-20190924-formal-roi.json.gz.b64")


def encode_array(arr: np.ndarray) -> list:
    return np.where(np.isfinite(arr), arr, None).tolist()


def read_tif(path: Path) -> dict:
    with rasterio.open(path) as src:
        bands = {}
        for i, desc in enumerate(src.descriptions, start=1):
            name = desc or f"B{i}"
            arr = src.read(i, masked=True).astype("float32").filled(np.nan)
            bands[name] = encode_array(arr)
        return {
            "file": str(path),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "height": src.height,
            "width": src.width,
            "bands": bands,
        }


def main() -> None:
    matches = sorted(ROOT.glob(f"*/2019/09/24/*_20190924_LST_QA.tif"))
    if not matches:
        raise FileNotFoundError("No MODIS v2 files found for formal ROI/date")

    payload = {
        "schema": "modis-v2-precalibration-snapshot-v1",
        "date": DATE,
        "files": [read_tif(p) for p in matches],
    }
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded, encoding="ascii")
    print(json.dumps({
        "snapshot": str(OUT),
        "files": [str(p) for p in matches],
        "raw_bytes": len(raw),
        "encoded_chars": len(encoded),
    }, indent=2))


if __name__ == "__main__":
    main()
