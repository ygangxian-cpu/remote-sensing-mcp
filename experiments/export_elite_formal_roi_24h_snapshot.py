from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.warp import reproject

ROI = [99.86, 38.67, 100.5, 39.43]
TARGET_SHAPE = (21, 18)
TARGET_TRANSFORM = rasterio.Affine(
    0.03393635517805412, 0.0, 99.86528182419725,
    0.0, -0.034221534633331885, 39.39776620081703,
)
TARGET_CRS = "EPSG:4326"
ROOT = Path("data/elite/china/2019/09/24")
OUT = Path("data/metadata/elite-formal-roi-20190924-24h.json.gz.b64")


def fill(arr: np.ndarray) -> np.ndarray:
    valid = np.isfinite(arr)
    if valid.all():
        return arr.astype("float32")
    work = np.where(valid, arr, 0).astype("float32")
    out = fillnodata(work, mask=valid.astype("uint8"), max_search_distance=200)
    return out.astype("float32")


def read_hour(hour: int) -> np.ndarray:
    path = ROOT / f"ELITE_FY4A_LST_20190924_{hour:02d}00_CHINA_K.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    xmin, ymin, xmax, ymax = ROI
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(
            xmin, ymin, xmax, ymax, transform=src.transform
        ).round_offsets().round_lengths()
        raw = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        finite = raw[np.isfinite(raw)]
        if finite.size and float(np.nanmedian(finite)) > 1000:
            raw = raw * 0.01
        finite = raw[np.isfinite(raw)]
        if finite.size and float(np.nanmedian(finite)) > 100:
            raw = raw - 273.15

        out = np.full(TARGET_SHAPE, np.nan, dtype="float32")
        reproject(
            source=np.where(np.isfinite(raw), raw, -9999.0).astype("float32"),
            destination=out,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            src_nodata=-9999.0,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return fill(out)


def main() -> None:
    stack = np.stack([read_hour(h) for h in range(24)], axis=0)
    payload = {
        "schema": "elite-formal-roi-hourly-v1",
        "date": "2019-09-24",
        "roi": ROI,
        "crs": TARGET_CRS,
        "transform": list(TARGET_TRANSFORM)[:6],
        "height": TARGET_SHAPE[0],
        "width": TARGET_SHAPE[1],
        "hours_utc": list(range(24)),
        "lst_c": stack.tolist(),
    }
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded, encoding="ascii")
    print(json.dumps({
        "snapshot": str(OUT),
        "shape": list(stack.shape),
        "mean_c": float(stack.mean()),
        "min_c": float(stack.min()),
        "max_c": float(stack.max()),
        "encoded_chars": len(encoded),
    }, indent=2))


if __name__ == "__main__":
    main()
