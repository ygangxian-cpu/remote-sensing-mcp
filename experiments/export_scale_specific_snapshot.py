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

DATE = "2019-09-24"
HOUR = 4
ROI = [99.86, 38.67, 100.5, 39.43]
ELITE = Path("data/elite/china/2019/09/24/ELITE_FY4A_LST_20190924_0400_CHINA_K.tif")

# Canonical 21 x 18 analysis grid used by the existing fair experiment.
TARGET_SHAPE = (21, 18)
TARGET_TRANSFORM = rasterio.Affine(
    0.03393635517805412, 0.0, 99.86528182419725,
    0.0, -0.034221534633331885, 39.39776620081703,
)
TARGET_CRS = "EPSG:4326"


def continuous_fill(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=True)
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if not valid.any():
        raise RuntimeError("array contains no valid pixels")
    work = np.where(valid, arr, 0).astype("float32")
    out = fillnodata(work, mask=valid.astype("uint8"), max_search_distance=200)
    if not np.isfinite(out).all():
        from scipy import ndimage
        _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
        nearest = arr[tuple(idx)]
        missing = ~np.isfinite(out)
        out[missing] = nearest[missing]
    return out.astype("float32")


def elite_target() -> np.ndarray:
    xmin, ymin, xmax, ymax = ROI
    with rasterio.open(ELITE) as src:
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
    return continuous_fill(out)


def find_era5() -> Path:
    matches = sorted(Path("data/era5_land/v1").glob(
        "*/2019/09/24/ERA5LAND_20190924_0400_UTC.tif"
    ))
    if not matches:
        raise FileNotFoundError("ERA5-Land 2019-09-24 04:00 cache not found")
    return matches[-1]


def era5_payload(path: Path) -> dict:
    with rasterio.open(path) as src:
        bands = {}
        for i, desc in enumerate(src.descriptions, start=1):
            name = desc or f"B{i}"
            arr = src.read(i, masked=True).astype("float32").filled(np.nan)
            bands[name] = np.where(np.isfinite(arr), arr, None).tolist()
        return {
            "source_path": str(path),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "bands": bands,
        }


def main() -> None:
    payload = {
        "schema": "scale-specific-rf-v1-bootstrap",
        "date": DATE,
        "hour_utc": HOUR,
        "roi": ROI,
        "target": {
            "source": str(ELITE),
            "grid_crs": TARGET_CRS,
            "grid_transform": list(TARGET_TRANSFORM)[:6],
            "shape": list(TARGET_SHAPE),
            "elite_lst_c": elite_target().tolist(),
        },
        "era5_land": era5_payload(find_era5()),
    }
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    print("SCALE_SPECIFIC_SNAPSHOT_B64_BEGIN")
    for i in range(0, len(encoded), 4000):
        print(encoded[i:i + 4000])
    print("SCALE_SPECIFIC_SNAPSHOT_B64_END")
    print(json.dumps({
        "payload_bytes": len(raw),
        "encoded_chars": len(encoded),
        "target_shape": payload["target"]["shape"],
        "era5_shape": [payload["era5_land"]["height"], payload["era5_land"]["width"]],
        "era5_bands": list(payload["era5_land"]["bands"]),
    }, indent=2))


if __name__ == "__main__":
    main()
