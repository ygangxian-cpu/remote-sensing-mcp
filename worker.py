from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
from pyhdf.SD import SD, SDC
import numpy as np
import requests
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

ZENODO_API = "https://zenodo.org/api"
KNOWN_RECORDS = {2019: 10672052, 2021: 8378354}
COFF = LOFF = 1373.5
CFAC = LFAC = 10233137.0
SAT_HEIGHT = 35785863.0
OUT_RES = 0.035932611365

JULIAN_PATTERN = re.compile(r"(?<!\d)(20\d{2})(\d{3})(\d{2})(\d{2})(?!\d)")

PATTERNS = [
    # YYYYMMDDHHMMSS — common FY-4A product timestamp form
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    # YYYYMMDDHHMM
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(?!\d)"),
    # YYYYMMDDHH
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(?!\d)"),
    # YYYYMMDD
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
]


def parse_ts(name: str):
    base = Path(name).name

    # ELITE filenames use YYYYDDDHHMM, where DDD is day-of-year.
    # Example: 20192440000 -> 2019, DOY 244, 00:00.
    m = JULIAN_PATTERN.search(base)
    if m:
        year, doy, hour, minute = m.groups()
        try:
            return datetime.strptime(
                f"{year}{doy}{hour}{minute}",
                "%Y%j%H%M",
            )
        except ValueError:
            pass

    for pattern in PATTERNS:
        m = pattern.search(base)
        if not m:
            continue
        p = [int(x) for x in m.groups()]
        try:
            if len(p) == 6:
                return datetime(*p)
            if len(p) == 5:
                return datetime(*p)
            if len(p) == 4:
                return datetime(p[0], p[1], p[2], p[3], 0)
            return datetime(p[0], p[1], p[2])
        except ValueError:
            pass
    return None


def months(start: datetime, end: datetime):
    y, m = start.year, start.month
    out = []
    while True:
        month_start = datetime(y, m, 1)
        if month_start >= end and (y, m) != (start.year, start.month):
            break
        out.append(f"{y:04d}{m:02d}")
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def record(year: int):
    rid = KNOWN_RECORDS.get(year)
    if rid:
        r = requests.get(f"{ZENODO_API}/records/{rid}", timeout=60)
        r.raise_for_status()
        return r.json()
    r = requests.get(
        f"{ZENODO_API}/records",
        params={"q": f"FY-4A AGRI seamless LST {year}", "size": 50, "sort": "mostrecent"},
        timeout=60,
    )
    r.raise_for_status()
    for hit in ((r.json().get("hits") or {}).get("hits") or []):
        title = str((hit.get("metadata") or {}).get("title", "")).lower()
        if "fy-4a/agri" in title and "seamless lst" in title and str(year) in title:
            return hit
    raise RuntimeError(f"No ELITE record found for {year}")


def file_entry(payload, filename: str):
    files = payload.get("files") or {}
    entries = files.get("entries", files) if isinstance(files, dict) else files
    items = list(entries.values()) if isinstance(entries, dict) else list(entries or [])
    for item in items:
        key = str(item.get("key") or item.get("filename") or "")
        if key == filename:
            links = item.get("links") or {}
            return {
                "size": int(item.get("size") or 0),
                "checksum": item.get("checksum"),
                "url": links.get("content")
                or links.get("self")
                or f"https://zenodo.org/records/{payload['id']}/files/{filename}?download=1",
            }
    raise RuntimeError(f"{filename} not found in record {payload.get('id')}")


def download(url: str, path: Path, size: int = 0, checksum: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        append = existing and r.status_code == 206
        with path.open("ab" if append else "wb") as f:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    if size and path.stat().st_size != size:
        raise RuntimeError(f"Size mismatch for {path.name}")
    if checksum and ":" in checksum:
        algo, expected = checksum.split(":", 1)
        if algo.lower() == "md5":
            h = hashlib.md5()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest().lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {path.name}")


def datasets(group, prefix=""):
    for key, value in group.items():
        name = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(value, h5py.Dataset):
            yield name, value
        elif isinstance(value, h5py.Group):
            yield from datasets(value, name)


def score(name: str, shape):
    n = name.lower()
    s = 0
    if "lst" in n:
        s += 100
    if "land" in n and "temperature" in n:
        s += 80
    if "temperature" in n or "temp" in n:
        s += 25
    if len(shape) == 2:
        s += 20
    if tuple(shape) == (2748, 2748):
        s += 30
    if any(k in n for k in ("qa", "qc", "flag", "lon", "lat")):
        s -= 100
    return s


def read_lst(path: Path):
    # Try HDF5 first.
    try:
        with h5py.File(path, "r") as h5:
            ranked = sorted(datasets(h5), key=lambda x: score(x[0], x[1].shape), reverse=True)
            if not ranked or score(ranked[0][0], ranked[0][1].shape) <= 0:
                raise RuntimeError(f"Cannot auto-detect LST dataset in {path}")
            name, ds = ranked[0]
            arr = np.asarray(ds[...], dtype="float32")
            attrs = {str(k): v for k, v in ds.attrs.items()}
        return arr, name, attrs
    except OSError:
        pass

    # ELITE .hdf archives are HDF4. Fall back to pyhdf.
    h4 = SD(str(path), SDC.READ)
    try:
        candidates = []
        for name, meta in h4.datasets().items():
            shape = tuple(meta[1]) if len(meta) > 1 else ()
            candidates.append((score(name, shape), name, shape))
        candidates.sort(reverse=True)
        if not candidates or candidates[0][0] <= 0:
            raise RuntimeError(f"Cannot auto-detect LST dataset in HDF4 file {path}")
        _, name, _ = candidates[0]
        ds = h4.select(name)
        arr = np.asarray(ds[:], dtype="float32")
        attrs = {str(k): v for k, v in ds.attributes().items()}
        return arr, name, attrs
    finally:
        h4.end()


def src_crs():
    return CRS.from_string(
        "+proj=geos +lon_0=104.7 +h=35785863 +x_0=0 +y_0=0 "
        "+a=6378137 +b=6356752.31414 +units=m +sweep=x +no_defs"
    )


def src_transform(width: int, height: int):
    coff = COFF if width == 2748 else (width - 1) / 2
    loff = LOFF if height == 2748 else (height - 1) / 2
    xs = math.radians((2**16) / CFAC) * SAT_HEIGHT
    ys = math.radians((2**16) / LFAC) * SAT_HEIGHT
    x0 = (0 - coff) * xs
    y0 = (loff - 0) * ys
    return Affine(xs, 0, x0 - xs / 2, 0, -ys, y0 + ys / 2)


def convert(path: Path, out: Path, bbox: list[float], unit: str):
    raw, ds_name, attrs = read_lst(path)
    invalid = ~np.isfinite(raw)
    for key in ("_FillValue", "FillValue", "fill_value", "missing_value"):
        v = attrs.get(key)
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, (int, float)):
            invalid |= raw == float(v)
    values = raw * 0.01
    invalid |= (values < 150) | (values > 400)
    if unit == "celsius":
        values = values - 273.15
    values = np.where(invalid, np.nan, values).astype("float32")

    xmin, ymin, xmax, ymax = bbox
    width = max(1, math.ceil((xmax - xmin) / OUT_RES))
    height = max(1, math.ceil((ymax - ymin) / OUT_RES))
    transform = from_origin(xmin, ymax, OUT_RES, OUT_RES)
    nodata = -9999.0
    dest = np.full((height, width), nodata, dtype="float32")
    reproject(
        source=values,
        destination=dest,
        src_transform=src_transform(values.shape[1], values.shape[0]),
        src_crs=src_crs(),
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        dst_nodata=nodata,
        resampling=Resampling.nearest,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as dst:
        dst.write(dest, 1)
        dst.set_band_description(1, "ELITE_FY4A_AGRI_LST")
        dst.update_tags(source_dataset=ds_name, scale_factor="0.01", output_unit=unit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--output-unit", choices=["celsius", "kelvin"], default="celsius")
    args = p.parse_args()

    start = datetime.fromisoformat(args.start_date.replace("Z", "+00:00")).replace(tzinfo=None)
    end = datetime.fromisoformat(args.end_date.replace("Z", "+00:00")).replace(tzinfo=None)
    if end <= start:
        raise ValueError("end_date must be after start_date")
    bbox = [float(x) for x in args.bbox.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")

    root = Path("output")
    cache = root / "cache"
    hdfs = root / "hdf"
    result_dir = root / "elite"
    converted = []
    archive_samples = []
    parsed_samples = []
    candidate_count = 0

    for ym in months(start, end):
        payload = record(int(ym[:4]))
        entry = file_entry(payload, f"{ym}.zip")
        archive = cache / f"{ym}.zip"
        download(entry["url"], archive, entry["size"], entry["checksum"])
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in {".hdf", ".h5", ".hdf5", ".he5"}:
                    continue
                candidate_count += 1
                if len(archive_samples) < 30:
                    archive_samples.append(info.filename)
                ts = parse_ts(info.filename)
                if ts is not None and len(parsed_samples) < 30:
                    parsed_samples.append({"name": info.filename, "timestamp": ts.isoformat()})
                if ts is None or not (start <= ts < end):
                    continue
                hdf = hdfs / Path(info.filename).name
                hdf.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, hdf.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                out = result_dir / f"ELITE_FY4A_LST_{ts.strftime('%Y%m%d_%H%M')}.tif"
                convert(hdf, out, bbox, args.output_unit)
                converted.append(str(out))
                hdf.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)

    root.mkdir(exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(
            {
                "product": "ELITE FY-4A/AGRI hourly 4 km seamless LST",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "bbox": bbox,
                "output_unit": args.output_unit,
                "count": len(converted),
                "files": converted,
                "candidate_hdf_count": candidate_count,
                "archive_filename_samples": archive_samples,
                "parsed_timestamp_samples": parsed_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not converted:
        raise RuntimeError(
            "ELITE archive was downloaded but produced zero GeoTIFFs. "
            f"candidate_hdf_count={candidate_count}; "
            f"sample_names={archive_samples[:5]}"
        )


if __name__ == "__main__":
    main()
