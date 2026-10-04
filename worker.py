from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
import requests
from affine import Affine
from pyhdf.SD import SD, SDC
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.warp import reproject

ZENODO_API = "https://zenodo.org/api"
KNOWN_RECORDS = {2019: 10672052, 2021: 8378354, 2022: 14864342}

COFF = LOFF = 1373.5
CFAC = LFAC = 10233137.0
SAT_HEIGHT = 35785863.0

# Repository cache: China extent in EPSG:4326, approximately native 4 km spacing.
CHINA_BBOX = [73.0, 18.0, 135.0, 54.0]
OUT_RES = 0.035932611365
FLOAT_NODATA = -9999.0
CACHE_NODATA = np.uint16(65535)
CACHE_SCALE = 0.01

JULIAN_PATTERN = re.compile(r"(?<!\d)(20\d{2})(\d{3})(\d{2})(\d{2})(?!\d)")
PATTERNS = [
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
]


def parse_ts(name: str) -> datetime | None:
    base = Path(name).name

    # ELITE uses YYYYDDDHHMM, where DDD is day-of-year.
    m = JULIAN_PATTERN.search(base)
    if m:
        year, doy, hour, minute = m.groups()
        try:
            return datetime.strptime(f"{year}{doy}{hour}{minute}", "%Y%j%H%M")
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


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def hourly_timestamps(start: datetime, end: datetime) -> list[datetime]:
    if end <= start:
        raise ValueError("end_date must be after start_date")
    if any((start.minute, start.second, start.microsecond, end.minute, end.second, end.microsecond)):
        raise ValueError("ELITE requests must start/end on exact hourly boundaries")

    out: list[datetime] = []
    current = start
    while current < end:
        out.append(current)
        current += timedelta(hours=1)
    return out


def record(year: int) -> dict[str, Any]:
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


def file_entry(payload: dict[str, Any], filename: str) -> dict[str, Any]:
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


def record_with_file(year: int, filename: str) -> dict[str, Any]:
    """Resolve the Zenodo record that actually owns one monthly archive.

    The FY-4B 2022.6-2023.12 release is published as multiple Zenodo records
    sharing the same dataset title, with one YYYYMM.zip per record.  Therefore
    a single hard-coded record id is not sufficient for arbitrary months.
    """
    primary = record(year)
    try:
        file_entry(primary, filename)
        return primary
    except RuntimeError:
        pass

    queries = [
        f'FY-4B/AGRI hourly 4km seamless LST {year}',
        'ELITE FY-4B AGRI seamless LST',
        'ELITE land surface temperature FY-4B AGRI hourly 4km seamless LST',
    ]
    seen: set[int] = set()
    for query in queries:
        response = requests.get(
            f"{ZENODO_API}/records",
            params={"q": query, "size": 100, "sort": "mostrecent"},
            timeout=90,
        )
        response.raise_for_status()
        hits = ((response.json().get("hits") or {}).get("hits") or [])
        for hit in hits:
            rid = int(hit.get("id") or 0)
            if rid in seen:
                continue
            seen.add(rid)
            title = str((hit.get("metadata") or {}).get("title", "")).lower()
            if "fy-4b/agri" not in title or "seamless lst" not in title:
                continue
            try:
                file_entry(hit, filename)
                return hit
            except RuntimeError:
                continue

    raise RuntimeError(
        f"{filename} was not found in any FY-4B ELITE seamless-LST Zenodo record "
        f"(searched {len(seen)} records)."
    )


def download_http(url: str, path: Path, size: int = 0, checksum: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        append = bool(existing and r.status_code == 206)
        with path.open("ab" if append else "wb") as dst:
            for chunk in r.iter_content(8 * 1024 * 1024):
                if chunk:
                    dst.write(chunk)

    if size and path.stat().st_size != size:
        raise RuntimeError(
            f"Size mismatch for {path.name}: got {path.stat().st_size}, expected {size}"
        )

    if checksum and ":" in checksum:
        algo, expected = checksum.split(":", 1)
        if algo.lower() == "md5":
            h = hashlib.md5()
            with path.open("rb") as src:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest().lower() != expected.lower():
                raise RuntimeError(f"Checksum mismatch for {path.name}")


def datasets(group, prefix: str = ""):
    for key, value in group.items():
        name = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(value, h5py.Dataset):
            yield name, value
        elif isinstance(value, h5py.Group):
            yield from datasets(value, name)


def score(name: str, shape) -> int:
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


def satellite_for_ts(ts: datetime) -> tuple[str, float]:
    # ELITE switches to FY-4B/AGRI for the 2022.6-2023.12 seamless 4 km product.
    # FY-4B was located at 133.0E during 2022.
    if ts >= datetime(2022, 6, 1):
        return "FY4B", 133.0
    return "FY4A", 104.7


def src_crs(ts: datetime) -> CRS:
    _, lon0 = satellite_for_ts(ts)
    return CRS.from_string(
        f"+proj=geos +lon_0={lon0} +h=35785863 +x_0=0 +y_0=0 "
        "+a=6378137 +b=6356752.31414 +units=m +sweep=x +no_defs"
    )


def src_transform(width: int, height: int) -> Affine:
    coff = COFF if width == 2748 else (width - 1) / 2
    loff = LOFF if height == 2748 else (height - 1) / 2
    xs = math.radians((2**16) / CFAC) * SAT_HEIGHT
    ys = math.radians((2**16) / LFAC) * SAT_HEIGHT
    x0 = (0 - coff) * xs
    y0 = (loff - 0) * ys
    return Affine(xs, 0, x0 - xs / 2, 0, -ys, y0 + ys / 2)


def lst_kelvin(path: Path):
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
    values = np.where(invalid, np.nan, values).astype("float32")
    return values, ds_name, attrs


def time_relevant_source_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    keywords = ("time", "date", "utc", "zone", "hour", "minute", "start", "end", "acquisition", "observation", "scan", "nominal")
    out: dict[str, str] = {}
    for key, value in attrs.items():
        text = f"{key} {value}".lower()
        if any(token in text for token in keywords):
            out[str(key)] = str(value)[:1000]
    return out


def china_cache_path(ts: datetime) -> Path:
    satellite, _ = satellite_for_ts(ts)
    return (
        Path("data")
        / "elite"
        / "china"
        / f"{ts:%Y}"
        / f"{ts:%m}"
        / f"{ts:%d}"
        / f"ELITE_{satellite}_LST_{ts:%Y%m%d_%H%M}_CHINA_K.tif"
    )


def write_china_cache(hdf_path: Path, out: Path, ts: datetime) -> dict[str, Any]:
    values, ds_name, source_attrs = lst_kelvin(hdf_path)
    satellite, subpoint_lon = satellite_for_ts(ts)
    source_time_attrs = time_relevant_source_attrs(source_attrs)
    xmin, ymin, xmax, ymax = CHINA_BBOX
    width = max(1, math.ceil((xmax - xmin) / OUT_RES))
    height = max(1, math.ceil((ymax - ymin) / OUT_RES))
    transform = from_origin(xmin, ymax, OUT_RES, OUT_RES)

    dest_kelvin = np.full((height, width), FLOAT_NODATA, dtype="float32")
    reproject(
        source=values,
        destination=dest_kelvin,
        src_transform=src_transform(values.shape[1], values.shape[0]),
        src_crs=src_crs(ts),
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs="EPSG:4326",
        dst_nodata=FLOAT_NODATA,
        resampling=Resampling.nearest,
    )

    valid = dest_kelvin != FLOAT_NODATA
    scaled = np.full((height, width), CACHE_NODATA, dtype="uint16")
    scaled[valid] = np.clip(
        np.rint(dest_kelvin[valid] / CACHE_SCALE),
        0,
        int(CACHE_NODATA) - 1,
    ).astype("uint16")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        nodata=int(CACHE_NODATA),
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(scaled, 1)
        dst.set_band_description(1, f"ELITE_{satellite}_AGRI_LST")
        dst.update_tags(
            source_dataset=ds_name,
            source_time_label=ts.isoformat(),
            source_time_standard="UTC",
            source_time_standard_verified="true",
            source_time_attrs=json.dumps(source_time_attrs, ensure_ascii=False, sort_keys=True),
            scale_factor=str(CACHE_SCALE),
            unit="kelvin",
            cache_extent="china",
            cache_bbox=",".join(str(x) for x in CHINA_BBOX),
            source_grid=f"{satellite}_AGRI_4KM_native_geostationary",
            source_satellite=satellite,
            sub_satellite_longitude_deg=str(subpoint_lon),
        )

    try:
        rio_copy(
            tmp,
            out,
            driver="COG",
            compress="DEFLATE",
            blocksize=512,
            overview_resampling="nearest",
        )
        tmp.unlink(missing_ok=True)
    except Exception:
        tmp.replace(out)

    return {
        "path": str(out),
        "size_bytes": out.stat().st_size,
        "width": width,
        "height": height,
        "valid_pixels": int(valid.sum()),
        "source_dataset": ds_name,
        "source_time_label": ts.isoformat(),
        "source_time_standard": "UTC",
        "source_time_attrs": source_time_attrs,
    }


def validate_roi_bbox(bbox: list[float]) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = bbox
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")

    cxmin, cymin, cxmax, cymax = CHINA_BBOX
    if xmin < cxmin or ymin < cymin or xmax > cxmax or ymax > cymax:
        raise ValueError(
            f"ROI must be inside the repository China cache bbox {CHINA_BBOX}; got {bbox}"
        )


def crop_china_cache(
    cache_path: Path,
    out: Path,
    bbox: list[float],
    output_unit: str,
) -> dict[str, Any]:
    xmin, ymin, xmax, ymax = bbox
    width = max(1, math.ceil((xmax - xmin) / OUT_RES))
    height = max(1, math.ceil((ymax - ymin) / OUT_RES))
    transform = from_origin(xmin, ymax, OUT_RES, OUT_RES)
    dest = np.full((height, width), FLOAT_NODATA, dtype="float32")

    satellite = "FY4B" if "FY4B" in cache_path.name else "FY4A"
    with rasterio.open(cache_path) as src:
        raw = src.read(1)
        source = raw.astype("float32") * CACHE_SCALE
        source[raw == src.nodata] = np.nan
        reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=FLOAT_NODATA,
            resampling=Resampling.nearest,
        )

    valid = dest != FLOAT_NODATA
    if output_unit == "celsius":
        dest[valid] -= 273.15

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
        nodata=FLOAT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as dst:
        dst.write(dest, 1)
        dst.set_band_description(1, f"ELITE_{satellite}_AGRI_LST")
        dst.update_tags(
            output_unit=output_unit,
            source_cache=str(cache_path),
            cache_scale_factor=str(CACHE_SCALE),
            bbox=",".join(str(x) for x in bbox),
        )

    vals = dest[valid]
    return {
        "path": str(out),
        "valid_pixels": int(valid.sum()),
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
    }


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()
    return value[:64] or "roi"


def bbox_hash(bbox: list[float]) -> str:
    canonical = ",".join(f"{x:.6f}" for x in bbox)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


def load_index(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "product": "ELITE AGRI hourly 4 km seamless LST (FY-4A/FY-4B by acquisition date)",
        "storage": "GitHub repository China cache",
        "bbox": CHINA_BBOX,
        "crs": "EPSG:4326",
        "resolution_degrees": OUT_RES,
        "dtype": "uint16",
        "scale_factor": CACHE_SCALE,
        "unit": "kelvin",
        "hours": {},
    }


def save_index(path: Path, index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--output-unit", choices=["celsius", "kelvin"], default="celsius")
    p.add_argument("--region-name", default="")
    args = p.parse_args()

    start = parse_datetime(args.start_date)
    end = parse_datetime(args.end_date)
    requested_hours = hourly_timestamps(start, end)

    bbox = [float(x) for x in args.bbox.split(",")]
    validate_roi_bbox(bbox)

    region_slug = slugify(args.region_name) if args.region_name else "roi"
    region_id = f"{region_slug}-{bbox_hash(bbox)}"

    output_root = Path("output")
    work_dir = output_root / "work"
    result_dir = output_root / "derived" / region_id
    index_path = Path("data") / "metadata" / "elite-index.json"
    index = load_index(index_path)

    cache_hits = 0
    cache_created: list[dict[str, Any]] = []
    archive_downloads: list[dict[str, Any]] = []

    missing_by_month: dict[str, list[datetime]] = {}
    for ts in requested_hours:
        cache = china_cache_path(ts)
        if cache.exists():
            cache_hits += 1
            index["hours"][ts.isoformat()] = str(cache)
        else:
            missing_by_month.setdefault(f"{ts:%Y%m}", []).append(ts)

    for ym, missing_hours in sorted(missing_by_month.items()):
        archive_name = f"{ym}.zip"
        payload = record_with_file(int(ym[:4]), archive_name)
        entry = file_entry(payload, archive_name)
        archive = work_dir / "raw" / f"{ym}.zip"
        download_http(entry["url"], archive, entry["size"], entry["checksum"])
        archive_downloads.append(
            {
                "year_month": ym,
                "source": "zenodo",
                "size_bytes": archive.stat().st_size,
                "temporary": True,
            }
        )

        wanted = set(missing_hours)
        found: dict[datetime, zipfile.ZipInfo] = {}
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in {
                    ".hdf",
                    ".h5",
                    ".hdf5",
                    ".he5",
                }:
                    continue
                ts = parse_ts(info.filename)
                if ts in wanted:
                    found[ts] = info

            missing_in_archive = sorted(wanted - set(found))
            if missing_in_archive:
                raise RuntimeError(
                    "Requested ELITE hours not found in archive: "
                    + ", ".join(x.isoformat() for x in missing_in_archive)
                )

            for ts in sorted(missing_hours):
                info = found[ts]
                hdf = work_dir / "hdf" / Path(info.filename).name
                hdf.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, hdf.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)

                cache = china_cache_path(ts)
                stats = write_china_cache(hdf, cache, ts)
                cache_created.append(
                    {
                        "timestamp": ts.isoformat(),
                        **stats,
                    }
                )
                index["hours"][ts.isoformat()] = str(cache)
                hdf.unlink(missing_ok=True)

        archive.unlink(missing_ok=True)

    if cache_created:
        index["updated_at"] = datetime.utcnow().isoformat() + "Z"
        save_index(index_path, index)
    elif not index_path.exists():
        save_index(index_path, index)

    outputs: list[dict[str, Any]] = []
    for ts in requested_hours:
        cache = china_cache_path(ts)
        if not cache.exists():
            raise RuntimeError(f"China cache was not produced: {cache}")

        suffix = "C" if args.output_unit == "celsius" else "K"
        satellite, _ = satellite_for_ts(ts)
        out = result_dir / f"ELITE_{satellite}_LST_{ts:%Y%m%d_%H%M}_{suffix}.tif"
        stats = crop_china_cache(cache, out, bbox, args.output_unit)
        outputs.append(
            {
                "timestamp": ts.isoformat(),
                "cache_path": str(cache),
                **stats,
            }
        )

    result = {
        "product": "ELITE AGRI hourly 4 km seamless LST (FY-4A/FY-4B by acquisition date)",
        "architecture": "Zenodo temporary archive -> GitHub China cache -> ROI artifact",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bbox": bbox,
        "region_name": args.region_name or None,
        "region_id": region_id,
        "output_unit": args.output_unit,
        "count": len(outputs),
        "files": outputs,
        "repository_cache": {
            "root": "data/elite/china",
            "china_bbox": CHINA_BBOX,
            "crs": "EPSG:4326",
            "resolution_degrees": OUT_RES,
            "dtype": "uint16",
            "scale_factor": CACHE_SCALE,
            "unit": "kelvin",
            "cache_hits": cache_hits,
            "cache_created": len(cache_created),
            "created_files": cache_created,
        },
        "temporary_archive_downloads": archive_downloads,
    }

    output_root.mkdir(exist_ok=True)
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shutil.rmtree(work_dir, ignore_errors=True)

    if not outputs:
        raise RuntimeError("ELITE processing produced zero ROI outputs")


if __name__ == "__main__":
    main()

