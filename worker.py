from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
import requests
from affine import Affine
from google.cloud import storage
from google.oauth2 import service_account
from pyhdf.SD import SD, SDC
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.warp import reproject

ZENODO_API = "https://zenodo.org/api"
KNOWN_RECORDS = {2019: 10672052, 2021: 8378354}

COFF = LOFF = 1373.5
CFAC = LFAC = 10233137.0
SAT_HEIGHT = 35785863.0
ROI_RES = 0.035932611365
NODATA = -9999.0

JULIAN_PATTERN = re.compile(r"(?<!\d)(20\d{2})(\d{3})(\d{2})(\d{2})(?!\d)")
PATTERNS = [
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
]


def parse_ts(name: str) -> datetime | None:
    base = Path(name).name
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


def months(start: datetime, end: datetime) -> list[str]:
    y, m = start.year, start.month
    out: list[str] = []
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


def download_http(url: str, path: Path, size: int = 0, checksum: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        append = bool(existing and r.status_code == 206)
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


def _credentials_from_env():
    raw = os.getenv("GCP_SERVICE_ACCOUNT_JSON", "")
    raw_b64 = os.getenv("GCP_SERVICE_ACCOUNT_JSON_BASE64", "")
    if not raw and raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        return None, None
    info = json.loads(raw)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return credentials, info.get("project_id")


class ObjectStore:
    """Optional long-lived GCS backing store."""

    def __init__(self) -> None:
        self.bucket_name = (
            os.getenv("REMOTE_DATA_BUCKET", "").strip()
            or os.getenv("GEE_GCS_BUCKET", "").strip()
        )
        self.enabled = bool(self.bucket_name)
        self.client = None
        self.bucket = None
        if self.enabled:
            credentials, project = _credentials_from_env()
            self.client = storage.Client(project=project, credentials=credentials)
            self.bucket = self.client.bucket(self.bucket_name)

    def uri(self, key: str) -> str | None:
        return f"gs://{self.bucket_name}/{key}" if self.enabled else None

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        return bool(self.bucket.blob(key).exists(self.client))

    def download(self, key: str, path: Path) -> bool:
        if not self.enabled or not self.exists(key):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(key).download_to_filename(str(path))
        return True

    def upload(
        self,
        path: Path,
        key: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        blob = self.bucket.blob(key)
        if metadata:
            blob.metadata = metadata
        blob.upload_from_filename(str(path), content_type=content_type)
        return self.uri(key)


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


def src_crs() -> CRS:
    return CRS.from_string(
        "+proj=geos +lon_0=104.7 +h=35785863 +x_0=0 +y_0=0 "
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
    return values, ds_name


def write_full_disk_cache(hdf_path: Path, out: Path) -> dict[str, Any]:
    values, ds_name = lst_kelvin(hdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.tif")
    data = np.where(np.isfinite(values), values, NODATA).astype("float32")

    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=src_crs(),
        transform=src_transform(data.shape[1], data.shape[0]),
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "ELITE_FY4A_AGRI_LST")
        dst.update_tags(
            source_dataset=ds_name,
            scale_factor="0.01",
            storage_layer="cache",
            cache_unit="kelvin",
            grid="FY4A_AGRI_4KM_native_geostationary",
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
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "dataset": ds_name,
    }


def crop_cache_to_roi(
    cache_path: Path,
    out: Path,
    bbox: list[float],
    output_unit: str,
) -> dict[str, Any]:
    xmin, ymin, xmax, ymax = bbox
    width = max(1, math.ceil((xmax - xmin) / ROI_RES))
    height = max(1, math.ceil((ymax - ymin) / ROI_RES))
    transform = from_origin(xmin, ymax, ROI_RES, ROI_RES)
    dest = np.full((height, width), NODATA, dtype="float32")

    with rasterio.open(cache_path) as src:
        source = src.read(1).astype("float32")
        source = np.where(source == src.nodata, np.nan, source)
        reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=NODATA,
            resampling=Resampling.nearest,
        )

    valid = dest != NODATA
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
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as dst:
        dst.write(dest, 1)
        dst.set_band_description(1, "ELITE_FY4A_AGRI_LST")
        dst.update_tags(
            output_unit=output_unit,
            storage_layer="derived",
            source_cache_unit="kelvin",
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


def raw_key(ym: str) -> str:
    return f"raw/elite/{ym[:4]}/{ym[4:6]}/{ym}.zip"


def cache_key(ts: datetime) -> str:
    return (
        f"cache/elite/{ts:%Y/%m/%d}/"
        f"ELITE_FY4A_LST_{ts:%Y%m%d_%H%M}_FULLDISK_K.tif"
    )


def derived_key(region_id: str, ts: datetime, output_unit: str) -> str:
    suffix = "C" if output_unit == "celsius" else "K"
    return (
        f"derived/elite/{region_id}/{ts:%Y/%m/%d}/"
        f"ELITE_FY4A_LST_{ts:%Y%m%d_%H%M}_{suffix}.tif"
    )


def ensure_raw_archive(
    store: ObjectStore,
    ym: str,
    work_dir: Path,
) -> tuple[Path, str]:
    local = work_dir / "raw" / f"{ym}.zip"
    key = raw_key(ym)
    if store.download(key, local):
        return local, "gcs_hit"

    payload = record(int(ym[:4]))
    source_entry = file_entry(payload, f"{ym}.zip")
    download_http(
        source_entry["url"],
        local,
        source_entry["size"],
        source_entry["checksum"],
    )
    store.upload(
        local,
        key,
        content_type="application/zip",
        metadata={
            "source": "Zenodo",
            "source_checksum": str(source_entry.get("checksum") or ""),
            "year_month": ym,
        },
    )
    return local, "zenodo_download"


def matching_infos(
    archive: Path,
    start: datetime,
    end: datetime,
) -> tuple[list[tuple[zipfile.ZipInfo, datetime]], int, list[str]]:
    matches: list[tuple[zipfile.ZipInfo, datetime]] = []
    candidate_count = 0
    samples: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".hdf", ".h5", ".hdf5", ".he5"}:
                continue
            candidate_count += 1
            if len(samples) < 10:
                samples.append(info.filename)
            ts = parse_ts(info.filename)
            if ts is not None and start <= ts < end:
                matches.append((info, ts))
    return matches, candidate_count, samples


def ensure_cache(
    store: ObjectStore,
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    ts: datetime,
    work_dir: Path,
) -> tuple[Path, str, str | None]:
    local_cache = work_dir / "cache" / f"ELITE_FY4A_LST_{ts:%Y%m%d_%H%M}_FULLDISK_K.tif"
    key = cache_key(ts)

    if store.download(key, local_cache):
        return local_cache, "cache_hit", store.uri(key)

    hdf = work_dir / "hdf" / Path(info.filename).name
    hdf.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as src, hdf.open("wb") as dst:
        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)

    write_full_disk_cache(hdf, local_cache)
    hdf.unlink(missing_ok=True)
    uri = store.upload(
        local_cache,
        key,
        content_type="image/tiff",
        metadata={
            "product": "ELITE FY-4A AGRI hourly 4 km LST",
            "timestamp": ts.isoformat(),
            "unit": "kelvin",
            "storage_layer": "cache",
        },
    )
    return local_cache, "cache_created", uri


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--bbox", required=True)
    p.add_argument("--output-unit", choices=["celsius", "kelvin"], default="celsius")
    p.add_argument("--region-name", default="")
    args = p.parse_args()

    start = datetime.fromisoformat(args.start_date.replace("Z", "+00:00")).replace(tzinfo=None)
    end = datetime.fromisoformat(args.end_date.replace("Z", "+00:00")).replace(tzinfo=None)
    if end <= start:
        raise ValueError("end_date must be after start_date")

    bbox = [float(x) for x in args.bbox.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = bbox
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")

    region_slug = slugify(args.region_name) if args.region_name else "roi"
    region_id = f"{region_slug}-{bbox_hash(bbox)}"

    output_root = Path("output")
    work_dir = output_root / "work"
    result_dir = output_root / "derived"
    store = ObjectStore()

    outputs: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    cache_hits = 0
    cache_created = 0
    derived_hits = 0
    derived_created = 0
    archive_candidate_count = 0
    archive_samples: list[str] = []

    for ym in months(start, end):
        archive, raw_source = ensure_raw_archive(store, ym, work_dir)
        raw_events.append(
            {
                "year_month": ym,
                "source": raw_source,
                "local_path": str(archive),
                "remote_uri": store.uri(raw_key(ym)),
            }
        )

        matches, candidate_count, samples = matching_infos(archive, start, end)
        archive_candidate_count += candidate_count
        archive_samples.extend(x for x in samples if x not in archive_samples)
        if not matches:
            continue

        with zipfile.ZipFile(archive) as zf:
            for info, ts in matches:
                d_key = derived_key(region_id, ts, args.output_unit)
                out = (
                    result_dir
                    / region_id
                    / f"ELITE_FY4A_LST_{ts:%Y%m%d_%H%M}_{'C' if args.output_unit == 'celsius' else 'K'}.tif"
                )

                if store.download(d_key, out):
                    derived_hits += 1
                    outputs.append(
                        {
                            "timestamp": ts.isoformat(),
                            "path": str(out),
                            "remote_uri": store.uri(d_key),
                            "derived_status": "derived_hit",
                        }
                    )
                    continue

                cache_path, cache_status, cache_uri = ensure_cache(
                    store, zf, info, ts, work_dir
                )
                if cache_status == "cache_hit":
                    cache_hits += 1
                else:
                    cache_created += 1

                stats = crop_cache_to_roi(
                    cache_path,
                    out,
                    bbox,
                    args.output_unit,
                )
                derived_created += 1
                remote_uri = store.upload(
                    out,
                    d_key,
                    content_type="image/tiff",
                    metadata={
                        "region_id": region_id,
                        "timestamp": ts.isoformat(),
                        "output_unit": args.output_unit,
                        "bbox": ",".join(str(x) for x in bbox),
                        "source_cache": cache_uri or cache_key(ts),
                        "storage_layer": "derived",
                    },
                )
                outputs.append(
                    {
                        "timestamp": ts.isoformat(),
                        "path": str(out),
                        "remote_uri": remote_uri,
                        "cache_uri": cache_uri,
                        "cache_status": cache_status,
                        "derived_status": "derived_created",
                        "valid_pixels": stats["valid_pixels"],
                        "min": stats["min"],
                        "max": stats["max"],
                    }
                )

                cache_path.unlink(missing_ok=True)

        archive.unlink(missing_ok=True)

    result = {
        "product": "ELITE FY-4A/AGRI hourly 4 km seamless LST",
        "architecture": "raw -> cache -> derived",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "bbox": bbox,
        "region_name": args.region_name or None,
        "region_id": region_id,
        "output_unit": args.output_unit,
        "count": len(outputs),
        "files": outputs,
        "storage": {
            "persistent": store.enabled,
            "backend": "gcs" if store.enabled else "ephemeral-local",
            "bucket": store.bucket_name or None,
            "raw_prefix": "raw/elite/",
            "cache_prefix": "cache/elite/",
            "derived_prefix": f"derived/elite/{region_id}/",
        },
        "stats": {
            "raw_events": raw_events,
            "cache_hits": cache_hits,
            "cache_created": cache_created,
            "derived_hits": derived_hits,
            "derived_created": derived_created,
            "archive_candidate_hdf_count": archive_candidate_count,
            "archive_filename_samples": archive_samples[:10],
        },
    }

    output_root.mkdir(exist_ok=True)
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shutil.rmtree(work_dir, ignore_errors=True)

    if not outputs:
        raise RuntimeError(
            "ELITE archive processing produced zero outputs. "
            f"candidate_hdf_count={archive_candidate_count}; "
            f"sample_names={archive_samples[:5]}"
        )


if __name__ == "__main__":
    main()
