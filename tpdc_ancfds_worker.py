from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

DATASET_ID = "4adbc070-afb3-4e9c-85a0-2ce68d1388ad"
DOI = "10.11888/RemoteSen.tpdc.303249"
DATASET_PAGE = f"https://data.tpdc.ac.cn/en/data/{DATASET_ID}"
API_BASE = "https://data.tpdc.ac.cn/file/file"
CACHE_VERSION = "v1"
SOURCE_SCALE_K = 0.1
SOURCE_NODATA = 0
OUTPUT_NODATA = -9999.0
BAND_NAMES = ["T_DIR", "T_NADIR", "T_HEMI"]


class TPDCClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "remote-sensing-mcp/0.11 TPDC-ANCFDS",
                "Accept": "application/json, application/octet-stream;q=0.9, */*;q=0.8",
            }
        )
        self._lst_folder: dict[str, Any] | None = None
        self._years: dict[str, dict[str, Any]] | None = None
        self._days: dict[str, dict[str, dict[str, Any]]] = {}

    def _get(self, path: str, **params: str) -> list[dict[str, Any]]:
        response = self.session.get(
            API_BASE + path,
            params=params,
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        if str(payload.get("code")) != "200":
            raise RuntimeError(f"TPDC API failed: {payload}")
        data = payload.get("data") or []
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected TPDC response: {payload}")
        return data

    def lst_folder(self) -> dict[str, Any]:
        if self._lst_folder is None:
            root = self._get("/getRootFileDataList", metadataId=DATASET_ID)
            self._lst_folder = next(
                (item for item in root if item.get("name") == "LST" and item.get("type") == "dir"),
                None,
            )
            if self._lst_folder is None:
                raise LookupError("TPDC ANCFDS-LST root folder was not found.")
        return self._lst_folder

    def year_folder(self, year: str) -> dict[str, Any]:
        if self._years is None:
            items = self._get("/getFileDataList", parentId=str(self.lst_folder()["id"]))
            self._years = {
                str(item.get("name")): item
                for item in items
                if item.get("type") == "dir"
            }
        folder = self._years.get(year)
        if folder is None:
            raise LookupError(f"ANCFDS-LST year {year} was not found on TPDC.")
        return folder

    def day_folder(self, day: datetime) -> dict[str, Any]:
        year = day.strftime("%Y")
        if year not in self._days:
            items = self._get(
                "/getFileDataList",
                parentId=str(self.year_folder(year)["id"]),
            )
            self._days[year] = {
                str(item.get("name")): item
                for item in items
                if item.get("type") == "dir"
            }
        key = day.strftime("%Y%m%d")
        folder = self._days[year].get(key)
        if folder is None:
            raise LookupError(f"ANCFDS-LST day {key} was not found on TPDC.")
        return folder

    def hourly_files(self, day: datetime) -> dict[int, dict[str, Any]]:
        items = self._get(
            "/getFileDataList",
            parentId=str(self.day_folder(day)["id"]),
        )
        prefix = f"FY4A_AGRI_{day:%Y%m%d}_"
        result: dict[int, dict[str, Any]] = {}
        for item in items:
            name = str(item.get("name") or "")
            if item.get("type") != "file" or not name.startswith(prefix):
                continue
            match = re.search(r"_(\d{2})0000_AllWeatherLST001\.tif$", name)
            if match:
                result[int(match.group(1))] = item
        return result

    def download(self, item: dict[str, Any], target: Path) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        url = API_BASE + "/batchDownloadByFileId"
        with self.session.post(
            url,
            params={"fileId": str(item["id"])},
            stream=True,
            timeout=(60, 900),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "json" in content_type.lower():
                raise RuntimeError(
                    f"TPDC returned JSON instead of a file: {response.text[:500]}"
                )
            total = 0
            with target.open("wb") as stream:
                for chunk in response.iter_content(2 * 1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        total += len(chunk)
        expected = int(item.get("size") or 0)
        if expected and total != expected:
            raise RuntimeError(
                f"Incomplete TPDC download for {item.get('name')}: expected {expected}, got {total}"
            )
        return total


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value[:10])


def parse_dates(start_date: str, end_date: str) -> list[datetime]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")
    days: list[datetime] = []
    current = start
    while current < end:
        days.append(current)
        current += timedelta(days=1)
    if len(days) > 31:
        raise ValueError("One ANCFDS-LST request may span at most 31 calendar days.")
    return days


def parse_hours(value: str) -> list[int]:
    text = value.strip().lower()
    if not text or text == "all":
        return list(range(24))
    hours = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not hours or any(hour < 0 or hour > 23 for hour in hours):
        raise ValueError("hours must be 'all' or comma-separated integers from 0 to 23")
    return hours


def parse_bbox(value: str) -> list[float]:
    bbox = [float(part) for part in value.split(",")]
    if len(bbox) != 4:
        raise ValueError("bbox must be xmin,ymin,xmax,ymax")
    xmin, ymin, xmax, ymax = bbox
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError("Invalid WGS84 bbox")
    return bbox


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-_").lower()[:64] or "roi"


def bbox_hash(bbox: list[float]) -> str:
    canonical = ",".join(f"{value:.6f}" for value in bbox)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]


def region_id(region_name: str, bbox: list[float]) -> str:
    return f"{slugify(region_name or 'roi')}-{bbox_hash(bbox)}"


def cache_path(
    rid: str,
    day: datetime,
    hour: int,
    output_unit: str,
) -> Path:
    suffix = "C" if output_unit == "celsius" else "K"
    return (
        Path("data")
        / "tpdc_ancfds"
        / CACHE_VERSION
        / rid
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{day:%d}"
        / f"ANCFDS_FY4A_{day:%Y%m%d}_{hour:02d}00_{suffix}.tif"
    )


def crop_and_convert(
    source: Path,
    target: Path,
    bbox: list[float],
    output_unit: str,
    source_item: dict[str, Any],
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        if src.count < 3:
            raise RuntimeError(f"Expected 3 ANCFDS bands, found {src.count}")
        if src.crs is None:
            raise RuntimeError("ANCFDS source GeoTIFF has no CRS.")

        requested = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox,
            densify_pts=21,
        )
        window = from_bounds(*requested, transform=src.transform)
        window = window.round_offsets().round_lengths()
        full = Window(0, 0, src.width, src.height)
        try:
            window = window.intersection(full)
        except Exception as exc:
            raise ValueError("Requested bbox does not intersect the ANCFDS source raster.") from exc
        if window.width <= 0 or window.height <= 0:
            raise ValueError("Requested bbox does not intersect the ANCFDS source raster.")

        raw = src.read(indexes=[1, 2, 3], window=window)
        valid = raw != SOURCE_NODATA
        values = raw.astype("float32") * SOURCE_SCALE_K
        if output_unit == "celsius":
            values -= 273.15
        values[~valid] = OUTPUT_NODATA

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            nodata=OUTPUT_NODATA,
            count=3,
            width=int(window.width),
            height=int(window.height),
            transform=src.window_transform(window),
            compress="deflate",
        )
        descriptions = [
            f"{name}_{'C' if output_unit == 'celsius' else 'K'}"
            for name in BAND_NAMES
        ]
        with rasterio.open(target, "w", **profile) as dst:
            dst.write(values)
            for idx, description in enumerate(descriptions, start=1):
                dst.set_band_description(idx, description)
            dst.update_tags(
                source_dataset="ANCFDS-LST",
                source_doi=DOI,
                source_dataset_id=DATASET_ID,
                source_file=str(source_item.get("name") or ""),
                source_file_id=str(source_item.get("id") or ""),
                source_file_path=str(source_item.get("path") or ""),
                source_scale_factor_kelvin=str(SOURCE_SCALE_K),
                source_nodata=str(SOURCE_NODATA),
                requested_bbox_wgs84=",".join(str(x) for x in bbox),
                output_unit=output_unit,
                band_1="T_dir",
                band_2="T_nadir",
                band_3="T_hemi",
            )

        valid_counts = [int(valid[index].sum()) for index in range(3)]
        stats = {}
        for index, name in enumerate(BAND_NAMES):
            band_valid = values[index][valid[index]]
            stats[name] = {
                "valid_pixels": int(band_valid.size),
                "min": float(band_valid.min()) if band_valid.size else None,
                "max": float(band_valid.max()) if band_valid.size else None,
                "mean": float(band_valid.mean()) if band_valid.size else None,
            }

        return {
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "crs": str(src.crs),
            "shape": [3, int(window.height), int(window.width)],
            "band_names": descriptions,
            "valid_pixels": valid_counts,
            "stats": stats,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--region-name", default="")
    parser.add_argument("--hours", default="all")
    parser.add_argument("--output-unit", choices=["celsius", "kelvin"], default="celsius")
    args = parser.parse_args()

    days = parse_dates(args.start_date, args.end_date)
    hours = parse_hours(args.hours)
    requested_files = len(days) * len(hours)
    if requested_files > 24:
        raise ValueError(
            "One ANCFDS-LST job is limited to 24 source hours because each TPDC "
            "full-domain GeoTIFF is roughly 100-140 MB. Split larger requests."
        )
    bbox = parse_bbox(args.bbox)
    rid = region_id(args.region_name, bbox)

    output_root = Path("output") / "tpdc_ancfds"
    work_root = Path("output") / "work" / "tpdc_ancfds"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    client = TPDCClient()
    records: list[dict[str, Any]] = []
    downloaded_bytes = 0

    for day in days:
        remote = client.hourly_files(day)
        missing_hours = [hour for hour in hours if hour not in remote]
        if missing_hours:
            raise LookupError(
                f"TPDC ANCFDS-LST {day:%Y-%m-%d} is missing requested hours: {missing_hours}"
            )

        for hour in hours:
            item = remote[hour]
            cache = cache_path(rid, day, hour, args.output_unit)
            artifact = output_root / cache.name
            cache_hit = cache.exists()
            source_bytes = 0

            if cache_hit:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache, artifact)
                with rasterio.open(cache) as src:
                    record = {
                        "path": str(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "crs": str(src.crs),
                        "shape": [src.count, src.height, src.width],
                        "band_names": list(src.descriptions),
                    }
            else:
                raw = work_root / str(item["name"])
                source_bytes = client.download(item, raw)
                downloaded_bytes += source_bytes
                record = crop_and_convert(
                    raw,
                    cache,
                    bbox,
                    args.output_unit,
                    item,
                )
                artifact.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache, artifact)
                raw.unlink(missing_ok=True)

            record.update(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "hour_utc": hour,
                    "cache_hit": cache_hit,
                    "cache_path": str(cache),
                    "source_file": item.get("name"),
                    "source_file_id": item.get("id"),
                    "source_size_bytes": int(item.get("size") or 0),
                    "source_bytes_downloaded_this_run": source_bytes,
                }
            )
            records.append(record)
            print(
                f"READY {day:%Y-%m-%d} {hour:02d}:00 UTC "
                f"cache_hit={cache_hit} output={artifact}",
                flush=True,
            )

    result = {
        "dataset": "ANCFDS-LST",
        "doi": DOI,
        "dataset_id": DATASET_ID,
        "dataset_page": DATASET_PAGE,
        "source": "TPDC public file API",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "hours": hours,
        "bbox_wgs84": bbox,
        "region_name": args.region_name or None,
        "region_id": rid,
        "output_unit": args.output_unit,
        "band_semantics": {
            "1": "T_dir: directional LST in the FY-4A/AGRI viewing direction",
            "2": "T_nadir: angular-normalized nadir LST",
            "3": "T_hemi: hemispherical-equivalent LST",
        },
        "source_encoding": {
            "dtype": "uint16",
            "nodata": SOURCE_NODATA,
            "scale_factor_kelvin": SOURCE_SCALE_K,
        },
        "requested_source_hours": requested_files,
        "source_bytes_downloaded_this_run": downloaded_bytes,
        "records": records,
    }
    (Path("output") / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
