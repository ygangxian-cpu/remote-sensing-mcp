from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import h5py
import requests
from pyhdf.SD import SD, SDC
from remotezip import RemoteZip

ZENODO_API = "https://zenodo.org/api/records/10672052"
MONTH = "201909.zip"
TARGET_TOKEN = "20192670400"  # 2019 DOY 267, 04:00 as encoded by current worker parser

TIME_KEYWORDS = (
    "time", "date", "utc", "zone", "hour", "minute", "start", "end",
    "acquisition", "observation", "scan", "nominal", "calendar",
)


def scalarize(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [scalarize(x) for x in value[:100]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def relevant_attrs(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        key = str(k)
        val = scalarize(v)
        text = f"{key} {val}".lower()
        if any(token in text for token in TIME_KEYWORDS):
            out[key] = val
    return out


def zenodo_month_url() -> str:
    payload = requests.get(ZENODO_API, timeout=60).json()
    files = payload.get("files") or []
    if isinstance(files, dict):
        files = list((files.get("entries") or files).values())
    for item in files:
        key = str(item.get("key") or item.get("filename") or "")
        if key == MONTH:
            links = item.get("links") or {}
            return (
                links.get("content")
                or links.get("self")
                or f"https://zenodo.org/records/10672052/files/{MONTH}?download=1"
            )
    raise RuntimeError(f"{MONTH} not found")


def inspect_hdf5(path: Path) -> dict:
    result = {"format": "HDF5", "root_attrs": {}, "objects": {}}
    with h5py.File(path, "r") as h5:
        result["root_attrs"] = {
            str(k): scalarize(v) for k, v in h5.attrs.items()
        }
        def visitor(name, obj):
            attrs = {str(k): scalarize(v) for k, v in obj.attrs.items()}
            rel = relevant_attrs(attrs)
            if rel:
                result["objects"][f"/{name}"] = {
                    "type": type(obj).__name__,
                    "attrs": rel,
                }
        h5.visititems(visitor)
    return result


def inspect_hdf4(path: Path) -> dict:
    h4 = SD(str(path), SDC.READ)
    try:
        result = {
            "format": "HDF4",
            "root_attrs": {str(k): scalarize(v) for k, v in h4.attributes().items()},
            "objects": {},
        }
        for name in h4.datasets():
            ds = h4.select(name)
            attrs = {str(k): scalarize(v) for k, v in ds.attributes().items()}
            rel = relevant_attrs(attrs)
            if rel:
                result["objects"][name] = {"type": "SDS", "attrs": rel}
        return result
    finally:
        h4.end()


def main() -> None:
    url = zenodo_month_url()
    with RemoteZip(url) as rz:
        names = rz.namelist()
        candidates = [
            name for name in names
            if TARGET_TOKEN in Path(name).name
            and Path(name).suffix.lower() in {".hdf", ".h5", ".hdf5", ".he5"}
        ]
        if not candidates:
            # Keep nearby filenames to diagnose the naming convention if the exact token differs.
            nearby = [
                name for name in names
                if "2019267" in Path(name).name
                and Path(name).suffix.lower() in {".hdf", ".h5", ".hdf5", ".he5"}
            ][:30]
            raise RuntimeError(
                f"No member containing {TARGET_TOKEN}; nearby={nearby}"
            )
        member = candidates[0]
        raw = rz.read(member)

    suffix = Path(member).suffix or ".hdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        temp_path = Path(tmp.name)

    try:
        try:
            inspected = inspect_hdf5(temp_path)
        except OSError:
            inspected = inspect_hdf4(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    root_relevant = relevant_attrs(inspected.get("root_attrs") or {})
    report = {
        "zenodo_record": 10672052,
        "archive": MONTH,
        "member": member,
        "member_size_bytes": len(raw),
        "current_parser_interpretation": "2019 DOY267 04:00, timezone unspecified",
        "format": inspected["format"],
        "root_attrs_all": inspected.get("root_attrs", {}),
        "root_attrs_time_relevant": root_relevant,
        "object_attrs_time_relevant": inspected.get("objects", {}),
        "finding": (
            "If no explicit UTC/time-zone attribute is present, the filename HHMM time standard "
            "remains unresolved and must be confirmed from authoritative product documentation/author."
        ),
    }
    Path("elite_source_time_metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
