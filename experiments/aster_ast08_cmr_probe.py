from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import requests

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
COLLECTIONS_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"

CASES = {
    "zhangye_20190924": {
        "date": "2019-09-24",
        "bbox": [99.86, 38.67, 100.50, 39.43],
    },
    "guilin_20220824": {
        "date": "2022-08-24",
        "bbox": [110.12447068021214, 24.99113120420509, 110.49277994670115, 25.341474165011704],
    },
}


def collection_info() -> dict:
    response = requests.get(
        COLLECTIONS_URL,
        params={"short_name": "AST_08", "version": "004", "page_size": 20},
        headers={"Accept": "application/json", "User-Agent": "remote-sensing-mcp/aster-probe"},
        timeout=60,
    )
    response.raise_for_status()
    entries = ((response.json().get("feed") or {}).get("entry") or [])
    return {
        "cmr_hits": int(response.headers.get("CMR-Hits", len(entries))),
        "entries": [
            {
                "id": e.get("id"),
                "short_name": e.get("short_name"),
                "version_id": e.get("version_id"),
                "data_center": e.get("data_center"),
                "dataset_id": e.get("dataset_id"),
            }
            for e in entries
        ],
    }


def global_day_count(start: str, end: str, provider: str | None) -> dict:
    params = {
        "short_name": "AST_08",
        "version": "004",
        "temporal": f"{start}T00:00:00Z,{end}T00:00:00Z",
        "page_size": 1,
    }
    if provider:
        params["provider"] = provider
    response = requests.get(
        CMR_URL,
        params=params,
        headers={"Accept": "application/json", "User-Agent": "remote-sensing-mcp/aster-probe"},
        timeout=60,
    )
    response.raise_for_status()
    entries = ((response.json().get("feed") or {}).get("entry") or [])
    return {
        "provider_filter": provider,
        "cmr_hits": int(response.headers.get("CMR-Hits", len(entries))),
        "sample_titles": [e.get("title") for e in entries[:1]],
    }


def query(start: str, end: str, bbox: list[float], provider: str | None = "LPCLOUD") -> dict:
    response = requests.get(
        CMR_URL,
        params={
            **({"provider": provider} if provider else {}),
            "short_name": "AST_08",
            "version": "004",
            "bounding_box": ",".join(str(x) for x in bbox),
            "temporal": f"{start}T00:00:00Z,{end}T00:00:00Z",
            "page_size": 200,
        },
        headers={
            "Accept": "application/json",
            "User-Agent": "remote-sensing-mcp/aster-probe",
        },
        timeout=60,
    )
    response.raise_for_status()
    entries = ((response.json().get("feed") or {}).get("entry") or [])
    return {
        "cmr_hits": int(response.headers.get("CMR-Hits", len(entries))),
        "entries": entries,
    }


def summarize_entry(entry: dict) -> dict:
    return {
        "concept_id": entry.get("id"),
        "granule_ur": entry.get("title"),
        "producer_granule_id": entry.get("producer_granule_id"),
        "start_time": entry.get("time_start"),
        "end_time": entry.get("time_end"),
        "day_night_flag": entry.get("day_night_flag"),
        "cloud_cover": entry.get("cloud_cover"),
        "boxes": entry.get("boxes") or [],
        "polygons": entry.get("polygons") or [],
    }


def main() -> None:
    out = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "product": "ASTER AST_08.004",
        "collection_info": collection_info(),
        "global_sanity": {
            "2019-09-24_lpccloud": global_day_count("2019-09-24", "2019-09-25", "LPCLOUD"),
            "2019-09-24_no_provider": global_day_count("2019-09-24", "2019-09-25", None),
            "2022-08-24_lpccloud": global_day_count("2022-08-24", "2022-08-25", "LPCLOUD"),
            "2022-08-24_no_provider": global_day_count("2022-08-24", "2022-08-25", None),
        },
        "cases": {},
    }
    for name, cfg in CASES.items():
        target = date.fromisoformat(cfg["date"])
        exact = query(target.isoformat(), (target + timedelta(days=1)).isoformat(), cfg["bbox"])
        exact_no_provider = query(target.isoformat(), (target + timedelta(days=1)).isoformat(), cfg["bbox"], None)
        nearby_start = target - timedelta(days=2)
        nearby_end = target + timedelta(days=3)
        nearby = query(nearby_start.isoformat(), nearby_end.isoformat(), cfg["bbox"])
        wide_start = target - timedelta(days=90)
        wide_end = target + timedelta(days=91)
        wide = query(wide_start.isoformat(), wide_end.isoformat(), cfg["bbox"])

        day_counts = Counter()
        for entry in nearby["entries"]:
            ts = entry.get("time_start")
            if ts:
                day_counts[str(ts)[:10]] += 1

        wide_summaries = [summarize_entry(e) for e in wide["entries"]]
        def scene_distance(scene: dict) -> float:
            ts = scene.get("start_time")
            if not ts:
                return float("inf")
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            target_dt = datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
            return abs((dt - target_dt).total_seconds())
        wide_summaries.sort(key=scene_distance)

        out["cases"][name] = {
            "target_date": cfg["date"],
            "bbox": cfg["bbox"],
            "exact_day_hit_count": exact["cmr_hits"],
            "exact_day_hit_count_no_provider": exact_no_provider["cmr_hits"],
            "exact_day_scenes": [summarize_entry(e) for e in exact["entries"]],
            "nearby_window": [nearby_start.isoformat(), nearby_end.isoformat()],
            "nearby_hit_count": nearby["cmr_hits"],
            "nearby_hits_by_date": dict(sorted(day_counts.items())),
            "nearby_scenes": [summarize_entry(e) for e in nearby["entries"]],
            "wide_window_days": 90,
            "wide_hit_count": wide["cmr_hits"],
            "nearest_scenes": wide_summaries[:10],
        }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open("aster_ast08_probe_result.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
