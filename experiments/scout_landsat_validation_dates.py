from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path

import ee
from google.oauth2 import service_account

ROI = [99.86, 38.67, 100.5, 39.43]
START = "2019-05-01"
END = "2019-11-01"
EXCLUDE_DATES = {"2019-09-24"}
COLLECTION = "LANDSAT/LC08/C02/T1_L2"


def init_ee() -> str:
    project = os.getenv("EE_PROJECT", "").strip() or "ee-ygangxian"
    cred_file = Path.home() / ".config" / "earthengine" / "credentials"
    if cred_file.exists():
        ee.Initialize(project=project)
        return project
    raw = os.getenv("EE_SERVICE_ACCOUNT_JSON", "").strip()
    raw_b64 = os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64", "").strip()
    if not raw and raw_b64:
        raw = base64.b64decode(raw_b64).decode("utf-8")
    if not raw:
        raise RuntimeError("Earth Engine credentials are missing")
    info = json.loads(raw)
    project = os.getenv("EE_PROJECT", "").strip() or info.get("project_id") or project
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(creds, project=project)
    return project


def scene_metrics(image, geom):
    qa = image.select("QA_PIXEL")
    lst = image.select("ST_B10")

    clear = ee.Image(1)
    for bit in [0, 1, 2, 3, 4, 5]:
        clear = clear.And(qa.bitwiseAnd(1 << bit).eq(0))
    clear_lst = clear.And(lst.mask().gt(0))

    native_ratio = (
        lst.mask()
        .gt(0)
        .unmask(0)
        .rename("native")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=120,
            bestEffort=True,
            maxPixels=10_000_000,
        )
        .get("native")
    )

    clear_ratio = (
        clear_lst
        .unmask(0)
        .rename("clear")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=120,
            bestEffort=True,
            maxPixels=10_000_000,
        )
        .get("clear")
    )

    return image.set({
        "roi_native_lst_ratio": native_ratio,
        "roi_clear_lst_ratio": clear_ratio,
    })


def choose_spread(candidates: list[dict], n: int = 5, min_days: int = 24) -> list[dict]:
    chosen: list[dict] = []
    ranked = sorted(
        candidates,
        key=lambda x: (
            x["roi_clear_lst_ratio"],
            x["roi_native_lst_ratio"],
            -x["cloud_cover"],
        ),
        reverse=True,
    )
    for item in ranked:
        dt = datetime.fromisoformat(item["date"])
        if all(abs((dt - datetime.fromisoformat(x["date"])).days) >= min_days for x in chosen):
            chosen.append(item)
            if len(chosen) >= n:
                break
    if len(chosen) < n:
        for item in ranked:
            if item not in chosen:
                chosen.append(item)
                if len(chosen) >= n:
                    break
    return sorted(chosen, key=lambda x: x["date"])


def main() -> None:
    project = init_ee()
    geom = ee.Geometry.Rectangle(ROI, proj="EPSG:4326", geodesic=False)

    col = (
        ee.ImageCollection(COLLECTION)
        .filterDate(START, END)
        .filterBounds(geom)
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .filter(ee.Filter.lte("CLOUD_COVER", 80))
        .map(lambda img: scene_metrics(img, geom))
        .sort("system:time_start")
    )

    count = int(col.size().getInfo())
    props = col.toList(count).map(
        lambda img: ee.Image(img).toDictionary([
            "LANDSAT_PRODUCT_ID",
            "LANDSAT_SCENE_ID",
            "SPACECRAFT_ID",
            "CLOUD_COVER",
            "WRS_PATH",
            "WRS_ROW",
            "system:time_start",
            "roi_native_lst_ratio",
            "roi_clear_lst_ratio",
        ])
    ).getInfo()

    rows = []
    for p in props:
        ts = datetime.utcfromtimestamp(float(p["system:time_start"]) / 1000.0)
        date = ts.strftime("%Y-%m-%d")
        rows.append({
            "date": date,
            "acquired_utc": ts.isoformat(),
            "product_id": p.get("LANDSAT_PRODUCT_ID"),
            "scene_id": p.get("LANDSAT_SCENE_ID"),
            "spacecraft": p.get("SPACECRAFT_ID"),
            "cloud_cover": float(p.get("CLOUD_COVER") or 0.0),
            "wrs_path": int(p.get("WRS_PATH") or 0),
            "wrs_row": int(p.get("WRS_ROW") or 0),
            "roi_native_lst_ratio": float(p.get("roi_native_lst_ratio") or 0.0),
            "roi_clear_lst_ratio": float(p.get("roi_clear_lst_ratio") or 0.0),
        })

    eligible = [
        r for r in rows
        if r["date"] not in EXCLUDE_DATES
        and r["roi_clear_lst_ratio"] >= 0.55
        and r["roi_native_lst_ratio"] >= 0.70
    ]
    chosen = choose_spread(eligible, n=5, min_days=24)

    report = {
        "project": project,
        "roi": ROI,
        "collection": COLLECTION,
        "search_window": [START, END],
        "selection_rule": {
            "exclude_dates": sorted(EXCLUDE_DATES),
            "minimum_roi_clear_lst_ratio": 0.55,
            "minimum_roi_native_lst_ratio": 0.70,
            "target_count": 5,
            "preferred_minimum_date_spacing_days": 24,
            "ranking": "ROI clear-LST ratio, then native-LST ratio, then lower scene cloud metadata",
        },
        "scene_count": len(rows),
        "eligible_count": len(eligible),
        "selected_dates": chosen,
        "all_scenes": rows,
    }

    Path("output").mkdir(exist_ok=True)
    Path("output/landsat_date_scout.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== SELECTED INDEPENDENT DATES ===")
    for r in chosen:
        print(
            f"{r['date']}  {r['acquired_utc']}  clear={r['roi_clear_lst_ratio']:.4f}  "
            f"native={r['roi_native_lst_ratio']:.4f}  cloud_meta={r['cloud_cover']:.2f}%  "
            f"{r['product_id']}"
        )
    print(f"scene_count={len(rows)} eligible={len(eligible)}")


if __name__ == "__main__":
    main()
