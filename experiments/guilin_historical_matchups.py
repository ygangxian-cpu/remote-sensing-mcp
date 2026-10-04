from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import landsat_worker
import modis_worker


START_DATE = "2020-01-01"
END_DATE = "2022-09-25"  # exclusive: target day 2022-09-25 is deliberately excluded
TARGET_DATE = "2022-09-25"
BBOX = [110.12447068021214, 24.99113120420509, 110.49277994670115, 25.341474165011704]
MAX_DATES = 20
LANDSAT_METADATA_CLOUD_MAX = 60.0
MIN_CLEAR_FRACTION_1KM = 0.70
MIN_MODIS_STRICT_FRACTION_1KM = 0.50
MAX_TIME_DIFF_MIN = 120.0

OUT = Path("experiments/guilin_historical_matchups")
WORK = Path("output/guilin_historical_matchups")
TEMPLATE_URL = (
    "https://raw.githubusercontent.com/yuyan3616/jiangchidu4kmto100m/"
    "4a2fa30a007bda79b61b6cb28a20948746757322/"
    "data_guilin_summer_20220925/Guilin_static_factors_1km.tif"
)


def download_template() -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "Guilin_static_factors_1km.tif"
    if path.exists():
        return path
    with requests.get(TEMPLATE_URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return path


def profile_dict(src: rasterio.io.DatasetReader) -> dict:
    return {
        "crs": src.crs,
        "transform": src.transform,
        "height": src.height,
        "width": src.width,
    }


def warp_average(src_arr: np.ndarray, src_profile: dict, dst_profile: dict, nodata: float = -9999.0) -> np.ndarray:
    src = np.where(np.isfinite(src_arr), src_arr, nodata).astype("float32")
    dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=src,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return dst


def warp_nearest(src_arr: np.ndarray, src_profile: dict, dst_profile: dict, nodata: float = -9999.0) -> np.ndarray:
    src = np.where(np.isfinite(src_arr), src_arr, nodata).astype("float32")
    dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=src,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return dst


def scene_candidates(region) -> list[dict]:
    collections = []
    for sat, collection_name in landsat_worker.COLLECTIONS.items():
        col = (
            ee.ImageCollection(collection_name)
            .filterDate(START_DATE, END_DATE)
            .filterBounds(region)
            .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
            .filter(ee.Filter.lte("CLOUD_COVER", LANDSAT_METADATA_CLOUD_MAX))
        )
        collections.append(col)

    merged = collections[0]
    for col in collections[1:]:
        merged = merged.merge(col)
    merged = merged.sort("system:time_start")

    count = int(merged.size().getInfo())
    if count == 0:
        return []

    items = merged.toList(count)
    rows = []
    for i in range(count):
        image = ee.Image(items.get(i))
        props = ee.Dictionary({
            "product_id": image.get("LANDSAT_PRODUCT_ID"),
            "scene_id": image.get("LANDSAT_SCENE_ID"),
            "spacecraft": image.get("SPACECRAFT_ID"),
            "cloud_cover": image.get("CLOUD_COVER"),
            "time": image.get("system:time_start"),
        }).getInfo()
        ts = datetime.fromtimestamp(float(props["time"]) / 1000.0, UTC)
        rows.append({
            "image": image,
            "product_id": str(props.get("product_id") or props.get("scene_id") or f"scene_{i}"),
            "scene_id": props.get("scene_id"),
            "spacecraft": props.get("spacecraft"),
            "cloud_cover": float(props.get("cloud_cover") or 0.0),
            "acquired_utc": ts,
            "date": ts.date().isoformat(),
        })

    # Keep at most one scene per calendar date, preferring lower metadata cloud cover.
    by_date = {}
    for row in sorted(rows, key=lambda r: (r["cloud_cover"], r["acquired_utc"])):
        by_date.setdefault(row["date"], row)

    unique = list(by_date.values())

    # Encourage temporal diversity: pick the best candidate within each YYYY-MM bucket first.
    by_month: dict[str, list[dict]] = {}
    for row in unique:
        by_month.setdefault(row["date"][:7], []).append(row)
    monthly_best = [
        min(group, key=lambda r: r["cloud_cover"])
        for _, group in sorted(by_month.items())
    ]
    monthly_best.sort(key=lambda r: (r["cloud_cover"], r["date"]))

    chosen = monthly_best[:MAX_DATES]
    if len(chosen) < MAX_DATES:
        used = {r["date"] for r in chosen}
        extras = sorted(
            [r for r in unique if r["date"] not in used],
            key=lambda r: (r["cloud_cover"], r["date"]),
        )
        chosen.extend(extras[: MAX_DATES - len(chosen)])

    return sorted(chosen, key=lambda r: r["date"])


def read_landsat_matchup(path: Path, dst_profile: dict) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype("float32")
        p = profile_dict(src)
        lst = data[0].filled(np.nan)
        qa = data[2].filled(np.nan)

    qa_int = np.where(np.isfinite(qa), qa, 0).astype("int64")
    clear = np.isfinite(lst) & np.isfinite(qa)
    for bit in [0, 1, 2, 3, 4, 5]:
        clear &= (qa_int & (1 << bit)) == 0

    lst_clear = np.where(clear, lst, np.nan)
    clear_fraction = warp_average(clear.astype("float32"), p, dst_profile)
    lst_1km = warp_average(lst_clear.astype("float32"), p, dst_profile)
    return lst_1km, clear_fraction


def read_modis_matchup(path: Path, dst_profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype("float32")
        p = profile_dict(src)
        lst = data[0].filled(np.nan)
        view_local = data[2].filled(np.nan)
        qc = data[4].filled(np.nan)

    qc_int = np.where(np.isfinite(qc), qc, 0).astype("int64")
    mandatory = qc_int & 3
    data_quality = (qc_int >> 2) & 3
    lst_error = (qc_int >> 6) & 3
    strict = (
        np.isfinite(lst)
        & np.isfinite(qc)
        & (mandatory <= 1)
        & (data_quality == 0)
        & (lst_error <= 2)
    )

    lst_strict = np.where(strict, lst, np.nan)
    view_strict = np.where(strict, view_local, np.nan)
    strict_fraction = warp_average(strict.astype("float32"), p, dst_profile)
    lst_1km = warp_average(lst_strict.astype("float32"), p, dst_profile)
    view_1km = warp_average(view_strict.astype("float32"), p, dst_profile)
    return lst_1km, strict_fraction, view_1km


def main() -> None:
    if TARGET_DATE >= END_DATE:
        pass
    else:
        raise RuntimeError("Target-date exclusion contract is inconsistent")

    project = landsat_worker.init_ee()
    region = ee.Geometry.Rectangle(BBOX, proj="EPSG:4326", geodesic=False)
    template_path = download_template()

    with rasterio.open(template_path) as src:
        dst_profile = profile_dict(src)
        landcover = src.read(4).astype("float32")

    candidates = scene_candidates(region)
    if len(candidates) < 6:
        raise RuntimeError(f"Only {len(candidates)} historical Landsat candidate dates found")

    OUT.mkdir(parents=True, exist_ok=True)
    matchup_rows: list[dict] = []
    date_rows: list[dict] = []

    for idx, row in enumerate(candidates, start=1):
        date = row["date"]
        if date == TARGET_DATE:
            raise RuntimeError("Target date leaked into historical candidate list")

        print(f"[{idx}/{len(candidates)}] {date} {row['product_id']} cloud={row['cloud_cover']:.1f}", flush=True)

        meta = {
            "product_id": row["product_id"],
            "scene_id": row["scene_id"],
            "spacecraft": row["spacecraft"],
            "cloud_cover": row["cloud_cover"],
            "acquired_utc": row["acquired_utc"].isoformat().replace("+00:00", "Z"),
        }
        ls_path = WORK / "landsat" / f"{row['product_id']}.tif"
        ls_path.parent.mkdir(parents=True, exist_ok=True)
        landsat_worker.download_scene(
            landsat_worker.prepare_scene(row["image"]),
            BBOX,
            ls_path,
            meta,
        )

        day = datetime.fromisoformat(date)
        modis_path = WORK / "modis" / f"MOD11A1_{day:%Y%m%d}.tif"
        modis_path.parent.mkdir(parents=True, exist_ok=True)
        details = modis_worker.download_one("terra", day, BBOX, modis_path)
        if details is None or not modis_path.exists():
            date_rows.append({
                "date": date,
                "product_id": row["product_id"],
                "cloud_cover_metadata": row["cloud_cover"],
                "status": "missing_modis",
                "matched_pixels": 0,
            })
            continue

        ls1, clear_frac = read_landsat_matchup(ls_path, dst_profile)
        mod1, mod_frac, view_local = read_modis_matchup(modis_path, dst_profile)

        # Convert MODIS local solar time to approximate UTC per cell using the ROI center longitude.
        center_lon = 0.5 * (BBOX[0] + BBOX[2])
        modis_view_utc = view_local - center_lon / 15.0
        landsat_hour = (
            row["acquired_utc"].hour
            + row["acquired_utc"].minute / 60.0
            + row["acquired_utc"].second / 3600.0
        )
        time_diff_min = np.abs(modis_view_utc - landsat_hour) * 60.0

        valid = (
            np.isfinite(ls1)
            & np.isfinite(mod1)
            & np.isfinite(landcover)
            & np.isfinite(clear_frac)
            & np.isfinite(mod_frac)
            & (clear_frac >= MIN_CLEAR_FRACTION_1KM)
            & (mod_frac >= MIN_MODIS_STRICT_FRACTION_1KM)
            & np.isfinite(time_diff_min)
            & (time_diff_min <= MAX_TIME_DIFF_MIN)
        )

        rr, cc = np.where(valid)
        for r, c in zip(rr.tolist(), cc.tolist()):
            matchup_rows.append({
                "date": date,
                "modis_c": float(mod1[r, c]),
                "landsat_1km_c": float(ls1[r, c]),
                "valid_fraction": float(clear_frac[r, c]),
                "modis_strict_fraction": float(mod_frac[r, c]),
                "landcover": int(round(float(landcover[r, c]))),
                "row_1km": int(r),
                "col_1km": int(c),
                "landsat_product_id": row["product_id"],
                "landsat_spacecraft": row["spacecraft"],
                "landsat_cloud_cover_metadata": float(row["cloud_cover"]),
                "landsat_acquired_utc": row["acquired_utc"].isoformat().replace("+00:00", "Z"),
                "modis_day_view_time_local_h": float(view_local[r, c]),
                "modis_day_view_time_utc_h_approx": float(modis_view_utc[r, c]),
                "time_diff_min_approx": float(time_diff_min[r, c]),
            })

        date_rows.append({
            "date": date,
            "product_id": row["product_id"],
            "spacecraft": row["spacecraft"],
            "cloud_cover_metadata": row["cloud_cover"],
            "landsat_acquired_utc": row["acquired_utc"].isoformat().replace("+00:00", "Z"),
            "status": "ok" if valid.sum() > 0 else "no_valid_matchups",
            "matched_pixels": int(valid.sum()),
            "mean_landsat_clear_fraction": float(np.nanmean(clear_frac)),
            "mean_modis_strict_fraction": float(np.nanmean(mod_frac)),
            "mean_time_diff_min_approx": float(np.nanmean(time_diff_min)),
        })

        ls_path.unlink(missing_ok=True)
        modis_path.unlink(missing_ok=True)

    matchups = pd.DataFrame(matchup_rows)
    dates = pd.DataFrame(date_rows)
    if matchups.empty:
        raise RuntimeError("No historical Landsat-MODIS matchup pixels survived QC")

    usable_dates = sorted(matchups["date"].unique().tolist())
    if len(usable_dates) < 6:
        raise RuntimeError(f"Only {len(usable_dates)} usable historical dates survived QC")

    matchups.to_csv(OUT / "guilin_landsat_modis_historical_matchups.csv", index=False, encoding="utf-8-sig")
    dates.to_csv(OUT / "guilin_historical_date_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment": "Guilin historical Landsat-MODIS matchup acquisition",
        "gee_project": project,
        "target_date_excluded": TARGET_DATE,
        "historical_start": START_DATE,
        "historical_end_exclusive": END_DATE,
        "bbox": BBOX,
        "candidate_dates": len(candidates),
        "usable_dates": len(usable_dates),
        "usable_date_list": usable_dates,
        "matchup_samples": int(len(matchups)),
        "landsat_metadata_cloud_max_percent": LANDSAT_METADATA_CLOUD_MAX,
        "landsat_min_clear_fraction_1km": MIN_CLEAR_FRACTION_1KM,
        "modis_min_strict_fraction_1km": MIN_MODIS_STRICT_FRACTION_1KM,
        "max_time_difference_minutes": MAX_TIME_DIFF_MIN,
        "modis_product": "MODIS/061/MOD11A1 Terra daytime",
        "modis_strict_qc": "bits0-1<=1; bits2-3==0; bits6-7<=2",
        "landsat_clear_mask": "QA_PIXEL bits0-5 == 0; water retained",
        "template_grid": "Guilin_static_factors_1km.tif from research repo main",
        "note": (
            "This file is acquisition/QC support for the cross-region experiment. "
            "The target date 2022-09-25 is excluded from calibration data."
        ),
    }
    (OUT / "guilin_historical_matchups_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(dates.to_string(index=False), flush=True)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    main()
