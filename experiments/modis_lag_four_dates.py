from __future__ import annotations

import argparse
import base64
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from google.oauth2 import service_account

ROI = [99.86528182419725, 38.67911397351706, 100.47613621740223, 39.39776620081703]
PIX1 = 0.00898315284124962
PROFILE = {
    "crs": "EPSG:4326",
    "transform": Affine(PIX1, 0.0, ROI[0], 0.0, -PIX1, ROI[3]),
    "width": 68,
    "height": 80,
}
SCENES = {
    "2019-05-19": "LC08_L2SP_133033_20190519_20200828_02_T1",
    "2019-07-22": "LC08_L2SP_133033_20190722_20200827_02_T1",
    "2019-10-26": "LC08_L2SP_133033_20191026_20200825_02_T1",
    "2019-12-13": "LC08_L2SP_133033_20191213_20201023_02_T1",
    "2020-09-10": "LC08_L2SP_133033_20200910_20200919_02_T1",
}
MODIS = {
    "terra": "MODIS/061/MOD11A1",
    "aqua": "MODIS/061/MYD11A1",
}
NODATA = -9999.0

def init_ee():
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
        raise RuntimeError("Earth Engine credentials missing")
    info = json.loads(raw)
    project = info.get("project_id") or project
    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ],
    )
    ee.Initialize(creds, project=project)
    return project

def download_exact(image: ee.Image, path: Path):
    t = PROFILE["transform"]
    url = ee.Image(image).getDownloadURL({
        "name": path.stem,
        "crs": PROFILE["crs"],
        "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "dimensions": [PROFILE["width"], PROFILE["height"]],
        "format": "GEO_TIFF",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

def landsat_1km(date: str, work: Path):
    product_id = SCENES[date]
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filter(ee.Filter.eq("LANDSAT_PRODUCT_ID", product_id))
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
    )
    if int(col.size().getInfo()) != 1:
        raise RuntimeError(f"Landsat scene not unique: {product_id}")
    img = ee.Image(col.first())
    qa = img.select("QA_PIXEL")
    clear = ee.Image(1)
    for bit in [0, 1, 2, 3, 4, 5]:
        clear = clear.And(qa.bitwiseAnd(1 << bit).eq(0))
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
    valid = clear.And(lst.mask().gt(0))
    lst1 = lst.updateMask(valid).reduceResolution(ee.Reducer.mean(), maxPixels=4096).rename("LST_C")
    frac = valid.unmask(0).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=4096).rename("VALID_FRAC")
    out = ee.Image.cat([lst1, frac]).toFloat().unmask(NODATA)
    path = work / f"landsat_{date}_1km.tif"
    download_exact(out, path)
    props = ee.Dictionary({
        "time": img.get("system:time_start"),
        "cloud_cover": img.get("CLOUD_COVER"),
    }).getInfo()
    acquired = datetime.utcfromtimestamp(float(props["time"]) / 1000.0).isoformat() + "Z"
    with rasterio.open(path) as ds:
        a = ds.read().astype("float32")
    l = a[0]; f = a[1]
    l[np.isclose(l, NODATA)] = np.nan
    f[np.isclose(f, NODATA)] = np.nan
    return l, f, acquired, float(props.get("cloud_cover") or 0.0)

def modis_day(platform: str, date: str, work: Path):
    day = datetime.fromisoformat(date)
    end = day + timedelta(days=1)
    col = (
        ee.ImageCollection(MODIS[platform])
        .filterDate(day.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        .filterBounds(ee.Geometry.Rectangle(ROI))
    )
    if int(col.size().getInfo()) < 1:
        raise RuntimeError(f"No MODIS {platform} for {date}")
    img = ee.Image(col.first())
    qc = img.select("QC_Day")
    mandatory = qc.bitwiseAnd(3).lte(1)
    data_quality = qc.rightShift(2).bitwiseAnd(3).eq(0)
    lst_error = qc.rightShift(6).bitwiseAnd(3).lte(2)
    mask = mandatory.And(data_quality).And(lst_error)
    lst = (
        img.select("LST_Day_1km")
        .multiply(0.02)
        .subtract(273.15)
        .updateMask(mask)
        .resample("bilinear")
        .rename("LST_C")
        .toFloat()
        .unmask(NODATA)
    )
    path = work / f"{platform}_{date}.tif"
    download_exact(lst, path)
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float32")
    a[np.isclose(a, NODATA)] = np.nan
    return a

def metric(y, x, mask):
    yy = y[mask].astype("float64")
    xx = x[mask].astype("float64")
    n = int(yy.size)
    if n < 3:
        return {"n": n, "pearson_r": None, "spearman_rho": None, "rmse_c": None, "bias_c": None}
    pr = float(np.corrcoef(xx, yy)[0,1])
    rx = pd.Series(xx).rank(method="average").to_numpy()
    ry = pd.Series(yy).rank(method="average").to_numpy()
    sr = float(np.corrcoef(rx, ry)[0,1])
    err = xx - yy
    return {
        "n": n,
        "pearson_r": pr,
        "spearman_rho": sr,
        "rmse_c": float(math.sqrt(np.mean(err**2))),
        "bias_c": float(np.mean(err)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, choices=sorted(SCENES))
    args = ap.parse_args()
    project = init_ee()
    date = args.date
    work = Path("output") / date / "work"
    out_dir = Path("output") / date
    out_dir.mkdir(parents=True, exist_ok=True)

    landsat, valid_frac, acquired_utc, cloud_cover = landsat_1km(date, work)
    landsat_mask = np.isfinite(landsat) & np.isfinite(valid_frac) & (valid_frac >= 0.70)

    d0 = datetime.fromisoformat(date)
    d1 = (d0 - timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    conclusions = {}

    for platform in ["terra", "aqua"]:
        m1 = modis_day(platform, d1, work)
        m0 = modis_day(platform, date, work)
        common = landsat_mask & np.isfinite(m1) & np.isfinite(m0)
        r1 = metric(landsat, m1, common)
        r0 = metric(landsat, m0, common)
        rows += [
            {"landsat_date": date, "platform": platform, "modis_date": d1, "lag_days": -1, **r1},
            {"landsat_date": date, "platform": platform, "modis_date": date, "lag_days": 0, **r0},
        ]
        delta = (r1["pearson_r"] - r0["pearson_r"]) if r1["pearson_r"] is not None and r0["pearson_r"] is not None else None
        conclusions[platform] = {
            "common_n": int(common.sum()),
            "pearson_d_minus_1": r1["pearson_r"],
            "pearson_d0": r0["pearson_r"],
            "delta_r": delta,
            "d_minus_1_better": bool(delta > 0) if delta is not None else None,
            "spearman_d_minus_1": r1["spearman_rho"],
            "spearman_d0": r0["spearman_rho"],
        }

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    summary = {
        "date": date,
        "product_id": SCENES[date],
        "landsat_acquired_utc": acquired_utc,
        "landsat_scene_cloud_cover_percent": cloud_cover,
        "formal_master_grid_roi": ROI,
        "formal_grid_shape": [PROFILE["height"], PROFILE["width"]],
        "landsat_min_valid_fraction": 0.70,
        "modis_qc": "bits0-1<=1; bits2-3==0; bits6-7<=2",
        "ee_project": project,
        "conclusions": conclusions,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
