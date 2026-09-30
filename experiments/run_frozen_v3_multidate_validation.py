from __future__ import annotations

import base64
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import ee
import h5py
import numpy as np
import pandas as pd
import rasterio
import requests
from affine import Affine
from google.oauth2 import service_account
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.transform import from_origin
from rasterio.warp import reproject
from remotezip import RemoteZip
from scipy import ndimage
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import worker as elite_worker

ROI = [99.86, 38.67, 100.5, 39.43]
DATES = [
    "2019-05-03",
    "2019-05-19",
    "2019-07-22",
    "2019-09-15",
    "2019-10-26",
]
CONTROL_DATE = "2019-09-24"
HOUR = 4
ELITE_RECORD_ID = 10672052

# Frozen v3 baseline implementation:
# yuyan3616/jiangchidu4kmto100m run_scale_specific_rf_v3.py
# Stage 1 = atmospheric/radiative + DEM + NDVI + landcover.
# Stage 2 = v2 absolute thermal-potential surface anomaly.
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": 42,
}

STATIC_FEATURES = [
    "dem", "slope", "aspect",
    "ndvi", "ndbi", "mndwi", "savi", "bsi", "ui",
    "albedo_bsa", "landcover", "lat_m", "lon_m",
]
MCP20 = STATIC_FEATURES + [
    "edrf", "t2_c", "td2_c", "wind_speed", "psfc_pa", "swdown_wm2", "glw_wm2",
]
STAGE1_V2 = [
    "t2_c", "td2_c", "wind_speed", "psfc_pa",
    "swdown_wm2", "glw_wm2", "edrf", "albedo_bsa",
]
STAGE1_V3 = STAGE1_V2 + ["dem", "ndvi", "landcover"]

# Exact formal grid anchored to the existing Landsat-v2 validation snapshot.
ORIGIN_X = 99.85942192857837
ORIGIN_Y = 39.43065108114228
PIX100 = 0.0008983152841195215
PROFILES = {
    "100m": {
        "crs": "EPSG:4326",
        "transform": Affine(PIX100, 0, ORIGIN_X, 0, -PIX100, ORIGIN_Y),
        "width": 714,
        "height": 847,
    },
    "1km": {
        "crs": "EPSG:4326",
        "transform": Affine(PIX100 * 10, 0, ORIGIN_X, 0, -PIX100 * 10, ORIGIN_Y),
        "width": 72,
        "height": 85,
    },
    "4km": {
        "crs": "EPSG:4326",
        "transform": Affine(PIX100 * 40, 0, ORIGIN_X, 0, -PIX100 * 40, ORIGIN_Y),
        "width": 18,
        "height": 21,
    },
}


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


def profile_dict(src):
    return {
        "crs": src.crs,
        "transform": src.transform,
        "width": src.width,
        "height": src.height,
    }


def reproject_array(arr, src_profile, dst_profile, resampling):
    out = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=np.where(np.isfinite(arr), arr, -9999.0).astype("float32"),
        destination=out,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=-9999.0,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return out


def fill_continuous(arr):
    arr = arr.astype("float32", copy=True)
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if not valid.any():
        raise RuntimeError("continuous predictor has no valid pixels")
    work = np.where(valid, arr, 0).astype("float32")
    out = fillnodata(work, mask=valid.astype("uint8"), max_search_distance=300)
    missing = ~np.isfinite(out)
    if missing.any():
        _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
        nearest = arr[tuple(idx)]
        out[missing] = nearest[missing]
    return out.astype("float32")


def fill_nearest(arr):
    arr = arr.astype("float32", copy=True)
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if not valid.any():
        raise RuntimeError("categorical predictor has no valid pixels")
    _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = arr[tuple(idx)]
    out[valid] = arr[valid]
    return out.astype("float32")


def download_ee_image(image, path: Path, region, scale=30):
    url = ee.Image(image).getDownloadURL({
        "name": path.stem,
        "region": region,
        "scale": scale,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_exact_grid(image, path: Path, profile: dict):
    t = profile["transform"]
    url = ee.Image(image).getDownloadURL({
        "name": path.stem,
        "crs": "EPSG:4326",
        "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "dimensions": [profile["width"], profile["height"]],
        "format": "GEO_TIFF",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)


def landsat_scene(date: str):
    start = ee.Date(date)
    end = start.advance(1, "day")
    geom = ee.Geometry.Rectangle(ROI, proj="EPSG:4326", geodesic=False)
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterDate(start, end)
        .filterBounds(geom)
        .filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
        .sort("CLOUD_COVER")
    )
    if int(col.size().getInfo()) < 1:
        raise RuntimeError(f"No Landsat L2SP scene on {date}")
    image = ee.Image(col.first())
    meta = ee.Dictionary({
        "product_id": image.get("LANDSAT_PRODUCT_ID"),
        "scene_id": image.get("LANDSAT_SCENE_ID"),
        "cloud_cover": image.get("CLOUD_COVER"),
        "time": image.get("system:time_start"),
        "wrs_path": image.get("WRS_PATH"),
        "wrs_row": image.get("WRS_ROW"),
    }).getInfo()
    ts = datetime.utcfromtimestamp(float(meta["time"]) / 1000.0)
    meta["acquired_utc"] = ts.isoformat()
    return image, meta


def clear_mask(image):
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 0).eq(0)
    for bit in [1, 2, 3, 4, 5]:
        mask = mask.And(qa.bitwiseAnd(1 << bit).eq(0))
    return mask


def spectral_indices(image):
    mask = clear_mask(image)
    sr = (
        image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
        .multiply(0.0000275)
        .add(-0.2)
    )
    sr = sr.updateMask(mask.And(sr.reduce(ee.Reducer.min()).gt(0)))
    ndvi = sr.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    ndbi = sr.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi")
    mndwi = sr.normalizedDifference(["SR_B3", "SR_B6"]).rename("mndwi")
    bsi = sr.expression(
        "((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))",
        {
            "swir1": sr.select("SR_B6"),
            "red": sr.select("SR_B4"),
            "nir": sr.select("SR_B5"),
            "blue": sr.select("SR_B2"),
        },
    ).rename("bsi")
    savi = sr.expression(
        "(nir - red) * 1.5 / (nir + red + 0.5)",
        {"nir": sr.select("SR_B5"), "red": sr.select("SR_B4")},
    ).rename("savi")
    ui = sr.normalizedDifference(["SR_B7", "SR_B5"]).rename("ui")
    return ee.Image.cat([ndvi, ndbi, mndwi, savi, bsi, ui]).toFloat()


def read_multiband(path: Path, names: list[str]):
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype("float32")
        p = profile_dict(src)
    if data.shape[0] != len(names):
        raise RuntimeError(f"{path}: expected {len(names)} bands, got {data.shape[0]}")
    return {name: data[i].filled(np.nan) for i, name in enumerate(names)}, p


def build_static_features(date: str, scene, work: Path):
    geom = ee.Geometry.Rectangle(ROI, proj="EPSG:4326", geodesic=False)

    # 1) Date-specific Landsat spectral indices, same formulation as frozen 2019-09-24 pipeline.
    idx_path = work / f"indices_{date}.tif"
    indices100 = spectral_indices(scene).reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=1024
    )
    download_exact_grid(indices100, idx_path, PROFILES["100m"])
    idx_names = ["ndvi", "ndbi", "mndwi", "savi", "bsi", "ui"]
    idx_raw, idx_profile = read_multiband(idx_path, idx_names)

    # 2) Terrain from SRTM.
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("dem")
    terrain = ee.Terrain.products(dem)
    terr = ee.Image.cat([
        dem,
        terrain.select("slope").rename("slope"),
        terrain.select("aspect").rename("aspect"),
    ]).toFloat()
    terr_path = work / f"terrain_{date}.tif"
    terr100 = terr.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
    download_exact_grid(terr100, terr_path, PROFILES["100m"])
    terr_raw, terr_profile = read_multiband(terr_path, ["dem", "slope", "aspect"])

    # 3) Same-day MCD43A3 black-sky shortwave albedo, strict mandatory QA==0.
    d0 = ee.Date(date)
    alb_col = (
        ee.ImageCollection("MODIS/061/MCD43A3")
        .filterDate(d0, d0.advance(1, "day"))
        .filterBounds(geom)
    )
    if int(alb_col.size().getInfo()) < 1:
        raise RuntimeError(f"No MCD43A3 on {date}")
    alb_src = ee.Image(alb_col.first())
    alb = (
        alb_src.select("Albedo_BSA_shortwave")
        .multiply(0.001)
        .rename("albedo_bsa")
        .updateMask(
            alb_src.select("BRDF_Albedo_Band_Mandatory_Quality_shortwave").eq(0)
        )
        .toFloat()
    )
    alb_path = work / f"albedo_{date}.tif"
    download_exact_grid(alb.resample("bilinear"), alb_path, PROFILES["100m"])
    alb_raw, alb_profile = read_multiband(alb_path, ["albedo_bsa"])

    # 4) WorldCover categorical factor on the exact 100m grid.
    lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover").toFloat()
    lc_path = work / f"landcover_{date}.tif"
    download_exact_grid(lc, lc_path, PROFILES["100m"])
    lc_raw, lc_profile = read_multiband(lc_path, ["landcover"])

    base100 = {}
    for name, arr in idx_raw.items():
        base100[name] = fill_continuous(arr)
    for name, arr in terr_raw.items():
        base100[name] = fill_continuous(arr)
    base100["albedo_bsa"] = fill_continuous(alb_raw["albedo_bsa"])
    base100["landcover"] = fill_nearest(
        reproject_array(lc_raw["landcover"], lc_profile, PROFILES["100m"], Resampling.nearest)
    )

    # Metric-coordinate predictors, generated directly on the exact formal grid.
    rows, cols = np.indices((PROFILES["100m"]["height"], PROFILES["100m"]["width"]))
    xs, ys = rasterio.transform.xy(PROFILES["100m"]["transform"], rows, cols, offset="center")
    lon = np.asarray(xs).reshape(rows.shape)
    lat = np.asarray(ys).reshape(rows.shape)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    lon_m, lat_m = tr.transform(lon, lat)
    base100["lon_m"] = np.asarray(lon_m, dtype="float32")
    base100["lat_m"] = np.asarray(lat_m, dtype="float32")

    all_features = {"100m": base100}
    for res in ["1km", "4km"]:
        dst = {}
        for name in STATIC_FEATURES:
            method = Resampling.mode if name == "landcover" else Resampling.average
            dst[name] = reproject_array(base100[name], PROFILES["100m"], PROFILES[res], method)
            dst[name] = fill_nearest(dst[name]) if name == "landcover" else fill_continuous(dst[name])
        all_features[res] = dst
    return all_features


def era5_image(date: str):
    ts = datetime.fromisoformat(f"{date}T04:00:00")
    start = ts.strftime("%Y-%m-%dT%H:%M:%S")
    end = (ts + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    src = ee.Image(ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end).first())
    return ee.Image.cat([
        src.select("temperature_2m").subtract(273.15).rename("t2_c"),
        src.select("dewpoint_temperature_2m").subtract(273.15).rename("td2_c"),
        src.select("u_component_of_wind_10m").rename("u10_mps"),
        src.select("v_component_of_wind_10m").rename("v10_mps"),
        src.select("surface_pressure").rename("psfc_pa"),
        src.select("surface_solar_radiation_downwards_hourly").divide(3600).rename("swdown_wm2"),
        src.select("surface_thermal_radiation_downwards_hourly").divide(3600).rename("glw_wm2"),
    ]).toFloat()


def add_edrf(features, static, profile, date: str):
    rows, cols = np.indices((profile["height"], profile["width"]))
    xs, ys = rasterio.transform.xy(profile["transform"], rows, cols, offset="center")
    lon = np.asarray(xs).reshape(rows.shape)
    lat = np.asarray(ys).reshape(rows.shape)

    doy = float(datetime.fromisoformat(date).strftime("%j"))
    utc_hour = HOUR + 0.5
    decl = 0.40928 * math.sin(2.0 * math.pi * (doy - 81.0) / 364.0)
    lat_r = np.deg2rad(lat)
    hour_angle = (lon / 15.0 + utc_hour - 12.0) * math.pi / 12.0
    sin_alt = (
        np.sin(lat_r) * math.sin(decl)
        + np.cos(lat_r) * math.cos(decl) * np.cos(hour_angle)
    )
    cos_alt = np.sqrt(np.maximum(0.0, 1.0 - sin_alt ** 2))
    solar_az = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat_r) - math.tan(decl) * np.cos(lat_r),
    ) + math.pi

    slope_r = np.deg2rad(static["slope"])
    aspect_r = np.deg2rad(static["aspect"])
    dssr_flat = np.maximum(sin_alt, 0)
    dssr_topo = np.maximum(
        np.cos(slope_r) * sin_alt
        + np.sin(slope_r) * cos_alt * np.cos(solar_az - aspect_r),
        0,
    )
    day = sin_alt > 0
    f_topo = np.where(day, dssr_topo / np.maximum(dssr_flat, 0.01745), 1.0)
    f_topo = np.clip(f_topo, 0, 5)
    ssr_topo = np.where(day, features["swdown_wm2"] * f_topo, 0)
    w_ssr = np.clip(sin_alt, 0, 1)
    features["edrf"] = fill_continuous(
        w_ssr * ssr_topo + (1.0 - w_ssr) * features["glw_wm2"]
    )


def build_dynamic_features(date: str, static_all, work: Path):
    img = era5_image(date)
    names = ["t2_c", "td2_c", "u10_mps", "v10_mps", "psfc_pa", "swdown_wm2", "glw_wm2"]
    out = {}
    for res in ["100m", "1km", "4km"]:
        path = work / f"era5_{date}_{res}.tif"
        download_exact_grid(img.resample("bilinear"), path, PROFILES[res])
        raw, p = read_multiband(path, names)
        dyn = {k: fill_continuous(v) for k, v in raw.items()}
        dyn["wind_speed"] = fill_continuous(
            np.sqrt(dyn.pop("u10_mps") ** 2 + dyn.pop("v10_mps") ** 2)
        )
        add_edrf(dyn, static_all[res], PROFILES[res], date)
        out[res] = dyn
    return out


def zenodo_month_url(yearmonth: str) -> str:
    payload = requests.get(
        f"https://zenodo.org/api/records/{ELITE_RECORD_ID}", timeout=60
    ).json()
    filename = f"{yearmonth}.zip"
    files = payload.get("files") or []
    if isinstance(files, dict):
        files = list((files.get("entries") or files).values())
    for item in files:
        key = str(item.get("key") or item.get("filename") or "")
        if key == filename:
            links = item.get("links") or {}
            return links.get("content") or links.get("self") or f"https://zenodo.org/records/{ELITE_RECORD_ID}/files/{filename}?download=1"
    raise RuntimeError(f"{filename} missing from Zenodo record")


def elite_target(date: str):
    dt = datetime.fromisoformat(date)
    token = f"{dt:%Y}{dt.strftime('%j')}{HOUR:02d}00"
    url = zenodo_month_url(dt.strftime("%Y%m"))
    with RemoteZip(url) as rz:
        candidates = [
            n for n in rz.namelist()
            if token in Path(n).name and Path(n).suffix.lower() in {".hdf", ".h5", ".hdf5", ".he5"}
        ]
        if not candidates:
            raise RuntimeError(f"ELITE source not found for {date} {HOUR:02d}:00; token={token}")
        member = candidates[0]
        raw = rz.read(member)

    with tempfile.NamedTemporaryFile(suffix=Path(member).suffix or ".hdf", delete=False) as tmp:
        tmp.write(raw)
        p = Path(tmp.name)
    try:
        values_k, ds_name, attrs = elite_worker.lst_kelvin(p)
    finally:
        p.unlink(missing_ok=True)

    values_c = values_k - 273.15
    src_profile = {
        "crs": elite_worker.src_crs(),
        "transform": elite_worker.src_transform(values_c.shape[1], values_c.shape[0]),
        "width": values_c.shape[1],
        "height": values_c.shape[0],
    }
    y4 = reproject_array(values_c, src_profile, PROFILES["4km"], Resampling.bilinear)
    y4 = fill_continuous(y4)
    return y4, {
        "source_member": member,
        "source_dataset": ds_name,
        "time_standard": "UTC",
        "target_hour_utc": HOUR,
    }


def landsat_reference(date: str, scene, work: Path):
    lst = (
        scene.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("lst_c")
        .updateMask(clear_mask(scene))
        .toFloat()
    )
    path = work / f"landsat_lst_{date}_100m.tif"
    lst100 = lst.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
    download_exact_grid(lst100, path, PROFILES["100m"])
    raw, _ = read_multiband(path, ["lst_c"])
    return raw["lst_c"].astype("float32")


def merge_features(static_all, dynamic_all):
    return {
        res: {**static_all[res], **dynamic_all[res]}
        for res in ["100m", "1km", "4km"]
    }


def matrix(features, names):
    return np.column_stack([features[n].reshape(-1) for n in names]).astype("float32")


def basic_metrics(ref, pred):
    valid = np.isfinite(ref) & np.isfinite(pred)
    y = ref[valid].astype("float64")
    p = pred[valid].astype("float64")
    return {
        "n_samples": int(valid.sum()),
        "r2": float(r2_score(y, p)),
        "rmse": float(mean_squared_error(y, p) ** 0.5),
        "mae": float(mean_absolute_error(y, p)),
        "bias": float(np.mean(p - y)),
    }


def distribution_metrics(ref, pred):
    valid = np.isfinite(ref) & np.isfinite(pred)
    y = ref[valid].astype("float64")
    p = pred[valid].astype("float64")
    ys = float(np.std(y))
    ps = float(np.std(p))
    yr = float(np.percentile(y, 95) - np.percentile(y, 5))
    pr = float(np.percentile(p, 95) - np.percentile(p, 5))
    return {
        "std_prediction": ps,
        "std_reference": ys,
        "std_ratio": ps / ys if ys else float("nan"),
        "p05_p95_ratio": pr / yr if yr else float("nan"),
    }


def parent_mean_10x(arr100, shape1):
    h1, w1 = shape1
    means = np.full((h1, w1), np.nan, dtype="float32")
    for r in range(h1):
        rs = slice(r * 10, min((r + 1) * 10, arr100.shape[0]))
        for c in range(w1):
            cs = slice(c * 10, min((c + 1) * 10, arr100.shape[1]))
            block = arr100[rs, cs]
            if np.isfinite(block).any():
                means[r, c] = float(np.nanmean(block))
    return means


def expand_parent_10x(arr1, shape100):
    rows = np.minimum(np.arange(shape100[0]) // 10, arr1.shape[0] - 1)
    cols = np.minimum(np.arange(shape100[1]) // 10, arr1.shape[1] - 1)
    return arr1[rows[:, None], cols[None, :]].astype("float32")


def subpixel_metrics(ref, pred):
    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    ref_parent = parent_mean_10x(ref, shape1)
    pred_parent = parent_mean_10x(pred, shape1)
    ra = ref - expand_parent_10x(ref_parent, ref.shape)
    pa = pred - expand_parent_10x(pred_parent, pred.shape)
    valid = np.isfinite(ra) & np.isfinite(pa)
    r = ra[valid].astype("float64")
    p = pa[valid].astype("float64")
    rs = float(np.std(r))
    ps = float(np.std(p))
    return {
        "subpixel_anomaly_std_ratio": ps / rs if rs else float("nan"),
        "subpixel_anomaly_pearson": float(np.corrcoef(r, p)[0, 1]) if r.size > 1 else float("nan"),
        "subpixel_anomaly_rmse": float(np.sqrt(np.mean((p - r) ** 2))),
    }


def conservation_metrics(pred, pred_profile, target, target_profile):
    agg = reproject_array(pred, pred_profile, target_profile, Resampling.average)
    return basic_metrics(target, agg)


def fit_direct(features, y4):
    X4 = matrix(features["4km"], MCP20)
    X100 = matrix(features["100m"], MCP20)
    y = y4.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(X4).all(axis=1)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X4[valid], y[valid])
    pred4 = rf.predict(X4).reshape(y4.shape)
    residual4 = y4 - pred4
    pred100 = rf.predict(X100).reshape((PROFILES["100m"]["height"], PROFILES["100m"]["width"]))
    pred100 += np.nan_to_num(
        reproject_array(residual4, PROFILES["4km"], PROFILES["100m"], Resampling.bilinear),
        nan=0.0,
    )
    return pred100.astype("float32"), None


def fit_ordinary(features, y4):
    X4 = matrix(features["4km"], MCP20)
    X1 = matrix(features["1km"], MCP20)
    X100 = matrix(features["100m"], MCP20)
    y = y4.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(X4).all(axis=1)
    rf1 = RandomForestRegressor(**RF_PARAMS)
    rf1.fit(X4[valid], y[valid])
    pred4 = rf1.predict(X4).reshape(y4.shape)
    residual4 = y4 - pred4
    y1 = rf1.predict(X1).reshape((PROFILES["1km"]["height"], PROFILES["1km"]["width"]))
    y1 += np.nan_to_num(
        reproject_array(residual4, PROFILES["4km"], PROFILES["1km"], Resampling.bilinear),
        nan=0.0,
    )
    rf2 = RandomForestRegressor(**RF_PARAMS)
    rf2.fit(X1, y1.reshape(-1))
    pred1 = rf2.predict(X1).reshape(y1.shape)
    residual1 = y1 - pred1
    pred100 = rf2.predict(X100).reshape((PROFILES["100m"]["height"], PROFILES["100m"]["width"]))
    pred100 += np.nan_to_num(
        reproject_array(residual1, PROFILES["1km"], PROFILES["100m"], Resampling.bilinear),
        nan=0.0,
    )
    return pred100.astype("float32"), y1.astype("float32")


def fit_stage1(features, y4, names):
    X4 = matrix(features["4km"], names)
    X1 = matrix(features["1km"], names)
    y = y4.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(X4).all(axis=1)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X4[valid], y[valid])
    pred4 = rf.predict(X4).reshape(y4.shape)
    residual4 = y4 - pred4
    y1 = rf.predict(X1).reshape((PROFILES["1km"]["height"], PROFILES["1km"]["width"]))
    y1 += np.nan_to_num(
        reproject_array(residual4, PROFILES["4km"], PROFILES["1km"], Resampling.bilinear),
        nan=0.0,
    )
    back4 = reproject_array(y1, PROFILES["1km"], PROFILES["4km"], Resampling.average)
    correction4 = y4 - back4
    y1 += np.nan_to_num(
        reproject_array(correction4, PROFILES["4km"], PROFILES["1km"], Resampling.bilinear),
        nan=0.0,
    )
    return y1.astype("float32")


def fit_stage2_v2(features, y4, stage1_y1):
    X4 = matrix(features["4km"], STATIC_FEATURES)
    X100 = matrix(features["100m"], STATIC_FEATURES)
    y = y4.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(X4).all(axis=1)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X4[valid], y[valid])
    raw100 = rf.predict(X100).reshape((PROFILES["100m"]["height"], PROFILES["100m"]["width"]))

    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    shape100 = (PROFILES["100m"]["height"], PROFILES["100m"]["width"])
    raw_parent = parent_mean_10x(raw100, shape1)
    surface_anomaly = raw100 - expand_parent_10x(raw_parent, shape100)

    smooth = reproject_array(stage1_y1, PROFILES["1km"], PROFILES["100m"], Resampling.bilinear)
    smooth_parent = parent_mean_10x(smooth, shape1)
    smooth_anomaly = smooth - expand_parent_10x(smooth_parent, shape100)

    pred100 = expand_parent_10x(stage1_y1, shape100) + smooth_anomaly + surface_anomaly
    final_parent = parent_mean_10x(pred100, shape1)
    pred100 += expand_parent_10x(stage1_y1 - final_parent, shape100)
    return pred100.astype("float32")


def evaluate(date: str, method: str, ref100, pred100, y4, parent1):
    row = {
        "date": date,
        "method": method,
        **basic_metrics(ref100, pred100),
        **distribution_metrics(ref100, pred100),
        **subpixel_metrics(ref100, pred100),
    }
    c4 = conservation_metrics(pred100, PROFILES["100m"], y4, PROFILES["4km"])
    row["conservation_4km_r2"] = c4["r2"]
    row["conservation_4km_rmse"] = c4["rmse"]
    if parent1 is not None:
        c1 = conservation_metrics(pred100, PROFILES["100m"], parent1, PROFILES["1km"])
        row["conservation_1km_r2"] = c1["r2"]
        row["conservation_1km_rmse"] = c1["rmse"]
    else:
        row["conservation_1km_r2"] = float("nan")
        row["conservation_1km_rmse"] = float("nan")
    return row


def process_date(date: str, work: Path):
    print(f"=== {date}: data preparation ===", flush=True)
    scene, landsat_meta = landsat_scene(date)
    static = build_static_features(date, scene, work)
    dynamic = build_dynamic_features(date, static, work)
    features = merge_features(static, dynamic)
    y4, elite_meta = elite_target(date)
    ref100 = landsat_reference(date, scene, work)

    print(
        f"{date} Landsat={landsat_meta['acquired_utc']} clear100={np.isfinite(ref100).sum()} "
        f"ELITE mean={np.nanmean(y4):.3f}",
        flush=True,
    )

    direct, _ = fit_direct(features, y4)
    ordinary, ordinary_y1 = fit_ordinary(features, y4)

    v2_stage1 = fit_stage1(features, y4, STAGE1_V2)
    v2_pred = fit_stage2_v2(features, y4, v2_stage1)

    v3_stage1 = fit_stage1(features, y4, STAGE1_V3)
    v3_pred = fit_stage2_v2(features, y4, v3_stage1)

    methods = {
        "direct_mcp20": (direct, None),
        "ordinary_cascade": (ordinary, ordinary_y1),
        "scale_specific_v2": (v2_pred, v2_stage1),
        "scale_specific_v3_frozen": (v3_pred, v3_stage1),
    }
    rows = [
        evaluate(date, name, ref100, pred, y4, parent)
        for name, (pred, parent) in methods.items()
    ]

    parent_ref = reproject_array(ref100, PROFILES["100m"], PROFILES["1km"], Resampling.average)
    parent_metrics = {
        "date": date,
        "v2_stage1": basic_metrics(parent_ref, v2_stage1),
        "v3_stage1": basic_metrics(parent_ref, v3_stage1),
        "ordinary_stage1": basic_metrics(parent_ref, ordinary_y1),
    }

    provenance = {
        "date": date,
        "landsat": landsat_meta,
        "elite": elite_meta,
        "formal_hour_utc": HOUR,
        "landsat_clear_pixels_100m": int(np.isfinite(ref100).sum()),
        "grid_shapes": {k: [v["height"], v["width"]] for k, v in PROFILES.items()},
    }
    return rows, parent_metrics, provenance


def main():
    project = init_ee()
    out = Path("output/multidate_v3_validation")
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    all_rows = []
    parents = []
    provenance = []
    dates = DATES + [CONTROL_DATE]
    for date in dates:
        rows, parent, prov = process_date(date, work)
        all_rows.extend(rows)
        parents.append(parent)
        provenance.append(prov)
        print(pd.DataFrame(rows)[["date","method","r2","rmse","mae","bias","std_ratio","subpixel_anomaly_std_ratio","subpixel_anomaly_pearson"]].to_string(index=False), flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "per_date_metrics.csv", index=False)

    indep = df[df["date"].isin(DATES)].copy()
    summary_rows = []
    for method, g in indep.groupby("method"):
        summary_rows.append({
            "method": method,
            "dates": int(g["date"].nunique()),
            "mean_r2": float(g["r2"].mean()),
            "median_r2": float(g["r2"].median()),
            "mean_rmse": float(g["rmse"].mean()),
            "median_rmse": float(g["rmse"].median()),
            "mean_mae": float(g["mae"].mean()),
            "mean_abs_bias": float(g["bias"].abs().mean()),
            "mean_std_ratio": float(g["std_ratio"].mean()),
            "mean_anomaly_std_ratio": float(g["subpixel_anomaly_std_ratio"].mean()),
            "mean_anomaly_pearson": float(g["subpixel_anomaly_pearson"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_rmse", "mean_r2"], ascending=[True, False])
    summary_df.to_csv(out / "independent_date_summary.csv", index=False)

    paired = []
    pivot = indep.pivot(index="date", columns="method", values=["r2","rmse","mae","subpixel_anomaly_std_ratio","subpixel_anomaly_pearson"])
    for date in DATES:
        rec = {"date": date}
        if date in pivot.index:
            for metric in ["r2","rmse","mae","subpixel_anomaly_std_ratio","subpixel_anomaly_pearson"]:
                rec[f"v3_minus_direct_{metric}"] = float(
                    pivot.loc[date, (metric, "scale_specific_v3_frozen")]
                    - pivot.loc[date, (metric, "direct_mcp20")]
                )
            rec["v3_beats_direct_rmse"] = bool(rec["v3_minus_direct_rmse"] < 0)
            rec["v3_beats_direct_r2"] = bool(rec["v3_minus_direct_r2"] > 0)
        paired.append(rec)
    pd.DataFrame(paired).to_csv(out / "v3_vs_direct_paired.csv", index=False)

    control = df[df["date"] == CONTROL_DATE].to_dict(orient="records")
    report = {
        "ee_project": project,
        "formal_roi": ROI,
        "formal_hour_utc": HOUR,
        "independent_dates": DATES,
        "control_date": CONTROL_DATE,
        "frozen_method": {
            "source_repo": "yuyan3616/jiangchidu4kmto100m",
            "script": "scripts/heihe_20190924/downscaling/run_scale_specific_rf_v3.py",
            "stage1_features": STAGE1_V3,
            "stage2": "v2 absolute thermal-potential surface anomaly",
            "rf_params": RF_PARAMS,
            "no_method_tuning_from_independent_dates": True,
        },
        "data_refresh_rule": (
            "Each date rebuilds Landsat spectral indices, same-day MCD43A3 BSA, "
            "04:00 UTC ERA5-Land and 04:00 UTC ELITE. Terrain, WorldCover and coordinate "
            "semantics remain fixed. Landsat LST is used only for post-prediction evaluation."
        ),
        "selection_origin": "ROI clear-LST scouting run 36653792591; 2019-09-24 excluded from selection",
        "independent_summary": summary_rows,
        "paired_v3_vs_direct": paired,
        "control_20190924": control,
        "provenance": provenance,
        "stage1_parent_metrics": parents,
        "interpretation_guard": (
            "The five selected dates are method-selection holdout diagnostics. "
            "No feature set, RF hyperparameter or model structure is changed based on their results."
        ),
    }
    (out / "run_summary.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print("=== INDEPENDENT-DATE SUMMARY ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("=== V3 VS DIRECT PAIRED ===", flush=True)
    print(pd.DataFrame(paired).to_string(index=False), flush=True)
    print("=== CONTROL 2019-09-24 ===", flush=True)
    print(pd.DataFrame(control)[["date","method","r2","rmse","mae","bias","std_ratio","subpixel_anomaly_std_ratio","subpixel_anomaly_pearson"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
