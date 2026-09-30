from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for p in [REPO_ROOT, EXPERIMENTS]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_frozen_v3_multidate_validation as md
import diagnose_elite_modis_multidate as em

TRAIN_DATES = [
    (datetime(2019, 9, 15) + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range(16)
]
EVAL_DATES = ["2019-09-15", "2019-09-24"]
PLATFORM_HOUR = {"terra": 5, "aqua": 7}
PLATFORMS = ["terra", "aqua"]
MIN_4KM_SUPPORT = 0.50

PARENT_RF = {
    "n_estimators": 600,
    "min_samples_leaf": 12,
    "max_features": 0.8,
    "n_jobs": -1,
    "random_state": 42,
}
LOCAL_RF = {
    "n_estimators": 600,
    "min_samples_leaf": 20,
    "max_features": 0.8,
    "n_jobs": -1,
    "random_state": 42,
}

PARENT_FEATURES = [
    "elite_c", "t2_c", "td2_c", "wind_speed", "psfc_pa",
    "swdown_wm2", "glw_wm2", "elite_minus_t2",
    "dem", "slope", "aspect", "albedo_bsa", "landcover",
    "lat_m", "lon_m", "utc_hour", "doy_sin", "doy_cos",
]
LOCAL_FEATURES = [
    "base_c", "base_anomaly_c", "dem_dev", "albedo_dev",
    "slope", "aspect", "landcover", "lat_m", "lon_m",
    "utc_hour", "doy_sin", "doy_cos",
]


def score(y, p):
    valid = np.isfinite(y) & np.isfinite(p)
    yy = np.asarray(y)[valid].astype("float64")
    pp = np.asarray(p)[valid].astype("float64")
    if yy.size < 3:
        return {
            "n": int(yy.size), "r2": np.nan, "pearson": np.nan,
            "rmse": np.nan, "mae": np.nan, "bias": np.nan,
        }
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    return {
        "n": int(yy.size),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "pearson": float(np.corrcoef(yy, pp)[0, 1]),
        "rmse": float(np.sqrt(np.mean((pp - yy) ** 2))),
        "mae": float(np.mean(np.abs(pp - yy))),
        "bias": float(np.mean(pp - yy)),
    }


def read_one(path: Path, name: str):
    raw, _ = md.read_multiband(path, [name])
    return raw[name]


def fixed_static_1km(work: Path):
    terr_path = work / "fixed_terrain_1km.tif"
    if not terr_path.exists():
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("dem")
        terrain = ee.Terrain.products(dem)
        image = ee.Image.cat([
            dem,
            terrain.select("slope").rename("slope"),
            terrain.select("aspect").rename("aspect"),
        ]).toFloat()
        md.download_exact_grid(image, terr_path, md.PROFILES["1km"])
    terr, _ = md.read_multiband(terr_path, ["dem", "slope", "aspect"])

    lc_path = work / "fixed_landcover_1km.tif"
    if not lc_path.exists():
        lc = (
            ee.ImageCollection("ESA/WorldCover/v200").first()
            .select("Map").rename("landcover").toFloat()
        )
        md.download_exact_grid(lc, lc_path, md.PROFILES["1km"])
    lc = read_one(lc_path, "landcover")

    out = {k: md.fill_continuous(v) for k, v in terr.items()}
    out["landcover"] = md.fill_nearest(lc)

    rows, cols = np.indices(
        (md.PROFILES["1km"]["height"], md.PROFILES["1km"]["width"])
    )
    xs, ys = rasterio.transform.xy(
        md.PROFILES["1km"]["transform"], rows, cols, offset="center"
    )
    lon = np.asarray(xs).reshape(rows.shape)
    lat = np.asarray(ys).reshape(rows.shape)
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    lon_m, lat_m = tr.transform(lon, lat)
    out["lon_m"] = np.asarray(lon_m, dtype="float32")
    out["lat_m"] = np.asarray(lat_m, dtype="float32")
    out["lon_deg"] = lon.astype("float32")
    return out


def aggregate_static_4km(static1):
    out = {}
    for name in ["dem", "slope", "aspect", "lat_m", "lon_m"]:
        out[name] = md.reproject_array(
            static1[name],
            md.PROFILES["1km"],
            md.PROFILES["4km"],
            Resampling.average,
        )
        out[name] = md.fill_continuous(out[name])
    out["landcover"] = md.reproject_array(
        static1["landcover"],
        md.PROFILES["1km"],
        md.PROFILES["4km"],
        Resampling.mode,
    )
    out["landcover"] = md.fill_nearest(out["landcover"])
    return out


def albedo_1km(date: str, work: Path):
    path = work / f"albedo_{date}_1km.tif"
    if not path.exists():
        d0 = ee.Date(date)
        geom = ee.Geometry.Rectangle(md.ROI, proj="EPSG:4326", geodesic=False)
        col = (
            ee.ImageCollection("MODIS/061/MCD43A3")
            .filterDate(d0, d0.advance(1, "day"))
            .filterBounds(geom)
        )
        if int(col.size().getInfo()) < 1:
            raise RuntimeError(f"No MCD43A3 on {date}")
        src = ee.Image(col.first())
        alb = (
            src.select("Albedo_BSA_shortwave")
            .multiply(0.001)
            .rename("albedo_bsa")
            .updateMask(
                src.select(
                    "BRDF_Albedo_Band_Mandatory_Quality_shortwave"
                ).eq(0)
            )
            .toFloat()
        )
        md.download_exact_grid(
            alb.resample("bilinear"),
            path,
            md.PROFILES["1km"],
            unmask_value=-9999.0,
        )
    return md.fill_continuous(read_one(path, "albedo_bsa"))


def era5_at(date: str, hour: int, work: Path):
    path = work / f"era5_{date}_{hour:02d}_4km.tif"
    names = [
        "t2_c", "td2_c", "u10_mps", "v10_mps",
        "psfc_pa", "swdown_wm2", "glw_wm2",
    ]
    if not path.exists():
        ts = datetime.fromisoformat(f"{date}T{hour:02d}:00:00")
        start = ts.strftime("%Y-%m-%dT%H:%M:%S")
        end = (ts + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
        src = ee.Image(
            ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(start, end)
            .first()
        )
        image = ee.Image.cat([
            src.select("temperature_2m").subtract(273.15).rename("t2_c"),
            src.select("dewpoint_temperature_2m")
            .subtract(273.15).rename("td2_c"),
            src.select("u_component_of_wind_10m").rename("u10_mps"),
            src.select("v_component_of_wind_10m").rename("v10_mps"),
            src.select("surface_pressure").rename("psfc_pa"),
            src.select("surface_solar_radiation_downwards_hourly")
            .divide(3600).rename("swdown_wm2"),
            src.select("surface_thermal_radiation_downwards_hourly")
            .divide(3600).rename("glw_wm2"),
        ]).toFloat()
        md.download_exact_grid(
            image.resample("bilinear"), path, md.PROFILES["4km"]
        )
    raw, _ = md.read_multiband(path, names)
    out = {k: md.fill_continuous(v) for k, v in raw.items()}
    out["wind_speed"] = md.fill_continuous(
        np.sqrt(out.pop("u10_mps") ** 2 + out.pop("v10_mps") ** 2)
    )
    return out


def expand_4_to_1(arr4):
    return md.reproject_array(
        arr4, md.PROFILES["4km"], md.PROFILES["1km"], Resampling.nearest
    )


def parent_average_1_to_4(arr1):
    return md.reproject_array(
        arr1, md.PROFILES["1km"], md.PROFILES["4km"], Resampling.average
    )


def date_angles(date: str):
    doy = float(datetime.fromisoformat(date).strftime("%j"))
    angle = 2.0 * math.pi * doy / 365.25
    return math.sin(angle), math.cos(angle)


def build_training_tables(work: Path):
    static1_fixed = fixed_static_1km(work)
    static4_fixed = aggregate_static_4km(static1_fixed)
    parent_frames = []
    local_frames = []
    provenance = []

    for date in TRAIN_DATES:
        print(f"=== training corpus {date} ===", flush=True)
        elite_hours, elite_meta = em.load_elite_hours(date)
        alb1 = albedo_1km(date, work)
        alb4 = md.reproject_array(
            alb1,
            md.PROFILES["1km"],
            md.PROFILES["4km"],
            Resampling.average,
        )
        alb4 = md.fill_continuous(alb4)
        dsin, dcos = date_angles(date)

        for platform in PLATFORMS:
            raw = em.read_exact_modis(date, platform, work)
            lst = raw["lst_day_c"]
            vt = raw["view_time_local_h"]
            qc = raw["qc_day"]
            valid = em.strict_modis_mask(lst, qc)
            utc = (vt - static1_fixed["lon_deg"] / 15.0) % 24.0
            valid &= np.isfinite(utc) & (utc >= 4.0) & (utc <= 8.0)
            exact1, _ = em.interpolate_elite(elite_hours, utc)
            valid &= np.isfinite(exact1)

            support4 = md.reproject_array(
                valid.astype("float32"),
                md.PROFILES["1km"],
                md.PROFILES["4km"],
                Resampling.average,
            )
            mod4 = parent_average_1_to_4(
                np.where(valid, lst, np.nan).astype("float32")
            )
            elite4 = parent_average_1_to_4(
                np.where(valid, exact1, np.nan).astype("float32")
            )
            utc4 = parent_average_1_to_4(
                np.where(valid, utc, np.nan).astype("float32")
            )
            keep4 = (
                np.isfinite(mod4)
                & np.isfinite(elite4)
                & np.isfinite(utc4)
                & (support4 >= MIN_4KM_SUPPORT)
            )

            dyn4 = era5_at(date, PLATFORM_HOUR[platform], work)
            rr, cc = np.indices(elite4.shape)
            pdata = {
                "date": np.full(int(keep4.sum()), date, dtype=object),
                "platform": np.full(
                    int(keep4.sum()), platform, dtype=object
                ),
                "row4": rr[keep4].astype("int16"),
                "col4": cc[keep4].astype("int16"),
                "modis_c": mod4[keep4].astype("float32"),
                "elite_c": elite4[keep4].astype("float32"),
                "delta_parent_c": (
                    mod4[keep4] - elite4[keep4]
                ).astype("float32"),
                "utc_hour": utc4[keep4].astype("float32"),
                "doy_sin": np.full(
                    int(keep4.sum()), dsin, dtype="float32"
                ),
                "doy_cos": np.full(
                    int(keep4.sum()), dcos, dtype="float32"
                ),
                "albedo_bsa": alb4[keep4].astype("float32"),
            }
            for name in [
                "dem", "slope", "aspect",
                "landcover", "lat_m", "lon_m",
            ]:
                pdata[name] = static4_fixed[name][keep4].astype("float32")
            for name in [
                "t2_c", "td2_c", "wind_speed",
                "psfc_pa", "swdown_wm2", "glw_wm2",
            ]:
                pdata[name] = dyn4[name][keep4].astype("float32")
            pdata["elite_minus_t2"] = (
                pdata["elite_c"] - pdata["t2_c"]
            ).astype("float32")
            parent_frames.append(pd.DataFrame(pdata))

            residual1 = np.where(
                valid, lst - exact1, np.nan
            ).astype("float32")
            residual4 = parent_average_1_to_4(residual1)
            residual_parent1 = expand_4_to_1(residual4)
            local_target = residual1 - residual_parent1

            elite_parent4 = parent_average_1_to_4(
                np.where(valid, exact1, np.nan).astype("float32")
            )
            elite_parent1 = expand_4_to_1(elite_parent4)
            base_anomaly = exact1 - elite_parent1

            dem4 = parent_average_1_to_4(static1_fixed["dem"])
            dem_dev = (
                static1_fixed["dem"] - expand_4_to_1(dem4)
            )
            alb4_for_dev = parent_average_1_to_4(alb1)
            alb_dev = alb1 - expand_4_to_1(alb4_for_dev)
            keep1 = (
                valid
                & np.isfinite(local_target)
                & np.isfinite(base_anomaly)
                & np.isfinite(dem_dev)
                & np.isfinite(alb_dev)
            )
            r1, c1 = np.indices(lst.shape)
            ldata = {
                "date": np.full(int(keep1.sum()), date, dtype=object),
                "platform": np.full(
                    int(keep1.sum()), platform, dtype=object
                ),
                "row1": r1[keep1].astype("int16"),
                "col1": c1[keep1].astype("int16"),
                "local_delta_c": local_target[keep1].astype("float32"),
                "base_c": exact1[keep1].astype("float32"),
                "base_anomaly_c": base_anomaly[keep1].astype("float32"),
                "dem_dev": dem_dev[keep1].astype("float32"),
                "albedo_dev": alb_dev[keep1].astype("float32"),
                "slope": static1_fixed["slope"][keep1].astype("float32"),
                "aspect": static1_fixed["aspect"][keep1].astype("float32"),
                "landcover": static1_fixed["landcover"][keep1]
                .astype("float32"),
                "lat_m": static1_fixed["lat_m"][keep1].astype("float32"),
                "lon_m": static1_fixed["lon_m"][keep1].astype("float32"),
                "utc_hour": utc[keep1].astype("float32"),
                "doy_sin": np.full(
                    int(keep1.sum()), dsin, dtype="float32"
                ),
                "doy_cos": np.full(
                    int(keep1.sum()), dcos, dtype="float32"
                ),
            }
            local_frames.append(pd.DataFrame(ldata))
            print(
                f"{date} {platform}: "
                f"parent n={int(keep4.sum())}, "
                f"local n={int(keep1.sum())}",
                flush=True,
            )

        provenance.append({
            "date": date,
            "elite_sources": elite_meta,
        })

    parent = pd.concat(parent_frames, ignore_index=True)
    local = pd.concat(local_frames, ignore_index=True)
    return parent, local, provenance


def fit_date_blocked(df, features, target, params):
    clean = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=features + [target, "date"])
        .reset_index(drop=True)
    )
    X = clean[features].to_numpy(dtype="float32")
    y = clean[target].to_numpy(dtype="float32")
    groups = clean["date"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    oof = np.full(len(clean), np.nan, dtype="float32")
    fold_rows = []

    for tr, te in logo.split(X, y, groups):
        model = RandomForestRegressor(**params)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te]).astype("float32")
        fold_rows.append({
            "date": str(groups[te][0]),
            **score(y[te], oof[te]),
        })

    full = RandomForestRegressor(**params)
    full.fit(X, y)
    return clean, oof, full, score(y, oof), fold_rows


def model_weights(platform_cv):
    raw = {}
    for platform, metrics in platform_cv.items():
        rmse = float(metrics["rmse"])
        raw[platform] = 1.0 / max(rmse * rmse, 1e-6)
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def fit_excluding_date(
    df, features, target, params, platform, excluded_date
):
    train = df[
        (df["platform"] == platform)
        & (df["date"] != excluded_date)
    ].copy()
    train = (
        train.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=features + [target])
    )
    model = RandomForestRegressor(**params)
    model.fit(
        train[features].to_numpy(dtype="float32"),
        train[target].to_numpy(dtype="float32"),
    )
    return model, int(len(train))


def target_parent_feature_arrays(date, y4, static4, dyn4):
    dsin, dcos = date_angles(date)
    return {
        "elite_c": y4,
        "t2_c": dyn4["t2_c"],
        "td2_c": dyn4["td2_c"],
        "wind_speed": dyn4["wind_speed"],
        "psfc_pa": dyn4["psfc_pa"],
        "swdown_wm2": dyn4["swdown_wm2"],
        "glw_wm2": dyn4["glw_wm2"],
        "elite_minus_t2": y4 - dyn4["t2_c"],
        "dem": static4["dem"],
        "slope": static4["slope"],
        "aspect": static4["aspect"],
        "albedo_bsa": static4["albedo_bsa"],
        "landcover": static4["landcover"],
        "lat_m": static4["lat_m"],
        "lon_m": static4["lon_m"],
        "utc_hour": np.full(y4.shape, 4.0, dtype="float32"),
        "doy_sin": np.full(y4.shape, dsin, dtype="float32"),
        "doy_cos": np.full(y4.shape, dcos, dtype="float32"),
    }


def predict_grid(model, arrays, names):
    X = np.column_stack([
        arrays[n].reshape(-1) for n in names
    ]).astype("float32")
    pred = model.predict(X).reshape(
        next(iter(arrays.values())).shape
    )
    return pred.astype("float32")


def local_feature_arrays(date, parent1, features1):
    dsin, dcos = date_angles(date)
    base4 = parent_average_1_to_4(parent1)
    base_anom = parent1 - expand_4_to_1(base4)
    dem4 = parent_average_1_to_4(features1["dem"])
    alb4 = parent_average_1_to_4(features1["albedo_bsa"])
    return {
        "base_c": parent1,
        "base_anomaly_c": base_anom,
        "dem_dev": features1["dem"] - expand_4_to_1(dem4),
        "albedo_dev": (
            features1["albedo_bsa"] - expand_4_to_1(alb4)
        ),
        "slope": features1["slope"],
        "aspect": features1["aspect"],
        "landcover": features1["landcover"],
        "lat_m": features1["lat_m"],
        "lon_m": features1["lon_m"],
        "utc_hour": np.full(
            parent1.shape, 4.0, dtype="float32"
        ),
        "doy_sin": np.full(
            parent1.shape, dsin, dtype="float32"
        ),
        "doy_cos": np.full(
            parent1.shape, dcos, dtype="float32"
        ),
    }


def zero_mean_local(local_delta):
    mean4 = parent_average_1_to_4(local_delta)
    centered = local_delta - expand_4_to_1(mean4)
    mean4b = parent_average_1_to_4(centered)
    return (
        centered - expand_4_to_1(mean4b)
    ).astype("float32")


def evaluate_variant(
    date, name, target4, parent1, pred100, ref100
):
    ref1 = md.reproject_array(
        ref100,
        md.PROFILES["100m"],
        md.PROFILES["1km"],
        Resampling.average,
    )
    ref4 = md.reproject_array(
        ref100,
        md.PROFILES["100m"],
        md.PROFILES["4km"],
        Resampling.average,
    )
    return {
        "date": date,
        "method": name,
        **{
            f"final100_{k}": v
            for k, v in md.basic_metrics(ref100, pred100).items()
        },
        **{
            f"final100_{k}": v
            for k, v in md.distribution_metrics(
                ref100, pred100
            ).items()
        },
        **{
            f"stage1_{k}": v
            for k, v in md.basic_metrics(ref1, parent1).items()
        },
        **{
            f"target4_{k}": v
            for k, v in md.basic_metrics(ref4, target4).items()
        },
        **md.subpixel_metrics(ref100, pred100),
    }


def evaluate_end_to_end(
    date, parent_df, local_df,
    parent_weights, local_weights, work
):
    print(f"=== end-to-end {date} ===", flush=True)
    scene, landsat_meta = md.landsat_scene(date)
    static = md.build_static_features(date, scene, work)
    dynamic = md.build_dynamic_features(date, static, work)
    features = md.merge_features(static, dynamic)
    y4_raw, elite_meta = md.elite_target(date)
    ref100 = md.landsat_reference(date, scene, work)

    raw_parent1 = md.fit_stage1(
        features, y4_raw, md.STAGE1_V3
    )

    static4 = {
        k: static["4km"][k]
        for k in [
            "dem", "slope", "aspect", "albedo_bsa",
            "landcover", "lat_m", "lon_m",
        ]
    }
    dyn4 = dynamic["4km"]
    p_arrays = target_parent_feature_arrays(
        date, y4_raw, static4, dyn4
    )

    parent_deltas = {}
    parent_train_n = {}
    for platform in PLATFORMS:
        model, n = fit_excluding_date(
            parent_df,
            PARENT_FEATURES,
            "delta_parent_c",
            PARENT_RF,
            platform,
            date,
        )
        parent_deltas[platform] = predict_grid(
            model, p_arrays, PARENT_FEATURES
        )
        parent_train_n[platform] = n

    parent_delta = sum(
        parent_weights[p] * parent_deltas[p]
        for p in PLATFORMS
    ).astype("float32")
    y4_corr = (y4_raw + parent_delta).astype("float32")
    parent_corr1 = md.fit_stage1(
        features, y4_corr, md.STAGE1_V3
    )

    def local_delta_for(parent1):
        arrays = local_feature_arrays(
            date, parent1, static["1km"]
        )
        deltas = {}
        ns = {}
        for platform in PLATFORMS:
            model, n = fit_excluding_date(
                local_df,
                LOCAL_FEATURES,
                "local_delta_c",
                LOCAL_RF,
                platform,
                date,
            )
            deltas[platform] = predict_grid(
                model, arrays, LOCAL_FEATURES
            )
            ns[platform] = n
        mixed = sum(
            local_weights[p] * deltas[p]
            for p in PLATFORMS
        )
        return zero_mean_local(mixed), ns

    local_raw, local_train_n = local_delta_for(raw_parent1)
    local_corr, _ = local_delta_for(parent_corr1)
    p2_parent1 = (raw_parent1 + local_raw).astype("float32")
    p3_parent1 = (
        parent_corr1 + local_corr
    ).astype("float32")

    variants = {
        "P0_current_v3": (y4_raw, raw_parent1),
        "P1_parent_only": (y4_corr, parent_corr1),
        "P2_local_only": (y4_raw, p2_parent1),
        "P3_hierarchical_parent_plus_local": (
            y4_corr, p3_parent1
        ),
    }

    rows = []
    for name, (target4, parent1) in variants.items():
        pred100 = md.fit_stage2_v2(
            features, target4, parent1
        )
        rows.append(
            evaluate_variant(
                date, name, target4,
                parent1, pred100, ref100
            )
        )

    diag = {
        "date": date,
        "landsat": landsat_meta,
        "elite": elite_meta,
        "parent_model_training_samples_excluding_date": (
            parent_train_n
        ),
        "local_model_training_samples_excluding_date": (
            local_train_n
        ),
        "parent_delta_mean_c": float(
            np.nanmean(parent_delta)
        ),
        "parent_delta_std_c": float(
            np.nanstd(parent_delta)
        ),
        "raw_target4_mean_c": float(
            np.nanmean(y4_raw)
        ),
        "corrected_target4_mean_c": float(
            np.nanmean(y4_corr)
        ),
    }
    return rows, diag


def main():
    md.init_ee()
    out = Path("output/parent_bias_v4_pilot")
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    parent, local, provenance = build_training_tables(work)
    parent.to_csv(out / "parent_matchups.csv", index=False)
    local.to_csv(out / "local_matchups.csv", index=False)

    cv_report = {"parent": {}, "local": {}}
    parent_platform_cv = {}
    local_platform_cv = {}
    fold_rows = []

    for platform in PLATFORMS:
        p = parent[
            parent["platform"] == platform
        ].copy()
        _, _, model, cv, folds = fit_date_blocked(
            p,
            PARENT_FEATURES,
            "delta_parent_c",
            PARENT_RF,
        )
        parent_platform_cv[platform] = cv
        cv_report["parent"][platform] = {
            "residual_cv": cv,
            "feature_importance": sorted(
                [
                    {
                        "feature": n,
                        "importance": float(v),
                    }
                    for n, v in zip(
                        PARENT_FEATURES,
                        model.feature_importances_,
                    )
                ],
                key=lambda x: x["importance"],
                reverse=True,
            ),
        }
        for row in folds:
            fold_rows.append({
                "component": "parent",
                "platform": platform,
                **row,
            })

        l = local[
            local["platform"] == platform
        ].copy()
        _, _, lmodel, lcv, lfolds = fit_date_blocked(
            l,
            LOCAL_FEATURES,
            "local_delta_c",
            LOCAL_RF,
        )
        local_platform_cv[platform] = lcv
        cv_report["local"][platform] = {
            "residual_cv": lcv,
            "feature_importance": sorted(
                [
                    {
                        "feature": n,
                        "importance": float(v),
                    }
                    for n, v in zip(
                        LOCAL_FEATURES,
                        lmodel.feature_importances_,
                    )
                ],
                key=lambda x: x["importance"],
                reverse=True,
            ),
        }
        for row in lfolds:
            fold_rows.append({
                "component": "local",
                "platform": platform,
                **row,
            })

    pd.DataFrame(fold_rows).to_csv(
        out / "date_blocked_cv_folds.csv",
        index=False,
    )

    parent_weights = model_weights(parent_platform_cv)
    local_weights = model_weights(local_platform_cv)

    rows = []
    diagnostics = []
    for date in EVAL_DATES:
        r, d = evaluate_end_to_end(
            date,
            parent,
            local,
            parent_weights,
            local_weights,
            work,
        )
        rows.extend(r)
        diagnostics.append(d)
        print(
            pd.DataFrame(r)[[
                "date", "method",
                "target4_r2", "target4_rmse",
                "target4_bias",
                "stage1_r2", "stage1_rmse",
                "stage1_bias",
                "final100_r2", "final100_rmse",
                "final100_bias",
                "subpixel_anomaly_std_ratio",
                "subpixel_anomaly_pearson",
            ]].to_string(index=False),
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        out / "end_to_end_metrics.csv",
        index=False,
    )

    paired = []
    for date in EVAL_DATES:
        g = metrics[
            metrics["date"] == date
        ].set_index("method")
        base = g.loc["P0_current_v3"]
        for method in [
            "P1_parent_only",
            "P2_local_only",
            "P3_hierarchical_parent_plus_local",
        ]:
            row = g.loc[method]
            paired.append({
                "date": date,
                "method": method,
                "delta_target4_rmse": float(
                    row["target4_rmse"]
                    - base["target4_rmse"]
                ),
                "delta_stage1_rmse": float(
                    row["stage1_rmse"]
                    - base["stage1_rmse"]
                ),
                "delta_final100_rmse": float(
                    row["final100_rmse"]
                    - base["final100_rmse"]
                ),
                "delta_final100_r2": float(
                    row["final100_r2"]
                    - base["final100_r2"]
                ),
                "delta_final100_bias": float(
                    row["final100_bias"]
                    - base["final100_bias"]
                ),
            })

    pd.DataFrame(paired).to_csv(
        out / "paired_vs_p0.csv",
        index=False,
    )

    report = {
        "status": "pilot",
        "purpose": (
            "Test hierarchical parent/local bias correction "
            "with date-blocked calibration before "
            "multi-season expansion."
        ),
        "train_dates": TRAIN_DATES,
        "end_to_end_eval_dates": EVAL_DATES,
        "leakage_control": (
            "For each Landsat end-to-end date, "
            "both parent and local correction models "
            "exclude the entire target date."
        ),
        "modis_time_matching": (
            "Per-pixel MODIS local solar view time -> UTC, "
            "then linear interpolation of ELITE hourly LST."
        ),
        "modis_qc": (
            "bits0-1<=1; bits2-3==0; bits6-7<=2"
        ),
        "parent_target": (
            "MODIS_4km - exact-time ELITE_4km"
        ),
        "local_target": (
            "(MODIS_1km - exact-time ELITE_1km) "
            "minus its 4km parent mean"
        ),
        "local_constraint": (
            "Predicted local correction is explicitly "
            "re-centered to zero 4km mean before application."
        ),
        "parent_features": PARENT_FEATURES,
        "local_features": LOCAL_FEATURES,
        "parent_rf": PARENT_RF,
        "local_rf": LOCAL_RF,
        "parent_platform_cv": parent_platform_cv,
        "local_platform_cv": local_platform_cv,
        "parent_platform_weights_inverse_cv_rmse2": (
            parent_weights
        ),
        "local_platform_weights_inverse_cv_rmse2": (
            local_weights
        ),
        "cv_report": cv_report,
        "end_to_end_metrics": rows,
        "paired_vs_p0": paired,
        "diagnostics": diagnostics,
        "provenance": provenance,
        "paper_guard": (
            "This 16-day September run is a mechanism pilot, "
            "not final evidence. The previously inspected "
            "Landsat dates are development diagnostics. "
            "Any promoted v4 method must be retrained on a "
            "multi-season MODIS corpus and evaluated on new "
            "Landsat dates and/or station observations."
        ),
    }
    (out / "run_summary.json").write_text(
        json.dumps(report, indent=2, default=float),
        encoding="utf-8",
    )

    print(
        "=== DATE-BLOCKED PARENT CV ===",
        flush=True,
    )
    print(
        json.dumps(
            parent_platform_cv, indent=2
        ),
        flush=True,
    )
    print(
        "=== DATE-BLOCKED LOCAL CV ===",
        flush=True,
    )
    print(
        json.dumps(
            local_platform_cv, indent=2
        ),
        flush=True,
    )
    print("=== PLATFORM WEIGHTS ===", flush=True)
    print(
        json.dumps({
            "parent": parent_weights,
            "local": local_weights,
        }, indent=2),
        flush=True,
    )
    print("=== PAIRED VS P0 ===", flush=True)
    print(
        pd.DataFrame(paired).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
