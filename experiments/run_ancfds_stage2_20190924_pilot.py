from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from scipy import ndimage
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATE = "2019-09-24"
HOUR = 4
ROI = [99.86, 38.67, 100.5, 39.43]

# 与 frozen-v3 / ILC pilot 保持一致的正式验证网格。
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

STATIC_FEATURES = [
    "dem", "slope", "aspect",
    "ndvi", "ndbi", "mndwi", "savi", "bsi", "ui",
    "albedo_bsa", "landcover", "lat_m", "lon_m",
]

RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": 42,
}

ROOT = Path(".")
OUT = Path("output/ancfds_stage2_20190924")
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "ancfds": ROOT / "data/tpdc_ancfds/v1/zhangye-2e599e4070/2019/09/24/ANCFDS_FY4A_20190924_0400_C.tif",
    "landsat": ROOT / "data/landsat_c2_l2/v1/zhangye-c6c86510ae/2019/09/24/LC08_L2SP_133033_20190924_20200826_02_T1_L2_QC.tif",
    "surface": ROOT / "data/scaling_factors/v1/zhangye-c6c86510ae/surface/20190924_20190924/LANDSAT_SCALING_FACTORS_100M.tif",
    "terrain": ROOT / "data/scaling_factors/v1/zhangye-c6c86510ae/static/SRTM_TERRAIN_100M.tif",
    "landcover": ROOT / "data/scaling_factors/v1/zhangye-c6c86510ae/static/WORLDCOVER_2021_100M.tif",
    "albedo": ROOT / "data/scaling_factors/v1/zhangye-c6c86510ae/albedo/2019/09/24/MCD43A3_20190924_BSA_WSA_QC.tif",
    "elite": ROOT / "data/elite/china/2019/09/24/ELITE_FY4A_LST_20190924_0400_CHINA_K.tif",
}

SURFACE_FALLBACK = [
    "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2",
    "NDVI", "EVI", "FVC", "MNDWI", "NDBI", "BSI", "NDMI",
]
LANDSAT_FALLBACK = [
    "LST_C", "ST_QA_K", "SR_B2", "SR_B3", "SR_B4", "SR_B5",
    "SR_B6", "SR_B7", "QA_PIXEL", "QA_RADSAT",
]
TERRAIN_FALLBACK = ["DEM_M", "SLOPE_DEG", "ASPECT_DEG"]
ALBEDO_FALLBACK = ["BSA_SHORTWAVE", "WSA_SHORTWAVE", "ALBEDO_QA"]
ANCFDS_FALLBACK = ["T_DIR_C", "T_NADIR_C", "T_HEMI_C"]

# 前序实验已报告的同日结果，用于图上做参照。
PRIOR_R2 = {
    "P0_1km": 0.606,
    "ILC-v1_1km": 0.666,
    "P0_100m": 0.543,
    "ILC-v1_100m": 0.600,
}


def ensure_inputs() -> None:
    missing = [str(p) for p in PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required cached inputs:\n" + "\n".join(missing))


def profile_dict(src) -> dict:
    return {
        "crs": src.crs,
        "transform": src.transform,
        "width": src.width,
        "height": src.height,
    }


def read_named(path: Path, fallback: list[str] | None = None):
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype("float32")
        p = profile_dict(src)
        descriptions = list(src.descriptions)
        tags = src.tags()
        scales = list(src.scales)
        dtypes = list(src.dtypes)

    names = []
    for i in range(data.shape[0]):
        name = descriptions[i] if i < len(descriptions) else None
        if not name and fallback and i < len(fallback):
            name = fallback[i]
        if not name:
            name = f"BAND_{i + 1}"
        names.append(str(name).upper())

    result = {}
    for i, name in enumerate(names):
        arr = data[i].filled(np.nan).astype("float32")
        arr[np.isclose(arr, -9999.0, atol=1e-3)] = np.nan
        result[name] = arr

    meta = {
        "profile": p,
        "descriptions": names,
        "tags": tags,
        "scales": scales,
        "dtypes": dtypes,
    }
    return result, meta


def choose_band(mapping: dict[str, np.ndarray], candidates: list[str]) -> np.ndarray:
    upper = {k.upper(): k for k in mapping}
    for candidate in candidates:
        key = upper.get(candidate.upper())
        if key is not None:
            return mapping[key]
    for candidate in candidates:
        token = candidate.upper()
        for key_upper, key in upper.items():
            if token in key_upper:
                return mapping[key]
    raise KeyError(f"Could not find any of {candidates}; available={list(mapping)}")


def reproject_array(arr, src_profile, dst_profile, resampling):
    src = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    out = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=src,
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
    return arr[tuple(idx)].astype("float32")


def safe_ratio(num, den):
    out = np.full_like(num, np.nan, dtype="float32")
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-6)
    out[valid] = num[valid] / den[valid]
    return out


def build_static_features():
    surface, smeta = read_named(PATHS["surface"], SURFACE_FALLBACK)
    sp = smeta["profile"]

    def s(name):
        return reproject_array(choose_band(surface, [name]), sp, PROFILES["100m"], Resampling.bilinear)

    red = s("RED")
    nir = s("NIR")
    swir2 = s("SWIR2")

    base100 = {
        "ndvi": fill_continuous(s("NDVI")),
        "ndbi": fill_continuous(s("NDBI")),
        "mndwi": fill_continuous(s("MNDWI")),
        "bsi": fill_continuous(s("BSI")),
        "savi": fill_continuous(safe_ratio((nir - red) * 1.5, nir + red + 0.5)),
        "ui": fill_continuous(safe_ratio(swir2 - nir, swir2 + nir)),
    }

    terrain, tmeta = read_named(PATHS["terrain"], TERRAIN_FALLBACK)
    tp = tmeta["profile"]
    base100["dem"] = fill_continuous(
        reproject_array(choose_band(terrain, ["DEM_M", "DEM", "ELEVATION"]), tp, PROFILES["100m"], Resampling.average)
    )
    base100["slope"] = fill_continuous(
        reproject_array(choose_band(terrain, ["SLOPE_DEG", "SLOPE"]), tp, PROFILES["100m"], Resampling.average)
    )
    base100["aspect"] = fill_continuous(
        reproject_array(choose_band(terrain, ["ASPECT_DEG", "ASPECT"]), tp, PROFILES["100m"], Resampling.average)
    )

    albedo, ameta = read_named(PATHS["albedo"], ALBEDO_FALLBACK)
    ap = ameta["profile"]
    base100["albedo_bsa"] = fill_continuous(
        reproject_array(choose_band(albedo, ["BSA_SHORTWAVE", "ALBEDO_BSA"]), ap, PROFILES["100m"], Resampling.bilinear)
    )

    landcover, lcmeta = read_named(PATHS["landcover"], ["LANDCOVER"])
    lcp = lcmeta["profile"]
    base100["landcover"] = fill_nearest(
        reproject_array(choose_band(landcover, ["LANDCOVER", "MAP"]), lcp, PROFILES["100m"], Resampling.nearest)
    )

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
            arr = reproject_array(base100[name], PROFILES["100m"], PROFILES[res], method)
            dst[name] = fill_nearest(arr) if name == "landcover" else fill_continuous(arr)
        all_features[res] = dst
    return all_features, {
        "surface_descriptions": smeta["descriptions"],
        "terrain_descriptions": tmeta["descriptions"],
        "albedo_descriptions": ameta["descriptions"],
        "landcover_descriptions": lcmeta["descriptions"],
    }


def load_landsat_reference():
    bands, meta = read_named(PATHS["landsat"], LANDSAT_FALLBACK)
    lst = choose_band(bands, ["LST_C"])

    # 当前仓库中的 2019-09-24 是 legacy v1 cache：只有 8 个 band，
    # QA_PIXEL 没有保留下来；该版本的 LST_C 已在下载阶段做过研究 QC。
    # 若后续切到 v2 cache，则自动使用原始 QA_PIXEL bits 0..5 做 clear mask。
    qa_key = next((k for k in bands if k.upper() == "QA_PIXEL"), None)
    if qa_key is not None:
        qa = bands[qa_key]
        clear = np.isfinite(lst) & np.isfinite(qa)
        q = np.zeros_like(qa, dtype="uint32")
        q[np.isfinite(qa)] = np.rint(qa[np.isfinite(qa)]).astype("uint32")
        for bit in [0, 1, 2, 3, 4, 5]:
            clear &= (q & (1 << bit)) == 0
        meta["reference_mask_mode"] = "QA_PIXEL bits 0..5 == 0"
    else:
        clear = np.isfinite(lst)
        meta["reference_mask_mode"] = "legacy v1 cache finite LST_C (already QC-filtered)"

    ref_native = np.where(clear, lst, np.nan).astype("float32")
    ref100 = reproject_array(ref_native, meta["profile"], PROFILES["100m"], Resampling.average)
    ref100[(ref100 < -60) | (ref100 > 80)] = np.nan
    return ref100, meta


def load_elite_target():
    with rasterio.open(PATHS["elite"]) as src:
        raw = src.read(1, masked=True).astype("float32").filled(np.nan)
        p = profile_dict(src)
        tags = src.tags()
        scales = list(src.scales)
        dtype = src.dtypes[0]
    finite = raw[np.isfinite(raw)]
    med = float(np.median(finite)) if finite.size else float("nan")
    tag_scale = None
    for key in ["scale_factor", "source_scale_factor_kelvin"]:
        if key in tags:
            try:
                tag_scale = float(tags[key])
                break
            except Exception:
                pass
    if tag_scale is not None:
        values_k = raw * tag_scale
    elif med > 1000:
        values_k = raw * 0.01
    elif 150 < med < 400:
        values_k = raw
    elif scales and scales[0] not in (None, 1.0):
        values_k = raw * float(scales[0])
    else:
        raise RuntimeError(f"Cannot infer ELITE scale: dtype={dtype}, median={med}, tags={tags}")
    values_c = values_k - 273.15
    y4 = fill_continuous(reproject_array(values_c, p, PROFILES["4km"], Resampling.bilinear))
    return y4, {"dtype": dtype, "median_raw": med, "tags": tags, "scales": scales}


def load_ancfds_parents():
    bands, meta = read_named(PATHS["ancfds"], ANCFDS_FALLBACK)
    p = meta["profile"]
    mapping = {
        "T_DIR": choose_band(bands, ["T_DIR_C", "T_DIR"]),
        "T_NADIR": choose_band(bands, ["T_NADIR_C", "T_NADIR"]),
        "T_HEMI": choose_band(bands, ["T_HEMI_C", "T_HEMI"]),
    }
    parents = {}
    for name, arr in mapping.items():
        parent = reproject_array(arr, p, PROFILES["1km"], Resampling.bilinear)
        parents[name] = fill_continuous(parent)
    return parents, meta


def matrix(features, names):
    return np.column_stack([features[n].reshape(-1) for n in names]).astype("float32")


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


def build_surface_anomaly(features, y4):
    X4 = matrix(features["4km"], STATIC_FEATURES)
    X100 = matrix(features["100m"], STATIC_FEATURES)
    y = y4.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(X4).all(axis=1)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X4[valid], y[valid])
    raw100 = rf.predict(X100).reshape((PROFILES["100m"]["height"], PROFILES["100m"]["width"])).astype("float32")

    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    shape100 = (PROFILES["100m"]["height"], PROFILES["100m"]["width"])
    raw_parent = parent_mean_10x(raw100, shape1)
    surface_anomaly = raw100 - expand_parent_10x(raw_parent, shape100)

    importance = pd.DataFrame({
        "feature": STATIC_FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return surface_anomaly.astype("float32"), raw100, importance


def apply_native_stage2(parent1, features, feature_names, include_smooth=True):
    """Train the 1 km -> 100 m RF on ANCFDS itself, then conserve each 1 km parent."""
    X1 = matrix(features["1km"], feature_names)
    X100 = matrix(features["100m"], feature_names)
    y = parent1.reshape(-1)
    valid = np.isfinite(y) & np.isfinite(X1).all(axis=1)
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X1[valid], y[valid])
    raw100 = rf.predict(X100).reshape(
        (PROFILES["100m"]["height"], PROFILES["100m"]["width"])
    ).astype("float32")

    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    shape100 = (PROFILES["100m"]["height"], PROFILES["100m"]["width"])
    raw_parent = parent_mean_10x(raw100, shape1)
    local_anomaly = raw100 - expand_parent_10x(raw_parent, shape100)

    pred100 = expand_parent_10x(parent1, shape100) + local_anomaly
    if include_smooth:
        smooth = reproject_array(
            parent1, PROFILES["1km"], PROFILES["100m"], Resampling.bilinear
        )
        smooth_parent = parent_mean_10x(smooth, shape1)
        pred100 += smooth - expand_parent_10x(smooth_parent, shape100)

    final_parent = parent_mean_10x(pred100, shape1)
    pred100 += expand_parent_10x(parent1 - final_parent, shape100)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return pred100.astype("float32"), importance


def parent_only_from_1km(parent1):
    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    shape100 = (PROFILES["100m"]["height"], PROFILES["100m"]["width"])
    smooth = reproject_array(parent1, PROFILES["1km"], PROFILES["100m"], Resampling.bilinear)
    smooth_parent = parent_mean_10x(smooth, shape1)
    smooth_anomaly = smooth - expand_parent_10x(smooth_parent, shape100)
    pred100 = expand_parent_10x(parent1, shape100) + smooth_anomaly
    final_parent = parent_mean_10x(pred100, shape1)
    pred100 += expand_parent_10x(parent1 - final_parent, shape100)
    return pred100.astype("float32")


def apply_stage2(parent1, surface_anomaly):
    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    shape100 = (PROFILES["100m"]["height"], PROFILES["100m"]["width"])
    smooth = reproject_array(parent1, PROFILES["1km"], PROFILES["100m"], Resampling.bilinear)
    smooth_parent = parent_mean_10x(smooth, shape1)
    smooth_anomaly = smooth - expand_parent_10x(smooth_parent, shape100)

    pred100 = expand_parent_10x(parent1, shape100) + smooth_anomaly + surface_anomaly
    final_parent = parent_mean_10x(pred100, shape1)
    pred100 += expand_parent_10x(parent1 - final_parent, shape100)
    return pred100.astype("float32")


def harmonize_parent_to_elite(parent1, y4, iterations=3):
    """Use ANCFDS only for within-4km structure while retaining ELITE 4km means.

    This mirrors the ILC-v1 conservation idea more closely than feeding raw
    ANCFDS absolute temperatures directly into Stage2.
    """
    h = parent1.astype("float32", copy=True)
    for _ in range(iterations):
        back4 = reproject_array(h, PROFILES["1km"], PROFILES["4km"], Resampling.average)
        correction4 = y4 - back4
        correction1 = reproject_array(
            correction4, PROFILES["4km"], PROFILES["1km"], Resampling.nearest
        )
        h = h + np.nan_to_num(correction1, nan=0.0).astype("float32")
    return h.astype("float32")


def basic_metrics(ref, pred):
    valid = np.isfinite(ref) & np.isfinite(pred)
    y = ref[valid].astype("float64")
    p = pred[valid].astype("float64")
    if y.size < 2:
        return {
            "n": int(y.size), "r2": float("nan"), "rmse": float("nan"),
            "mae": float("nan"), "bias": float("nan"), "ubrmse": float("nan"),
            "pearson": float("nan"),
        }
    bias = float(np.mean(p - y))
    rmse = float(mean_squared_error(y, p) ** 0.5)
    centered = (p - y) - bias
    return {
        "n": int(y.size),
        "r2": float(r2_score(y, p)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y, p)),
        "bias": bias,
        "ubrmse": float(np.sqrt(np.mean(centered ** 2))),
        "pearson": float(np.corrcoef(y, p)[0, 1]),
    }


def subpixel_metrics(ref, pred):
    shape1 = (PROFILES["1km"]["height"], PROFILES["1km"]["width"])
    ref_parent = parent_mean_10x(ref, shape1)
    pred_parent = parent_mean_10x(pred, shape1)
    ra = ref - expand_parent_10x(ref_parent, ref.shape)
    pa = pred - expand_parent_10x(pred_parent, pred.shape)
    valid = np.isfinite(ra) & np.isfinite(pa)
    r = ra[valid].astype("float64")
    p = pa[valid].astype("float64")
    rs = float(np.std(r)) if r.size else float("nan")
    ps = float(np.std(p)) if p.size else float("nan")
    return {
        "subpixel_anomaly_std_ratio": ps / rs if r.size and rs else float("nan"),
        "subpixel_anomaly_pearson": float(np.corrcoef(r, p)[0, 1]) if r.size > 1 else float("nan"),
        "subpixel_anomaly_rmse": float(np.sqrt(np.mean((p - r) ** 2))) if r.size else float("nan"),
    }


def write_tif(path: Path, arr: np.ndarray, profile: dict, description: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(np.isfinite(arr), arr, -9999.0).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff",
        height=profile["height"], width=profile["width"], count=1,
        dtype="float32", crs=profile["crs"], transform=profile["transform"],
        nodata=-9999.0, compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)


def plot_spatial(ref100, preds, parent_only):
    valid_ref = ref100[np.isfinite(ref100)]
    lo, hi = np.nanpercentile(valid_ref, [2, 98])
    err_lim = 8.0

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    im = axes[0, 0].imshow(ref100, vmin=lo, vmax=hi)
    axes[0, 0].set_title("Landsat reference 100 m")
    axes[0, 0].axis("off")

    order = ["T_DIR", "T_NADIR", "T_HEMI"]
    for j, name in enumerate(order, start=1):
        axes[0, j].imshow(preds[name], vmin=lo, vmax=hi)
        m = basic_metrics(ref100, preds[name])
        axes[0, j].set_title(f"{name} + Stage2\nR²={m['r2']:.3f}, RMSE={m['rmse']:.2f}°C")
        axes[0, j].axis("off")

    axes[1, 0].imshow(parent_only["T_NADIR"], vmin=lo, vmax=hi)
    pm = basic_metrics(ref100, parent_only["T_NADIR"])
    axes[1, 0].set_title(f"T_NADIR parent-only\nR²={pm['r2']:.3f}, RMSE={pm['rmse']:.2f}°C")
    axes[1, 0].axis("off")

    for j, name in enumerate(order, start=1):
        err = preds[name] - ref100
        eim = axes[1, j].imshow(err, vmin=-err_lim, vmax=err_lim)
        axes[1, j].set_title(f"{name} error (prediction - Landsat)")
        axes[1, j].axis("off")

    fig.colorbar(im, ax=axes[0, :], shrink=0.7, label="LST (°C)")
    fig.colorbar(eim, ax=axes[1, :], shrink=0.7, label="Error (°C)")
    fig.suptitle("ANCFDS 1 km → frozen Stage2 → 100 m, 2019-09-24 04 UTC", fontsize=14)
    fig.savefig(OUT / "01_spatial_comparison.png", dpi=180)
    plt.close(fig)


def plot_scatter(ref100, preds):
    order = ["T_DIR", "T_NADIR", "T_HEMI"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, name in zip(axes, order):
        valid = np.isfinite(ref100) & np.isfinite(preds[name])
        x = ref100[valid]
        y = preds[name][valid]
        hb = ax.hexbin(x, y, gridsize=70, mincnt=1, bins="log")
        lo = float(np.nanpercentile(np.concatenate([x, y]), 1))
        hi = float(np.nanpercentile(np.concatenate([x, y]), 99))
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        m = basic_metrics(ref100, preds[name])
        ax.set_title(f"{name}: R²={m['r2']:.3f}, RMSE={m['rmse']:.2f}°C")
        ax.set_xlabel("Landsat LST (°C)")
        ax.set_ylabel("Prediction LST (°C)")
        fig.colorbar(hb, ax=ax, label="log10 count")
    fig.suptitle("100 m validation scatter")
    fig.savefig(OUT / "02_scatter_comparison.png", dpi=180)
    plt.close(fig)


def plot_r2_comparison(metrics_df):
    new_1 = {}
    new_100 = {}
    for name in ["T_DIR", "T_NADIR", "T_HEMI"]:
        new_1[name] = float(metrics_df.loc[metrics_df["method"] == f"{name}_1km_parent", "r2"].iloc[0])
        new_100[name] = float(metrics_df.loc[metrics_df["method"] == f"{name}_stage2_100m", "r2"].iloc[0])

    labels = ["P0", "ILC-v1", "ANCFDS T_DIR", "ANCFDS T_NADIR", "ANCFDS T_HEMI"]
    values1 = [PRIOR_R2["P0_1km"], PRIOR_R2["ILC-v1_1km"], new_1["T_DIR"], new_1["T_NADIR"], new_1["T_HEMI"]]
    values100 = [PRIOR_R2["P0_100m"], PRIOR_R2["ILC-v1_100m"], new_100["T_DIR"], new_100["T_NADIR"], new_100["T_HEMI"]]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, values1, width, label="1 km R²")
    ax.bar(x + width / 2, values100, width, label="100 m R²")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("R²")
    ax.set_ylim(min(0.0, min(values100 + values1) - 0.05), min(1.0, max(values100 + values1) + 0.12))
    ax.set_title("2019-09-24 R² comparison (prior pilot vs ANCFDS replacement)")
    ax.legend()
    for xpos, v in zip(x - width / 2, values1):
        ax.text(xpos, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xpos, v in zip(x + width / 2, values100):
        ax.text(xpos, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(OUT / "03_r2_comparison_with_prior.png", dpi=180)
    plt.close(fig)


def plot_parent_stage2_delta(metrics_df):
    order = ["T_DIR", "T_NADIR", "T_HEMI"]
    parent = [float(metrics_df.loc[metrics_df.method == f"{n}_parent_only_100m", "rmse"].iloc[0]) for n in order]
    stage2 = [float(metrics_df.loc[metrics_df.method == f"{n}_stage2_100m", "rmse"].iloc[0]) for n in order]
    x = np.arange(3)
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.bar(x - width / 2, parent, width, label="Parent-only 100 m")
    ax.bar(x + width / 2, stage2, width, label="+ frozen Stage2 anomaly")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Does Stage2 improve the ANCFDS 1 km parent?")
    ax.legend()
    for xpos, v in zip(x - width / 2, parent):
        ax.text(xpos, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    for xpos, v in zip(x + width / 2, stage2):
        ax.text(xpos, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    fig.savefig(OUT / "04_parent_vs_stage2_rmse.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(importance):
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("Frozen Stage2 thermal-potential RF")
    fig.savefig(OUT / "05_stage2_feature_importance.png", dpi=180)
    plt.close(fig)


def plot_harmonized_comparison(metrics_df):
    order = ["T_DIR", "T_NADIR", "T_HEMI"]
    labels = ["P0", "ILC-v1"] + [f"Harmonized {n}" for n in order]
    values1 = [
        PRIOR_R2["P0_1km"], PRIOR_R2["ILC-v1_1km"],
        *[
            float(metrics_df.loc[
                metrics_df["method"] == f"{n}_HARM_1km_parent", "r2"
            ].iloc[0])
            for n in order
        ],
    ]
    values100 = [
        PRIOR_R2["P0_100m"], PRIOR_R2["ILC-v1_100m"],
        *[
            float(metrics_df.loc[
                metrics_df["method"] == f"{n}_HARM_stage2_100m", "r2"
            ].iloc[0])
            for n in order
        ],
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.7), constrained_layout=True)
    ax.bar(x - width / 2, values1, width, label="1 km R²")
    ax.bar(x + width / 2, values100, width, label="100 m R²")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("R²")
    ax.set_title("ILC-like test: ANCFDS local structure + ELITE 4 km absolute parent")
    ax.legend()
    ymin = min(0.0, min(values1 + values100) - 0.05)
    ymax = min(1.0, max(values1 + values100) + 0.12)
    ax.set_ylim(ymin, ymax)
    for xpos, v in zip(x - width / 2, values1):
        ax.text(xpos, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for xpos, v in zip(x + width / 2, values100):
        ax.text(xpos, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(OUT / "06_harmonized_r2_comparison.png", dpi=180)
    plt.close(fig)


def plot_harmonized_spatial(ref100, preds):
    order = ["T_DIR", "T_NADIR", "T_HEMI"]
    valid_ref = ref100[np.isfinite(ref100)]
    lo, hi = np.nanpercentile(valid_ref, [2, 98])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for j, name in enumerate(order):
        axes[0, j].imshow(preds[name], vmin=lo, vmax=hi)
        m = basic_metrics(ref100, preds[name])
        axes[0, j].set_title(
            f"{name} harmonized + Stage2\nR²={m['r2']:.3f}, RMSE={m['rmse']:.2f}°C"
        )
        axes[0, j].axis("off")
        err = preds[name] - ref100
        axes[1, j].imshow(err, vmin=-8, vmax=8)
        axes[1, j].set_title(f"{name} error")
        axes[1, j].axis("off")
    fig.suptitle("ANCFDS local anomaly + ELITE parent → Stage2 → 100 m")
    fig.savefig(OUT / "07_harmonized_spatial.png", dpi=180)
    plt.close(fig)


def plot_native_stage2_comparison(metrics_df):
    order = ["T_DIR", "T_NADIR", "T_HEMI"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, name in zip(axes, order):
        methods = [
            (f"{name}_parent_only_100m", "Parent-only"),
            (f"{name}_stage2_100m", "Frozen ELITE Stage2"),
            (f"{name}_NATIVE_stage2_100m", "Native RF"),
            (f"{name}_NATIVE_NOXY_stage2_100m", "Native RF no-XY"),
        ]
        vals = [
            float(metrics_df.loc[metrics_df["method"] == key, "r2"].iloc[0])
            for key, _ in methods
        ]
        labels = [label for _, label in methods]
        x = np.arange(len(labels))
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(name)
        ax.set_ylabel("100 m R²")
        ax.set_ylim(min(0, min(vals) - 0.05), min(1.0, max(vals) + 0.12))
        for xpos, v in zip(x, vals):
            ax.text(xpos, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("ANCFDS-native 1 km → 100 m retraining vs frozen Stage2")
    fig.savefig(OUT / "08_native_stage2_r2.png", dpi=180)
    plt.close(fig)


def plot_best_native_spatial(ref100, native_preds, metrics_df):
    candidates = []
    for key, arr in native_preds.items():
        method = f"{key}_100m"
        row = metrics_df.loc[metrics_df["method"] == method]
        if len(row):
            candidates.append((float(row["r2"].iloc[0]), key, arr))
    if not candidates:
        return
    candidates.sort(reverse=True, key=lambda x: x[0])
    r2, key, pred = candidates[0]
    m = basic_metrics(ref100, pred)
    valid_ref = ref100[np.isfinite(ref100)]
    lo, hi = np.nanpercentile(valid_ref, [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].imshow(ref100, vmin=lo, vmax=hi)
    axes[0].set_title("Landsat reference")
    axes[0].axis("off")
    axes[1].imshow(pred, vmin=lo, vmax=hi)
    axes[1].set_title(f"{key}\nR²={m['r2']:.3f}, RMSE={m['rmse']:.2f}°C")
    axes[1].axis("off")
    axes[2].imshow(pred - ref100, vmin=-8, vmax=8)
    axes[2].set_title("Error")
    axes[2].axis("off")
    fig.suptitle("Best ANCFDS-native Stage2 candidate")
    fig.savefig(OUT / "09_best_native_spatial.png", dpi=180)
    plt.close(fig)


def landcover_metrics(ref100, preds, landcover100):
    rows = []
    classes = sorted(int(x) for x in np.unique(landcover100[np.isfinite(landcover100)]))
    for cls in classes:
        mask = np.isfinite(ref100) & (np.rint(landcover100).astype(int) == cls)
        if int(mask.sum()) < 100:
            continue
        for name, pred in preds.items():
            m = basic_metrics(np.where(mask, ref100, np.nan), np.where(mask, pred, np.nan))
            rows.append({"landcover": cls, "method": name, **m})
    return pd.DataFrame(rows)


def main():
    ensure_inputs()

    print("=== Load static features ===", flush=True)
    features, feature_meta = build_static_features()
    print("=== Load Landsat reference ===", flush=True)
    ref100, landsat_meta = load_landsat_reference()
    print(f"Landsat clear 100m pixels: {np.isfinite(ref100).sum()}", flush=True)

    print("=== Load ELITE 4km Stage2 training target ===", flush=True)
    y4, elite_meta = load_elite_target()
    print(f"ELITE 4km mean: {np.nanmean(y4):.3f} C", flush=True)

    print("=== Load ANCFDS parents ===", flush=True)
    parents, ancfds_meta = load_ancfds_parents()
    harmonized_parents = {
        name: harmonize_parent_to_elite(parent, y4)
        for name, parent in parents.items()
    }

    print("=== Train frozen Stage2 RF once ===", flush=True)
    surface_anomaly, raw100, importance = build_surface_anomaly(features, y4)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    ref1 = reproject_array(ref100, PROFILES["100m"], PROFILES["1km"], Resampling.average)
    rows = []
    preds = {}
    parent_only = {}

    for name, parent in parents.items():
        pm = basic_metrics(ref1, parent)
        rows.append({
            "date": DATE, "hour_utc": HOUR, "method": f"{name}_1km_parent",
            "resolution": "1km", **pm,
        })

        po = parent_only_from_1km(parent)
        parent_only[name] = po
        pom = basic_metrics(ref100, po)
        rows.append({
            "date": DATE, "hour_utc": HOUR, "method": f"{name}_parent_only_100m",
            "resolution": "100m", **pom, **subpixel_metrics(ref100, po),
        })

        pred = apply_stage2(parent, surface_anomaly)
        preds[name] = pred
        m = basic_metrics(ref100, pred)
        rows.append({
            "date": DATE, "hour_utc": HOUR, "method": f"{name}_stage2_100m",
            "resolution": "100m", **m, **subpixel_metrics(ref100, pred),
        })

        conservation = basic_metrics(parent, parent_mean_10x(pred, parent.shape))
        rows[-1]["conservation_1km_rmse"] = conservation["rmse"]
        rows[-1]["conservation_1km_bias"] = conservation["bias"]

        write_tif(OUT / f"{name.lower()}_parent_1km.tif", parent, PROFILES["1km"], f"{name}_C")
        write_tif(OUT / f"{name.lower()}_parent_only_100m.tif", po, PROFILES["100m"], f"{name}_PARENT_ONLY_C")
        write_tif(OUT / f"{name.lower()}_stage2_100m.tif", pred, PROFILES["100m"], f"{name}_STAGE2_C")

    harmonized_preds = {}
    harmonized_parent_only = {}
    for name, parent in harmonized_parents.items():
        pm = basic_metrics(ref1, parent)
        cons4 = basic_metrics(
            y4,
            reproject_array(parent, PROFILES["1km"], PROFILES["4km"], Resampling.average),
        )
        rows.append({
            "date": DATE, "hour_utc": HOUR, "method": f"{name}_HARM_1km_parent",
            "resolution": "1km", **pm,
            "conservation_4km_rmse": cons4["rmse"],
            "conservation_4km_bias": cons4["bias"],
        })

        po = parent_only_from_1km(parent)
        harmonized_parent_only[name] = po
        pom = basic_metrics(ref100, po)
        rows.append({
            "date": DATE, "hour_utc": HOUR, "method": f"{name}_HARM_parent_only_100m",
            "resolution": "100m", **pom, **subpixel_metrics(ref100, po),
        })

        pred = apply_stage2(parent, surface_anomaly)
        harmonized_preds[name] = pred
        m = basic_metrics(ref100, pred)
        rows.append({
            "date": DATE, "hour_utc": HOUR, "method": f"{name}_HARM_stage2_100m",
            "resolution": "100m", **m, **subpixel_metrics(ref100, pred),
        })
        conservation1 = basic_metrics(parent, parent_mean_10x(pred, parent.shape))
        rows[-1]["conservation_1km_rmse"] = conservation1["rmse"]
        rows[-1]["conservation_1km_bias"] = conservation1["bias"]

        write_tif(
            OUT / f"{name.lower()}_harm_parent_1km.tif",
            parent, PROFILES["1km"], f"{name}_HARM_C"
        )
        write_tif(
            OUT / f"{name.lower()}_harm_stage2_100m.tif",
            pred, PROFILES["100m"], f"{name}_HARM_STAGE2_C"
        )

    native_preds = {}
    native_importance_rows = []
    no_xy = [n for n in STATIC_FEATURES if n not in {"lat_m", "lon_m"}]
    for name, parent in parents.items():
        for variant, feature_names in [
            ("NATIVE", STATIC_FEATURES),
            ("NATIVE_NOXY", no_xy),
        ]:
            pred, imp = apply_native_stage2(
                parent, features, feature_names, include_smooth=True
            )
            key = f"{name}_{variant}_stage2"
            native_preds[key] = pred
            m = basic_metrics(ref100, pred)
            rows.append({
                "date": DATE,
                "hour_utc": HOUR,
                "method": f"{key}_100m",
                "resolution": "100m",
                **m,
                **subpixel_metrics(ref100, pred),
            })
            conservation1 = basic_metrics(parent, parent_mean_10x(pred, parent.shape))
            rows[-1]["conservation_1km_rmse"] = conservation1["rmse"]
            rows[-1]["conservation_1km_bias"] = conservation1["bias"]
            imp = imp.copy()
            imp.insert(0, "variant", variant)
            imp.insert(0, "band", name)
            native_importance_rows.append(imp)
            write_tif(
                OUT / f"{name.lower()}_{variant.lower()}_stage2_100m.tif",
                pred,
                PROFILES["100m"],
                f"{name}_{variant}_C",
            )

    pd.concat(native_importance_rows, ignore_index=True).to_csv(
        OUT / "native_stage2_feature_importance.csv", index=False
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    lc100 = features["100m"]["landcover"]
    lc_preds = {f"{k}_raw_stage2": v for k, v in preds.items()}
    lc_preds.update({f"{k}_harm_stage2": v for k, v in harmonized_preds.items()})
    lcdf = landcover_metrics(ref100, lc_preds, lc100)
    lcdf.to_csv(OUT / "landcover_metrics.csv", index=False)

    plot_spatial(ref100, preds, parent_only)
    plot_scatter(ref100, preds)
    plot_r2_comparison(metrics)
    plot_parent_stage2_delta(metrics)
    plot_feature_importance(importance)
    plot_harmonized_comparison(metrics)
    plot_harmonized_spatial(ref100, harmonized_preds)
    plot_native_stage2_comparison(metrics)
    plot_best_native_spatial(ref100, native_preds, metrics)

    raw_1 = metrics[
        metrics["method"].isin([f"{n}_1km_parent" for n in parents])
    ].sort_values("r2", ascending=False).iloc[0]
    raw_100 = metrics[
        metrics["method"].isin([f"{n}_stage2_100m" for n in parents])
    ].sort_values("r2", ascending=False).iloc[0]
    harm_1 = metrics[
        metrics["method"].isin([f"{n}_HARM_1km_parent" for n in parents])
    ].sort_values("r2", ascending=False).iloc[0]
    harm_100 = metrics[
        metrics["method"].isin([f"{n}_HARM_stage2_100m" for n in parents])
    ].sort_values("r2", ascending=False).iloc[0]
    best_1_row = harm_1
    best_100_row = harm_100
    native_rows = metrics[
        metrics["method"].str.contains("_NATIVE") &
        metrics["method"].str.endswith("_100m")
    ].sort_values("r2", ascending=False)
    best_native = native_rows.iloc[0]

    summary = {
        "experiment": "ANCFDS 1km replacement -> frozen Stage2 -> 100m",
        "date": DATE,
        "hour_utc": HOUR,
        "roi": ROI,
        "design": (
            "Keep the prior frozen Stage2 thermal-potential RF unchanged: train the "
            "100m surface anomaly from ELITE 4km + static factors, replace only the "
            "Stage1/ILC 1km parent field with each ANCFDS band, and enforce exact "
            "1km parent conservation after Stage2."
        ),
        "prior_reported_r2": PRIOR_R2,
        "best_raw_ancfds_1km": {
            "method": str(raw_1["method"]),
            "r2": float(raw_1["r2"]),
            "rmse": float(raw_1["rmse"]),
            "mae": float(raw_1["mae"]),
            "bias": float(raw_1["bias"]),
        },
        "best_raw_ancfds_100m": {
            "method": str(raw_100["method"]),
            "r2": float(raw_100["r2"]),
            "rmse": float(raw_100["rmse"]),
            "mae": float(raw_100["mae"]),
            "bias": float(raw_100["bias"]),
        },
        "best_harmonized_ancfds_1km": {
            "method": str(harm_1["method"]),
            "r2": float(harm_1["r2"]),
            "rmse": float(harm_1["rmse"]),
            "mae": float(harm_1["mae"]),
            "bias": float(harm_1["bias"]),
        },
        "best_harmonized_ancfds_100m": {
            "method": str(harm_100["method"]),
            "r2": float(harm_100["r2"]),
            "rmse": float(harm_100["rmse"]),
            "mae": float(harm_100["mae"]),
            "bias": float(harm_100["bias"]),
        },
        "best_native_ancfds_100m": {
            "method": str(best_native["method"]),
            "r2": float(best_native["r2"]),
            "rmse": float(best_native["rmse"]),
            "mae": float(best_native["mae"]),
            "bias": float(best_native["bias"]),
            "delta_r2_vs_prior_ilc_100m": float(
                best_native["r2"] - PRIOR_R2["ILC-v1_100m"]
            ),
        },
        "delta_vs_prior_ilc_r2": {
            "1km": float(best_1_row["r2"] - PRIOR_R2["ILC-v1_1km"]),
            "100m": float(best_100_row["r2"] - PRIOR_R2["ILC-v1_100m"]),
        },
        "clear_landsat_100m_pixels": int(np.isfinite(ref100).sum()),
        "ancfds_input": {
            "path": str(PATHS["ancfds"]),
            "bands": ancfds_meta["descriptions"],
            "tags": ancfds_meta["tags"],
        },
        "landsat_input": {
            "path": str(PATHS["landsat"]),
            "bands": landsat_meta["descriptions"],
            "tags": landsat_meta["tags"],
        },
        "elite_input": {
            "path": str(PATHS["elite"]),
            **elite_meta,
        },
        "feature_sources": feature_meta,
        "notes": [
            "This is a same-day pilot at 04:00 UTC, matching the frozen validation hour.",
            "T_DIR, T_NADIR and T_HEMI are all tested rather than choosing a band in advance.",
            "Raw-direct and ILC-like harmonized variants are both evaluated.",
            "Harmonized means: preserve ANCFDS within-4km structure but force the 4km parent mean back to ELITE, matching the zero-mean ILC philosophy.",
            "A native Stage2 ablation is also trained directly at 1 km on each ANCFDS band, with both full static features and a no-lat/lon variant.",
            "The comparison chart includes the previously reported 2019-09-24 P0/ILC-v1 R2 values.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    write_tif(OUT / "landsat_reference_100m.tif", ref100, PROFILES["100m"], "LANDSAT_LST_C")
    write_tif(OUT / "stage2_surface_anomaly_100m.tif", surface_anomaly, PROFILES["100m"], "STAGE2_SURFACE_ANOMALY_C")

    print("\n=== METRICS ===", flush=True)
    print(metrics.to_string(index=False), flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
