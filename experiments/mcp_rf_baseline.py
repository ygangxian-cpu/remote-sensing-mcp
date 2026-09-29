from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATE = "2019-09-24"
HOUR = 4
ROI = [99.86, 38.67, 100.5, 39.43]
RID = "zhangye-experiment-2e599e4070"

# Legacy high-precision reconstructed FY-4A target exported from
# yuyan3616/jiangchidu4kmto100m for 2019-09-24 04:00 UTC.
OLD_TARGET_VALUES = np.array([[32.47857666015625,32.47857666015625,32.885772705078125,31.675384521484375,32.809112548828125,32.809112548828125,32.764312744140625,32.1815185546875,33.479644775390625,33.479644775390625,33.974609375,33.248077392578125,33.248077392578125,32.8409423828125,33.038848876953125,33.1490478515625,33.1490478515625,33.149017333984375],[32.1484375,32.1484375,32.412384033203125,31.79595947265625,31.79595947265625,30.8714599609375,30.1561279296875,30.21148681640625,30.21148681640625,31.870033264160156,32.685997009277344,31.576171875,31.576171875,32.962127685546875,33.13775634765625,33.14892578125,33.14892578125,32.345794677734375],[31.02593994140625,32.1484375,32.412384033203125,31.79595947265625,31.79595947265625,30.8714599609375,30.1561279296875,30.21148681640625,30.21148681640625,30.645702362060547,31.056581497192383,31.576171875,31.576171875,32.962127685546875,33.13775634765625,33.14892578125,33.14892578125,32.345794677734375],[32.02801513671875,32.379730224609375,32.050262451171875,31.02679443359375,31.02679443359375,30.322601318359375,31.232208251953125,31.049072265625,31.049072265625,30.45501708984375,30.2783203125,30.9388427734375,30.9388427734375,33.4249267578125,33.80975341796875,34.0189208984375,34.0189208984375,33.20458984375],[33.226959228515625,33.226959228515625,33.18267822265625,30.94989013671875,30.94989013671875,29.749725341796875,30.025482177734375,30.79541015625,30.79541015625,30.3883056640625,31.35321044921875,29.29827880859375,29.29827880859375,31.620513916015625,33.9085693359375,33.9085693359375,34.513702392578125,34.6126708984375],[33.226959228515625,33.226959228515625,33.18267822265625,30.94989013671875,30.94989013671875,29.749725341796875,30.025482177734375,30.025482177734375,30.79541015625,30.3883056640625,31.35321044921875,31.35321044921875,29.29827880859375,31.620513916015625,33.9085693359375,33.9085693359375,34.513702392578125,34.6126708984375],[30.5859375,32.79730224609375,33.5460205078125,33.5460205078125,31.52069091796875,30.200286865234375,30.245574951171875,30.245574951171875,29.427520751953125,29.05657958984375,30.057647705078125,30.057647705078125,29.911346435546875,28.978851318359375,31.697052001953125,31.697052001953125,34.304351806640625,35.492340087890625],[29.000885009765625,31.41119384765625,33.545806884765625,33.545806884765625,32.8082275390625,31.278717041015625,30.5201416015625,30.5201416015625,28.9351806640625,27.900909423828125,28.4180908203125,28.4180908203125,29.2330322265625,28.517120361328125,28.814056396484375,28.814056396484375,32.8525390625,34.700164794921875],[29.000885009765625,31.41119384765625,33.545806884765625,33.545806884765625,32.8082275390625,31.278717041015625,30.5201416015625,30.5201416015625,28.9351806640625,27.900909423828125,28.4180908203125,28.4180908203125,29.2330322265625,28.517120361328125,28.814056396484375,28.814056396484375,32.8525390625,34.700164794921875],[30.992767333984375,31.751556396484375,33.332672119140625,33.332672119140625,33.9415283203125,32.092803955078125,29.91485595703125,29.91485595703125,30.090728759765625,29.881591796875,29.867584228515625,29.867584228515625,29.694610595703125,29.255126953125,29.255126953125,28.14312744140625,30.46527099609375,32.7532958984375],[33.486083984375,33.5194091796875,33.732208251953125,33.732208251953125,32.69775390625,31.79193115234375,31.79193115234375,29.93603515625,30.343963623046875,29.749053955078125,29.749053955078125,28.63873291015625,28.847076416015625,30.18939208984375,30.18939208984375,28.0491943359375,27.526763916015625,29.98028564453125],[32.389617919921875,33.5194091796875,33.5194091796875,33.732208251953125,32.69775390625,31.79193115234375,31.79193115234375,29.93603515625,30.343963623046875,29.749053955078125,29.749053955078125,28.63873291015625,28.847076416015625,30.18939208984375,30.18939208984375,28.0491943359375,27.526763916015625,29.98028564453125],[32.389617919921875,32.40069580078125,32.40069580078125,32.092529296875,33.6292724609375,33.621856689453125,33.621856689453125,31.043365478515625,28.89581298828125,29.149169921875,29.149169921875,29.122467041015625,29.143798828125,29.01239013671875,29.01239013671875,29.155731201171875,29.07843017578125,29.27642822265625],[29.199310302734375,28.142730712890625,28.142730712890625,28.835723876953125,30.383544921875,32.290374755859375,32.290374755859375,32.70849609375,30.3427734375,29.34173583984375,29.34173583984375,30.420440673828125,29.404541015625,26.877105712890625,26.877105712890625,27.603302001953125,28.4449462890625,28.35699462890625],[25.741546630859375,25.5435791015625,25.5435791015625,25.52142333984375,30.383544921875,32.290374755859375,32.290374755859375,32.70849609375,30.3427734375,30.3427734375,29.34173583984375,30.420440673828125,29.404541015625,29.404541015625,26.877105712890625,27.603302001953125,28.4449462890625,28.4449462890625],[25.741546630859375,25.741546630859375,25.5435791015625,25.52142333984375,26.636260986328125,26.636260986328125,30.46380615234375,31.850555419921875,31.564605712890625,31.564605712890625,29.837310791015625,29.045318603515625,28.09307861328125,28.09307861328125,26.970916748046875,27.88397216796875,28.47796630859375,28.47796630859375],[23.269500732421875,23.269500732421875,23.071441650390625,23.5302734375,25.436370849609375,25.436370849609375,28.846527099609375,29.932525634765625,30.46435546875,30.46435546875,30.96661376953125,29.693634033203125,28.378814697265625,28.378814697265625,26.97076416015625,27.28985595703125,29.06646728515625,29.06646728515625],[24.047313690185547,22.53582763671875,20.79443359375,21.50604248046875,24.33624267578125,24.33624267578125,27.240325927734375,29.932525634765625,30.46435546875,30.46435546875,30.96661376953125,29.693634033203125,28.378814697265625,28.378814697265625,26.97076416015625,27.28985595703125,29.06646728515625,29.06646728515625],[23.269824981689453,21.9097900390625,20.79443359375,21.50604248046875,24.33624267578125,24.33624267578125,27.240325927734375,27.43829345703125,27.61114501953125,27.61114501953125,29.338287353515625,29.86944580078125,30.47442626953125,30.47442626953125,29.170654296875,28.04852294921875,28.76971435546875,28.76971435546875],[21.86200523376465,21.25119400024414,22.209396362304688,24.512359619140625,24.512359619140625,24.2371826171875,27.405364990234375,26.239227294921875,26.239227294921875,25.620025634765625,26.54412841796875,28.05108642578125,28.05108642578125,28.30401611328125,28.83209228515625,28.329742431640625,28.329742431640625,28.53875732421875],[22.178577423095703,22.676671981811523,23.693418502807617,23.783601760864258,23.745635986328125,24.872100830078125,25.40008544921875,26.019195556640625,26.019195556640625,24.6881103515625,23.995147705078125,25.575958251953125,28.05108642578125,28.30401611328125,28.83209228515625,28.329742431640625,28.329742431640625,28.53875732421875]], dtype="float32")
OLD_TARGET_TRANSFORM = [0.03393635517805412,0,99.86528182419725,0,-0.034221534633331885,39.39776620081703]
OLD_TARGET_CRS = "EPSG:4326"

OUT = Path("output/mcp_downscaling_baseline")

ELITE = Path("data/elite/china/2019/09/24/ELITE_FY4A_LST_20190924_0400_CHINA_K.tif")
SURFACE = Path(
    f"data/scaling_factors/v1/{RID}/surface/20190924_20190924/"
    "LANDSAT_SCALING_FACTORS_100M.tif"
)
TERRAIN = Path(f"data/scaling_factors/v1/{RID}/static/SRTM_TERRAIN_100M.tif")
LANDCOVER = Path(f"data/scaling_factors/v1/{RID}/static/WORLDCOVER_2021_100M.tif")
ALBEDO = Path(
    f"data/scaling_factors/v1/{RID}/albedo/2019/09/24/"
    "MCD43A3_20190924_BSA_WSA_QC.tif"
)
ERA5 = Path(
    f"data/era5_land/v1/{RID}/2019/09/24/"
    "ERA5LAND_20190924_0400_UTC.tif"
)

RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": 42,
}


def band_index(src: rasterio.DatasetReader, name: str) -> int:
    names = [d or "" for d in src.descriptions]
    if name not in names:
        raise KeyError(f"{name} not found in {src.name}; bands={names}")
    return names.index(name) + 1


def profile_like(src: rasterio.DatasetReader) -> dict:
    return {
        "crs": src.crs,
        "transform": src.transform,
        "width": src.width,
        "height": src.height,
    }


def continuous_fill(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=True)
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if not valid.any():
        raise RuntimeError("Predictor contains no valid pixels")
    work = np.where(valid, arr, 0).astype("float32")
    filled = fillnodata(work, mask=valid.astype("uint8"), max_search_distance=200)
    remaining = ~np.isfinite(filled)
    if remaining.any():
        from scipy import ndimage
        _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
        nearest = arr[tuple(idx)]
        filled[remaining] = nearest[remaining]
    return filled.astype("float32")


def nearest_fill(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=True)
    valid = np.isfinite(arr)
    if valid.all():
        return arr
    if not valid.any():
        raise RuntimeError("Categorical predictor contains no valid pixels")
    from scipy import ndimage
    _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = arr[tuple(idx)]
    out[valid] = arr[valid]
    return out.astype("float32")


def reproject_band(
    src_path: Path,
    band_name: str,
    dst_profile: dict,
    resampling: Resampling = Resampling.bilinear,
    fill: bool = True,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        idx = band_index(src, band_name)
        data = src.read(idx, masked=True).astype("float32")
        source = data.filled(np.nan)
        out = np.full(
            (dst_profile["height"], dst_profile["width"]),
            np.nan,
            dtype="float32",
        )
        src_nodata = src.nodata
        if src_nodata is None:
            src_nodata = -9999.0
        source2 = np.where(np.isfinite(source), source, src_nodata).astype("float32")
        reproject(
            source=source2,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    if fill:
        if resampling == Resampling.nearest:
            return nearest_fill(out)
        return continuous_fill(out)
    return out


def crop_elite() -> tuple[np.ndarray, dict]:
    xmin, ymin, xmax, ymax = ROI
    with rasterio.open(ELITE) as src:
        window = rasterio.windows.from_bounds(
            xmin, ymin, xmax, ymax, transform=src.transform
        ).round_offsets().round_lengths()
        raw = src.read(1, window=window, masked=True).astype("float32")
        arr = raw.filled(np.nan)
        # ELITE repository cache stores UInt16 Kelvin with scale_factor=0.01.
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmedian(finite)) > 1000:
            arr = arr * 0.01
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmedian(finite)) > 100:
            arr = arr - 273.15
        profile = {
            "crs": src.crs,
            "transform": src.window_transform(window),
            "width": int(window.width),
            "height": int(window.height),
        }
    arr = continuous_fill(arr)
    return arr.astype("float32"), profile


def grid_1km(profile100: dict) -> dict:
    transform = profile100["transform"]
    factor = 10
    return {
        "crs": profile100["crs"],
        "transform": rasterio.Affine(
            transform.a * factor,
            transform.b,
            transform.c,
            transform.d,
            transform.e * factor,
            transform.f,
        ),
        "width": math.ceil(profile100["width"] / factor),
        "height": math.ceil(profile100["height"] / factor),
    }


def reproject_array(
    arr: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    resampling: Resampling,
) -> np.ndarray:
    src_nodata = -9999.0
    source = np.where(np.isfinite(arr), arr, src_nodata).astype("float32")
    out = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=source,
        destination=out,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=src_nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return out


def coordinates_utm(profile100: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices((profile100["height"], profile100["width"]))
    xs, ys = rasterio.transform.xy(
        profile100["transform"], rows, cols, offset="center"
    )
    shape = rows.shape
    lon = np.asarray(xs, dtype="float64").reshape(shape)
    lat = np.asarray(ys, dtype="float64").reshape(shape)
    transformer = Transformer.from_crs(profile100["crs"], "EPSG:32647", always_xy=True)
    xm, ym = transformer.transform(lon, lat)
    return (
        lon,
        lat,
        np.asarray(xm, dtype="float32").reshape(shape),
        np.asarray(ym, dtype="float32").reshape(shape),
    )


def build_features_100m() -> tuple[dict[str, np.ndarray], dict]:
    with rasterio.open(SURFACE) as src:
        p100 = profile_like(src)

    blue = reproject_band(SURFACE, "BLUE", p100)
    red = reproject_band(SURFACE, "RED", p100)
    nir = reproject_band(SURFACE, "NIR", p100)
    swir2 = reproject_band(SURFACE, "SWIR2", p100)

    ndvi = reproject_band(SURFACE, "NDVI", p100)
    ndbi = reproject_band(SURFACE, "NDBI", p100)
    mndwi = reproject_band(SURFACE, "MNDWI", p100)
    bsi = reproject_band(SURFACE, "BSI", p100)
    savi = continuous_fill(1.5 * (nir - red) / np.maximum(nir + red + 0.5, 1e-6))
    ui = continuous_fill((swir2 - nir) / np.where(np.abs(swir2 + nir) < 1e-6, 1e-6, swir2 + nir))

    dem = reproject_band(TERRAIN, "DEM_M", p100)
    slope = reproject_band(TERRAIN, "SLOPE_DEG", p100)
    aspect = reproject_band(TERRAIN, "ASPECT_DEG", p100)
    lc = reproject_band(LANDCOVER, "LANDCOVER", p100, Resampling.nearest)
    bsa = reproject_band(ALBEDO, "BSA_SHORTWAVE", p100)

    t2 = reproject_band(ERA5, "T2_C", p100)
    td2 = reproject_band(ERA5, "TD2_C", p100)
    u10 = reproject_band(ERA5, "U10_MPS", p100)
    v10 = reproject_band(ERA5, "V10_MPS", p100)
    psfc = reproject_band(ERA5, "PSFC_PA", p100)
    swdown = reproject_band(ERA5, "SWDOWN_WM2", p100)
    glw = reproject_band(ERA5, "GLW_WM2", p100)
    wind_speed = continuous_fill(np.sqrt(u10 ** 2 + v10 ** 2))

    lon, lat, lon_m, lat_m = coordinates_utm(p100)

    # Reproduce the terrain-aware EDRF used by the existing experiment.
    doy = 267.0
    utc_hour = HOUR + 0.5
    decl = 0.40928 * math.sin(2.0 * math.pi * (doy - 81.0) / 364.0)
    lat_r = np.deg2rad(lat)
    hour_angle = (lon / 15.0 + utc_hour - 12.0) * math.pi / 12.0
    sin_alt = (
        np.sin(lat_r) * math.sin(decl)
        + np.cos(lat_r) * math.cos(decl) * np.cos(hour_angle)
    )
    cos_alt = np.sqrt(np.maximum(0.0, 1.0 - sin_alt**2))
    solar_az = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat_r) - math.tan(decl) * np.cos(lat_r),
    ) + math.pi
    slope_r = np.deg2rad(slope)
    aspect_r = np.deg2rad(aspect)
    dssr_flat = np.maximum(sin_alt, 0)
    dssr_topo = np.maximum(
        np.cos(slope_r) * sin_alt
        + np.sin(slope_r) * cos_alt * np.cos(solar_az - aspect_r),
        0,
    )
    day = sin_alt > 0
    f_topo = np.where(day, dssr_topo / np.maximum(dssr_flat, 0.01745), 1.0)
    f_topo = np.clip(f_topo, 0, 5)
    ssr_topo = np.where(day, swdown * f_topo, 0)
    w_ssr = np.clip(sin_alt, 0, 1)
    edrf = continuous_fill(w_ssr * ssr_topo + (1.0 - w_ssr) * glw)

    features = {
        "dem": dem,
        "slope": slope,
        "aspect": aspect,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "mndwi": mndwi,
        "savi": savi,
        "bsi": bsi,
        "ui": ui,
        "albedo_bsa": bsa,
        "landcover": lc,
        "lat_m": lat_m,
        "lon_m": lon_m,
        "edrf": edrf,
        "t2_c": t2,
        "td2_c": td2,
        "wind_speed": wind_speed,
        "psfc_pa": psfc,
        "swdown_wm2": swdown,
        "glw_wm2": glw,
    }
    for name, arr in features.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"Feature still contains gaps after filling: {name}")
    return features, p100


def aggregate_features(
    features100: dict[str, np.ndarray],
    p100: dict,
    dst_profile: dict,
) -> dict[str, np.ndarray]:
    out = {}
    for name, arr in features100.items():
        method = Resampling.nearest if name == "landcover" else Resampling.average
        agg = reproject_array(arr, p100, dst_profile, method)
        out[name] = nearest_fill(agg) if name == "landcover" else continuous_fill(agg)
    return out


def matrix(features: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.column_stack([features[name].reshape(-1) for name in names]).astype("float32")


def metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
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


def old_target_4km() -> tuple[np.ndarray, dict]:
    profile = {
        "crs": OLD_TARGET_CRS,
        "transform": rasterio.Affine(*OLD_TARGET_TRANSFORM),
        "width": int(OLD_TARGET_VALUES.shape[1]),
        "height": int(OLD_TARGET_VALUES.shape[0]),
    }
    return OLD_TARGET_VALUES.copy(), profile


def target_stats(arr: np.ndarray) -> dict:
    x = arr[np.isfinite(arr)].astype("float64")
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def target_pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[valid].reshape(-1), b[valid].reshape(-1))[0, 1])


def landsat_reference(p100: dict) -> tuple[np.ndarray, int]:
    candidates = sorted(
        Path(f"data/landsat_c2_l2/v2/{RID}/2019/09/24").glob("*_L2_RAW_QA.tif")
    )
    if not candidates:
        raise FileNotFoundError("No Landsat v2 file found; run landsat_worker.py first")
    path = candidates[0]
    with rasterio.open(path) as src:
        lst = src.read(band_index(src, "LST_C"), masked=True).astype("float32").filled(np.nan)
        qa = src.read(band_index(src, "QA_PIXEL"), masked=True).astype("float32").filled(np.nan)
        qa_i = np.where(np.isfinite(qa), qa, 0).astype("int32")
        clear = np.isfinite(lst) & np.isfinite(qa)
        for bit in [0, 1, 2, 3, 4, 5]:
            clear &= (qa_i & (1 << bit)) == 0
        src_lst = np.where(clear, lst, -9999.0).astype("float32")
        ref = np.full((p100["height"], p100["width"]), np.nan, dtype="float32")
        reproject(
            source=src_lst,
            destination=ref,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-9999.0,
            dst_transform=p100["transform"],
            dst_crs=p100["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return ref, int(np.isfinite(ref).sum())


def run_feature_set(
    target_label: str,
    label: str,
    names: list[str],
    features100: dict[str, np.ndarray],
    p100: dict,
    f1_all: dict[str, np.ndarray],
    p1: dict,
    f4_all: dict[str, np.ndarray],
    p4: dict,
    y4: np.ndarray,
    ref100: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    X4 = matrix(f4_all, names)
    X1 = matrix(f1_all, names)
    X100 = matrix(features100, names)
    y4f = y4.reshape(-1)

    valid4 = np.isfinite(y4f) & np.isfinite(X4).all(axis=1)
    if valid4.sum() < 30:
        raise RuntimeError(f"Too few 4km training pixels: {valid4.sum()}")

    rf_direct = RandomForestRegressor(**RF_PARAMS)
    rf_direct.fit(X4[valid4], y4f[valid4])
    pred4 = rf_direct.predict(X4)
    res4 = (y4f - pred4).reshape(y4.shape)
    direct = rf_direct.predict(X100).reshape((p100["height"], p100["width"]))
    direct += np.nan_to_num(reproject_array(res4, p4, p100, Resampling.bilinear), nan=0.0)

    rf1 = RandomForestRegressor(**RF_PARAMS)
    rf1.fit(X4[valid4], y4f[valid4])
    pred4_l1 = rf1.predict(X4)
    res4_l1 = (y4f - pred4_l1).reshape(y4.shape)
    y1 = rf1.predict(X1).reshape((p1["height"], p1["width"]))
    y1 += np.nan_to_num(reproject_array(res4_l1, p4, p1, Resampling.bilinear), nan=0.0)

    y1f = y1.reshape(-1)
    valid1 = np.isfinite(y1f) & np.isfinite(X1).all(axis=1)
    rf2 = RandomForestRegressor(**RF_PARAMS)
    rf2.fit(X1[valid1], y1f[valid1])
    pred1 = rf2.predict(X1)
    res1 = (y1f - pred1).reshape(y1.shape)
    cascade = rf2.predict(X100).reshape((p100["height"], p100["width"]))
    cascade += np.nan_to_num(reproject_array(res1, p1, p100, Resampling.bilinear), nan=0.0)

    rows = [
        {"target_source": target_label, "feature_set": label, "method": "direct_4km_to_100m", **metrics(ref100, direct)},
        {"target_source": target_label, "feature_set": label, "method": "cascaded_4km_to_1km_to_100m", **metrics(ref100, cascade)},
    ]
    diagnostics = {
        "feature_names": names,
        "n_features": len(names),
        "training_pixels_4km": int(valid4.sum()),
        "training_pixels_1km": int(valid1.sum()),
        "train_metrics_4km": metrics(y4f[valid4], pred4[valid4]),
        "train_metrics_1km": metrics(y1f[valid1], pred1[valid1]),
    }
    return rows, {"direct": direct, "cascade": cascade}, diagnostics


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    elite_native, elite_profile = crop_elite()
    legacy_target, p4 = old_target_4km()

    # Put ELITE on the exact legacy 4 km grid so target_source is the only target-side change.
    elite_target = reproject_array(
        elite_native, elite_profile, p4, Resampling.bilinear
    )
    elite_target = continuous_fill(elite_target)

    features100, p100 = build_features_100m()
    p1 = grid_1km(p100)
    f1_all = aggregate_features(features100, p100, p1)
    f4_all = aggregate_features(features100, p100, p4)
    ref100, ref_valid = landsat_reference(p100)

    base14 = [
        "dem", "slope", "aspect",
        "ndvi", "ndbi", "mndwi", "savi", "bsi", "ui",
        "albedo_bsa", "landcover", "lat_m", "lon_m", "edrf",
    ]
    mcp20 = base14 + [
        "t2_c", "td2_c", "wind_speed", "psfc_pa", "swdown_wm2", "glw_wm2",
    ]

    targets = {
        "elite": elite_target,
        "legacy_reconstructed": legacy_target,
    }

    all_rows = []
    predictions = {}
    diagnostics = {}

    for target_label, y4 in targets.items():
        predictions[target_label] = {}
        diagnostics[target_label] = {}
        for label, names in [("base14", base14), ("mcp20", mcp20)]:
            rows, preds, diag = run_feature_set(
                target_label, label, names,
                features100, p100, f1_all, p1, f4_all, p4, y4, ref100
            )
            all_rows.extend(rows)
            predictions[target_label][label] = preds
            diagnostics[target_label][label] = diag

    pd.DataFrame(all_rows).to_csv(OUT / "metrics.csv", index=False)

    target_comparison = {
        "elite": target_stats(elite_target),
        "legacy_reconstructed": target_stats(legacy_target),
        "pearson_r": target_pearson(elite_target, legacy_target),
        "elite_vs_legacy_metrics": metrics(legacy_target, elite_target),
    }

    meta = {
        "date": DATE,
        "hour_utc": HOUR,
        "roi": ROI,
        "target_grid": {
            "shape": [p4["height"], p4["width"]],
            "crs": str(p4["crs"]),
            "transform": list(p4["transform"])[:6],
        },
        "target_comparison": target_comparison,
        "predictor_sources": {
            "surface": str(SURFACE),
            "terrain": str(TERRAIN),
            "landcover": str(LANDCOVER),
            "albedo": str(ALBEDO),
            "era5": str(ERA5),
        },
        "grid_shapes": {
            "4km": [p4["height"], p4["width"]],
            "1km": [p1["height"], p1["width"]],
            "100m": [p100["height"], p100["width"]],
        },
        "landsat_clear_100m_pixels": ref_valid,
        "experiments": diagnostics,
        "validation": all_rows,
        "comparison_note": (
            "Target-source ablation holds ROI, 4 km grid, MCP predictors, Landsat v2 reference, "
            "RF parameters and residual correction fixed. Only the 4 km LST target source changes."
        ),
    }
    (OUT / "run_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    profile = {
        "driver": "GTiff",
        "height": p100["height"],
        "width": p100["width"],
        "count": 1,
        "dtype": "float32",
        "crs": p100["crs"],
        "transform": p100["transform"],
        "nodata": -9999.0,
        "compress": "deflate",
    }
    outputs = [("landsat_reference_100m.tif", ref100)]
    for target_label, feature_sets in predictions.items():
        for label, preds in feature_sets.items():
            outputs.append((f"{target_label}_{label}_direct_100m.tif", preds["direct"]))
            outputs.append((f"{target_label}_{label}_cascaded_100m.tif", preds["cascade"]))
    for name, arr in outputs:
        with rasterio.open(OUT / name, "w", **profile) as dst:
            dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    run()
