from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

DATE = "2019-09-24"
ERA5_DIR = Path("data/era5_land/v1/zhangye-formal-buffered-4ba50c0b09/2019/09/24")
ANCFDS_DIR = Path("data/tpdc_ancfds/v1/zhangye-2e599e4070/2019/09/24")
SURFACE = Path("data/scaling_factors/v1/zhangye-formal-master-955b44ac26/surface/20190924_20190924/LANDSAT_SCALING_FACTORS_100M.tif")
TERRAIN = Path("data/scaling_factors/v1/zhangye-formal-master-955b44ac26/static/SRTM_TERRAIN_100M.tif")
OUT = Path("output/experiments/lw_lst_diurnal_correlation")

ERA5_SW_BAND = 6
ERA5_LW_BAND = 7
ANCFDS_TNADIR_BAND = 2
SURFACE_NDVI_BAND = 7
TERRAIN_DEM_BAND = 1
NODATA = -9999.0


def read_band(path: Path, band: int) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(band, masked=True).astype("float32").filled(np.nan)
        profile = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }
    return arr, profile


def align(arr: np.ndarray, src: dict, dst: dict, method=Resampling.bilinear) -> np.ndarray:
    out = np.full((dst["height"], dst["width"]), np.nan, dtype="float32")
    reproject(
        source=arr,
        destination=out,
        src_transform=src["transform"],
        src_crs=src["crs"],
        dst_transform=dst["transform"],
        dst_crs=dst["crs"],
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=method,
    )
    return out


def ndvi_emissivity(ndvi: np.ndarray) -> np.ndarray:
    # Same NDVI-emissivity family used by the current research code:
    # bare/low vegetation ~0.97, dense vegetation ~0.99, smooth transition.
    x = np.asarray(ndvi, dtype="float64")
    pv = np.clip((x - 0.20) / (0.86 - 0.20), 0.0, 1.0) ** 2
    eps = 0.97 * (1.0 - pv) + 0.99 * pv
    eps[~np.isfinite(x)] = np.nan
    return eps.astype("float32")


def pixel_size_m(profile: dict) -> tuple[float, float]:
    t = profile["transform"]
    lat0 = t.f + t.e * profile["height"] / 2
    dy = abs(t.e) * 111_320.0
    dx = abs(t.a) * 111_320.0 * math.cos(math.radians(lat0))
    return max(dx, 1.0), max(dy, 1.0)


def sky_view_factor(dem: np.ndarray, profile: dict, max_distance_m: float = 10000.0,
                    azimuth_count: int = 16) -> np.ndarray:
    """Approximate SVF from DEM horizon angles.

    This is intentionally labelled an SVF proxy, not a full terrain longwave
    balance. It accounts for sky obstruction only; it does not invent
    surrounding-slope temperature or emitted longwave.
    """
    z = np.asarray(dem, dtype="float64")
    h, w = z.shape
    dx, dy = pixel_size_m(profile)
    step = max(dx, dy)
    max_steps = max(1, int(max_distance_m / step))
    horizon2_sum = np.zeros((h, w), dtype="float64")
    valid_dirs = np.zeros((h, w), dtype="float64")

    for az in np.linspace(0.0, 2.0 * math.pi, azimuth_count, endpoint=False):
        east, north = math.sin(az), math.cos(az)
        max_tan = np.full((h, w), -np.inf, dtype="float64")
        seen = set()
        for k in range(1, max_steps + 1):
            dist_nom = k * step
            dc = int(round(east * dist_nom / dx))
            dr = int(round(-north * dist_nom / dy))
            if (dr, dc) == (0, 0) or (dr, dc) in seen:
                continue
            seen.add((dr, dc))
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            if r0 >= r1 or c0 >= c1:
                continue
            rr0, rr1 = r0 + dr, r1 + dr
            cc0, cc1 = c0 + dc, c1 + dc
            dist = math.hypot(dc * dx, dr * dy)
            base = z[r0:r1, c0:c1]
            nbr = z[rr0:rr1, cc0:cc1]
            tan_h = (nbr - base) / max(dist, 1.0)
            block = max_tan[r0:r1, c0:c1]
            ok = np.isfinite(base) & np.isfinite(nbr)
            block[ok] = np.maximum(block[ok], tan_h[ok])

        ok = np.isfinite(max_tan)
        horizon = np.zeros_like(max_tan)
        horizon[ok] = np.maximum(0.0, np.arctan(max_tan[ok]))
        horizon2_sum[ok] += np.cos(horizon[ok]) ** 2
        valid_dirs[ok] += 1.0

    svf = np.divide(horizon2_sum, valid_dirs, out=np.full_like(horizon2_sum, np.nan),
                    where=valid_dirs > 0)
    return np.clip(svf, 0.0, 1.0).astype("float32")


def pearson(x, y) -> float:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def lagged_r(x: np.ndarray, y: np.ndarray, lag: int, mask: np.ndarray | None = None) -> float:
    # Correlate radiation at t-lag with LST at t.
    if lag == 0:
        xx, yy = x, y
        mm = np.ones(len(y), dtype=bool) if mask is None else mask.copy()
    else:
        xx, yy = x[:-lag], y[lag:]
        mm = np.ones(len(yy), dtype=bool) if mask is None else mask[lag:].copy()
    return pearson(xx[mm], yy[mm])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--svf-max-distance-m", type=float, default=10000.0)
    ap.add_argument("--svf-azimuths", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    required = [SURFACE, TERRAIN]
    required += [ERA5_DIR / f"ERA5LAND_20190924_{h:02d}00_UTC.tif" for h in range(24)]
    required += [ANCFDS_DIR / f"ANCFDS_FY4A_20190924_{h:02d}00_C.tif" for h in range(24)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing cached MCP inputs:\n" + "\n".join(missing))

    ndvi, p100 = read_band(SURFACE, SURFACE_NDVI_BAND)
    dem, pdem = read_band(TERRAIN, TERRAIN_DEM_BAND)
    if (pdem["height"], pdem["width"]) != (p100["height"], p100["width"]):
        dem = align(dem, pdem, p100)

    eps100 = ndvi_emissivity(ndvi)
    svf100 = sky_view_factor(
        dem, p100,
        max_distance_m=args.svf_max_distance_m,
        azimuth_count=args.svf_azimuths,
    )

    rows = []
    spatial_rows = []
    for h in range(24):
        era_path = ERA5_DIR / f"ERA5LAND_20190924_{h:02d}00_UTC.tif"
        lst_path = ANCFDS_DIR / f"ANCFDS_FY4A_20190924_{h:02d}00_C.tif"
        sw, pe = read_band(era_path, ERA5_SW_BAND)
        lw, _ = read_band(era_path, ERA5_LW_BAND)
        lst, pl = read_band(lst_path, ANCFDS_TNADIR_BAND)

        sw100 = align(sw, pe, p100)
        lw100 = align(lw, pe, p100)
        lw_abs100 = eps100 * lw100
        # Terrain-aware sky component proxy: physically conservative naming.
        lw_sky_svf100 = eps100 * svf100 * lw100

        lw1 = align(lw_abs100, p100, pl, Resampling.average)
        lwsvf1 = align(lw_sky_svf100, p100, pl, Resampling.average)
        sw1 = align(sw100, p100, pl, Resampling.average)

        rows.append({
            "hour_utc": h,
            "hour_bjt": (h + 8) % 24,
            "lst_tnadir_mean_c": float(np.nanmean(lst)),
            "swdown_mean_wm2": float(np.nanmean(sw)),
            "glw_mean_wm2": float(np.nanmean(lw)),
            "emissivity_mean": float(np.nanmean(eps100)),
            "svf_mean": float(np.nanmean(svf100)),
            "lw_abs_mean_wm2": float(np.nanmean(lw_abs100)),
            "lw_abs_svf_sky_mean_wm2": float(np.nanmean(lw_sky_svf100)),
        })
        spatial_rows.append({
            "hour_utc": h,
            "hour_bjt": (h + 8) % 24,
            "r_spatial_swdown_vs_lst": pearson(sw1.ravel(), lst.ravel()),
            "r_spatial_lw_abs_vs_lst": pearson(lw1.ravel(), lst.ravel()),
            "r_spatial_lw_abs_svf_sky_vs_lst": pearson(lwsvf1.ravel(), lst.ravel()),
        })

    df = pd.DataFrame(rows)
    sdf = pd.DataFrame(spatial_rows)
    sw = df["swdown_mean_wm2"].to_numpy(float)
    lw = df["glw_mean_wm2"].to_numpy(float)
    lwa = df["lw_abs_mean_wm2"].to_numpy(float)
    lwsvf = df["lw_abs_svf_sky_mean_wm2"].to_numpy(float)
    lst = df["lst_tnadir_mean_c"].to_numpy(float)

    # Data-defined night: no meaningful incoming shortwave.
    night = sw <= 1.0
    day = ~night

    summary = {
        "date": DATE,
        "target_lst": "ANCFDS T_nadir 1 km, native ROI mean",
        "radiation_source": "ERA5-Land hourly",
        "night_definition": "ROI mean SWDOWN_WM2 <= 1.0",
        "night_hours_utc": df.loc[night, "hour_utc"].astype(int).tolist(),
        "day_hours_utc": df.loc[day, "hour_utc"].astype(int).tolist(),
        "svf": {
            "method": "16-direction DEM horizon proxy unless CLI overrides",
            "max_distance_m": args.svf_max_distance_m,
            "azimuth_count": args.svf_azimuths,
            "mean": float(np.nanmean(svf100)),
            "p05": float(np.nanpercentile(svf100, 5)),
            "p95": float(np.nanpercentile(svf100, 95)),
            "important_note": "SVF-modulated sky longwave only; not full terrain longwave balance because surrounding-slope temperature is not assumed.",
        },
        "temporal_pearson_r": {},
        "lagged_temporal_r": {},
        "spatial_r_mean": {
            "all_swdown": float(sdf["r_spatial_swdown_vs_lst"].mean()),
            "all_lw_abs": float(sdf["r_spatial_lw_abs_vs_lst"].mean()),
            "all_lw_abs_svf_sky": float(sdf["r_spatial_lw_abs_svf_sky_vs_lst"].mean()),
            "day_lw_abs": float(sdf.loc[day, "r_spatial_lw_abs_vs_lst"].mean()),
            "night_lw_abs": float(sdf.loc[night, "r_spatial_lw_abs_vs_lst"].mean()),
            "night_lw_abs_svf_sky": float(sdf.loc[night, "r_spatial_lw_abs_svf_sky_vs_lst"].mean()),
        },
    }

    for name, x in {
        "SWDOWN": sw,
        "GLW": lw,
        "LW_abs": lwa,
        "LW_abs_SVF_sky": lwsvf,
    }.items():
        summary["temporal_pearson_r"][name] = {
            "all24": pearson(x, lst),
            "day": pearson(x[day], lst[day]),
            "night": pearson(x[night], lst[night]),
        }
        summary["lagged_temporal_r"][name] = {
            f"lag_{lag}h": lagged_r(x, lst, lag)
            for lag in range(0, 7)
        }

    df.to_csv(out / "hourly_curve.csv", index=False, encoding="utf-8-sig")
    sdf.to_csv(out / "spatial_correlations.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Normalized curves: compare shape/phase without unit-scale dominance.
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    def z(v):
        v = np.asarray(v, dtype=float)
        return (v - np.nanmean(v)) / np.nanstd(v)
    x = df["hour_bjt"].to_numpy(int)
    order = np.argsort(x)
    ax.plot(x[order], z(lst)[order], marker="o", label="ANCFDS T_nadir LST")
    ax.plot(x[order], z(sw)[order], marker=".", label="SWDOWN")
    ax.plot(x[order], z(lwa)[order], marker=".", label="LW_abs = emissivity * GLW")
    ax.plot(x[order], z(lwsvf)[order], marker=".", label="LW_abs_SVF_sky")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Beijing time (UTC+8)")
    ax.set_ylabel("standardized value (z-score)")
    ax.set_title("2019-09-24 diurnal curves: radiation vs ANCFDS T_nadir")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out / "diurnal_curve_radiation_vs_lst.png", dpi=220, facecolor="white")
    plt.close(fig)

    # Correlation bars split by all/day/night.
    c = summary["temporal_pearson_r"]
    names = list(c)
    pos = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.bar(pos - width, [c[n]["all24"] for n in names], width, label="24h")
    ax.bar(pos, [c[n]["day"] for n in names], width, label="day")
    ax.bar(pos + width, [c[n]["night"] for n in names], width, label="night")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1, 1)
    ax.set_title("Temporal correlation with ANCFDS T_nadir")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "temporal_correlation_day_night.png", dpi=220, facecolor="white")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
