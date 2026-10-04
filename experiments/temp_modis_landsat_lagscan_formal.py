from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODIS_DIR = ROOT / "data" / "modis_lst" / "v2" / "zhangye-2e599e4070"
LANDSAT = (
    ROOT / "data" / "landsat_c2_l2" / "v3" / "zhangye-2e599e4070"
    / "2019" / "09" / "24"
    / "LC08_L2SP_133033_20190924_20200826_02_T1_L2_LST_QA.tif"
)
OUT = ROOT / "experiments" / "results" / "formal_modis_landsat_lagscan"

TARGET_DATE = date(2019, 9, 24)
ROI = [99.86, 38.67, 100.50, 39.43]
CENTER_LON = (ROI[0] + ROI[2]) / 2.0
LANDSAT_CLEAR_FRACTION_MIN = 0.80
FILE_RE = re.compile(r"^(MOD11A1|MYD11A1)_(\d{8})_LST_QA\.tif$")
PLATFORM = {"MOD11A1": "terra", "MYD11A1": "aqua"}


def profile(src) -> dict:
    return {
        "crs": src.crs,
        "transform": src.transform,
        "width": src.width,
        "height": src.height,
        "nodata": src.nodata,
    }


def reproject_array(
    source: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    *,
    resampling: Resampling,
    src_nodata: float | None = np.nan,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    dst = np.full(
        (dst_profile["height"], dst_profile["width"]),
        dst_nodata,
        dtype="float32",
    )
    reproject(
        source=source.astype("float32"),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=src_nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def read_landsat() -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(LANDSAT) as src:
        data = src.read(masked=True).astype("float32")
        p = profile(src)
    lst = data[0]
    qa = data[2]
    lst_data = lst.filled(np.nan)
    qa_data = qa.filled(np.nan)

    valid = np.isfinite(lst_data) & np.isfinite(qa_data)
    q = np.where(np.isfinite(qa_data), qa_data, 0).astype("int32")
    for bit in [0, 1, 2, 3, 4, 5]:
        valid &= (q & (1 << bit)) == 0
    return lst_data, valid, p


def read_modis(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(masked=True).astype("float32")
        p = profile(src)
    lst = data[0].filled(np.nan)
    view = data[2].filled(np.nan)
    qc = data[4].filled(np.nan)

    valid = np.isfinite(lst) & np.isfinite(qc)
    q = np.where(np.isfinite(qc), qc, 0).astype("int32")
    valid &= (q & 0b11) <= 1
    valid &= ((q >> 2) & 0b11) == 0
    valid &= ((q >> 6) & 0b11) <= 2
    return lst, view, valid, p


def aggregate_landsat(
    lst: np.ndarray,
    clear: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    src_lst = np.where(clear, lst, -9999.0).astype("float32")
    avg_lst = reproject_array(
        src_lst,
        src_profile,
        dst_profile,
        resampling=Resampling.average,
        src_nodata=-9999.0,
        dst_nodata=np.nan,
    )
    clear_fraction = reproject_array(
        clear.astype("float32"),
        src_profile,
        dst_profile,
        resampling=Resampling.average,
        src_nodata=None,
        dst_nodata=np.nan,
    )
    avg_lst[clear_fraction < LANDSAT_CLEAR_FRACTION_MIN] = np.nan
    return avg_lst, clear_fraction


def modis_to_grid(
    lst: np.ndarray,
    view: np.ndarray,
    valid: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    src_lst = np.where(valid, lst, -9999.0).astype("float32")
    src_view = np.where(valid & np.isfinite(view), view, -9999.0).astype("float32")
    out_lst = reproject_array(
        src_lst,
        src_profile,
        dst_profile,
        resampling=Resampling.nearest,
        src_nodata=-9999.0,
        dst_nodata=np.nan,
    )
    out_view = reproject_array(
        src_view,
        src_profile,
        dst_profile,
        resampling=Resampling.nearest,
        src_nodata=-9999.0,
        dst_nodata=np.nan,
    )
    return out_lst, out_view


def metrics(ref: np.ndarray, pred: np.ndarray, extra_mask: np.ndarray | None = None) -> dict:
    valid = np.isfinite(ref) & np.isfinite(pred)
    if extra_mask is not None:
        valid &= extra_mask
    y = ref[valid].astype("float64")
    x = pred[valid].astype("float64")
    n = int(y.size)
    out = {
        "n": n,
        "landsat_mean_c": float(y.mean()) if n else None,
        "modis_mean_c": float(x.mean()) if n else None,
    }
    if n < 3 or np.std(y) == 0 or np.std(x) == 0:
        out.update({
            "pearson_r": None,
            "pearson_r2": None,
            "spearman_rho": None,
            "identity_r2": None,
            "rmse_c": None,
            "mae_c": None,
            "bias_modis_minus_landsat_c": None,
            "slope_modis_from_landsat": None,
            "intercept_modis_from_landsat": None,
        })
        return out

    r = float(np.corrcoef(y, x)[0, 1])
    rho = float(spearmanr(y, x).statistic)
    residual = x - y
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum(residual ** 2))
    slope, intercept = np.polyfit(y, x, 1)
    out.update({
        "pearson_r": r,
        "pearson_r2": r * r,
        "spearman_rho": rho,
        "identity_r2": float(1.0 - sse / sst) if sst else None,
        "rmse_c": float(math.sqrt(np.mean(residual ** 2))),
        "mae_c": float(np.mean(np.abs(residual))),
        "bias_modis_minus_landsat_c": float(np.mean(residual)),
        "slope_modis_from_landsat": float(slope),
        "intercept_modis_from_landsat": float(intercept),
    })
    return out


def collect_paths(platform: str) -> dict[date, Path]:
    prefix = "MOD11A1" if platform == "terra" else "MYD11A1"
    paths: dict[date, Path] = {}
    for path in sorted(MODIS_DIR.glob("2019/09/*/*.tif")):
        m = FILE_RE.match(path.name)
        if not m or m.group(1) != prefix:
            continue
        obs = datetime.strptime(m.group(2), "%Y%m%d").date()
        if date(2019, 9, 15) <= obs <= date(2019, 9, 30):
            paths[obs] = path
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ls_lst, ls_clear, ls_profile = read_landsat()

    pair_rows = []
    primary_rows = []
    primary_summary = {}
    best_by_platform = {}
    grid_summary = {}

    fig_scatter, axes = plt.subplots(2, 2, figsize=(11, 10))

    for row_idx, platform in enumerate(["terra", "aqua"]):
        paths = collect_paths(platform)
        if TARGET_DATE not in paths:
            raise RuntimeError(f"Missing D0 MODIS for {platform}")

        _, _, _, canonical_profile = read_modis(paths[TARGET_DATE])
        landsat_1km, clear_fraction = aggregate_landsat(
            ls_lst, ls_clear, ls_profile, canonical_profile
        )
        ls_valid = np.isfinite(landsat_1km)
        grid_summary[platform] = {
            "grid_width": canonical_profile["width"],
            "grid_height": canonical_profile["height"],
            "grid_crs": str(canonical_profile["crs"]),
            "landsat_cells_meeting_clear_fraction": int(ls_valid.sum()),
            "landsat_grid_cells_total": int(ls_valid.size),
            "landsat_clear_fraction_min": LANDSAT_CLEAR_FRACTION_MIN,
            "landsat_mean_clear_fraction_over_retained": (
                float(np.nanmean(clear_fraction[ls_valid])) if ls_valid.any() else None
            ),
        }

        arrays: dict[date, np.ndarray] = {}
        views: dict[date, np.ndarray] = {}
        for obs_date, path in sorted(paths.items()):
            lst, view, strict, src_profile = read_modis(path)
            arr, view_arr = modis_to_grid(
                lst, view, strict, src_profile, canonical_profile
            )
            arrays[obs_date] = arr
            views[obs_date] = view_arr
            m = metrics(landsat_1km, arr)
            valid = np.isfinite(landsat_1km) & np.isfinite(arr)
            vt = view_arr[valid & np.isfinite(view_arr)]
            view_local_mean = float(vt.mean()) if vt.size else None
            view_utc_approx = (
                float(np.mod(view_local_mean - CENTER_LON / 15.0, 24.0))
                if view_local_mean is not None else None
            )
            pair_rows.append({
                "platform": platform,
                "modis_date": obs_date.isoformat(),
                "landsat_date": TARGET_DATE.isoformat(),
                "lag_days": (obs_date - TARGET_DATE).days,
                "mask_scope": "pairwise_strict_modis_and_landsat_clear80",
                "landsat_retained_cells": int(ls_valid.sum()),
                "pair_valid_fraction_of_landsat_retained": (
                    float(m["n"] / ls_valid.sum()) if ls_valid.sum() else None
                ),
                "view_time_local_mean_h": view_local_mean,
                "view_time_utc_approx_mean_h": view_utc_approx,
                **m,
            })

        dminus1 = TARGET_DATE - timedelta(days=1)
        if dminus1 not in arrays:
            raise RuntimeError(f"Missing D-1 MODIS for {platform}")
        common = (
            np.isfinite(landsat_1km)
            & np.isfinite(arrays[dminus1])
            & np.isfinite(arrays[TARGET_DATE])
        )
        m_prev = metrics(landsat_1km, arrays[dminus1], common)
        m_same = metrics(landsat_1km, arrays[TARGET_DATE], common)

        for lag, obs_date, m in [
            (-1, dminus1, m_prev),
            (0, TARGET_DATE, m_same),
        ]:
            primary_rows.append({
                "platform": platform,
                "modis_date": obs_date.isoformat(),
                "landsat_date": TARGET_DATE.isoformat(),
                "lag_days": lag,
                "mask_scope": "common_Dminus1_D0_strict_modis_landsat_clear80",
                "common_valid_pixels": int(common.sum()),
                **m,
            })

        primary_summary[platform] = {
            "common_valid_pixels": int(common.sum()),
            "pearson_r_Dminus1": m_prev["pearson_r"],
            "pearson_r_D0": m_same["pearson_r"],
            "delta_pearson_r_Dminus1_minus_D0": (
                m_prev["pearson_r"] - m_same["pearson_r"]
            ),
            "spearman_rho_Dminus1": m_prev["spearman_rho"],
            "spearman_rho_D0": m_same["spearman_rho"],
            "delta_spearman_Dminus1_minus_D0": (
                m_prev["spearman_rho"] - m_same["spearman_rho"]
            ),
            "rmse_Dminus1_c": m_prev["rmse_c"],
            "rmse_D0_c": m_same["rmse_c"],
            "bias_Dminus1_c": m_prev["bias_modis_minus_landsat_c"],
            "bias_D0_c": m_same["bias_modis_minus_landsat_c"],
            "Dminus1_has_higher_pearson": bool(
                m_prev["pearson_r"] > m_same["pearson_r"]
            ),
        }

        for col_idx, (lag, obs_date, m) in enumerate([
            (-1, dminus1, m_prev),
            (0, TARGET_DATE, m_same),
        ]):
            ax = axes[row_idx, col_idx]
            x = landsat_1km[common]
            y = arrays[obs_date][common]
            ax.scatter(x, y, s=12, alpha=0.55)
            lo = float(min(np.nanpercentile(x, 1), np.nanpercentile(y, 1)))
            hi = float(max(np.nanpercentile(x, 99), np.nanpercentile(y, 99)))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            ax.set_title(
                f"{platform.title()} lag {lag}: r={m['pearson_r']:.3f}, n={m['n']}"
            )
            ax.set_xlabel("Landsat clear-aggregated LST (°C)")
            ax.set_ylabel("MODIS strict-QC Day LST (°C)")

    pair_df = pd.DataFrame(pair_rows).sort_values(["platform", "lag_days"])
    primary_df = pd.DataFrame(primary_rows)
    pair_df.to_csv(OUT / "lag_metrics_pairwise.csv", index=False)
    primary_df.to_csv(OUT / "dminus1_vs_d0_common_mask.csv", index=False)

    ranking = []
    for platform, g in pair_df.groupby("platform"):
        gg = g[(g["n"] >= 3) & g["pearson_r"].notna()].sort_values(
            ["pearson_r", "n"], ascending=[False, False]
        )
        for rank, (_, row) in enumerate(gg.iterrows(), start=1):
            ranking.append({
                "platform": platform,
                "rank_by_pearson": rank,
                "modis_date": row["modis_date"],
                "lag_days": int(row["lag_days"]),
                "pearson_r": float(row["pearson_r"]),
                "pearson_r2": float(row["pearson_r2"]),
                "spearman_rho": float(row["spearman_rho"]),
                "n": int(row["n"]),
                "pair_valid_fraction_of_landsat_retained": float(
                    row["pair_valid_fraction_of_landsat_retained"]
                ),
            })
        if len(gg):
            top = gg.iloc[0]
            best_by_platform[platform] = {
                "modis_date": top["modis_date"],
                "lag_days": int(top["lag_days"]),
                "pearson_r": float(top["pearson_r"]),
                "n": int(top["n"]),
            }
    pd.DataFrame(ranking).to_csv(OUT / "lag_ranking.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for platform, g in pair_df.groupby("platform"):
        g = g.sort_values("lag_days")
        ax.plot(g["lag_days"], g["pearson_r"], marker="o", label=platform.title())
    ax.axvline(-1, linestyle="--", linewidth=1, label="D-1")
    ax.axvline(0, linestyle=":", linewidth=1, label="D0")
    ax.set_xlabel("MODIS date lag relative to Landsat 2019-09-24 (days)")
    ax.set_ylabel("Spatial Pearson r")
    ax.set_title("Formal ROI: MODIS Day LST vs Landsat LST lag scan")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_lag_scan_pearson.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig_scatter.suptitle("Formal ROI D-1 vs D0 on identical common valid pixels")
    fig_scatter.tight_layout()
    fig_scatter.savefig(
        OUT / "02_dminus1_vs_d0_scatter.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig_scatter)

    summary = {
        "experiment": "formal_modis_lag1_vs_landsat_correlation",
        "target_landsat_date": TARGET_DATE.isoformat(),
        "roi": ROI,
        "modis_products": ["MOD11A1.061", "MYD11A1.061"],
        "modis_cache_version": "v2",
        "modis_qc": "bits0-1<=1; bits2-3==0; bits6-7<=2",
        "landsat_product": LANDSAT.name,
        "landsat_clear_mask": "QA_PIXEL bits 0,1,2,3,4,5 == 0; water retained",
        "landsat_1km_aggregation": (
            "Average clear 30 m Landsat LST into each platform's D0 MODIS 1 km grid; "
            f"retain cells with clear fraction >= {LANDSAT_CLEAR_FRACTION_MIN:.2f}."
        ),
        "primary_test": (
            "D-1 vs D0 compared on identical common valid cells for each platform. "
            "Primary statistic is spatial Pearson r; Spearman rho is secondary."
        ),
        "grid_summary": grid_summary,
        "primary_Dminus1_vs_D0": primary_summary,
        "best_pairwise_lag_by_platform": best_by_platform,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 正式 ROI：MODIS 前一日 vs 当日 Landsat 相关性实验",
        "",
        f"- ROI：{ROI}",
        f"- Landsat 日期：{TARGET_DATE.isoformat()}",
        "- MODIS：v2 QA-preserving cache，分析阶段应用严格 QC。",
        "- Landsat：v3 原始 LST+QA，30 m 晴空像元聚合至 MODIS 1 km；1 km 晴空覆盖率 >=80%。",
        "- D-1 与 D0 使用完全相同的共同有效像元。",
        "",
        "## D-1 vs D0",
        "",
    ]
    for platform, s in primary_summary.items():
        lines.append(
            f"- {platform}: r(D-1)={s['pearson_r_Dminus1']:.4f}, "
            f"r(D0)={s['pearson_r_D0']:.4f}, "
            f"Δr={s['delta_pearson_r_Dminus1_minus_D0']:+.4f}, "
            f"n={s['common_valid_pixels']}, "
            f"D-1更高={s['Dminus1_has_higher_pearson']}"
        )
    lines.extend(["", "## 全日期最高 Pearson r", ""])
    for platform, s in best_by_platform.items():
        lines.append(
            f"- {platform}: {s['modis_date']} (lag={s['lag_days']:+d}), "
            f"r={s['pearson_r']:.4f}, n={s['n']}"
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(primary_summary, indent=2, ensure_ascii=False))
    print(json.dumps(best_by_platform, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
