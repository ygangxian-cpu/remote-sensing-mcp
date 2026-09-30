from __future__ import annotations

import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from remotezip import RemoteZip
from rasterio.enums import Resampling

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for p in [REPO_ROOT, EXPERIMENTS]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import worker as elite_worker
import run_frozen_v3_multidate_validation as md

DATES = md.DATES + [md.CONTROL_DATE]
PLATFORMS = {
    "terra": "MODIS/061/MOD11A1",
    "aqua": "MODIS/061/MYD11A1",
}
ELITE_HOURS = [4, 5, 6, 7, 8]
MIN_4KM_SUPPORT = 0.50


def strict_modis_mask(lst, qc):
    q = np.asarray(qc, dtype="int32")
    mandatory = q & 3
    data_quality = (q >> 2) & 3
    lst_error = (q >> 6) & 3
    return (
        np.isfinite(lst)
        & np.isfinite(qc)
        & (mandatory <= 1)
        & (data_quality == 0)
        & (lst_error <= 2)
    )


def basic_metrics(ref, pred):
    valid = np.isfinite(ref) & np.isfinite(pred)
    y = ref[valid].astype("float64")
    p = pred[valid].astype("float64")
    if y.size < 2:
        return {
            "n": int(y.size), "r2": float("nan"), "pearson": float("nan"),
            "rmse": float("nan"), "mae": float("nan"), "bias": float("nan"),
            "ref_mean": float("nan"), "pred_mean": float("nan"),
        }
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": int(y.size),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson": float(np.corrcoef(y, p)[0, 1]),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "bias": float(np.mean(p - y)),
        "ref_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
    }


def modis_image(date, platform):
    d0 = ee.Date(date)
    col = (
        ee.ImageCollection(PLATFORMS[platform])
        .filterDate(d0, d0.advance(1, "day"))
        .filterBounds(ee.Geometry.Rectangle(md.ROI, proj="EPSG:4326", geodesic=False))
    )
    if int(col.size().getInfo()) < 1:
        raise RuntimeError(f"No {platform} MODIS on {date}")
    src = ee.Image(col.first())
    return ee.Image.cat([
        src.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("lst_day_c"),
        src.select("Day_view_time").multiply(0.1).rename("view_time_local_h"),
        src.select("QC_Day").rename("qc_day").toFloat(),
    ]).toFloat()


def read_exact_modis(date, platform, work):
    path = work / f"{platform}_{date}_day.tif"
    md.download_exact_grid(
        modis_image(date, platform),
        path,
        md.PROFILES["1km"],
        unmask_value=-9999.0,
    )
    raw, _ = md.read_multiband(path, ["lst_day_c", "view_time_local_h", "qc_day"])
    return raw


def elite_month_url(date):
    dt = datetime.fromisoformat(date)
    return md.zenodo_month_url(dt.strftime("%Y%m"))


def load_elite_hours(date):
    dt = datetime.fromisoformat(date)
    doy = dt.strftime("%j")
    url = elite_month_url(date)
    result = {}
    meta = {}

    with RemoteZip(url) as rz:
        names = rz.namelist()
        for hour in ELITE_HOURS:
            token = f"{dt:%Y}{doy}{hour:02d}00"
            candidates = [
                n for n in names
                if token in Path(n).name
                and Path(n).suffix.lower() in {".hdf", ".h5", ".hdf5", ".he5"}
            ]
            if not candidates:
                raise RuntimeError(f"ELITE source missing {date} {hour:02d}:00")
            member = candidates[0]
            raw = rz.read(member)
            with tempfile.NamedTemporaryFile(
                suffix=Path(member).suffix or ".hdf", delete=False
            ) as tmp:
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
            result[hour] = md.fill_continuous(
                md.reproject_array(
                    values_c,
                    src_profile,
                    md.PROFILES["1km"],
                    Resampling.bilinear,
                )
            )
            meta[hour] = {"member": member, "dataset": ds_name}
    return result, meta


def lon_grid(profile):
    rows, cols = np.indices((profile["height"], profile["width"]))
    xs, ys = rasterio.transform.xy(profile["transform"], rows, cols, offset="center")
    return np.asarray(xs, dtype="float64").reshape(rows.shape)


def interpolate_elite(hours, utc_time):
    out = np.full(utc_time.shape, np.nan, dtype="float32")
    nearest = np.full(utc_time.shape, np.nan, dtype="float32")
    for h0 in ELITE_HOURS[:-1]:
        m = (utc_time >= h0) & (utc_time < h0 + 1)
        if not np.any(m):
            continue
        w = utc_time[m] - h0
        out[m] = (
            hours[h0][m] * (1.0 - w)
            + hours[h0 + 1][m] * w
        )
        choose1 = w >= 0.5
        vals = hours[h0][m].copy()
        vals[choose1] = hours[h0 + 1][m][choose1]
        nearest[m] = vals

    exact8 = np.isclose(utc_time, 8.0)
    if np.any(exact8):
        out[exact8] = hours[8][exact8]
        nearest[exact8] = hours[8][exact8]
    return out, nearest


def aggregate_common_support(modis, elite, valid):
    m = np.where(valid, modis, np.nan).astype("float32")
    e = np.where(valid, elite, np.nan).astype("float32")
    support = md.reproject_array(
        valid.astype("float32"),
        md.PROFILES["1km"],
        md.PROFILES["4km"],
        Resampling.average,
    )
    m4 = md.reproject_array(m, md.PROFILES["1km"], md.PROFILES["4km"], Resampling.average)
    e4 = md.reproject_array(e, md.PROFILES["1km"], md.PROFILES["4km"], Resampling.average)
    keep = np.isfinite(m4) & np.isfinite(e4) & (support >= MIN_4KM_SUPPORT)
    return np.where(keep, m4, np.nan), np.where(keep, e4, np.nan), support


def main():
    md.init_ee()
    out = Path("output/elite_modis_multidate")
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    lon = lon_grid(md.PROFILES["1km"])
    rows = []
    provenance = []

    for date in DATES:
        print(f"=== {date}: loading ELITE hours ===", flush=True)
        elite_hours, elite_meta = load_elite_hours(date)

        for platform in ["terra", "aqua"]:
            raw = read_exact_modis(date, platform, work)
            lst = raw["lst_day_c"]
            vt = raw["view_time_local_h"]
            qc = raw["qc_day"]
            valid = strict_modis_mask(lst, qc)

            utc = (vt - lon / 15.0) % 24.0
            valid &= np.isfinite(utc) & (utc >= 4.0) & (utc <= 8.0)

            exact, nearest = interpolate_elite(elite_hours, utc)
            common = valid & np.isfinite(exact)

            m1 = basic_metrics(
                np.where(common, lst, np.nan),
                np.where(common, exact, np.nan),
            )
            n1 = basic_metrics(
                np.where(common, lst, np.nan),
                np.where(common, nearest, np.nan),
            )

            mod4, elite4, support = aggregate_common_support(lst, exact, common)
            m4 = basic_metrics(mod4, elite4)

            mean_utc = float(np.nanmean(np.where(common, utc, np.nan))) if np.any(common) else float("nan")
            row = {
                "date": date,
                "platform": platform,
                "mean_view_utc_h": mean_utc,
                "strict_pixels_1km": int(np.sum(common)),
                "formal4_cells_support_ge_50pct": int(np.sum(np.isfinite(mod4))),
                **{f"exact_1km_{k}": v for k, v in m1.items()},
                **{f"nearest_1km_{k}": v for k, v in n1.items()},
                **{f"exact_4km_{k}": v for k, v in m4.items()},
            }
            rows.append(row)
            print(
                f"{date} {platform} utc={mean_utc:.3f} "
                f"n1={m1['n']} exact1(r2={m1['r2']:.3f},rmse={m1['rmse']:.3f},bias={m1['bias']:+.3f}) "
                f"n4={m4['n']} exact4(r2={m4['r2']:.3f},rmse={m4['rmse']:.3f},bias={m4['bias']:+.3f})",
                flush=True,
            )

        provenance.append({
            "date": date,
            "elite_hours_utc": ELITE_HOURS,
            "elite_sources": elite_meta,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "elite_modis_exact_time_metrics.csv", index=False)

    # Day-level pooled Terra+Aqua indication; this is descriptive only.
    pooled = []
    for date, g in df.groupby("date"):
        pooled.append({
            "date": date,
            "mean_exact4_rmse": float(g["exact_4km_rmse"].mean()),
            "mean_exact4_abs_bias": float(g["exact_4km_bias"].abs().mean()),
            "mean_exact4_pearson": float(g["exact_4km_pearson"].mean()),
            "total_strict_1km_pixels": int(g["strict_pixels_1km"].sum()),
        })
    pd.DataFrame(pooled).to_csv(out / "date_summary.csv", index=False)

    report = {
        "purpose": (
            "Independent third-sensor cross-check of ELITE coarse thermal state on the same "
            "dates used for Landsat multi-date diagnostics."
        ),
        "dates": DATES,
        "roi": md.ROI,
        "modis_qc": "bits0-1<=1; bits2-3==0; bits6-7<=2",
        "time_matching": "per-pixel local solar view time -> UTC = local - lon/15, then linear ELITE hourly interpolation",
        "four_km_support_rule": f">={MIN_4KM_SUPPORT:.0%} strict MODIS support",
        "landsat_not_used": True,
        "metrics": rows,
        "date_summary": pooled,
        "provenance": provenance,
    }
    (out / "run_summary.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print("=== ELITE-MODIS DATE SUMMARY ===", flush=True)
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
