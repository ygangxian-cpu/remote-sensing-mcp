from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.enums import Resampling

import mcp_rf_baseline as b

OUT = Path("output/mcp_downscaling_diagnostics")


def stats(arr: np.ndarray) -> dict:
    x = arr[np.isfinite(arr)].astype("float64")
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "range_p05_p95": float(np.percentile(x, 95) - np.percentile(x, 5)),
    }


def pearson(a: np.ndarray, c: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(c)
    if valid.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[valid].reshape(-1), c[valid].reshape(-1))[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    y4, p4 = b.crop_elite()
    features100, p100 = b.build_features_100m()
    p1 = b.grid_1km(p100)
    f4 = b.aggregate_features(features100, p100, p4)
    f1 = b.aggregate_features(features100, p100, p1)
    landsat100, ref_valid = b.landsat_reference(p100)

    # Sensor consistency upper bound: aggregate Landsat to the exact ELITE grid.
    landsat4 = b.reproject_array(landsat100, p100, p4, Resampling.average)
    valid_sensor = np.isfinite(y4) & np.isfinite(landsat4)

    sensor_metrics = b.metrics(landsat4[valid_sensor], y4[valid_sensor])
    sensor_pearson = pearson(landsat4, y4)

    base14 = [
        "dem", "slope", "aspect",
        "ndvi", "ndbi", "mndwi", "savi", "bsi", "ui",
        "albedo_bsa", "landcover", "lat_m", "lon_m", "edrf",
    ]
    mcp20 = base14 + [
        "t2_c", "td2_c", "wind_speed", "psfc_pa", "swdown_wm2", "glw_wm2",
    ]

    rows = []
    prediction_stats = {}
    coarse_consistency = {}

    for label, names in [("base14", base14), ("mcp20", mcp20)]:
        result_rows, preds, diag = b.run_feature_set(
            label, names, features100, p100, f1, p1, f4, p4, y4, landsat100
        )
        rows.extend(result_rows)
        prediction_stats[label] = {}
        coarse_consistency[label] = {}
        for method, pred in preds.items():
            prediction_stats[label][method] = {
                "prediction": stats(pred),
                "landsat": stats(landsat100),
                "pearson_with_landsat": pearson(pred, landsat100),
                "std_ratio_pred_to_landsat": stats(pred)["std"] / stats(landsat100)["std"],
                "p05_p95_ratio_pred_to_landsat": (
                    stats(pred)["range_p05_p95"] / stats(landsat100)["range_p05_p95"]
                ),
            }
            pred4 = b.reproject_array(pred, p100, p4, Resampling.average)
            valid = np.isfinite(pred4) & np.isfinite(y4)
            coarse_consistency[label][method] = {
                "metrics_vs_elite_4km": b.metrics(y4[valid], pred4[valid]),
                "pearson_vs_elite_4km": pearson(y4, pred4),
            }

    # Feature association on the coarse training grid.
    feature_rows = []
    y4f = y4.reshape(-1)
    l4f = landsat4.reshape(-1)
    for name, arr in f4.items():
        x = arr.reshape(-1)
        feature_rows.append({
            "feature": name,
            "pearson_with_elite_4km": pearson(x, y4f),
            "pearson_with_landsat_4km": pearson(x, l4f),
            "std_4km": stats(x)["std"],
        })
    feature_rows.sort(
        key=lambda r: abs(r["pearson_with_elite_4km"])
        if np.isfinite(r["pearson_with_elite_4km"]) else -1,
        reverse=True,
    )

    report = {
        "sensor_consistency_4km": {
            "n_pixels": int(valid_sensor.sum()),
            "elite_stats": stats(y4),
            "landsat_aggregated_stats": stats(landsat4),
            "metrics_elite_vs_landsat4": sensor_metrics,
            "pearson_r": sensor_pearson,
            "pearson_r2": sensor_pearson ** 2 if np.isfinite(sensor_pearson) else None,
        },
        "fine_scale_reference": {
            "landsat_clear_pixels": ref_valid,
            "landsat_100m_stats": stats(landsat100),
        },
        "prediction_variance": prediction_stats,
        "reaggregation_consistency": coarse_consistency,
        "feature_correlations_4km": feature_rows,
        "interpretation_keys": {
            "sensor_consistency": (
                "If ELITE 4 km and Landsat aggregated to 4 km correlate poorly, the cross-sensor "
                "target/reference mismatch is already present before downscaling."
            ),
            "variance_ratio": (
                "A prediction/Landsat standard-deviation ratio well below 1 means the model is "
                "too smooth and cannot reproduce fine-scale thermal contrast."
            ),
            "reaggregation": (
                "High reaggregation consistency with ELITE plus low Landsat agreement means the "
                "algorithm conserves the coarse target but the target does not contain the fine "
                "thermal structure needed for independent Landsat validation."
            ),
        },
    }

    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(feature_rows).to_csv(OUT / "feature_correlations.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "validation_metrics.csv", index=False)

    print("=== SENSOR CONSISTENCY 4KM ===")
    print(json.dumps(report["sensor_consistency_4km"], indent=2))
    print("=== PREDICTION VARIANCE ===")
    print(json.dumps(report["prediction_variance"], indent=2))
    print("=== REAGGREGATION CONSISTENCY ===")
    print(json.dumps(report["reaggregation_consistency"], indent=2))
    print("=== TOP FEATURE CORRELATIONS ===")
    print(pd.DataFrame(feature_rows).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
