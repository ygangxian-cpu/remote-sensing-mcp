from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for p in [REPO_ROOT, EXPERIMENTS]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_frozen_v3_multidate_validation as md

HOLDOUT_PRODUCTS = {
    "2019-07-06": "LC08_L2SP_133033_20190706_20200827_02_T1",
    "2019-08-14": "LC08_L2SP_134033_20190814_20200827_02_T1",
    "2019-10-10": "LC08_L2SP_133033_20191010_20200825_02_T1",
}

RESOLUTIONS = ["100m", "1km", "4km"]


def profile_json(profile: dict) -> dict:
    t = profile["transform"]
    return {
        "crs": str(profile["crs"]),
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "width": int(profile["width"]),
        "height": int(profile["height"]),
    }


def build_one(date: str, product_id: str, out: Path, work: Path) -> dict:
    print(f"=== DATA PLANE SNAPSHOT {date} ===", flush=True)

    # Fix the pre-reserved Landsat product without using any validation metric.
    md.SELECTED_PRODUCTS[date] = product_id

    scene, landsat_meta = md.landsat_scene(date)
    static = md.build_static_features(date, scene, work)
    dynamic = md.build_dynamic_features(date, static, work)
    features = md.merge_features(static, dynamic)
    y4, elite_meta = md.elite_target(date)
    ref100 = md.landsat_reference(date, scene, work)

    arrays: dict[str, np.ndarray] = {
        "target__elite_lst_c": y4.astype("float32"),
        "reference__landsat_lst_c": ref100.astype("float32"),
    }
    feature_names = {}
    for res in RESOLUTIONS:
        names = sorted(features[res].keys())
        feature_names[res] = names
        for name in names:
            arrays[f"feature__{res}__{name}"] = np.asarray(
                features[res][name], dtype="float32"
            )

    npz_path = out / f"ilc_v1_holdout_{date}.npz"
    np.savez_compressed(npz_path, **arrays)

    manifest = {
        "date": date,
        "formal_hour_utc": md.HOUR,
        "formal_roi": md.ROI,
        "landsat_product_id": product_id,
        "landsat": landsat_meta,
        "elite": elite_meta,
        "profiles": {
            res: profile_json(md.PROFILES[res])
            for res in RESOLUTIONS
        },
        "feature_names": feature_names,
        "arrays": sorted(arrays.keys()),
        "role": (
            "Data Plane standardized holdout snapshot. "
            "No downscaling method is fitted or evaluated here."
        ),
        "holdout_guard": (
            "The date was reserved before ILC-v1 was frozen. "
            "This builder prepares inputs only and emits no performance metric."
        ),
    }
    manifest_path = out / f"ilc_v1_holdout_{date}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=float),
        encoding="utf-8",
    )

    print(
        f"{date}: wrote {npz_path.name} "
        f"({npz_path.stat().st_size / 1024 / 1024:.1f} MiB)",
        flush=True,
    )
    return {
        "date": date,
        "product_id": product_id,
        "npz": npz_path.name,
        "manifest": manifest_path.name,
        "npz_size_bytes": int(npz_path.stat().st_size),
    }


def main() -> None:
    md.init_ee()
    out = Path("output/ilc_v1_holdout_data")
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    rows = [
        build_one(date, product, out, work)
        for date, product in HOLDOUT_PRODUCTS.items()
    ]
    (out / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "status": "data_only",
                "method": "ILC-v1",
                "reserved_holdouts": list(HOLDOUT_PRODUCTS),
                "snapshots": rows,
                "source_builder": "remote-sensing-mcp",
                "no_metrics_computed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== HOLDOUT DATA SNAPSHOTS READY ===", flush=True)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
