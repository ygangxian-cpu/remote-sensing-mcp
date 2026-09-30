from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_frozen_v3_multidate_validation as md


def profile_json(profile: dict) -> dict:
    t = profile["transform"]
    return {
        "crs": str(profile["crs"]),
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "width": int(profile["width"]),
        "height": int(profile["height"]),
    }


def main() -> None:
    date = "2019-09-24"
    product = "LC08_L2SP_133033_20190924_20200826_02_T1"
    md.SELECTED_PRODUCTS[date] = product

    md.init_ee()

    out = Path("output/ilc_v1_parity_reference_data")
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    print("=== BUILD FRESH 2019-09-24 FORMAL SNAPSHOT ===", flush=True)
    scene, landsat_meta = md.landsat_scene(date)
    static = md.build_static_features(date, scene, work)
    dynamic = md.build_dynamic_features(date, static, work)
    features = md.merge_features(static, dynamic)
    y4, elite_meta = md.elite_target(date)
    ref100 = md.landsat_reference(date, scene, work)

    arrays = {
        "target__elite_lst_c": y4.astype("float32"),
        "reference__landsat_lst_c": ref100.astype("float32"),
    }
    feature_names = {}
    for res in ["4km", "1km", "100m"]:
        names = sorted(features[res].keys())
        feature_names[res] = names
        for name in names:
            arrays[f"feature__{res}__{name}"] = np.asarray(
                features[res][name], dtype="float32"
            )

    np.savez_compressed(out / "ilc_v1_parity_reference_2019-09-24.npz", **arrays)

    manifest = {
        "date": date,
        "landsat_product_id": product,
        "landsat": landsat_meta,
        "elite": elite_meta,
        "profiles": {
            res: profile_json(md.PROFILES[res])
            for res in ["4km", "1km", "100m"]
        },
        "feature_names": feature_names,
        "role": (
            "Fresh Data Plane reference snapshot generated with the same "
            "feature-construction code used by the original parent/local pilot."
        ),
        "no_method_metrics_computed": True,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float),
        encoding="utf-8",
    )

    print("PARITY_REFERENCE_DATA_READY", flush=True)


if __name__ == "__main__":
    main()
