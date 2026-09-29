from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import rasterio

import mcp_rf_baseline as b


def main() -> None:
    with rasterio.open(b.SURFACE) as src:
        p100 = b.profile_like(src)

    ref, n_clear = b.landsat_reference(p100)
    clear = np.isfinite(ref)

    meta = {
        "schema": "landsat-v2-clear-validation-v1",
        "date": b.DATE,
        "roi": b.ROI,
        "source_rule": "LST_C where QA_PIXEL bits 0,1,2,3,4,5 are all zero; water preserved; QA_RADSAT not used as blanket LST mask",
        "source_dir": f"data/landsat_c2_l2/v2/{b.RID}/2019/09/24",
        "grid": {
            "crs": str(p100["crs"]),
            "transform": list(p100["transform"])[:6],
            "height": int(p100["height"]),
            "width": int(p100["width"]),
        },
        "clear_pixels": int(n_clear),
        "total_pixels": int(ref.size),
    }

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        lst_c=np.where(clear, ref, np.nan).astype("float32"),
        clear_mask=clear.astype("uint8"),
        metadata_json=np.array(json.dumps(meta, separators=(",", ":"))),
    )
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    print("LANDSAT_V2_VALIDATION_B64_BEGIN")
    for i in range(0, len(encoded), 4000):
        print(encoded[i:i + 4000])
    print("LANDSAT_V2_VALIDATION_B64_END")
    print(json.dumps({
        "encoded_chars": len(encoded),
        "npz_bytes": len(buf.getvalue()),
        "clear_pixels": int(n_clear),
        "total_pixels": int(ref.size),
        "shape": list(ref.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
