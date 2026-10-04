from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modis_worker


BBOX = [110.12447068021214, 24.99113120420509, 110.49277994670115, 25.341474165011704]
DATE = "2022-08-24"
OUT = Path("experiments/guilin_20220824_modis")


def main() -> None:
    modis_worker.init_ee()
    OUT.mkdir(parents=True, exist_ok=True)
    day = datetime.fromisoformat(DATE)
    out = OUT / "Guilin_MOD11A1_20220824_LST_QA.tif"
    details = modis_worker.download_one("terra", day, BBOX, out)
    if details is None:
        raise RuntimeError("No Terra MOD11A1 scene found for 2022-08-24")

    with rasterio.open(out) as src:
        desc = list(src.descriptions)
        tags = src.tags()
        shape = [src.height, src.width]
        crs = str(src.crs)
        bounds = list(src.bounds)

    summary = {
        "date": DATE,
        "product": "MODIS/061/MOD11A1",
        "platform": "terra",
        "bbox": BBOX,
        "file": str(out),
        "descriptions": desc,
        "shape": shape,
        "crs": crs,
        "bounds": bounds,
        "tags": tags,
        "details": details,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
