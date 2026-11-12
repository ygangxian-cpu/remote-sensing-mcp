#!/usr/bin/env python3
"""Two actual satellite images for GTNNWAR Zhangye, with Fig. 9's Kelvin range.

No Landsat temperature is used to modify or calibrate MODIS. This is purely
an illustration of input/reference differences under comparable display scales.
"""
from pathlib import Path
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from PIL import Image

ROOT=Path("results/gtnnwar-zhangye-20181023")
ROOT.mkdir(parents=True, exist_ok=True)
roi=(100.188333,38.6975,100.767778,39.150278)
land_path="data/landsat_c2_l2/v3/gtnnwar-zhangye-20181023-a5d90c376e/2018/10/23/LC08_L2SP_133033_20181023_20200830_02_T1_L2_LST_QA.tif"
l2files=glob.glob("data/modis_l2_swath/v1/gtnnwar-zhangye-20181022-*/2018/10/22/MOD11_L2_20181022_*_UTC_C.tif")
assert Path(land_path).exists(),land_path
assert len(l2files)>=1,"MODIS 22 Oct L2 file not cached"
l2_path=l2files[0]

def raster(path,is_landsat):
    with rasterio.open(path) as s:
        a=s.read(1,masked=True).filled(np.nan).astype("float32")
        if is_landsat:
            qc=s.read(3,masked=True).filled(-9999).astype("int32")
            valid=(qc>=0)&((qc&63)==0)
        else:
            qc=s.read(2,masked=True).filled(-9999).astype("int32")
            valid=(qc>=0)&((qc&3)<=1)&(((qc>>2)&3)==0)&(((qc>>6)&3)<=2)
        valid&=np.isfinite(a)&(a>-80)&(a<90)
        t=np.where(valid,a+273.15,np.nan)
        b=s.bounds
        return dict(a=t,extent=[b.left,b.right,b.bottom,b.top],
                    n=int(valid.sum()),valid_pct=round(100*valid.mean(),4),
                    mean_K=round(float(np.nanmean(t)),2),
                    max_K=round(float(np.nanmax(t)),2),
                    min_K=round(float(np.nanmin(t)),2),
                    pixels_above_304K=int(np.sum(t>=304)),
                    pixels_above_312K=int(np.sum(t>=312)))
land=raster(land_path,True)
l2=raster(l2_path,False)

# Fig 9 PDF reports tick labels 280,288,296,304,312 K and clipped extremes.
# Palette approximates printed green-yellow-orange-red legend, not pixel-level
# reproduction of the journal's color encoding.
colors=[(0,"#008f00"),(.25,"#62bf00"),(.5,"#e6eb00"),(.75,"#ff9c00"),(.9,"#ff4200"),(1,"#da0303")]
cmap=LinearSegmentedColormap.from_list("paper_approx",colors,N=256)
cmap.set_bad("#f3f3f3")
fig,ax=plt.subplots(1,2,figsize=(12,5.5),layout="constrained")
for a,d,ttl in [
    (ax[0],land,"(a) Landsat 8 C2 L2 | 2018-10-23 03:55 UTC"),
    (ax[1],l2,"(b) Terra MOD11_L2 | 2018-10-22 04:18 UTC"),
]:
    im=a.imshow(d["a"],origin="upper",extent=d["extent"],
                cmap=cmap,vmin=280,vmax=312,interpolation="nearest")
    a.set_title(ttl,fontsize=11,pad=8)
    a.set_xlim(roi[0],roi[2]);a.set_ylim(roi[1],roi[3])
    a.set_xlabel("Longitude (°E)")
    a.set_ylabel("Latitude (°N)")
    a.text(.02,.03,f'Min {d["min_K"]:.2f} K   Max {d["max_K"]:.2f} K',
           ha="left",va="bottom",transform=a.transAxes,fontsize=9,
           bbox=dict(facecolor="white",alpha=.85,edgecolor="none",pad=3))
cb=fig.colorbar(im,ax=ax.ravel().tolist(),orientation="horizontal",
                ticks=[280,288,296,304,312],pad=.075,shrink=.77,aspect=32)
cb.set_label("LST (K) | identical 280-312 K range as paper Fig. 9")
png=ROOT/"landsat23_vs_modis_l2_22_paper_scale.png"
fig.savefig(png,dpi=160,facecolor="white",bbox_inches="tight")
plt.close(fig)

im=Image.open(png).convert("RGB")
im.thumbnail((1450,800),Image.Resampling.LANCZOS)
preview=ROOT/"landsat23_vs_modis_l2_22_paper_scale_preview.png"
im.quantize(colors=176,method=Image.Quantize.FASTOCTREE,dither=Image.Dither.NONE).save(preview,optimize=True)
stats={
 "source": "real satellite retrievals from MCP repo",
 "reference":land_path,
 "modis":l2_path,
 "date_note":"Landsat 2018-10-23 vs MODIS 2018-10-22 are on different dates",
 "paper_fig9_legend_K":[280,288,296,304,312],
 "plot_colormap":"approximation to paper green-yellow-orange-red legend",
 "landsat":{k:v for k,v in land.items() if k not in ("a","extent")},
 "modis":{k:v for k,v in l2.items() if k not in ("a","extent")},
 "note":"Raw products are not identical to paper processing (Landsat paper uses MW retrieval; MODIS paper may gap-fill)."
}
(ROOT/"landsat23_vs_modis_l2_22_paper_scale_stats.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(stats,indent=2,ensure_ascii=False))
