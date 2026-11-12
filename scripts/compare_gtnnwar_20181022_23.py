#!/usr/bin/env python3
"""Compare 2018-10-22/23 MODIS LST with the 2018-10-23 Landsat validation reference.

Datasets are actual remote-sensing rasters acquired by remote-sensing-mcp,
not a synthetic downscaled LST image or reproduction of paper Fig. 9(b).
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROI = (100.188333, 38.697500, 100.767778, 39.150278)
OUT = Path("results/gtnnwar-zhangye-20181023")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ("Landsat_20181023", "Landsat 8 C2 L2 | 23 Oct 03:55 UTC",
     "data/landsat_c2_l2/v3/gtnnwar-zhangye-20181023-*/2018/10/23/*_L2_LST_QA.tif", "landsat"),
    ("L2_20181023", "MOD11_L2 | 23 Oct", 
     "data/modis_l2_swath/v1/gtnnwar-zhangye-20181023-*/2018/10/23/*.tif", "l2"),
    ("L2_20181022", "MOD11_L2 | 22 Oct",
     "output/modis_l2_swath/**/MOD11_L2_20181022_*_C.tif", "l2"),
    ("A1_20181023", "MOD11A1 | 23 Oct",
     "data/modis_lst/v2/gtnnwar-zhangye-20181023-*/2018/10/23/MOD11A1_20181023_LST_QA.tif", "daily"),
    ("A1_20181022", "MOD11A1 | 22 Oct",
     "output/modis_lst/**/MOD11A1_20181022_LST_QA.tif", "daily"),
]


def strict_modis_mask(q):
    q = np.where(np.isfinite(q), q, -9999).astype("int32")
    return (q >= 0) & ((q & 3) <= 1) & (((q >> 2) & 3) == 0) & (((q >> 6) & 3) <= 2)


def view_time(src, kind, lon):
    if kind == "landsat":
        return "03:55 UTC"
    band = 8 if kind == "l2" else 3
    if src.count < band:
        return "time unknown"
    v = src.read(band, masked=True).filled(np.nan).astype("float32")
    v = v[np.isfinite(v) & (v >= 0) & (v < 24)]
    if not len(v):
        return "time unknown"
    h = float(np.mean(v))
    if kind == "daily":
        h = (h - lon / 15.0) % 24
    minute = round(h * 60) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d} UTC (ROI mean)"


def read_image(path, kind):
    with rasterio.open(path) as src:
        temp = src.read(1, masked=True).filled(np.nan).astype("float32")
        qc_band = {"landsat": 3, "l2": 2, "daily": 5}[kind]
        qc = src.read(qc_band, masked=True).filled(np.nan).astype("float32")
        valid = np.isfinite(temp) & (temp > -80) & (temp < 90)
        if kind == "landsat":
            qa = np.where(np.isfinite(qc), qc, -9999).astype("int32")
            valid &= (qa >= 0) & ((qa & 63) == 0)
        else:
            valid &= strict_modis_mask(qc)
        array = np.where(valid, temp, np.nan)
        bounds = src.bounds
        return dict(array=array, transform=src.transform, crs=src.crs,
                    extent=[bounds.left,bounds.right,bounds.bottom,bounds.top],
                    time=view_time(src,kind,(ROI[0]+ROI[2])/2), path=str(path),
                    valid_pixels=int(valid.sum()), total_pixels=int(temp.size),
                    valid_fraction=float(valid.mean()),
                    mean_c=float(np.nanmean(array)) if valid.any() else None,
                    min_c=float(np.nanmin(array)) if valid.any() else None,
                    max_c=float(np.nanmax(array)) if valid.any() else None)


def matchup(reference, coarse):
    """Reference aggregated to coarse pixel grid, using only native QA-valid Landsat."""
    a = np.full(coarse["array"].shape, np.nan, dtype="float32")
    reproject(
        source=reference["array"], destination=a,
        src_transform=reference["transform"], src_crs=reference["crs"],
        src_nodata=np.nan, dst_transform=coarse["transform"],
        dst_crs=coarse["crs"], dst_nodata=np.nan,
        resampling=Resampling.average
    )
    b=coarse["array"]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return {"match_pixels":int(mask.sum())}
    x=a[mask].astype("float64")
    y=b[mask].astype("float64")
    d=y-x
    return {
        "match_pixels":int(mask.sum()),
        "pearson_r":float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None,
        "rmse_c":float(np.sqrt(np.mean(d*d))),
        "mae_c":float(np.mean(np.abs(d))),
        "bias_c":float(np.mean(d)),
        "reference_agg_mean_c":float(np.mean(x)),
        "coarse_overlap_mean_c":float(np.mean(y)),
        "reference_agg_std_c":float(np.std(x)),
        "coarse_overlap_std_c":float(np.std(y)),
    }


loaded, report = {}, {}
for code, title, pat, kind in SOURCES:
    paths=sorted(glob.glob(pat,recursive=True))
    if not paths:
        report[code]={"status":"not_retrieved","pattern":pat}
        continue
    try:
        data=read_image(paths[0],kind)
        loaded[code]=data
        report[code]={k:v for k,v in data.items() if k not in {"array","transform","crs","extent"}}
        report[code]["status"]="ok" if data["valid_pixels"] else "no_valid_QA_pixels"
    except Exception as e:
        report[code]={"status":"read_error","path":paths[0],"error":str(e)}

ref=loaded.get("Landsat_20181023")
if ref is not None:
    for code in ["L2_20181023","L2_20181022","A1_20181023","A1_20181022"]:
        if code in loaded:
            report[code]["vs_landsat_23"]=matchup(ref,loaded[code])

# Scientific comparison: all maps use the same physical color scale and exact ROI.
vals = [d["array"][np.isfinite(d["array"])] for d in loaded.values() if np.isfinite(d["array"]).any()]
if vals:
    both=np.concatenate([v[::max(1,len(v)//100000)] for v in vals])
    vmin=float(np.percentile(both,1))
    vmax=float(np.percentile(both,99))
else:
    vmin,vmax=0.0,35.0
if vmax<=vmin:
    vmax=vmin+1
report["_metadata"]={
    "ROI_WGS84":list(ROI),
    "reference":"Landsat 8 C2 L2 ST_B10; cloud/cirrus/shadow/snow/fill QA excluded",
    "daily_product":"MOD11A1 is an additional comparison, not the paper's stated MOD11_L2",
    "method":"Landsat QA-valid temperature area-averaged to each MODIS source grid",
    "temperature_units":"degrees Celsius",
    "shared_display_vmin_c":vmin,"shared_display_vmax_c":vmax,
    "caution":"Different dates and/or overpass times; metrics are descriptive only, not downscaling validation scores."
}
(OUT/"modis_22_23_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
with (OUT/"modis_22_23_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
    fields=["product","utc_time","valid_fraction","mean_c","match_pixels","pearson_r","rmse_c","mae_c","bias_c"]
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for code in ["L2_20181023","L2_20181022","A1_20181023","A1_20181022"]:
        row=report.get(code,{})
        m=row.get("vs_landsat_23",{})
        w.writerow(dict(product=code,utc_time=row.get("time"),valid_fraction=row.get("valid_fraction"),
                        mean_c=row.get("mean_c"),**{k:m.get(k) for k in fields[4:]}))

positions=[
    ("Landsat_20181023",(0,0)),
    ("L2_20181023",(0,1)),
    ("L2_20181022",(0,2)),
    ("A1_20181023",(1,1)),
    ("A1_20181022",(1,2))
]
fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
for code,(r,c) in positions:
    ax=axes[r,c]
    data=loaded.get(code)
    if data is None or not np.isfinite(data["array"]).any():
        ax.set_facecolor("#eeeeee")
        ax.text(0.5,0.5,"Not retrieved / no QA-valid pixels",ha="center",va="center",
                transform=ax.transAxes,wrap=True,fontsize=11)
        ttl=next(x[1] for x in SOURCES if x[0]==code)
    else:
        ax.imshow(data["array"], extent=data["extent"], origin="upper",
                  vmin=vmin, vmax=vmax,cmap="RdYlGn_r",interpolation="nearest")
        ttl=next(x[1] for x in SOURCES if x[0]==code)
        ttl+="\n"+data["time"]+" | valid {:.1%}".format(data["valid_fraction"])
    ax.set_title(ttl,fontsize=11)
    ax.set_xlim(ROI[0],ROI[2]);ax.set_ylim(ROI[1],ROI[3])
    ax.set_xlabel("Longitude (°E)",fontsize=9)
    ax.set_ylabel("Latitude (°N)",fontsize=9)
    ax.tick_params(labelsize=8)
ax=axes[1,0]
ax.axis("off")
ax.set_title("Compared with 23 Oct Landsat\n(aggregated to each MODIS grid)",fontsize=11)
rows=[]
for code in ["L2_20181023","L2_20181022","A1_20181023","A1_20181022"]:
    m=report.get(code,{}).get("vs_landsat_23",{})
    if m.get("pearson_r") is not None:
        rows.append([code.replace("_201810"," 10/"), f'{m["pearson_r"]:+.3f}',
                     f'{m["rmse_c"]:.2f}',f'{m["bias_c"]:+.2f}',str(m["match_pixels"])])
    else:
        rows.append([code.replace("_201810"," 10/"),"—","—","—",str(m.get("match_pixels","—"))])
tbl=ax.table(cellText=rows,colLabels=["Product","r","RMSE °C","Bias °C","n"],loc="center",
             colWidths=[0.34,0.15,0.18,0.18,0.15],cellLoc="center")
tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,1.85)
ax.text(0.0,0.19,"Time/date differences are not corrected.\n"
        "These metrics do not measure downscaling accuracy.",transform=ax.transAxes,
        va="top",ha="left",fontsize=9,color="#555555")
sm=plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin,vmax=vmax),cmap="RdYlGn_r")
sm.set_array([])
fig.colorbar(sm,ax=axes.ravel().tolist(),orientation="horizontal",shrink=0.66,
             aspect=48,pad=0.035,label="LST (°C) | identical display scale in all panels")
fig.suptitle("Zhangye GTNNWAR paper study area | MODIS 22 vs 23 Oct 2018 | Landsat reference",
             fontsize=15)
out=OUT/"modis_22_23_vs_landsat_23.png"
fig.savefig(out,dpi=155,bbox_inches="tight",facecolor="white")
plt.close(fig)
print(json.dumps(report,ensure_ascii=False,indent=2))
print("Saved",out)
