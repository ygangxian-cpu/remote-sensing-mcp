from __future__ import annotations

import base64, csv, json, math, os, shutil
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import rasterio
import requests
from google.oauth2 import service_account
from pyproj import Transformer

PROJECT_DEFAULT = "ee-ygangxian"
BBOX = [31.00, 29.80, 31.55, 30.35]
CRS = "EPSG:32636"
START, END = "2025-06-01", "2025-09-01"
NODATA = -9999.0

S2 = "COPERNICUS/S2_SR_HARMONIZED"
SRTM = "USGS/SRTMGL1_003"
WC = "ESA/WorldCover/v200"
MODIS = "MODIS/061/MOD11A1"

REF = ["blue","green","red","nir","swir1","swir2"]
IDX = ["ndvi","evi","savi","ndwi","mndwi","ndmi","ndbi","ui","bsi","nbr"]
TERRAIN = ["elevation","slope","aspect"]
WC_MAP = {
    10:"wc_tree_cover_frac",20:"wc_shrubland_frac",30:"wc_grassland_frac",
    40:"wc_cropland_frac",50:"wc_built_up_frac",60:"wc_bare_sparse_frac",
    70:"wc_snow_ice_frac",80:"wc_water_frac",90:"wc_herbaceous_wetland_frac",
    95:"wc_mangroves_frac",100:"wc_moss_lichen_frac",
}
WC_BANDS = list(WC_MAP.values())
BANDS = REF + IDX + TERRAIN + WC_BANDS
assert len(BANDS) == 30

def init_ee():
    project = os.getenv("EE_PROJECT","").strip() or PROJECT_DEFAULT
    cred = Path.home()/".config/earthengine/credentials"
    if cred.exists():
        ee.Initialize(project=project)
        return project
    raw = os.getenv("EE_SERVICE_ACCOUNT_JSON","").strip()
    if not raw:
        b64 = os.getenv("EE_SERVICE_ACCOUNT_JSON_BASE64","").strip()
        if b64:
            raw = base64.b64decode(b64).decode()
    if not raw:
        raise RuntimeError("Earth Engine credentials are missing")
    info = json.loads(raw)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/earthengine","https://www.googleapis.com/auth/cloud-platform"],
    )
    project = os.getenv("EE_PROJECT","").strip() or info.get("project_id") or PROJECT_DEFAULT
    ee.Initialize(credentials, project=project)
    return project

def snap_bounds():
    w,s,e,n=BBOX
    tf=Transformer.from_crs("EPSG:4326",CRS,always_xy=True)
    pts=[tf.transform(w,s),tf.transform(w,n),tf.transform(e,s),tf.transform(e,n)]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return (
        math.floor(min(xs)/1000)*1000,
        math.floor(min(ys)/1000)*1000,
        math.ceil(max(xs)/1000)*1000,
        math.ceil(max(ys)/1000)*1000,
    )

XMIN,YMIN,XMAX,YMAX=snap_bounds()

def proj(scale):
    return ee.Projection(CRS,[float(scale),0.0,XMIN,0.0,-float(scale),YMAX])

def roi():
    return ee.Geometry.Rectangle([XMIN,YMIN,XMAX,YMAX],proj=CRS,geodesic=False)

def mask_s2(img):
    scl=img.select("SCL"); qa=img.select("QA60")
    m=scl.eq(2).Or(scl.eq(4)).Or(scl.eq(5)).Or(scl.eq(6))
    m=m.And(qa.bitwiseAnd(1<<10).eq(0)).And(qa.bitwiseAnd(1<<11).eq(0))
    return img.select(["B2","B3","B4","B8","B11","B12"],REF).multiply(1e-4).updateMask(m)

def sdiv(a,b):
    return a.divide(b.where(b.abs().lt(1e-6),1e-6))

def with_indices(x):
    b,g,r,n,s1,s2=[x.select(v) for v in REF]
    out=[
        sdiv(n.subtract(r),n.add(r)).rename("ndvi"),
        sdiv(n.subtract(r).multiply(2.5),n.add(r.multiply(6)).subtract(b.multiply(7.5)).add(1)).rename("evi"),
        sdiv(n.subtract(r).multiply(1.5),n.add(r).add(0.5)).rename("savi"),
        sdiv(g.subtract(n),g.add(n)).rename("ndwi"),
        sdiv(g.subtract(s1),g.add(s1)).rename("mndwi"),
        sdiv(n.subtract(s1),n.add(s1)).rename("ndmi"),
        sdiv(s1.subtract(n),s1.add(n)).rename("ndbi"),
        sdiv(s2.subtract(n),s2.add(n)).rename("ui"),
        sdiv(s1.add(r).subtract(n.add(b)),s1.add(r).add(n).add(b)).rename("bsi"),
        sdiv(n.subtract(s2),n.add(s2)).rename("nbr"),
    ]
    return x.addBands(out)

def avg(img,p,maxpx=65536):
    return img.reduceResolution(reducer=ee.Reducer.mean(),maxPixels=maxpx).reproject(p)

def build():
    p20,p100,p1k=proj(20),proj(100),proj(1000)
    c=(ee.ImageCollection(S2).filterBounds(roi()).filterDate(START,END)
       .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE",40)).map(mask_s2))
    s2n=int(c.size().getInfo())
    if s2n<1: raise RuntimeError("No S2 scenes")
    med=c.median().resample("bilinear").reproject(p20).clip(roi())
    s2_100=avg(with_indices(med).select(REF+IDX),p100)

    dem100=avg(ee.Image(SRTM).select("elevation"),p100,4096).rename("elevation")
    ter=ee.Terrain.products(dem100)
    terrain=dem100.addBands([ter.select("slope").rename("slope"),ter.select("aspect").rename("aspect")])

    wc=ee.ImageCollection(WC).first().select("Map")
    frac=ee.Image.cat([avg(wc.eq(k).toFloat().rename(v),p100,4096).rename(v) for k,v in WC_MAP.items()])
    stack100=s2_100.addBands(terrain).addBands(frac).select(BANDS).toFloat().clip(roi())

    ordinary=[x for x in BANDS if x!="aspect"]
    coarse=avg(stack100.select(ordinary),p1k,4096)
    a=stack100.select("aspect").multiply(math.pi/180)
    asp=avg(a.sin(),p1k,4096).atan2(avg(a.cos(),p1k,4096)).multiply(180/math.pi).add(360).mod(360).rename("aspect")
    stack1k=coarse.addBands(asp).select(BANDS).toFloat().clip(roi())

    def bits(v,a,b):
        mask=ee.Number(1).leftShift(ee.Number(b).subtract(a).add(1)).subtract(1)
        return v.rightShift(a).bitwiseAnd(mask)
    def qcmask(img):
        qc=img.select("QC_Day"); raw=img.select("LST_Day_1km")
        m=bits(qc,0,1).lte(1).And(bits(qc,2,3).eq(0)).And(bits(qc,6,7).lte(2)).And(raw.gt(0))
        return raw.multiply(0.02).subtract(273.15).rename("modis_lst_c").updateMask(m)
    mc=ee.ImageCollection(MODIS).filterBounds(roi()).filterDate(START,END).map(qcmask)
    mn=int(mc.size().getInfo())
    modis=mc.median().resample("bilinear").reproject(p1k).clip(roi())
    return stack100,stack1k,modis,s2n,mn

def download(image,band,scale,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tr=[float(scale),0.0,XMIN,0.0,-float(scale),YMAX]
    url=ee.Image(image).select(band).unmask(NODATA).getDownloadURL({
        "name":path.stem,"region":roi(),"crs":CRS,"crs_transform":tr,"format":"GEO_TIFF"
    })
    with requests.get(url,stream=True,timeout=300) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(2*1024*1024):
                if chunk: f.write(chunk)

def stack(paths,names,out):
    with rasterio.open(paths[0]) as s:
        prof=s.profile.copy(); shape=(s.height,s.width); crs=s.crs; transform=s.transform
    prof.update(count=len(paths),dtype="float32",nodata=NODATA,compress="deflate",predictor=3)
    with rasterio.open(out,"w",**prof) as d:
        for i,(p,n) in enumerate(zip(paths,names),1):
            with rasterio.open(p) as s:
                if (s.height,s.width)!=shape or s.crs!=crs or s.transform!=transform:
                    raise RuntimeError(f"grid mismatch: {p}")
                a=s.read(1).astype("float32")
            d.write(a,i); d.set_band_description(i,n)

def audit(p100,p1k,pmod,s2n,mn):
    out={"created_at_utc":datetime.now(timezone.utc).isoformat(),"predictors":BANDS,
         "s2_scene_count":s2n,"modis_scene_count":mn,"checks":{}}
    with rasterio.open(p100) as f, rasterio.open(p1k) as c:
        out["grid100"]={"width":f.width,"height":f.height,"transform":list(f.transform)[:6],"crs":str(f.crs)}
        out["grid1k"]={"width":c.width,"height":c.height,"transform":list(c.transform)[:6],"crs":str(c.crs)}
        out["checks"]["band_count"]=f.count==30 and c.count==30
        out["checks"]["exact_10x_nesting"]=f.width==c.width*10 and f.height==c.height*10 and f.transform.c==c.transform.c and f.transform.f==c.transform.f
        ids=[BANDS.index(x)+1 for x in WC_BANDS]
        fr=f.read(ids).astype(float)
        valid=np.all(np.isfinite(fr)&(fr!=NODATA),axis=0)
        vals=fr[:,valid]; sums=vals.sum(axis=0)
        out["worldcover_fraction_sum"]={"n":int(sums.size),"mean":float(sums.mean()),"p01":float(np.percentile(sums,1)),"p99":float(np.percentile(sums,99))}
        out["checks"]["fraction_bounds"]=bool(np.nanmin(vals)>=-1e-6 and np.nanmax(vals)<=1.000001)
        out["checks"]["fraction_sum"]=bool(np.mean(np.abs(sums-1)<=0.02)>=0.99)

    rows=[]
    with rasterio.open(p1k) as c, rasterio.open(pmod) as m:
        y=m.read(1).astype(float)
        for i,name in enumerate(BANDS,1):
            x=c.read(i).astype(float)
            ok=np.isfinite(x)&np.isfinite(y)&(x!=NODATA)&(y!=NODATA)
            xv,yv=x[ok],y[ok]
            rr=float(np.corrcoef(xv,yv)[0,1]) if xv.size>=3 and xv.std()>0 and yv.std()>0 else None
            rows.append({"predictor":name,"pearson_r":rr,"n":int(xv.size)})
    rows.sort(key=lambda z:abs(z["pearson_r"]) if z["pearson_r"] is not None else -1,reverse=True)
    out["top_correlations"]=rows[:10]
    key={r["predictor"]:r for r in rows}
    out["key_correlations"]={k:key[k] for k in ["swir2","ndbi","ndvi","mndwi","wc_built_up_frac","wc_bare_sparse_frac"]}
    out["checks"]["all_passed"]=all(out["checks"].values())
    return out,rows

def main():
    project=init_ee()
    root=Path("output/hamdi_2026"); tmp=root/"bands"
    shutil.rmtree(root,ignore_errors=True); tmp.mkdir(parents=True)
    s100,s1k,modis,s2n,mn=build()
    p100=[]; p1k=[]
    for b in BANDS:
        a=tmp/f"100m_{b}.tif"; c=tmp/f"1km_{b}.tif"
        download(s100,b,100,a); download(s1k,b,1000,c)
        p100.append(a); p1k.append(c)
    mt=tmp/"modis.tif"; download(modis,"modis_lst_c",1000,mt)
    f100=root/"HAMDI_CAIRO_2025_SUMMER_PREDICTORS_100M.tif"
    f1k=root/"HAMDI_CAIRO_2025_SUMMER_PREDICTORS_1KM.tif"
    fm=root/"MOD11A1_2025_SUMMER_MEDIAN_DAY_1KM_C.tif"
    stack(p100,BANDS,f100); stack(p1k,BANDS,f1k); shutil.copy2(mt,fm)
    summary,rows=audit(f100,f1k,fm,s2n,mn)
    summary["earth_engine_project"]=project
    (root/"audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (root/"predictor_modis_correlations.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["predictor","pearson_r","n"]); w.writeheader(); w.writerows(rows)
    shutil.rmtree(tmp,ignore_errors=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
