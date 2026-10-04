from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

BOUNDS = [99.86528182419725, 38.67911397351706, 100.47613621740223, 39.39776620081703]
H1, W1 = 80, 68
H100, W100 = 800, 680
TARGET_HOUR = 4
KNN = 80
ALPHA = 0.50
FEATURES = ["landcover", "dem", "albedo_bsa", "ndvi", "slope", "ui", "mndwi"]
XGB = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
)
C3_GLOBAL = (1.1349225619453363, -0.8893386588200753)
C3_GROUPS = {
    10: (1.032667187009931, -2.231727062239422),
    20: (1.164012553213395, -0.541645565854741),
    30: (1.1313549006480768, -1.0466180491837533),
    40: (1.089142725765745, -0.7373163629153986),
    50: (1.1537965070343497, -1.5590445207735217),
    60: (1.1694459660062833, -0.5049753218411099),
    80: (0.9871670329351451, -2.1456398008386093),
}

def profile_for(h: int, w: int) -> dict:
    return {
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": from_bounds(*BOUNDS, w, h),
        "height": h, "width": w,
    }

P1 = profile_for(H1, W1)
P100 = profile_for(H100, W100)

def fill_nearest(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype="float32").copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    if bad.all():
        raise RuntimeError("array is entirely missing")
    inds = distance_transform_edt(bad, return_distances=False, return_indices=True)
    x[bad] = x[tuple(inds[:, bad])]
    return x.astype("float32")

def reproj(a, src_profile, dst_profile, method=Resampling.bilinear):
    dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=np.asarray(a, dtype="float32"),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=method,
    )
    return dst

def tif_profile(src) -> dict:
    return {"crs": src.crs, "transform": src.transform, "height": src.height, "width": src.width}

def descs(src):
    return [(d or "").upper() for d in src.descriptions]

def read_band(path: Path, names: list[str], fallback: int | None = None):
    with rasterio.open(path) as src:
        ds = descs(src)
        idx = None
        for name in names:
            n = name.upper()
            for i, d in enumerate(ds, start=1):
                if d == n or n in d:
                    idx = i
                    break
            if idx:
                break
        if idx is None:
            if fallback is None:
                raise RuntimeError(f"Cannot find {names} in {path}; bands={ds}")
            idx = fallback
        a = src.read(idx, masked=True).astype("float32").filled(np.nan)
        return a, tif_profile(src), ds

def find_one(root: Path, token: str) -> Path:
    xs = [p for p in root.rglob("*.tif") if token.lower() in p.name.lower()]
    if len(xs) != 1:
        raise RuntimeError(f"Expected one *{token}* tif, got {len(xs)}: {xs[:10]}")
    return xs[0]

def extract_hour(path: Path) -> int | None:
    m = re.search(r"_(\d{8})_(\d{2})00(?:_|\.|$)", path.name)
    if m:
        return int(m.group(2))
    m = re.search(r"_(\d{2})00(?:_|\.|$)", path.name)
    return int(m.group(1)) if m else None

def hourly_files(root: Path, token: str) -> dict[int, Path]:
    out = {}
    for p in root.rglob("*.tif"):
        if token.lower() not in p.name.lower():
            continue
        h = extract_hour(p)
        if h is not None:
            out[h] = p
    if len(out) != 24:
        raise RuntimeError(f"Expected 24 {token} hourly tifs, got {len(out)}; keys={sorted(out)}")
    return out

def read_to_grid(path: Path, names: list[str], grid: dict, fallback=None, method=Resampling.bilinear):
    a, p, _ = read_band(path, names, fallback)
    return fill_nearest(reproj(a, p, grid, method))

def block_mean_100_to_1(a: np.ndarray) -> np.ndarray:
    return np.nanmean(a.reshape(H1, 10, W1, 10), axis=(1, 3)).astype("float32")

def apply_c3(a: np.ndarray, lc: np.ndarray) -> np.ndarray:
    s, b = C3_GLOBAL
    out = a.astype("float32") * np.float32(s) + np.float32(b)
    lci = np.rint(lc).astype("int16")
    for cls, (s, b) in C3_GROUPS.items():
        m = lci == cls
        out[m] = a[m] * np.float32(s) + np.float32(b)
    return out.astype("float32")

def lonlat_grid(profile):
    h, w = profile["height"], profile["width"]
    rr, cc = np.meshgrid(np.arange(h)+0.5, np.arange(w)+0.5, indexing="ij")
    t = profile["transform"]
    lon = t.c + t.a*cc + t.b*rr
    lat = t.f + t.d*cc + t.e*rr
    return lon.astype("float64"), lat.astype("float64")

def solar_geometry(profile, when):
    lon, lat = lonlat_grid(profile)
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute/60 + when.second/3600
    gamma = 2*math.pi/365*(doy-1+(hour-12)/24)
    eqtime = 229.18*(0.000075+0.001868*math.cos(gamma)-0.032077*math.sin(gamma)
                    -0.014615*math.cos(2*gamma)-0.040849*math.sin(2*gamma))
    decl = (0.006918-0.399912*math.cos(gamma)+0.070257*math.sin(gamma)
            -0.006758*math.cos(2*gamma)+0.000907*math.sin(2*gamma)
            -0.002697*math.cos(3*gamma)+0.00148*math.sin(3*gamma))
    tsm = (hour*60 + eqtime + 4*lon) % 1440
    ha = np.deg2rad(tsm/4 - 180)
    latr = np.deg2rad(lat)
    cosz = np.sin(latr)*math.sin(decl)+np.cos(latr)*math.cos(decl)*np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    alt = np.arcsin(cosz)
    az = np.arctan2(np.sin(ha), np.cos(ha)*np.sin(latr)-math.tan(decl)*np.cos(latr)) + math.pi
    return cosz.astype("float32"), alt.astype("float32"), np.mod(az, 2*math.pi).astype("float32")

def pixel_size_m(profile):
    lon, lat = lonlat_grid(profile)
    lat0 = float(np.nanmean(lat))
    t = profile["transform"]
    return abs(t.a)*111320*max(math.cos(math.radians(lat0)),0.1), abs(t.e)*110540

def terrain_shadow_fast(dem, profile, alt, az, max_distance_m=20000.0):
    z = dem.astype("float64")
    h, w = z.shape
    dx, dy = pixel_size_m(profile)
    step = max(1.0, min(dx, dy))
    mean_az = float(np.nanmean(az))
    east, north = math.sin(mean_az), math.cos(mean_az)
    offsets, seen = [], set()
    for k in range(1, max(2, int(math.ceil(max_distance_m/step)))+1):
        d = k*step
        dc = int(round(east*d/max(dx,1e-6)))
        dr = int(round(-north*d/max(dy,1e-6)))
        if (dr,dc) == (0,0) or (dr,dc) in seen:
            continue
        seen.add((dr,dc)); offsets.append((dr,dc))
    tan_alt = np.tan(alt.astype("float64"))
    shadow = (~np.isfinite(tan_alt) | (tan_alt <= 0))
    for dr, dc in offsets:
        r0, r1 = max(0,-dr), min(h,h-dr)
        c0, c1 = max(0,-dc), min(w,w-dc)
        if r0 >= r1 or c0 >= c1:
            continue
        rr0, rr1, cc0, cc1 = r0+dr, r1+dr, c0+dc, c1+dc
        dist = math.hypot(dc*dx, dr*dy)
        hit = ((z[rr0:rr1,cc0:cc1]-z[r0:r1,c0:c1])/dist > tan_alt[r0:r1,c0:c1])
        shadow[r0:r1,c0:c1] |= np.isfinite(hit) & hit
    return shadow.astype("float32")

def sw_abs_100(static, swdown, when):
    cosz, alt, az = solar_geometry(P100, when)
    slope = np.deg2rad(static["slope"].astype("float64"))
    aspect = np.deg2rad(static["aspect"].astype("float64"))
    sin_alt = np.clip(cosz.astype("float64"),0,1)
    cos_alt = np.sqrt(np.maximum(0,1-sin_alt**2))
    cos_i = np.maximum(np.cos(slope)*sin_alt + np.sin(slope)*cos_alt*np.cos(az-aspect), 0)
    shadow = terrain_shadow_fast(static["dem"], P100, alt, az)
    illum = np.where((sin_alt>0)&(shadow<0.5), cos_i, 0)
    tc = np.where(sin_alt>0.02, illum/np.maximum(sin_alt,0.02), 0)
    tc = np.clip(tc,0,3)
    alb = static["albedo_bsa"].astype("float64")
    if np.nanmedian(alb) > 1.5:
        alb /= 100
    alb = np.clip(alb,0.02,0.95)
    return np.maximum(0,(1-alb)*swdown.astype("float64")*tc).astype("float32"), float(np.nanmean(np.rad2deg(alt)))

def surface_stack(root: Path):
    surf = find_one(root, "LANDSAT_SCALING_FACTORS_100M")
    terrain = find_one(root, "SRTM_TERRAIN_100M")
    lcfile = find_one(root, "WORLDCOVER_2021_100M")
    albs = [p for p in root.rglob("*.tif") if "MCD43A3_" in p.name]
    if len(albs) != 1:
        raise RuntimeError(f"Expected one albedo tif, got {len(albs)}")
    ndvi = read_to_grid(surf, ["NDVI"], P100)
    mndwi = read_to_grid(surf, ["MNDWI"], P100)
    nir = read_to_grid(surf, ["NIR"], P100)
    swir2 = read_to_grid(surf, ["SWIR2"], P100)
    ui = np.divide(swir2-nir, swir2+nir, out=np.zeros_like(nir), where=np.abs(swir2+nir)>1e-6).astype("float32")
    dem = read_to_grid(terrain, ["DEM_M","DEM"], P100)
    slope = read_to_grid(terrain, ["SLOPE_DEG","SLOPE"], P100)
    aspect = read_to_grid(terrain, ["ASPECT_DEG","ASPECT"], P100)
    albedo = read_to_grid(albs[0], ["BSA_SHORTWAVE"], P100)
    lc100 = read_to_grid(lcfile, ["LANDCOVER"], P100, 1, Resampling.nearest)
    lc100 = np.rint(lc100).astype("float32")
    return dict(landcover=lc100, dem=dem, albedo_bsa=albedo, ndvi=ndvi, slope=slope, aspect=aspect, ui=ui, mndwi=mndwi)

def build_matrices(s100):
    s1 = {}
    for k,v in s100.items():
        if k == "landcover":
            # nearest-like parent label from block centre; source is categorical.
            s1[k] = v.reshape(H1,10,W1,10)[:,5,:,5].astype("float32")
        else:
            s1[k] = block_mean_100_to_1(v)
    classes = sorted(set(np.unique(s1["landcover"]).astype(int)) | set(np.unique(s100["landcover"]).astype(int)))
    cols1, cols100 = [], []
    for name in FEATURES:
        a1, a100 = s1[name], s100[name]
        if name == "landcover":
            for cls in classes:
                cols1.append((a1.reshape(-1,1)==cls).astype("float32"))
                cols100.append((a100.reshape(-1,1)==cls).astype("float32"))
        else:
            cols1.append(a1.reshape(-1,1).astype("float32"))
            cols100.append(a100.reshape(-1,1).astype("float32"))
    return s1, np.concatenate(cols1,axis=1), np.concatenate(cols100,axis=1)

def projected_xy(profile):
    lon, lat = lonlat_grid(profile)
    tr = Transformer.from_crs("EPSG:4326","EPSG:32647",always_xy=True)
    x,y = tr.transform(lon,lat)
    return np.column_stack([np.asarray(x).reshape(-1),np.asarray(y).reshape(-1)])

def write_tif(path, a, profile):
    path.parent.mkdir(parents=True,exist_ok=True)
    p = dict(driver="GTiff",height=profile["height"],width=profile["width"],count=1,
             dtype="float32",crs=profile["crs"],transform=profile["transform"],
             nodata=-9999.0,compress="deflate",predictor=3)
    with rasterio.open(path,"w",**p) as dst:
        dst.write(np.where(np.isfinite(a),a,-9999).astype("float32"),1)

def metrics(ref,pred):
    m=np.isfinite(ref)&np.isfinite(pred)
    y=ref[m].astype("float64"); p=pred[m].astype("float64")
    e=p-y; sst=np.sum((y-y.mean())**2); sse=np.sum(e**2)
    return dict(n=int(y.size),r2=float(1-sse/sst),rmse=float(np.sqrt(np.mean(e**2))),
                mae=float(np.mean(np.abs(e))),bias=float(np.mean(e)),
                pearson_r=float(np.corrcoef(y,p)[0,1]))

def anomaly_r(ref,pred):
    ys=[]; ps=[]
    for r in range(H1):
        for c in range(W1):
            y=ref[r*10:(r+1)*10,c*10:(c+1)*10]
            p=pred[r*10:(r+1)*10,c*10:(c+1)*10]
            m=np.isfinite(y)&np.isfinite(p)
            if m.sum()<3: continue
            yy=y[m]; pp=p[m]
            ys.append(yy-yy.mean()); ps.append(pp-pp.mean())
    if not ys: return float("nan")
    y=np.concatenate(ys); p=np.concatenate(ps)
    return float(np.corrcoef(y,p)[0,1])

def landsat_ref(root: Path):
    p = find_one(root, "landsat_target_lst_qa")
    with rasterio.open(p) as src:
        ds=descs(src)
        li = next((i for i,d in enumerate(ds,1) if "LST" in d and "QA" not in d),1)
        qi = next((i for i,d in enumerate(ds,1) if "QA_PIXEL" in d or d=="QA"),2 if src.count>=2 else None)
        lst = src.read(li,masked=True).astype("float32").filled(np.nan)
        if np.nanmedian(lst)>150: lst=lst-273.15
        if qi is not None:
            qa=src.read(qi,masked=True).filled(65535).astype("uint16")
            clear=(qa & np.uint16(0b111111))==0
            lst[~clear]=np.nan
        pp=tif_profile(src)
    return reproj(lst,pp,P100,Resampling.average).astype("float32")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--date",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--workers",type=int,default=int(os.getenv("SA_WORKERS","2")))
    args=ap.parse_args()
    root=Path(args.data_root); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    date=args.date

    s100=surface_stack(root)
    s1,X1_static,X100_static=build_matrices(s100)

    anc=hourly_files(root,"ANCFDS_FY4A")
    parents={}
    for h,p in anc.items():
        a,sp,_=read_band(p,["T_NADIR_C","T_NADIR"],2)
        med=float(np.nanmedian(a))
        if med>1000: a=a*0.1-273.15
        elif med>150: a=a-273.15
        a=fill_nearest(reproj(a,sp,P1,Resampling.bilinear))
        parents[h]=apply_c3(a,s1["landcover"])

    era=hourly_files(root,"ERA5LAND")
    dynamic={}; rows=[]; daylight=[]
    for h in range(24):
        sw=read_to_grid(era[h],["SWDOWN_WM2","SWDOWN"],P100,None,Resampling.bilinear)
        when=datetime.fromisoformat(f"{date}T{h:02d}:00:00+00:00").astimezone(timezone.utc)
        # avoid terrain work at night
        _,alt,_=solar_geometry(P100,when)
        mean_alt=float(np.nanmean(np.rad2deg(alt)))
        if mean_alt<=0:
            rows.append({"hour_utc":h,"mean_solar_altitude_deg":mean_alt,"is_daylight":False})
            continue
        swa,mean_alt=sw_abs_100(s100,sw,when)
        if float(np.nanmean(swa))<=1:
            rows.append({"hour_utc":h,"mean_solar_altitude_deg":mean_alt,"is_daylight":False})
            continue
        dynamic[h]={"sw_abs_100m":fill_nearest(swa),"sw_abs_1km":block_mean_100_to_1(swa)}
        daylight.append(h)
        rows.append({"hour_utc":h,"mean_solar_altitude_deg":mean_alt,
                     "mean_sw_abs_wm2":float(np.nanmean(swa)),"is_daylight":True})
    if TARGET_HOUR not in daylight:
        raise RuntimeError(f"Target hour {TARGET_HOUR} is not daylight: {daylight}")
    z100=np.zeros((H100,W100),dtype="float32"); z1=np.zeros((H1,W1),dtype="float32")
    for h in daylight:
        dynamic[h]["lag_100m"]=dynamic[h-1]["sw_abs_100m"] if h-1 in dynamic else z100
        dynamic[h]["lag_1km"]=dynamic[h-1]["sw_abs_1km"] if h-1 in dynamic else z1
    pd.DataFrame(rows).to_csv(out/"daylight_hours.csv",index=False)

    def mat(base,h,scale):
        a=dynamic[h][f"sw_abs_{scale}"].reshape(-1,1)
        l=dynamic[h][f"lag_{scale}"].reshape(-1,1)
        return np.concatenate([base,a,l],axis=1).astype("float32")
    X1h={h:mat(X1_static,h,"1km") for h in daylight}
    y={h:parents[h].reshape(-1).astype("float32") for h in daylight}
    ystack=np.stack([y[h] for h in daylight])
    valid=np.isfinite(ystack).all(axis=0)&np.isfinite(X1_static).all(axis=1)
    centers=np.flatnonzero(valid)
    if len(centers)<=KNN+1: raise RuntimeError("Insufficient centres")

    coords=projected_xy(P1); cc=coords[centers]; tree=cKDTree(cc)
    d,n=tree.query(cc,k=KNN+1)
    d=d[:,1:]; ng=centers[n[:,1:]]; bw=d[:,-1]

    poolX=np.concatenate([X1h[h] for h in daylight])
    pooly=np.concatenate([y[h] for h in daylight])
    vg=np.isfinite(pooly)&np.isfinite(poolX).all(axis=1)
    gm=XGBRegressor(**{**XGB,"n_jobs":-1})
    gm.fit(poolX[vg],pooly[vg])
    X100t=mat(X100_static,TARGET_HOUR,"100m")
    pg=gm.predict(X100t).reshape(H100,W100).astype("float32")

    local=np.full((H100,W100),np.nan,dtype="float32")
    otrue=np.full((len(daylight),len(centers)),np.nan,dtype="float32")
    opred=np.full_like(otrue,np.nan)
    params={**XGB,"n_jobs":1}

    def one(j):
        central=int(centers[j]); nbr=ng[j]
        q=d[j]/max(float(bw[j]),1e-9); sw=np.where(q<1,(1-q*q)**2,0).astype("float32")
        xl=np.concatenate([X1h[h][nbr] for h in daylight])
        yl=np.concatenate([y[h][nbr] for h in daylight])
        ww=np.tile(sw,len(daylight))
        v=np.isfinite(yl)&np.isfinite(xl).all(axis=1)&(ww>0)
        model=XGBRegressor(**params); model.fit(xl[v],yl[v],sample_weight=ww[v])
        xc=np.stack([X1h[h][central] for h in daylight])
        po=model.predict(xc).astype("float32")
        r,c=divmod(central,W1)
        child=[]
        for rr in range(r*10,(r+1)*10):
            child.extend(range(rr*W100+c*10,rr*W100+(c+1)*10))
        pc=model.predict(X100t[np.asarray(child,dtype="int64")]).reshape(10,10).astype("float32")
        return j,r,c,np.asarray([y[h][central] for h in daylight],dtype="float32"),po,pc

    done=0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for j,r,c,yt,yp,pc in ex.map(one,range(len(centers))):
            otrue[:,j]=yt; opred[:,j]=yp
            local[r*10:(r+1)*10,c*10:(c+1)*10]=pc
            done+=1
            if done%250==0 or done==len(centers):
                print(f"SA-XGBoost local models: {done}/{len(centers)}",flush=True)
    if not np.isfinite(local).all(): raise RuntimeError("Local prediction has gaps")
    pred=(ALPHA*local+(1-ALPHA)*pg).astype("float32")

    ref=landsat_ref(root)
    met=metrics(ref,pred); met["subpixel_anomaly_r"]=anomaly_r(ref,pred)
    met.update({"date":date,"method":"SA-XGBoost","k_neighbors":KNN,"alpha":ALPHA,
                "n_local_centres":int(len(centers)),"daylight_hours":daylight,
                "c3_model_id":"ancfds_landsat_c3_24dates_v2"})
    om=metrics(otrue,opred)
    met["oob_r2"]=om["r2"]; met["oob_rmse"]=om["rmse"]

    write_tif(out/"SA_XGBoost_100m.tif",pred,P100)
    write_tif(out/"Global_XGBoost_100m.tif",pg,P100)
    write_tif(out/"ANCFDS_C3_parent_1km.tif",parents[TARGET_HOUR],P1)
    write_tif(out/"Landsat_reference_100m.tif",ref,P100)
    (out/"metrics.json").write_text(json.dumps(met,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame([met]).to_csv(out/"metrics.csv",index=False,encoding="utf-8-sig")

    validref=ref[np.isfinite(ref)]
    lo,hi=np.percentile(validref,[2,98])
    err=pred-ref
    elim=float(np.nanpercentile(np.abs(err),98))
    fig,ax=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
    im=ax[0].imshow(parents[TARGET_HOUR],vmin=lo,vmax=hi,cmap="turbo"); ax[0].set_title("ANCFDS C3 1 km")
    ax[1].imshow(ref,vmin=lo,vmax=hi,cmap="turbo"); ax[1].set_title("Landsat 100 m")
    ax[2].imshow(pred,vmin=lo,vmax=hi,cmap="turbo"); ax[2].set_title("SA-XGBoost 100 m")
    ax[3].imshow(err,vmin=-elim,vmax=elim,cmap="coolwarm"); ax[3].set_title("SA-XGBoost - Landsat")
    for a in ax: a.axis("off")
    fig.colorbar(im,ax=ax[:3],fraction=0.02,pad=0.01,label="°C")
    fig.suptitle(f"{date} | R²={met['r2']:.3f}, RMSE={met['rmse']:.3f} °C")
    fig.savefig(out/"comparison.png",dpi=220,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(json.dumps(met,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
