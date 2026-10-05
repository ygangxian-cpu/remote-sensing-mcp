# 2019-09-24 长波辐射–LST 日变化诊断

分支：`experiment/lw-lst-diurnal-correlation`

目的：只验证动态辐射量与 ANCFDS `T_nadir` 的关系，不修改 MCP 正式下载接口，也不修改降尺度主流程。

## 数据

全部直接复用 MCP 仓库已有缓存，不重新下载：

- ERA5-Land 24 h：`SWDOWN_WM2`、`GLW_WM2`
- ANCFDS 24 h：band 2 `T_nadir`
- 100 m Landsat scaling factors：NDVI
- 100 m SRTM terrain：DEM

## 比较量

1. `SWDOWN`
2. `GLW`
3. `LW_abs = emissivity(NDVI) * GLW`
4. `LW_abs_SVF_sky = emissivity(NDVI) * SVF(DEM) * GLW`

第 4 项严格称为“SVF 调制的天空下行长波吸收代理”，不是完整 terrain longwave balance，因为实验不假设周围坡面的温度，也不使用目标 LST 构造向上长波。

## 验证

- 24 h 区域均值曲线；
- 全天 / 白天 / 夜间 Pearson r；
- 0–6 h lag correlation：比较辐射在 t-lag 与 LST(t) 的关系；
- 逐小时 1 km 空间相关；
- 夜间由数据定义：ROI mean `SWDOWN_WM2 <= 1 W/m²`。

## 本地运行

```bash
python experiments/lw_lst_diurnal_correlation.py
```

依赖：`numpy pandas rasterio matplotlib`。

输出：

```text
output/experiments/lw_lst_diurnal_correlation/
├── hourly_curve.csv
├── spatial_correlations.csv
├── summary.json
├── diurnal_curve_radiation_vs_lst.png
└── temporal_correlation_day_night.png
```

本实验只用于决定是否值得把长波动态因子带入 24 h SA-XGBoost；在验证完成前，不改正式模型合同。
