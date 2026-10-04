# 正式 ROI：MODIS 前一日 vs 当日 Landsat 相关性实验

- ROI：[99.86, 38.67, 100.5, 39.43]
- Landsat 日期：2019-09-24
- MODIS：v2 QA-preserving cache，分析阶段应用严格 QC。
- Landsat：v3 原始 LST+QA，30 m 晴空像元聚合至 MODIS 1 km；1 km 晴空覆盖率 >=80%。
- D-1 与 D0 使用完全相同的共同有效像元。

## D-1 vs D0

- terra: r(D-1)=0.9503, r(D0)=0.8541, Δr=+0.0962, n=6052, D-1更高=True
- aqua: r(D-1)=0.9017, r(D0)=0.9240, Δr=-0.0223, n=6028, D-1更高=False

## 全日期最高 Pearson r

- aqua: 2019-09-26 (lag=+2), r=0.9380, n=6052
- terra: 2019-09-25 (lag=+1), r=0.9538, n=6052
