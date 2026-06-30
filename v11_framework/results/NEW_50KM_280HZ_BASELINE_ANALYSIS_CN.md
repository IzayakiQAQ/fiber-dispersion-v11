# 2026-06-29 50km / 280Hz 原始对照数据分析

## 1. 数据定位

数据目录：

```text
E:\lzy\测试结果\2026.6.29 50km 280Hz
```

当前可直接分析的导出内容：

| pair | histogram count | sample period | duration |
|---|---:|---:|---:|
| ch3_ch1 | 1000 | 10 s | 10000 s = 2.78 h |
| ch4_ch2 | 1000 | 10 s | 10000 s = 2.78 h |

这批数据适合作为论文中的 **50km 未介入 DCM 原始对照**。它的核心价值不是证明算法输出，而是给出同距离、同滤波带宽下的未补偿色散展宽基线。

## 2. 原始峰形与稳定性

由 `singlepeak_peak_quality_gaussian.csv` 统计得到：

| pair | total rate | Gaussian FWHM | center TDEV@10s | center drift, last 10% - first 10% |
|---|---:|---:|---:|---:|
| ch3_ch1 | 331.25 Hz | 502.58 ps | 5.640 ps | -114.16 ps |
| ch4_ch2 | 297.42 Hz | 509.37 ps | 5.691 ps | -108.52 ps |
| pair mean | 314.34 Hz | 505.97 ps | 5.666 ps | -111.34 ps |

结论：

1. 这批数据的总符合率约 300 Hz 量级，符合“280Hz 原始对照”的定位。
2. 原始 50km 峰宽约 506 ps，显著宽于 DCM 补偿后的约 55.6 ps，也宽于 V11 输出的约 51.6 ps。
3. 单峰中心 TDEV@10s 约 5.67 ps，可作为未补偿中心稳定性基线。

## 3. 时间门内有效符合率

直接比较 full-window count rate 容易误导，因为宽峰会把相关符合摊到很宽的时间范围。更适合论文的是比较峰中心附近固定时间门内的有效符合率。

从 raw histogram 重新积分得到：

| pair | ±50ps | ±100ps | ±200ps | ±500ps | FWHM window |
|---|---:|---:|---:|---:|---:|
| ch3_ch1 | 41.13 Hz | 82.22 Hz | 160.35 Hz | 238.13 Hz | 193.18 Hz |
| ch4_ch2 | 39.25 Hz | 78.41 Hz | 154.71 Hz | 232.06 Hz | 190.19 Hz |
| pair mean | 40.19 Hz | 80.32 Hz | 157.53 Hz | 235.09 Hz | 191.68 Hz |

这个结果说明：未补偿时虽然 full-window 计数约 314 Hz，但在 ±100ps 这样更接近高精度时间判定的窗口内，有效相关符合只有约 80 Hz。色散补偿的目标不是凭空增加光子数，而是把相关符合重新集中到窄时间窗口内。

## 4. 与 DCM 硬件补偿的公平性

已知 DCM 模块插入损耗为 4.45 dB，对应线性因子：

```text
10^(4.45 / 10) = 2.786
```

因此：

```text
100 Hz after DCM ~= 278.61 Hz without DCM
```

DCM 2025-07-24 数据中，±100ps 峰中心有效符合率约 115.00 Hz。若只按 4.45 dB 损耗折算，其无模块等效计数为：

```text
115.00 Hz * 2.786 ~= 320.41 Hz
```

这与新测 no-DCM 原始数据的 full-window pair mean `314.34 Hz` 非常接近。因此，2026-06-29 的 50km / 280Hz 数据可以作为 DCM 100Hz 数据的损耗折算对照，逻辑上是合理的。

需要在论文中说明：两组数据不是同一天同一次切换实验，严格公平性仍弱于“同光源、同功率、同日切换 DCM/no-DCM”的 paired measurement。但当前计数率折算已经足以作为阶段性论文对比依据。

## 5. 建议论文主对比表

| condition | role | count-rate口径 | FWHM | TDEV@10s |
|---|---|---:|---:|---:|
| 50km no-DCM, 2026-06-29 | 原始色散基线 | full-window 314.34 Hz, ±100ps 80.32 Hz | 505.97 ps | 5.666 ps |
| 50km DCM, 2025-07-24 | 硬件补偿参考 | ±100ps 115.00 Hz | 55.63 ps | clock 2.130 ps |
| 50km V11 algorithm | 算法补偿结果 | 暂不报告同批计数率 | 51.61 ps | 2.813 ps |
| 0km reference | 近似无色散目标 | 视对应实验而定 | 48.74 ps | 1.110 ps |

推荐叙述：

```text
未补偿 50km 传输使相关峰展宽至约 506 ps，导致 ±100ps 中心时间门内的有效符合率仅约 80 Hz。
硬件 DCM 将峰宽压缩至约 56 ps，并把 TDEV@10s 降至约 2.13 ps。
V11 算法补偿在不引入 DCM 插入损耗的条件下，将峰宽恢复至约 52 ps，并将 TDEV@10s 压缩至约 2.81 ps。
```

## 6. 对 V11 论文最有利的表述

这批新数据让论文对比逻辑更完整：

1. no-DCM 280Hz 数据证明 50km 色散确实造成宽峰，FWHM 约 506 ps。
2. DCM 100Hz 数据证明真实硬件补偿可以把峰压到约 56 ps，但代价是 4.45 dB 插入损耗。
3. V11 数据证明软件补偿可以达到与 DCM 接近的峰宽量级，同时避免额外硬件模块。

当前最稳妥的创新性定位是：

```text
V11 is a single-histogram, teacher-guided algorithmic dispersion compensation method.
It restores the broadened 50km temporal-correlation peak to a DCM-like and 0km-like width, while preserving picosecond-level timing stability and avoiding the insertion loss of a physical dispersion compensation module.
```

## 7. 后续若要进一步增强严谨性

最好补一个同日 paired measurement：

1. no-DCM 50km / 0.8nm，导出 1000 张 10s histogram。
2. DCM 50km / 0.8nm，保持光源、滤波、探测器设置不变，再导出 1000 张 10s histogram。
3. 对两组同时报告 full-window rate、±100ps rate、FWHM、TDEV 曲线。

如果暂时不补测，当前 2026-06-29 no-DCM 数据依然可以作为论文中的原始基线，只要在图注中明确它和 DCM 数据是损耗折算对照，而不是同日切换实验。
