# DCM 50km / 0.8nm / 100Hz 数据分析

## 1. 数据来源

目录：

```text
E:\lzy\测试结果\2025.7.24 50km 24h 0.8nm 100Hz 色散补偿模块
```

可直接分析的文件包括：

- `data.csv`：1000 个 clock correction 点。
- `0km.csv`：包含 `time, ch1-ch4, ch2-ch3, clock correction` 四列。
- `histograms_raw_ps/`：1000 张原始直方图，文件名为 `hist_raw_00001.csv` 到 `hist_raw_01000.csv`。

当前 CSV/直方图覆盖：

```text
1000 points * 10 s = 10000 s = 2.78 h
```

因此，本报告分析的是当前已经导出的 2.78h 直方图/clock 序列，不等价于完整 24h 原始 `.ttbin` 全量分析。

## 2. Clock 稳定性

`clock correction = (ch1-ch4 - ch2-ch3) / 2`

统计结果：

| 指标 | 数值 |
|---|---:|
| mean | -398.122 ps |
| std | 2.247 ps |
| min | -404.330 ps |
| median | -398.147 ps |
| max | -389.981 ps |
| TDEV@10s | 2.130 ps |

TDEV 曲线：

| tau | TDEV ps |
|---:|---:|
| 10s | 2.130 |
| 20s | 1.151 |
| 30s | 0.747 |
| 60s | 0.365 |
| 100s | 0.229 |
| 200s | 0.111 |
| 300s | 0.077 |
| 600s | 0.038 |
| 1000s | 0.023 |
| 2000s | 0.012 |
| 3000s | 0.007 |

结论：DCM 硬件补偿后，50km / 0.8nm / 约 100Hz 条件下已经达到约 2.13 ps 的 TDEV@10s。

## 3. 峰形与符合计数

直方图总积分包含全 65536 ps 窗口内的背景和随机符合，因此不能直接作为有效峰符合率。更合理的符合率看峰中心附近局部窗口。

| 指标 | mean | median | p05 | p95 |
|---|---:|---:|---:|---:|
| full-window rate | 405.24 Hz | 394.80 Hz | 344.39 Hz | 488.52 Hz |
| peak ±50ps rate | 108.30 Hz | 106.40 Hz | 89.20 Hz | 131.00 Hz |
| peak ±100ps rate | 115.00 Hz | 112.90 Hz | 94.70 Hz | 139.31 Hz |
| peak ±200ps rate | 116.01 Hz | 113.70 Hz | 95.60 Hz | 140.91 Hz |
| FWHM-window rate | 82.54 Hz | 81.70 Hz | 62.20 Hz | 104.50 Hz |

峰宽：

| 指标 | 数值 |
|---|---:|
| FWHM mean | 55.633 ps |
| FWHM std | 8.570 ps |
| FWHM median | 56.000 ps |
| FWHM p05-p95 | 41.000 - 69.000 ps |
| local RMS width mean | 61.885 ps |

结论：目录名中的 `100Hz` 与峰中心 ±50ps 到 ±200ps 的局部有效符合率一致；全窗口 405Hz 主要包含背景/随机符合，不宜直接作为有效峰计数。

## 4. 漂移与相关性

前 10% 与后 10% 的均值变化：

| 指标 | 数值 |
|---|---:|
| count rate first 10% | 351.50 Hz |
| count rate last 10% | 478.52 Hz |
| clock first 10% | -398.734 ps |
| clock last 10% | -397.501 ps |
| clock drift | +1.233 ps |

相关性：

| 相关项 | correlation |
|---|---:|
| clock vs full-window count rate | 0.122 |
| clock vs histogram center | -0.134 |

结论：在这段 2.78h 数据里，clock correction 的慢漂移约 1.23 ps，且与计数率/直方图中心的线性相关性较弱。

## 5. 与 V11 结果的论文比较意义

当前 DCM 硬件补偿结果：

```text
TDEV@10s ~= 2.13 ps
FWHM ~= 55.63 ps
peak ±100ps effective count rate ~= 115 Hz
```

V11 50km / 0.8nm test 结果：

```text
TDEV@10s ~= 2.81 ps
FWHM ~= 51.61 ps
```

可以作为论文中的硬件 DCM 参考基线：

1. DCM 是真实物理模块，时间稳定性更优，TDEV@10s 约 2.13 ps。
2. V11 是无 DCM/算法补偿路线，峰宽已经接近 DCM 和 0km target 的量级。
3. 二者可以构成“硬件补偿 vs 算法补偿”的对比：DCM 牺牲插入损耗，V11 避免额外硬件损耗但当前 TDEV 稍弱。

## 6. 后续建议

若要把这组数据作为论文强证据，建议补两件事：

1. 用完整 24h `.ttbin` 重新导出全量 clock/直方图，确认 24h TDEV 曲线。
2. 同一光强/同一计数率下，对比三组：
   - 未补偿 50km / 0.8nm
   - DCM 硬件补偿
   - V11 算法补偿

这样可以正式说明：DCM 是硬件上限参考，V11 是低损耗/无额外模块的算法替代方案。
