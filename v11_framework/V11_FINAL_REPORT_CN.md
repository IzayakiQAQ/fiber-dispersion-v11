# V11 色散补偿模型整理报告

## 1. 路线定位

V11 不再继续把 V10 no-harm 模型作为主模型追，而是以已经验证有效的 V1.65 强补偿模型作为 teacher/baseline，再补充峰形合理性约束。

当前主线为：

```text
单张居中直方图
  -> V1.65 strong teacher
  -> Gaussian width guard
  -> 补偿后直方图
```

推理时只需要一张直方图，不使用序列输入，不使用 Mamba。

## 2. 为什么这样做

V1.65 在同一 V10 centered test split 上已经表现出强补偿能力：

```text
50km / 0.8nm input TDEV@10s: 32.890 ps
V1.65 teacher TDEV@10s:      2.816 ps
```

但它存在峰形过窄的问题：

```text
V1.65 teacher FWHM: 35.027
0km target FWHM:    48.736
teacher / target:   0.719
```

这在论文中容易被质疑为“过补偿”或“只追求 TDEV，牺牲了峰形物理合理性”。因此 V11 的目标是：

```text
保持 V1.65 的时间稳定性补偿能力，同时把峰宽恢复到 0km target 附近。
```

## 3. 数据和评估设置

- 数据来源：V10 centered 1024-pair tensor manifest
- 场景：50km / 0.8nm
- 参考：0km / 0.8nm
- split：blocked non-overlap
- train：pair 1-512
- val：pair 513-768
- test：pair 769-1024
- 评价中心：`window_centroid`
- TDEV：TDEV@10s
- 评估时使用 `shifts.csv` 恢复整体 shift

主配置：

```text
v11_framework/configs/v11_50km_0p8_teacher_width_guard.yaml
```

主输出：

```text
v11_framework/artifacts/v11_50km_0p8_teacher_width_guard_v1
```

## 4. 主结果

| split | input TDEV ps | teacher TDEV ps | V11 TDEV ps | target TDEV ps | teacher FWHM | V11 FWHM | target FWHM | V11/target FWHM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 32.625 | 2.604 | 2.599 | 1.181 | 35.094 | 51.582 | 48.795 | 1.068 |
| test | 32.890 | 2.816 | 2.813 | 1.110 | 35.027 | 51.613 | 48.736 | 1.073 |

V11 的 test TDEV 仍维持在 V1.65 teacher 级别：

```text
teacher: 2.816 ps
V11:     2.813 ps
```

同时峰宽从过窄状态修正到接近 0km target：

```text
teacher FWHM: 35.027
V11 FWHM:     51.613
target FWHM:  48.736
```

## 5. 可用于论文的表述

建议将当前方法表述为：

```text
physics-guided teacher compensation with target-aware peak-width guard
```

中文可以写为：

```text
基于物理先验 teacher 的目标峰宽约束色散补偿方法
```

核心创新点可以组织为：

1. 以强物理补偿 teacher 提供色散压缩先验，避免从零训练导致补偿幅度不足。
2. 引入目标峰宽约束，使补偿结果同时满足时间稳定性与峰形物理合理性。
3. 使用 shift-restored TDEV 和 FWHM 双指标评价，避免只用训练集重建误差说明补偿效果。

## 6. 负结果与边界

尝试过单直方图 shift correction head：

```text
v11_framework/V11_SHIFT_HEAD_EXPERIMENT_CN.md
```

普通 CNN shift head 和 pseudo residual shift head 均未优于 V11 width guard：

| 模型 | val TDEV ps | test TDEV ps | test FWHM |
|---|---:|---:|---:|
| V11 width guard | 2.599 | 2.813 | 51.613 |
| neural shift head | 2.606 | 2.818 | 51.625 |
| pseudo shift head | 2.615 | 2.819 | 51.615 |

因此当前主模型不采用 shift head。

额外诊断显示，手工峰形特征的 ridge 校正在 train+val -> test 设置下可把 test TDEV 诊断性降到约 2.09ps，但该结果未作为最终模型，因为还需要更严格的 test 隔离和可复现实验设计。

## 7. 当前结论

当前 V11 已经能支撑阶段性论文/汇报结论：

```text
50km / 0.8nm 色散展宽输入经过 V11 补偿后，
TDEV@10s 由 32.890 ps 降至 2.813 ps，
峰宽由 224.166 压缩并恢复到接近 0km target 的 51.613，
实现了时间稳定性提升与峰形合理性约束的同步满足。
```

如果后续继续冲击 2ps 以下，建议另开 V12 或 V11.5，采用显式峰形特征校准的 residual timing correction，而不是继续堆自由 CNN shift head。
