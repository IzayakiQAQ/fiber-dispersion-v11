# Fiber Dispersion Compensation V17

本仓库整理用于论文写作和复现实验的色散补偿模型。当前主线是 **V17 单直方图自标定似然读数器**：

```text
single histogram
  -> quality Gaussian initial center
  -> estimate effective broadening from peak width
  -> select matched training template
  -> Poisson/Fisher score center correction
  -> corrected t1, t2, t0
```

核心目标不是依赖长时间序列做后处理，而是在实验时输入单张符合峰直方图，直接得到补偿后的中心读数。长时间数据只用于训练、验证和评估 TDEV。

## Current Main Result

外部测试数据：

```text
E:\lzy\测试结果\2026.6.29 50km 280Hz
```

这批 1000 张直方图没有参与 V10/V11/V15/V17 模板训练。

| Method | TDEV@10s | Clock std | Notes |
|---|---:|---:|---|
| Raw quality Gaussian center | 4.098 ps | 4.163 ps | 原始宽峰中心读数 |
| V16 fixed diagnostic reader | 2.346 ps | 2.442 ps | 外部诊断参数，非严格泛化 |
| V17 self-calibrated reader | 2.340 ps | 2.450 ps | 单图估计等效展宽并自动定参 |

V17 在该外部数据上自动选择：

| Quantity | Value |
|---|---:|
| selected labels | `d025km_bw0p8nm`, `d050km_bw0p8nm` |
| selected smooth | `100 ps`, `120 ps` |
| selected background | `0.001` |
| mean input Gaussian FWHM | `505.97 ps` |
| mean selected template FWHM | `353.90 ps` |
| mean automatic blend | `1.716` |
| mean automatic clip | `48.07 ps` |

Main report:

```text
v17_framework/results/V17_SELF_CALIBRATED_LIKELIHOOD_READER_20260629_50KM_280HZ_RESULT_CN.md
```

Main summary:

```text
v17_framework/results/v17_self_calibrated_20260629_50km_280hz/summary.json
```

## Scientific Motivation

50 km 色散会显著展宽符合峰。宽峰不会只改变 FWHM，也会降低单次中心读数精度，因此 TDEV 变差。单纯把峰形压窄并不一定降低 TDEV，因为如果补偿网络引入中心偏移噪声，稳定性反而会变差。

V17 的基本判断是：

1. 直方图峰形包含等效色散或等效展宽状态。
2. 训练数据中的不同距离/带宽样本可以提供峰形模板先验。
3. 中心读数应由物理似然给出，而不是由神经网络自由回归。
4. 神经网络或自标定模块应负责选择模板、展宽和修正强度，而不是直接任意移动中心。

因此，V17 把问题写成：

```text
histogram shape -> effective broadening state -> likelihood reader parameters -> bounded Fisher correction
```

## V17 Method

V17 使用 V10 预处理后的居中训练直方图构建模板库。默认模板来源：

```text
E:\lzy\测试结果\补偿数据
```

默认模板标签：

```text
d025km_bw0p8nm
d050km_bw0p8nm
d075km_bw0p8nm
d100km_bw0p8nm
```

每个模板会生成多组候选：

```text
smooth = 20, 40, 60, 80, 100, 120, 160, 200 ps
background = 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01
direction = 1, 2
```

### Single-Histogram Self Calibration

> 本节描述旧 V17 likelihood/Fisher-score 中心读数基线，不是最终 RL 直方图补偿算子。

对每张输入直方图：

1. 使用已有窄窗口 Gaussian 质量读数作为初始中心。
2. 截取初始中心附近局部窗口。
3. 从输入峰宽估计等效模板宽度：

```text
target_template_fwhm = 0.67 * input_fwhm
```

4. 用宽度一致性、背景先验和 Poisson likelihood 选择候选模板。
5. 用模板导数计算 Fisher score 的一阶中心修正。
6. 根据输入峰宽和模板峰宽自动确定修正强度：

```text
blend = 1.2 * input_fwhm / selected_template_fwhm
clip  = 0.095 * input_fwhm
```

7. 输出修正后的 `t1`, `t2`, `t0`。

V17 不使用随机 sub-bin shift augmentation。

## Repository Layout

```text
v17_framework/
  build_template_bank.py        # Build template bank from V10 centered tensors
  likelihood_reader.py          # Template selection and Poisson/Fisher readout
  run_external_50km_280hz.py    # External 50km/280Hz inference entry
  v17_common.py                 # Shared IO, TDEV, FWHM, CSV utilities
  README.md                     # V17 short technical note
  results/
    V17_SELF_CALIBRATED_LIKELIHOOD_READER_20260629_50KM_280HZ_RESULT_CN.md
    v17_self_calibrated_20260629_50km_280hz/
      summary.json
      time_t1_t2_t0_four_columns.csv      # ignored by git, generated locally

v11_framework/
  Legacy V11 teacher-guided width-guard baseline and reports.
```

Large local data, checkpoints, tensors, CSV outputs and figures are intentionally ignored by git.

## Reproduce

Use a Python environment with `numpy`, `scipy`, `matplotlib`, and `torch`.

Build the V17 template bank:

```powershell
python .\v17_framework\build_template_bank.py
```

Run the external 50 km / 280 Hz evaluation:

```powershell
python .\v17_framework\run_external_50km_280hz.py
```

Optional: use another fixed-axis histogram dataset:

```powershell
python .\v17_framework\run_external_50km_280hz.py `
  --source-root "E:\lzy\测试结果\your_external_dataset" `
  --output-dir ".\v17_framework\results\your_output_dir"
```

The output directory contains:

```text
raw_time_t1_t2_t0_quality.csv
time_t1_t2_t0_four_columns.csv
pair_detail.csv
processed_example_histogram_00001.png
summary.json
```

## Paper-Oriented Description

可在论文中表述为：

> We propose a self-calibrated physics-constrained likelihood reader for single-histogram dispersion compensation. Instead of directly regressing the time center, the method estimates an effective broadening state from the measured coincidence peak, selects a matched template from training data, and computes a bounded Poisson/Fisher score correction. This design preserves center information while improving the statistical efficiency of the center estimator.

中文表述：

> 本文提出一种单直方图自标定的物理约束似然读数器。该方法不直接由神经网络自由回归时间中心，而是根据符合峰宽度估计等效色散展宽状态，在训练模板库中选择匹配的峰形模板，并通过 Poisson/Fisher score 给出受限中心修正，从而在保持中心信息的同时降低宽峰条件下的中心读数方差。

## What This Result Supports

V17 结果支持以下结论：

1. 宽峰导致的中心读数方差确实是 50 km / 280 Hz 数据 TDEV 偏大的主要来源之一。
2. 单张直方图包含可用于估计等效展宽状态的信息。
3. 模板似然读数器可以在不使用时间序列后处理的情况下把 TDEV@10s 从 `4.10 ps` 降到 `2.34 ps`。
4. 只压窄峰形是不够的，必须同时优化中心读数统计量。

## Legacy V17 Parameter Status

以下常数只用于旧 V17 likelihood/Fisher-score 中心修正：

```text
target_template_fraction = 0.67
blend_scale = 1.2
clip_fraction = 0.095
```

`0.67` 选择目标模板宽度，`1.2` 放大 Fisher-score 位移，`0.095` 限制该位移。最终 RL 管线不使用这三个参数；它们不参与 broad-PSF 构造、RL 更新、目标 PSF 卷积或输出中心读取。论文中若保留，应放入 legacy baseline 或 Supplement，而不是最终方法参数表。

## Legacy V11

`v11_framework` 保留早期 V11 teacher-guided width-guard baseline。它用于说明纯神经网络补偿峰形与中心稳定性之间的矛盾，也可作为论文中的 negative/legacy baseline。
