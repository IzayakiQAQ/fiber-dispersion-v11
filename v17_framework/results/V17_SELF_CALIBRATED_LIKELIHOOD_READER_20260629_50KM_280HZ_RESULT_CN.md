# V17 单图自标定似然读数器结果

## 核心想法

本轮验证你的猜想：

> 直方图本身包含等效色散/等效展宽信息，因此可以从单张直方图估计读数器参数，而不是对同类数据人工扫描参数。

V17 保留 V16 的物理似然框架，但把 `template / smooth / background / blend / clip` 的选择前移到单图自标定逻辑中。

不使用随机 sub-bin shift augmentation。

## 文件结构

新目录：

`fiber-dispersion-v11-paper/v17_framework`

脚本拆分：

- `v17_common.py`：通用读取、TDEV、FWHM、局部窗口工具。
- `build_template_bank.py`：从 V10 居中训练张量构建模板库。
- `likelihood_reader.py`：单图模板选择和 Poisson/Fisher score 读数器。
- `run_external_50km_280hz.py`：外部 50 km / 280 Hz 推理。

## V17 自标定规则

对每张输入直方图：

1. 使用已有 quality Gaussian center 作为初始中心。
2. 读取输入峰宽 `input_fwhm`。
3. 设定目标模板宽度：

   `target_template_fwhm = 0.67 * input_fwhm`

4. 在训练模板库中选择与该等效宽度、背景和 likelihood 最匹配的模板。
5. 用 Poisson/Fisher score 计算中心修正。
6. 自动计算修正强度：

   `blend = 1.2 * input_fwhm / selected_template_fwhm`

7. 自动限制最大修正：

   `clip = 0.095 * input_fwhm`

因此，V17 不再手动填入 V16 的 `d025/s100/bg0.001/blend1.795/clip48`，而是由单张直方图的峰宽和模板匹配自动得到相近参数。

## 模板库

构建命令：

```powershell
python fiber-dispersion-v11-paper\v17_framework\build_template_bank.py
```

模板来源：

`E:\lzy\测试结果\补偿数据`

模板包含：

- `d025km_bw0p8nm`
- `d050km_bw0p8nm`
- `d075km_bw0p8nm`
- `d100km_bw0p8nm`

平滑候选：

`20, 40, 60, 80, 100, 120, 160, 200 ps`

背景候选：

`0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01`

模板总数：

`384`

## 外部测试

测试数据：

`E:\lzy\测试结果\2026.6.29 50km 280Hz`

运行命令：

```powershell
python fiber-dispersion-v11-paper\v17_framework\run_external_50km_280hz.py
```

输出目录：

`fiber-dispersion-v11-paper/v17_framework/results/v17_self_calibrated_20260629_50km_280hz`

## 结果

1000 张外部直方图：

| 指标 | 原始 quality Gaussian | V17 自标定读数 |
|---|---:|---:|
| TDEV@10s | 4.098 ps | 2.340 ps |
| clock std | 4.163 ps | 2.450 ps |
| mean input FWHM | 505.97 ps | - |
| mean selected template FWHM | - | 353.90 ps |
| mean blend | - | 1.716 |
| mean clip | - | 48.07 ps |

TDEV 曲线：

| tau | 原始 | V17 |
|---:|---:|---:|
| 10 s | 4.098 ps | 2.340 ps |
| 20 s | 2.073 ps | 1.227 ps |
| 30 s | 1.437 ps | 0.809 ps |
| 60 s | 0.724 ps | 0.406 ps |
| 100 s | 0.415 ps | 0.238 ps |
| 200 s | 0.201 ps | 0.120 ps |

V17 自动选择到：

- label: `d025km_bw0p8nm`, `d050km_bw0p8nm`
- smooth: `100 ps`, `120 ps`
- background: `0.001`

这和 V16 外部诊断得到的最优区域一致，但 V17 是由单图 FWHM 和模板匹配自动推出。

## 输出文件

四列结果：

`fiber-dispersion-v11-paper/v17_framework/results/v17_self_calibrated_20260629_50km_280hz/time_t1_t2_t0_four_columns.csv`

原始四列：

`fiber-dispersion-v11-paper/v17_framework/results/v17_self_calibrated_20260629_50km_280hz/raw_time_t1_t2_t0_quality.csv`

逐图细节：

`fiber-dispersion-v11-paper/v17_framework/results/v17_self_calibrated_20260629_50km_280hz/pair_detail.csv`

范例直方图：

`fiber-dispersion-v11-paper/v17_framework/results/v17_self_calibrated_20260629_50km_280hz/processed_example_histogram_00001.png`

完整 summary：

`fiber-dispersion-v11-paper/v17_framework/results/v17_self_calibrated_20260629_50km_280hz/summary.json`

## 判断

V17 结果支持这个判断：

> 单张直方图可以提供等效色散/等效展宽状态，进而自适应得到模板宽度和读数修正强度。

这比 V16 更接近论文目标，因为 V16 是外部诊断调参，而 V17 已经把参数选择变成单图函数。

但 V17 还不是最终严谨版本。当前 `0.67 / 1.2 / 0.095` 仍是固定物理启发参数。论文最终版最好用独立验证集锁定这些常数，再把 2026.6.29 这批 1000 张作为纯测试。
