# Fiber Dispersion Compensation

本仓库用于单直方图光纤色散补偿与论文结果复现。当前主线是**无状态 Richardson-Lucy（RL）直方图到直方图补偿器**：系统每次输入一张固定横轴符合直方图，模型直接输出一张非负、计数守恒的补偿直方图。

```text
one raw histogram
  -> Gaussian coarse localization and 2049-bin crop
  -> edge-background subtraction and normalization
  -> direction-specific RL deconvolution
  -> convolution with direction-specific physical 0 km target PSF
  -> restore background and total count
  -> one compensated histogram
```

推理不使用相邻直方图、整段钟差序列或运行级统计量，也不在输出后执行 bounded center correction。最终方法不是模板加权混合，也不是 V1.65/V11 神经网络输出。

## Locked Operator

| Quantity | Final value |
|---|---:|
| RL iterations | `512` |
| Background bins | each edge `160 bins` |
| RL ratio clip | `8.0` |
| Latent probability floor | `1e-8` |
| Local histogram length | `2049 bins` (`-1024` to `+1024 ps`) |
| Physical 0 km target | direction 1 `166 ps`, direction 2 `160 ps` |
| Center readout window | peak `+/-180 bins` |

两个传播方向分别保存独立的 `broad_psf` 和 `target_psf`。`broad_psf` 由校准段前500张直方图按方向独立做背景扣除、概率归一化和平均得到。

最终方向中心为：

```text
raw Gaussian coarse absolute center
  + compensated local background-subtracted center of mass
  - local-window midpoint
```

因此，最终中心既不是对补偿直方图再次做 Gaussian fit，也没有 Gaussian fit 后的 bounded correction。补偿后 FWHM 则由输出直方图的亚 bin 半高交点独立计算。

## External Result

本地外部评估使用1000组 `50 km / 280 Hz` 数据。下表数值来自实际输出直方图；实验数据和派生结果不随 Git 上传。

| Metric | Before | Direct RL output |
|---|---:|---:|
| TDEV@10 s, full 1000 | 4.098 ps | 2.380 ps |
| TDEV@10 s, held-out 501-1000 | 4.036 ps | 2.485 ps |
| Median FWHM | 506.0 ps | 174.1 ps |
| Width reduction | 1.00x | 2.91x |
| Stability improvement | 1.00x | 1.72x |

独立外部数据上的 `1.6 ps` 目标尚未达到，因此仓库不作该项声明。

论文硬件对比的条件级结果为：硬件色散补偿模块在 `100 Hz` 下 TDEV@10 s 为 `2.130 ps`，指定 ch1-ch3 直方图中位 FWHM 为 `187.1 ps`。硬件 FWHM 与 TDEV 属于相同实验条件，但不是1000组逐样本配对数据。

## 0 km Reference

`build_fig5_0km_reference.py` 按 Fig.5 相同口径导出逐峰对齐、概率归一化、平均和峰值归一化的0 km参考数据。

本地结果使用 `2026.3.5 0km 2m单边_Merged`，每个方向8640张直方图：

| Curve | Half-maximum FWHM |
|---|---:|
| Measured direction 1 | 198.67 ps |
| Measured direction 2 | 169.36 ps |
| Measured two-direction aligned average | 182.26 ps |
| Corrected physical target average | 163.74 ps |

实测0 km曲线和校正物理目标曲线在源数据中分列保存。`163.74 ps` 是算法目标，不应表述成一批新的独立实测结果。

## Repository Layout

```text
v17_framework/
  direct_histogram_compensator.py          # Core nonnegative RL operator
  public_compensated_histogram_operator.py # Minimal public inference API
  run_direct_histogram_external_1000.py    # Calibration, selection and 1000-group evaluation
  build_physical_0km_target_psf.py         # Corrected physical 0 km target builder
  build_paper_comparison_dcm_vs_direct.py  # Paper tables, curves and Fig.1-Fig.5
  build_fig5_0km_reference.py              # Fig.5-style 0 km aligned reference
  verify_final_delivery.py                 # Local model/result consistency audit
  test_direct_histogram_compensator.py     # Unit and batch-equivalence tests
  likelihood_reader.py                     # Legacy V17 Fisher-score baseline

v11_framework/
  Legacy V11 teacher-guided neural baseline.
```

## Reproduce

Use Python with `numpy`, `scipy`, `matplotlib`, and optionally CUDA-enabled `torch` for accelerated batch inference.

From the repository root:

```powershell
python .\v17_framework\run_direct_histogram_external_1000.py
python .\v17_framework\build_paper_comparison_dcm_vs_direct.py
python .\v17_framework\build_fig5_0km_reference.py
python .\v17_framework\verify_final_delivery.py
python -m pytest .\v17_framework\test_direct_histogram_compensator.py -q
```

The public deployment entry is:

```python
from pathlib import Path

from public_compensated_histogram_operator import infer_with_saved_model

compensated = infer_with_saved_model(
    raw_local_histogram,
    direction=1,
    model_path=Path("results/v24_direct_histogram_external_1000_physical_0km/direct_histogram_model.npz"),
)
```

The saved model and local result package must be supplied separately because experimental arrays are intentionally excluded from Git.

## Legacy Parameters

旧 V17 likelihood/Fisher-score 中心修正使用：

```text
target_template_fraction = 0.67
blend_scale = 1.2
clip_fraction = 0.095
```

三者分别选择目标模板宽度、放大 Fisher-score 位移、限制该位移。它们**不进入最终 RL 管线**，不参与 broad-PSF 构造、RL 更新、目标 PSF 卷积或补偿后中心读取；论文中若保留，应放入 legacy baseline 或 Supplement。

## Data Policy

Git 只保存代码和方法说明。原始/中间直方图、校准 PSF、拟合模型、CSV、NPZ、PDF、PNG及论文派生结果均保留在本机 `artifacts/` 和 `results/`，不上传到远端仓库。
