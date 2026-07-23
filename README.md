# Fiber Dispersion Compensation

本仓库用于单直方图光纤色散补偿与论文结果复现。当前正式发布版本是
**v24 无状态 Richardson-Lucy 直方图补偿器**。

## 当前版本

新项目请直接使用 [`v24_framework`](v24_framework/)。

```text
one raw histogram
  -> Gaussian coarse localization and 2049-bin crop
  -> direction-specific Richardson-Lucy deconvolution
  -> direction-specific physical 0 km target response
  -> one nonnegative, count-preserving compensated histogram
```

最终 v24：

- 每次只输入一张直方图，不使用相邻样本或整段钟差序列。
- 两个传播方向分别使用独立的 broad PSF 和 target PSF。
- Richardson-Lucy 固定为 `512` 次迭代。
- 每侧 `160 bins` 估计背景。
- 输出中心在补偿后局部峰 `+/-180 bins` 内做背景扣除质心。
- 不执行 bounded center correction。
- 不使用旧 V17 的 `eta=0.67`、`blend_scale=1.2`、
  `clip_fraction=0.095`。

仓库已包含最终双方向模型，因此克隆后安装 `numpy` 和 `scipy` 即可推理：

```powershell
python -m pip install -r .\v24_framework\requirements.txt

python .\v24_framework\run_inference.py input.csv `
  --direction 1 `
  --output-csv output_v24.csv
```

Python API：

```python
from v24_framework import V24Compensator

operator = V24Compensator()
compensated = operator.infer_local(raw_local_histogram, direction=1)
```

完整横轴直方图可使用 `operator.infer_full(...)`，程序会独立定位主峰并截取
2049-bin 模型窗口。

详细参数、输入格式、重新校准和验证方法见
[`v24_framework/README.md`](v24_framework/README.md) 与
[`v24_framework/MODEL_CARD.md`](v24_framework/MODEL_CARD.md)。

## 锁定结果

外部 1000 组 `50 km / 280 Hz` 评估：

| 指标 | 补偿前 | v24 输出 |
|---|---:|---:|
| 全部1000组 TDEV@10 s | 4.098 ps | 2.380 ps |
| 严格留出501-1000组 TDEV@10 s | 4.036 ps | 2.485 ps |
| 中位 FWHM | 506.0 ps | 174.1 ps |
| 宽度压缩 | 1.00x | 2.91x |
| 稳定性提升 | 1.00x | 1.72x |

独立外部数据上的 `1.6 ps` 目标尚未达到，因此本仓库不作该项声明。

## 仓库结构

```text
v24_framework/  # 当前完整发布：代码、模型、CLI、测试与重校准脚本
v17_framework/  # 论文开发过程、旧 V17 基线及审计材料
v11_framework/  # V11 teacher-guided 神经网络基线
```

## 验证

```powershell
python .\v24_framework\verify_release.py
python -m pytest .\v24_framework\tests -q
```

## 数据策略

Git 包含 v24 推理所需的小型锁定模型和物理目标 PSF。原始实验直方图、
1000组派生输出、论文图片及大型结果包不上传；重新校准时通过命令行提供
本地实验数据路径。
