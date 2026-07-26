# Fiber Dispersion Compensation

本仓库用于量子双向时间同步中的单直方图光纤色散补偿与论文结果复现。当前正式版本为
[`v24_framework`](v24_framework/)；`v17_framework` 和 `v11_framework` 保留用于审计历史方法。

## V24 锁定基线

锁定基线对每幅直方图独立执行：

```text
原始直方图
  -> Gaussian 粗定位与 2049-bin 截取
  -> 方向独立 Richardson-Lucy 反卷积
  -> 方向独立 0 km 物理目标响应卷积
  -> 非负、计数守恒的补偿直方图
```

它不使用相邻直方图、整段钟差序列或 bounded center correction。外部 1000 组
`50 km / 280 Hz` 锁定结果为：FWHM 中位数从 506.0 ps 降至 174.1 ps，完整序列
TDEV@10 s 从 4.098 ps 降至 2.380 ps。独立的 1.6 ps 目标尚未达到，因此不作该项声明。

```powershell
python -m pip install -r .\v24_framework\requirements.txt
python .\v24_framework\run_inference.py input.csv `
  --direction 1 `
  --output-csv output_v24.csv
```

## Physics-Informed 扩展

v24 现包含一个可选的 physics-informed 扩展，用于从实测条件校准物理响应流形，并在
推理时仅根据当前一幅直方图选择有效补偿响应。模型包含：

- C46 连续泵浦、PPLN 级联 SHG/type-0 SPDC；
- C57/C35 能量反关联光子对；
- 两个标称 Gaussian WSS 强度滤波器；
- 普通单模光纤二/三阶谱相位；
- 方向相关等效探测/时间标记 IRF；
- Poisson 符合计数与边缘背景。

校准直接读取 1 ps/bin、10 s 积分的直方图，不需要原始事件时间戳：

```powershell
python .\v24_framework\run_physics_calibration.py `
  --dataset-root "E:\lzy\测试结果\补偿数据" `
  --calibration-layout channel_subdirectories `
  --calibration-layout pair_subdirectories `
  --holdout-length-km 125 `
  --holdout-bandwidth-nm 10
```

然后对单幅完整横坐标直方图推理：

```powershell
python .\v24_framework\run_physics_inference.py input.csv `
  --direction 1 `
  --calibration-json .\v24_framework\results\physics_informed_calibration\physics_calibration.json `
  --output-csv output_physics_v24.csv
```

数据布局会作为响应族单独审计，不能通过统一物理模型的旧处理布局不会被静默混入标定。
超出时间窗的物理候选会被拒绝；低计数、形状失配或理想 Fisher 增益不足时，no-harm gate
返回原直方图。

详细公式、边界和验证规则见
[`v24_framework/PHYSICS_INFORMED.md`](v24_framework/PHYSICS_INFORMED.md)。

## 验证

```powershell
python .\v24_framework\verify_release.py
python -m pytest .\v24_framework\tests -q
```

Git 只保存代码、小型锁定模型和文档。原始实验直方图、派生结果、图像和大型标定输出均不上传。
