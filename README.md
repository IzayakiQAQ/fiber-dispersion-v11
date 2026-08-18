# Fiber Dispersion Compensation V25

当前唯一发布版为 [`v25_framework`](v25_framework/)。V25 是 physics-informed Neural-PSF 色散补偿框架：物理模型提供可外推的 PSF 基线，小型 MLP 使用独立实验条件学习受限宽度残差，再由冻结的方向 PSF 完成 Richardson-Lucy 解卷积和 0 km 响应重建。

核心约束：

- 神经网络输入只有长度、带宽和方向，不读取待评估直方图；
- 异常实验条件通过探测器宽度下限和物理一致性门控排除；
- 网络超出实验覆盖范围时连续退回物理模型；
- 一幅输入直方图对应一幅非负、计数守恒的补偿直方图；
- 不使用相邻样本、序列滤波或 bounded center correction；
- 训练权重和部署配置分别使用 SHA-256 锁定；
- Git 只包含技术原理、代码和合成性质测试，不包含数据、权重或结果。

```powershell
python -m pip install -r .\requirements.txt
python -m pytest .\v25_framework\tests -q
```

训练、冻结、单直方图推理和 1000 组盲评命令见 [`v25_framework/README.md`](v25_framework/README.md)。
