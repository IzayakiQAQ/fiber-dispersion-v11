# Fiber Dispersion Compensation V25

当前发布版为 [`v25_framework`](v25_framework/)。仓库当前版本只保留技术原理、Python 实现和性质测试，不包含实验数据、结果曲线、训练权重、冻结模型或论文过程材料。

V25 使用独立校准的物理参数生成两个传播方向的展宽 PSF 与 0 km 目标 PSF，并在配置冻结后对每幅 coincidence histogram 独立执行 Richardson-Lucy 解卷积和目标响应重建。

核心约束：

- 一幅输入直方图对应一幅补偿直方图；
- 评估数据不参与 PSF 或超参数构造；
- 不使用相邻直方图、时序滤波或 bounded center correction；
- 输出非负且计数守恒；
- 冻结配置使用 SHA-256 校验。

安装与使用：

```powershell
python -m pip install -r .\requirements.txt
python -m pytest .\v25_framework\tests -q
```

完整原理、冻结命令和推理 API 见 [`v25_framework/README.md`](v25_framework/README.md)。
