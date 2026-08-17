# V25 Frozen-Physics Dispersion Compensation

V25 是单一、无状态的 coincidence-histogram 色散补偿实现。仓库只发布原理与代码；实验直方图、校准结果、冻结配置、输出曲线和模型数据均由使用者在本地生成，不进入 Git。

## 技术栈

- Python 3.10+
- NumPy：数组、FFT 频率网格与数值计算
- SciPy：Gaussian IRF、FFT 卷积和 Richardson-Lucy 更新
- pytest：性质测试

## 唯一算法流程

```text
独立校准的物理参数
  -> 冻结链路长度、WSS 带宽、方向 IRF 与算法常量
  -> 为两个传播方向分别生成 broad PSF 和 0 km target PSF
  -> SHA-256 锁定配置

当前单张直方图
  -> 平滑粗定位与固定 2049-bin 局部窗口
  -> 边缘背景估计
  -> 使用冻结 broad PSF 的 Richardson-Lucy 解卷积
  -> 与冻结 0 km target PSF 重卷积
  -> 非负、计数守恒的补偿直方图
  -> 从补偿直方图直接计算本次中心
```

评估直方图只作为当前输入，不参与 PSF 构造、迭代数选择或中心残差学习。算法不使用相邻直方图、钟差序列滤波、bounded correction 或同一评估数据的经验模板。

## 物理模型

代码中的确定性前向模型包含：

- CW C46 泵浦和能量反关联 C57/C35 双光子谱；
- 标称 Gaussian WSS 强度滤波；
- 普通单模光纤的二阶色散及可选三阶项；
- 两个传播方向独立的等效探测/时间标记 IRF；
- 由 `L=0` 生成的方向相关目标响应。

Fisher 信息只用于检查冻结目标响应相对展宽响应是否具有更高的平移信息量，不用于根据测试直方图修正输出。

## 安装

```powershell
python -m pip install -r .\v25_framework\requirements.txt
```

## 冻结配置

独立校准 JSON 可以直接包含物理参数，也可以使用 `{"parameters": {...}}`。冻结命令没有评估数据入口：

```powershell
python -m v25_framework.freeze `
  --calibration-json .\independent_calibration\physics_parameters.json `
  --length-km 50 `
  --bandwidth-nm 0.8 `
  --iterations 512 `
  --output .\v25_framework\frozen\v25_50km_0p8nm.json
```

命令同时生成 `.sha256` 文件。推理默认拒绝缺失或不匹配的哈希。

## 单直方图推理

输入 CSV 支持一列 `count`，或两列 `time_ps,count`：

```powershell
python -m v25_framework.run_inference .\input.csv `
  --frozen-config .\v25_framework\frozen\v25_50km_0p8nm.json `
  --direction 1 `
  --output-csv .\v25_framework\outputs\compensated.csv
```

Python API：

```python
from v25_framework import V25Compensator

operator = V25Compensator.from_frozen_json("frozen_config.json")
result = operator.infer_full(histogram, direction=1, time_ps=absolute_axis_ps)

compensated_histogram = result.compensated
compensated_axis_ps = result.time_ps
single_center_ps = result.center_ps
```

`infer_full` 返回定位后的局部补偿直方图；`infer_local` 接受已经截取的固定长度局部直方图。两个接口均为纯单样本推理。

## 代码结构

```text
config.py          冻结配置、严格字段校验和 SHA-256
physics.py         双光子谱、光纤色散、IRF 与方向 PSF
compensator.py     定位、RL 解卷积、目标重建和中心估计
freeze.py          独立物理参数冻结入口
run_inference.py   单直方图 CSV 推理入口
tests/             非负性、计数守恒、收窄和哈希测试
```

## 验证

```powershell
python -m pytest .\v25_framework\tests -q
```

测试只使用代码生成的合成概率分布，不包含实验数据。
