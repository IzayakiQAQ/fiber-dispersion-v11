# V25 Physics-Informed Neural PSF Compensation

V25 用神经网络补全未测长度/带宽条件下的 PSF，但不让待评估直方图参与 PSF、迭代数或参考响应构造。仓库只发布原理和代码；实验数据、训练产物、冻结权重和结果均保留在本地并由 `.gitignore` 排除。

## 方法边界

神经网络学习的是实验响应相对物理模型的受限残差：

\[
s_\theta=\exp\left[f_\theta(L,B,d)\right],\qquad
p_{\rm NPSF}(t)=\frac{1}{s_\theta}
p_{\rm phy}\left(\frac{t}{s_\theta};L,B,d\right).
\]

其中 `L` 为光纤长度、`B` 为 WSS 带宽、`d` 为传播方向。`f_theta` 是 `4-12-8-1` tanh MLP，宽度尺度被限制在 `1/1.5` 到 `1.5`。距离可信训练条件越远，网络残差越连续衰减到零，因此超出覆盖范围时自动退回物理 PSF。

网络不读取评估直方图，也不直接生成任意 2049-bin 曲线。这保证了 PSF 非负、归一化、中心不漂移，并避免把异常处理结果学习成物理规律。

## 算法流程

```text
独立实验条件集
  -> 每个长度/带宽/方向抽取代表性直方图
  -> Gaussian 宽度测量与物理一致性门控
  -> 物理 PSF 宽度作为基线
  -> MLP 学习 log(W_measured / W_physics)
  -> 整组条件留一验证
  -> 冻结网络、训练清单和 SHA-256

指定待部署条件 (L, B, direction)
  -> CW-SPDC + Gaussian WSS + SMF + IRF 生成物理 PSF
  -> 冻结 MLP 预测 PSF 残差
  -> 生成方向独立 broad PSF 与 0 km target PSF

当前单张直方图
  -> 单样本粗定位和固定 2049-bin 局部窗口
  -> Poisson 边缘背景均值
  -> 冻结 broad PSF 的 Richardson-Lucy 解卷积
  -> 冻结 0 km target PSF 重卷积
  -> 非负、计数守恒的补偿直方图
  -> 对补偿直方图执行冻结 target PSF 的 Poisson 中心拟合
```

推理不使用相邻直方图、钟差序列滤波、run-level 均值、bounded correction 或同一评估数据的经验模板。

## 物理模型

- C46 CW 泵浦与能量反关联 C57/C35 双光子谱；
- 标称 Gaussian WSS 强度滤波；
- 普通单模光纤二阶色散和可选三阶色散；
- 两个方向独立的探测器/时间标记等效 IRF；
- `L=0` 方向响应作为目标 PSF。

需要注意：输出峰形变窄会提高输出曲线的形式 Fisher 信息，但确定性后处理不能凭空增加原始观测包含的真实 Fisher 信息。稳定性最终仍受输入展宽、有效符合计数、背景和 PSF 失配限制。

## 安装

```powershell
python -m pip install -r .\v25_framework\requirements.txt
```

## 训练 Neural-PSF

```powershell
python -m v25_framework.train_neural_psf `
  --dataset-root "E:\path\to\independent_calibration" `
  --output-dir .\v25_framework\artifacts\neural_psf `
  --samples-per-direction 12
```

训练器输出 `neural_psf_model.npz`、`condition_audit.csv` 和 `training_summary.json`。模型文件仅包含 NumPy 可读取的冻结 MLP 权重。

## 生成未测条件数据

```powershell
python -m v25_framework.generate_virtual_dataset `
  --training-summary .\v25_framework\artifacts\neural_psf\training_summary.json `
  --neural-model .\v25_framework\artifacts\neural_psf\neural_psf_model.npz `
  --lengths-km "0,25,50,75,100,125" `
  --bandwidths-nm "0.2,0.4,0.8,2" `
  --count-rates-hz "50,100,280" `
  --bins 16385 `
  --output .\v25_framework\artifacts\virtual_conditions.npz
```

输出同时包含直方图和精确中心标签。符合计数率只进入 Poisson 采样强度，不参与 PSF 预测。

## 冻结部署条件

```powershell
python -m v25_framework.freeze `
  --calibration-json .\v25_framework\artifacts\neural_psf\training_summary.json `
  --neural-model .\v25_framework\artifacts\neural_psf\neural_psf_model.npz `
  --length-km 50 `
  --bandwidth-nm 0.8 `
  --iterations 512 `
  --output .\v25_framework\frozen\50km_0p8nm.json
```

冻结目录包含配置、网络和两个 SHA-256 校验。任何权重或配置变化都会使推理拒绝运行。

## 推理

单直方图：

```powershell
python -m v25_framework.run_inference .\input.csv `
  --frozen-config .\v25_framework\frozen\50km_0p8nm.json `
  --direction 1 `
  --output-csv .\v25_framework\outputs\compensated.csv
```

1000 组盲评：

```powershell
python -m v25_framework.run_external_1000 `
  --source-root "E:\path\to\fixed-axis-histograms" `
  --frozen-config .\v25_framework\frozen\50km_0p8nm.json `
  --output-dir .\v25_framework\outputs\blind_1000
```

## 代码结构

```text
config.py             配置、模型路径和双重 SHA-256
dataset.py            独立条件发现、宽度拟合和采样
neural_psf.py         NumPy MLP 推理、覆盖门控和 PSF 变换
train_neural_psf.py   质量门控、训练和整组条件留一验证
generate_virtual_dataset.py  未测条件与计数率的虚拟实验数据
physics.py            双光子谱、色散、IRF 和方向 PSF
compensator.py        单样本/批量 RL、重建和 Poisson 中心
freeze.py             网络与物理参数冻结
run_inference.py      单直方图入口
run_external_1000.py  固定配置的 1000 组盲评入口
tests/                数值性质与哈希测试
```

## 验证

```powershell
python -m pytest .\v25_framework\tests -q
```

单元测试只使用代码生成的分布，不包含实验数据。
