# V24 Physics RL + Poisson/Fisher 残差流程

## 1. 目标与约束

该扩展用于在物理 RL 已经生成补偿直方图之后，提高单个直方图的中心估计效率，同时保持以下部署约束：

- 一张原始直方图输入，一张补偿直方图输出；
- 每个方向独立处理，不读取相邻时刻直方图；
- 不使用运行均值、卡尔曼滤波或两点因果稳定器；
- 不使用旧版 bounded center correction；
- 所有中心残差都来自当前原始直方图的 Poisson 似然；
- Fisher 门只负责低信息量回退，不负责把钟差向固定值收缩。

完整在线流程为：

```text
one measured histogram
  -> coarse localization and fixed local crop
  -> physics response selection / known physical condition
  -> direction-specific physics RL deconvolution
  -> physical 0 km target-PSF convolution
  -> count-preserving physics compensated histogram
  -> raw-histogram Poisson template center
  -> Fisher no-harm gate
  -> translate the compensated histogram to the accepted center
  -> one final compensated histogram and its direct center
```

## 2. 离线标定

### 2.1 数据划分

按时间顺序或独立实验划分标定集与严格留出集。参数选择只能读取标定集。50 km / 280 Hz 审计使用前 500 组构造模型，后 500 组只做最终验收。

### 2.2 方向独立的 broad-response 模板

对每个方向，先用物理 RL 中心或另一种固定、单图中心估计器对标定直方图做亚 bin 配准。设第 `i` 张原始直方图为 `h_i(t)`，边缘 Poisson 背景为 `b_i`，配准残差为 `delta_i`，则模板为

```text
p(t) = Normalize{ G_sigma * sum_i max[h_i(t + delta_i) - b_i, 0] }.
```

`G_sigma` 是固定高斯模板平滑。平滑尺度必须由交叉拟合标定选择，不能根据严格留出集调整。50 km / 280 Hz 数据选择 `sigma=12 bins`。

### 2.3 交叉功率 Fisher 审计

把标定数据再分成两个互不重叠的模板 `p_A`、`p_B`。普通单模板 Fisher 会把有限计数造成的随机纹理当作峰导数，产生过于乐观的下界。使用

```text
I_cross = sum_k N^2 p'_A,k p'_B,k / (b + N p_k)

sigma_clock >= 0.5 sqrt(1 / I_1 + 1 / I_2)
```

只保留两半模板中可重复的平移信息。若 `I_cross` 为负或不稳定，说明模板统计量不足，不能据此声明亚 CRLB 稳定性。

## 3. 在线 Poisson 中心

对当前原始直方图 `h_k`，给定方向模板 `p_k(mu)`、信号计数 `N` 和边缘背景 `b`：

```text
lambda_k(mu) = b + N p_k(mu)

ell(mu) = sum_k [h_k log lambda_k(mu) - lambda_k(mu)]

S(mu) = sum_k [h_k / lambda_k(mu) - 1] d lambda_k / d mu

I(mu) = sum_k [d lambda_k / d mu]^2 / lambda_k(mu).
```

使用受限 Newton 步 `mu <- mu + clip(S/I)` 估计当前单图中心。这里的限制只保护数值迭代步长，不是把最终中心限制在旧中心附近的 bounded correction。

## 4. Fisher 门与直方图输出

当信号计数和 Fisher 信息同时超过离线锁定阈值时，接受 Poisson 中心；否则保留物理 RL 中心：

```text
center_accepted = poisson_center,  if N >= N_min and I >= I_min
                  physics_center, otherwise.
```

最终输出不是一列修正中心，而是把物理 RL 直方图平移

```text
Delta = center_accepted - center_physics

h_final(t) = h_physics(t - Delta),
```

随后恢复总计数并直接从 `h_final` 计算最终中心。该变换保持非负、保持积分计数，并且只依赖当前原始/补偿直方图。

## 5. RL 后 Fisher 的正确解释

原始计数可建模为独立 Poisson 变量，但 RL 输出 `y=g(h)` 的 bin 不再相互独立。其局部协方差近似为

```text
C_y = J_g diag(lambda) J_g^T.
```

因此不能把补偿后更窄的 FWHM 直接代入独立 Poisson 公式。正确局部形式为

```text
I_out = (d mu_y / d theta)^T C_y^-1 (d mu_y / d theta),
```

并满足确定性数据处理的 `I_out <= I_in`。软件反卷积可以改善估计器效率和峰形，但不能按 FWHM 缩窄倍数凭空增加输入 Fisher 信息。

## 6. 50 km / 280 Hz 锁定结果

| 指标 | 补偿前 | 物理 RL | Physics RL + Poisson/Fisher |
|---|---:|---:|---:|
| FWHM 中位数 | 506.028 ps | 155.897 ps | 155.900 ps |
| 10 s TDEV，全 1000 组 | 4.098 ps | 2.811 ps | 2.365 ps |
| 10 s TDEV，严格留出 501--1000 | 4.036 ps | 2.829 ps | 2.490 ps |
| 100 s TDEV | 0.415 ps | 0.286 ps | 0.237 ps |
| 1000 s TDEV | 0.0420 ps | 0.0293 ps | 0.0242 ps |

该批数据的交叉功率 Fisher 下界约为 `2.39--2.45 ps`。平滑 1.5 ps 的普通单模板公式可给出约 1.72 ps，但两半模板导数相关性只有 0.22/0.34；去除不可重复纹理后，下界回到 2.45 ps。因此 v24 不声明当前 280 Hz 数据已经达到 1.8 ps。

## 7. 达到 1.8 ps 的条件

仍值得继续验证的方向是：

1. 获取与 506 ps 宽峰状态完全一致、独立于测试集的高统计量 broad-PSF 标定；
2. 独立标定 TDC DNL/INL，再构造固定 Poisson 模板；
3. 用注入的已知动态钟差验证算法不会压制真实时间变化；
4. 提高有效符合计数，或在探测前执行真实的光学色散压缩。

按严格留出结果的散粒噪声标度，1.8 ps 约需要当前 `1.91` 倍有效计数，即约 `535 Hz`；1.6 ps 约需要 `678 Hz`。运行均值或动态先验能够产生更低的表观 TDEV，但不属于本模块的色散补偿声明。

## 8. 代码接口

```python
from v24_framework.physics_informed import (
    FisherResidualConfig,
    PhysicsFisherCompensationPipeline,
    PhysicsFisherResidualCorrector,
)

# Offline: histograms has shape (2, calibration_samples, odd_bins).
corrector = PhysicsFisherResidualCorrector.calibrate(
    histograms,
    coarse_centers_ps,
    physics_alignment_centers_ps,
    FisherResidualConfig(
        template_smoothing_sigma_bins=12.0,
        minimum_fisher_information_per_ps2=0.04,
    ),
)
corrector.save("physics_fisher_residual_model.npz")

# Online: raw_local and physics_rl_output are one current direction only.
result = corrector.align_compensated_histogram(
    raw_local,
    physics_rl_output,
    direction=1,
    coarse_center_ps=coarse_center_ps,
)
final_histogram = result.compensated_counts
final_center_ps = result.center_ps

# 或将 PhysicsAdaptiveCompensator 与残差校正器封装为单次完整推理。
pipeline = PhysicsFisherCompensationPipeline(physics_operator, corrector)
result = pipeline.infer(
    raw_full_histogram,
    direction=1,
    absolute_time_ps=full_axis_ps,
)
```

该模块是可选扩展，不修改 `direct_histogram_model_v24.npz`，也不改变锁定 v24 基线的复现结果。
