# V11 Shift Correction Head 实验记录

## 1. 实验目的

在不引入 Mamba、不使用序列输入的前提下，尝试给 V11 增加单张直方图的中心扰动修正：

```text
single histogram
  -> V1.65 teacher
  -> V11 width guard
  -> shift correction head
  -> compensated histogram
```

推理时仍然只输入单张直方图。训练时利用 pair 关系计算 shift-restored clock / TDEV。

## 2. 已完成实验

### 2.1 Neural shift head v1

配置：

```text
v11_framework/configs/v11_50km_0p8_shift_head.yaml
```

输出：

```text
v11_framework/artifacts/v11_50km_0p8_shift_head_v1
```

结果：

| split | V11 width guard TDEV ps | shift head TDEV ps | target TDEV ps | FWHM |
|---|---:|---:|---:|---:|
| val | 2.599 | 2.606 | 1.181 | 51.615 |
| test | 2.813 | 2.818 | 1.110 | 51.625 |

结论：无改善。模型基本学到接近常数的小 shift，batch-level TDEV loss 对单张直方图 CNN 头的监督太弱。

### 2.2 Neural shift head + pseudo residual

配置：

```text
v11_framework/configs/v11_50km_0p8_shift_head_pseudo.yaml
```

输出：

```text
v11_framework/artifacts/v11_50km_0p8_shift_head_pseudo_v1
```

结果：

| split | V11 width guard TDEV ps | pseudo shift TDEV ps | target TDEV ps | FWHM |
|---|---:|---:|---:|---:|
| val | 2.599 | 2.615 | 1.181 | 51.590 |
| test | 2.813 | 2.819 | 1.110 | 51.615 |

结论：仍无改善。直接把 pair residual 分配成 dir1/dir2 的伪标签，会让训练集 loss 略降，但不能泛化到 val/test。

## 3. 关键诊断

绝对 restored center residual 里存在约 -5242ps 的大常数项。这是全局 shift/recovery 基准差，不应该让模型学习。

真正影响 TDEV 的是该常数附近约 3ps 的波动：

```text
test residual std ~= 3.06 ps
```

因此 shift correction head 只能学习小扰动，不能重做整体对齐。

## 4. 手工特征 ridge 诊断

为了判断单张峰形里是否存在可学习信号，临时用以下手工特征做 ridge 回归：

```text
input peak/area/width/skew/asymmetry/tail
guarded peak/area/width/skew/asymmetry/tail
```

pair 级预测形式：

```text
clock_correction = 0.5 * (head(hist_dir1) - head(hist_dir2))
```

诊断结果：

| 训练数据 | test TDEV ps | baseline test TDEV ps | 说明 |
|---|---:|---:|---|
| train only | 2.696 | 2.813 | 有小幅泛化收益，但 val 不稳定 |
| train + val | 2.094 | 2.813 | 接近 2ps，说明手工峰形特征有明显信号 |

注意：train+val 结果应视为路线诊断，不应直接当最终论文结果；需要把超参数选择和 test 隔离后重新固化。

## 5. 当前结论

通用 CNN shift head 不是当前最佳路线。它没有稳定学到中心扰动。

但单张直方图的峰形特征确实包含可用于 TDEV 修正的信息，尤其是峰宽、偏斜、左右尾部不对称等局部统计量。

下一步更合适的 V11 路线应是：

```text
V1.65 teacher
  -> width guard
  -> feature-calibrated shift head
```

其中 shift head 不再从完整 65536 点 CNN 自由学习，而是显式输入物理可解释特征：

```text
center offset
local width
skewness
left/right area ratio
tail asymmetry
teacher/guarded feature difference
```

这条路线更适合论文表述，因为它能解释为：

```text
peak-shape-informed residual timing correction
```

而不是一个不可解释的黑盒中心平移。
