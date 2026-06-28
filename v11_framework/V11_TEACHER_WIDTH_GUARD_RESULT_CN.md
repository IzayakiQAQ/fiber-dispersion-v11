# V11 Teacher Width Guard 训练结果

## 1. 当前版本定位

V11 第一版采用单张直方图直接补偿，不使用 Mamba 或序列输入。

结构为：

```text
input histogram
  -> frozen V1.65 strong teacher
  -> trainable symmetric Gaussian width guard
  -> compensated histogram
```

该版本的目标不是从零替代 V1.65，而是先解决 V1.65 的主要论文风险：强补偿后峰形过窄。

## 2. 训练设置

- 场景：50km / 0.8nm
- 目标：0km / 0.8nm
- 数据：V10 centered 1024 tensor manifest
- split：blocked non-overlap
- train：1-512
- val：513-768
- test：769-1024
- teacher：V1.65 train-physics semantics
- 训练参数：8 epochs, batch size 32, AdamW lr 2e-2

输出目录：

```text
v11_framework/artifacts/v11_50km_0p8_teacher_width_guard_v1
```

## 3. 主要结果

| split | input TDEV ps | teacher TDEV ps | V11 TDEV ps | target TDEV ps | teacher FWHM | V11 FWHM | target FWHM | V11/target FWHM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 32.625 | 2.604 | 2.599 | 1.181 | 35.094 | 51.582 | 48.795 | 1.068 |
| test | 32.890 | 2.816 | 2.813 | 1.110 | 35.027 | 51.613 | 48.736 | 1.073 |

最终 guard 参数：

| direction | sigma | mix |
|---|---:|---:|
| dir1 | 22.162 | 0.927 |
| dir2 | 21.581 | 0.923 |

## 4. 结论

V11 width guard 成功把 V1.65 teacher 的过窄峰修回到接近 0km target 的范围：

```text
test FWHM: 35.03 -> 51.61
target FWHM: 48.74
```

同时 TDEV 没有被破坏：

```text
teacher TDEV: 2.816 ps
V11 TDEV:     2.813 ps
```

因此，V11 第一版已经证明：

1. V1.65 的强补偿能力可以作为稳定 teacher 保留。
2. 过窄峰不是不可避免的，可以通过对称峰宽 guard 修正。
3. 峰宽修正与 TDEV 稳定性可以同时成立。

## 5. 当前不足

V11 width guard 主要改变峰宽，不主动改变峰中心。因此它能修正 W/FWHM，但不能显著突破 V1.65 teacher 的 TDEV 上限。

当前 test TDEV 仍在 2.81ps 左右，没有达到 2ps 目标。

如果继续追 2ps，下一步不应该再只做峰宽修正，而应加入：

```text
single-histogram center perturbation / shift correction head
```

该模块仍然保持单张直方图推理，但训练时用 shift-restored clock loss，让模型学习每张直方图中可预测的中心扰动。

## 6. 论文表述建议

当前 V11 可以作为一个清晰的 ablation：

```text
V1.65 teacher:
  strong TDEV compression, but over-narrow peak.

V11 width guard:
  preserves teacher-level TDEV while restoring physically plausible peak width.
```

这比单纯报告 V1.65 更适合论文，因为它说明模型不仅追求 TDEV 下降，也约束了补偿后峰形的物理合理性。
