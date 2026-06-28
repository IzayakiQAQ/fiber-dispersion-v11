# V11 Framework

V11 is the cleaned teacher-guided compensation route used after the V10
no-harm experiments. The main model is intentionally single-histogram
inference:

```text
one centered histogram
  -> frozen V1.65 strong teacher
  -> trainable Gaussian width guard
  -> compensated histogram
```

No Mamba or sequence input is used. Pair information is only used during
evaluation/training to compute clock and TDEV metrics.

## Main Result

The current recommended run directory is:

```text
v11_framework/artifacts/v11_50km_0p8_teacher_width_guard_v1
```

For `50km / 0.8nm` test split `769-1024`:

| model | TDEV@10s ps | FWHM | FWHM / 0km target |
|---|---:|---:|---:|
| input | 32.890 | 224.166 | 4.601 |
| V1.65 teacher | 2.816 | 35.027 | 0.719 |
| V11 width guard | 2.813 | 51.613 | 1.073 |
| 0km target | 1.110 | 48.736 | 1.000 |

The main contribution is that V11 keeps the V1.65-level timing compensation
while correcting the over-narrow peak shape back to the 0km reference range.

## Files

- `models_v11.py`: V1.65 teacher replica, Gaussian width guard, and archived
  shift-head experiment modules.
- `train_v11_teacher_width_guard.py`: main V11 training/evaluation entry.
- `configs/v11_50km_0p8_teacher_width_guard.yaml`: main 50km/0.8nm config.
- `V11_FINAL_REPORT_CN.md`: final Chinese report for thesis/presentation use.
- `V11_TEACHER_WIDTH_GUARD_RESULT_CN.md`: detailed width-guard result note.
- `V11_SHIFT_HEAD_EXPERIMENT_CN.md`: archived negative experiment note.

## Reproduce Main Training

Use a Python environment with PyTorch installed:

```powershell
python .\v11_framework\train_v11_teacher_width_guard.py
```

Dry run:

```powershell
python .\v11_framework\train_v11_teacher_width_guard.py --dry-run
```

Artifacts and checkpoints are intentionally ignored by git.
