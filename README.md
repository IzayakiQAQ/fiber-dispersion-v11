# Fiber Dispersion V11 Paper Project

This repository is a clean V11-only extraction for paper writing and
reproducible discussion. It excludes the historical V8/V9/V10 experiments,
large tensor data, local checkpoints, and run artifacts.

## Main Model

```text
single centered histogram
  -> frozen V1.65 strong teacher
  -> trainable Gaussian width guard
  -> compensated histogram
```

The model does not use Mamba or sequence input. Pair information is used only
for training/evaluation metrics such as shift-restored TDEV.

## Main Result

For the `50km / 0.8nm` test split:

| model | TDEV@10s ps | FWHM | FWHM / 0km target |
|---|---:|---:|---:|
| input | 32.890 | 224.166 | 4.601 |
| V1.65 teacher | 2.816 | 35.027 | 0.719 |
| V11 width guard | 2.813 | 51.613 | 1.073 |
| 0km target | 1.110 | 48.736 | 1.000 |

See `v11_framework/V11_FINAL_REPORT_CN.md` for the Chinese report text.

## Expected Local Files

Large data/checkpoints are not tracked by git. To reproduce training locally,
place or configure:

```text
data/new_compensation_data_tensor_centered_1024/manifest.json
checkpoints/v165_teacher/adaptivecheck_compensator_dir1.pt
checkpoints/v165_teacher/adaptivecheck_compensator_dir2.pt
```

You can also edit the YAML configs under `v11_framework/configs/` to point to
your local data/checkpoint locations.

## Commands

Dry run:

```powershell
python .\v11_framework\train_v11_teacher_width_guard.py --dry-run
```

Train/evaluate:

```powershell
python .\v11_framework\train_v11_teacher_width_guard.py
```

The archived shift-head experiments are kept for negative-result discussion:

```powershell
python .\v11_framework\train_v11_shift_head.py --dry-run
```
