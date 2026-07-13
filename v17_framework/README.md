# V17 Self-Calibrated Likelihood Reader

V17 is a single-histogram effective-dispersion likelihood reader.

It keeps the V16 physical idea, but removes the manually fixed V16 parameters from normal inference. For each input histogram, V17 estimates an effective broadening state from the measured peak width, selects a matched template from the training template bank, and derives the Fisher-score correction strength from the ratio between input FWHM and selected template FWHM.

No random sub-bin shift augmentation is used.

## Files

- `v17_common.py`: shared IO, TDEV, FWHM, local-window and CSV utilities.
- `build_template_bank.py`: builds the template bank from V10 centered training tensors.
- `likelihood_reader.py`: single-histogram template selection and Poisson/Fisher score readout.
- `run_external_50km_280hz.py`: runs V17 on fixed-axis external 50 km / 280 Hz histograms.
- `build_fig5_0km_reference.py`: exports a Fig. 5-style aligned 0 km reference package from local data.

## Build Template Bank

```powershell
python v17_framework\build_template_bank.py
```

Default output:

`v17_framework/artifacts/template_bank_v1`

The default bank uses V10 centered tensors from:

`E:\lzy\测试结果\补偿数据`

Included labels:

- `d025km_bw0p8nm`
- `d050km_bw0p8nm`
- `d075km_bw0p8nm`
- `d100km_bw0p8nm`

## Run External 50 km / 280 Hz

```powershell
python v17_framework\run_external_50km_280hz.py
```

Default input:

`E:\lzy\测试结果\2026.6.29 50km 280Hz`

Default output:

`v17_framework/results/v17_self_calibrated_20260629_50km_280hz`

Main output:

`time_t1_t2_t0_four_columns.csv`

## Current Result

On the 1000 external histograms:

| Metric | Raw quality Gaussian | V17 |
|---|---:|---:|
| TDEV@10s | 4.098 ps | 2.340 ps |
| Clock std | 4.163 ps | 2.450 ps |

V17 selected:

- labels: `d025km_bw0p8nm`, `d050km_bw0p8nm`
- smooth values: `100 ps`, `120 ps`
- background: `0.001`
- mean automatic blend: `1.716`
- mean template FWHM: `353.9 ps`

## Interpretation

V17 supports the working hypothesis that a single histogram contains enough peak-shape information to estimate an effective dispersion/broadening state. The model does not need the run-level TDEV to choose the V16-style parameters. It reads the raw peak width, selects a compatible likelihood template, and computes a bounded center correction.

This is still a prototype. The constants `target_template_fraction`, `blend_scale`, and `clip_fraction` are fixed physical heuristics, not yet learned from an independent validation set. The next paper-clean step is to lock these constants using a separate validation dataset, then evaluate a held-out test set once.

## Direct Histogram Output

`public_compensated_histogram_operator.infer_with_saved_model(histogram,
direction)` is the final deployment API. It accepts one local fixed-axis
histogram and returns one nonnegative, count-preserving compensated histogram.
It uses Richardson-Lucy deconvolution followed by convolution with the corrected
physical 0 km target PSF. It is not a weighted template mixture and is not a
V1.65/V11 neural-network output.

The external 50 km / 280 Hz delivery is in
`results/v24_direct_histogram_external_1000_physical_0km`. Metrics read from the
actual output histograms are:

| Metric | Before | Direct output |
|---|---:|---:|
| Full 1000 TDEV@10s | 4.098 ps | 2.380 ps |
| Strict histograms 501-1000 TDEV@10s | 4.036 ps | 2.485 ps |
| Median FWHM | 506.0 ps | 174.1 ps |

The locked operator uses 512 RL iterations, 160 background bins at each edge,
and separately calibrated broad and target PSFs for the two propagation
directions. The final direction center is the raw Gaussian coarse center plus
the background-subtracted center of mass of the compensated local histogram in
a +/-180-bin window. No bounded center correction is applied after inference.

The legacy V17 constants `target_template_fraction=0.67`, `blend_scale=1.2`,
and `clip_fraction=0.095` are not used by this RL operator. They respectively
selected a target template width, scaled a Fisher-score center displacement,
and bounded that displacement in the earlier likelihood reader. They belong
only in the legacy baseline or Supplement, not in the final RL method.

The width reduction is 2.91x and the full-run stability improvement is 1.72x.
The 1.6 ps target is not claimed for this independent external dataset.

After placing the local result package under `results/`, run
`python v17_framework/verify_final_delivery.py` to verify the saved model
against representative entries and check the complete Fig. 5 source table.

Experimental histograms, fitted model arrays, generated figures, and derived
CSV tables are intentionally excluded from Git. They remain in the local
`artifacts/` and `results/` directories.
