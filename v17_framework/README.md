# Direct Histogram Dispersion Compensation

This workspace contains the final stateless histogram-to-histogram dispersion compensator and the earlier V17 likelihood-reader baseline.

## Final Method

`public_compensated_histogram_operator.infer_with_saved_model(histogram, direction, model_path)` is the deployment API. It accepts one 2049-bin local histogram and returns one nonnegative, count-preserving compensated histogram.

The operator performs:

1. Median background estimation from 160 bins at each edge.
2. Background subtraction and probability normalization.
3. 512 iterations of direction-specific Richardson-Lucy deconvolution.
4. Convolution with a direction-specific physical 0 km target PSF.
5. Background and total-count restoration.

The two directions use separately calibrated broad PSFs and target PSFs. The final center is the raw Gaussian coarse center plus the background-subtracted center of mass of the compensated local histogram within a +/-180-bin window. There is no post-output bounded center correction.

The legacy constants `target_template_fraction=0.67`, `blend_scale=1.2`, and `clip_fraction=0.095` are not used by the final RL operator. They belong only to the earlier Fisher-score center reader.

## Final Local Result

On 1000 external `50 km / 280 Hz` histogram pairs:

| Metric | Before | Direct RL output |
|---|---:|---:|
| Full TDEV@10 s | 4.098 ps | 2.380 ps |
| Held-out 501-1000 TDEV@10 s | 4.036 ps | 2.485 ps |
| Median FWHM | 506.0 ps | 174.1 ps |

The width reduction is 2.91x and the full-run stability improvement is 1.72x. The independent external `1.6 ps` target is not claimed.

## Main Files

- `direct_histogram_compensator.py`: core nonnegative RL implementation and center/FWHM utilities.
- `public_compensated_histogram_operator.py`: minimal single-histogram inference API.
- `run_direct_histogram_external_1000.py`: direction-specific calibration, RL-iteration selection, evaluation, and model export.
- `build_physical_0km_target_psf.py`: corrected direction-specific physical 0 km target builder.
- `build_paper_comparison_dcm_vs_direct.py`: paper tables, source CSV files, and Fig.1-Fig.5.
- `build_fig5_0km_reference.py`: Fig.5-style aligned 0 km source data and reference figure.
- `verify_final_delivery.py`: local consistency check between the saved model, compensated outputs, centers, and Fig.5 source table.
- `test_direct_histogram_compensator.py`: nonnegativity, count conservation, translation, batch-equivalence, and public-operator tests.
- `likelihood_reader.py`: legacy V17 Poisson/Fisher-score baseline.

## 0 km Fig.5 Reference

Run:

```powershell
python v17_framework\build_fig5_0km_reference.py
```

The local output directory is `results/fig5_0km_reference`. It contains:

- `fig5_0km_source_data_complete.csv`: complete -1400 ps to +1400 ps probability and peak-normalized curves.
- `fig5_0km_plot_data.csv`: -800 ps to +800 ps plotting table.
- `fig5_0km_reference.png` and `.pdf`: measured direction curves, measured aligned average, and corrected physical target.
- `summary.json` and `RESULT_CN.md`: width metrics and interpretation.

The current local source uses 8640 measured histograms per direction from `2026.3.5 0km 2m单边_Merged`. The measured two-direction aligned-average half-maximum FWHM is 182.26 ps; the corrected physical target average is 163.74 ps. These are kept as distinct curves.

## Reproduce

From the repository root:

```powershell
python .\v17_framework\run_direct_histogram_external_1000.py
python .\v17_framework\build_paper_comparison_dcm_vs_direct.py
python .\v17_framework\build_fig5_0km_reference.py
python .\v17_framework\verify_final_delivery.py
python -m pytest .\v17_framework\test_direct_histogram_compensator.py -q
```

Local defaults expect the measured datasets, target PSF, and saved model under the paths documented by each script. Supply the corresponding command-line path arguments when using another machine or dataset.

## Data Policy

Experimental histograms, fitted model arrays, generated figures, CSV/NPZ tables, and paper result packages are intentionally excluded from Git. They remain in local `artifacts/` and `results/` directories. The remote repository contains code and method documentation only.
