# V24 Direct Histogram Dispersion Compensation

`v24_framework` is the standalone final release of the stateless
histogram-to-histogram dispersion compensator. It includes the locked
direction-specific model, a Python API, a CSV command-line interface, tests,
and the external 1000-group recalibration/evaluation script.

The release does not use adjacent histograms, a clock-difference sequence, or
run-level post-processing. One histogram is independently transformed into one
nonnegative, count-preserving compensated histogram.

## Locked Method

```text
one raw histogram
  -> Gaussian coarse localization and 2049-bin crop
  -> edge-background subtraction and normalization
  -> direction-specific Richardson-Lucy deconvolution
  -> direction-specific physical 0 km target-PSF convolution
  -> background and count restoration
  -> one compensated histogram
```

| Parameter | Locked value |
|---|---:|
| Release | `v24` |
| Local input length | `2049 bins` |
| Histogram spacing | `1 ps/bin` |
| RL iterations | `512` |
| Background | median of `160 bins` at each edge |
| RL ratio clip | `8.0` |
| Latent probability floor | `1e-8` |
| Final-center window | `+/-180 bins` |
| Direction-specific broad PSF | yes |
| Direction-specific target PSF | yes |
| Post-output bounded correction | no |
| Legacy `eta/blend/clip` parameters | not used |

The final center is

```text
Gaussian coarse absolute center
  + compensated local background-subtracted center of mass
  - local-window midpoint
```

It is not followed by Gaussian fitting or bounded center correction.

## Install

From the repository root:

```powershell
python -m pip install -r .\v24_framework\requirements.txt
```

The tracked model is:

```text
v24_framework/models/direct_histogram_model_v24.npz
```

SHA-256:

```text
411db65754ae4ae8edfb04d0ac7850b2d3e4ae857ccd3458beb97cb9689e879f
```

## Python API

For a stream, load the model once:

```python
import numpy as np

from v24_framework import V24Compensator

operator = V24Compensator()

# One already-localized 2049-bin, 1 ps/bin histogram.
compensated = operator.infer_local(raw_local_histogram, direction=1)

# One full fixed-axis histogram. The method returns the compensated 2049-bin
# local histogram, its absolute axis, and the final absolute center.
compensated, absolute_axis_ps, center_ps = operator.infer_full(
    raw_full_histogram,
    direction=1,
    absolute_time_ps=full_axis_ps,
)
```

`direction` must be `1` or `2`; the two directions have different calibrated
broad and target PSFs.

## CSV Inference

Input may contain either `count` or `time_ps,count`. A header is accepted. A
full histogram is automatically localized and cropped; a 2049-bin histogram is
used directly.

```powershell
python .\v24_framework\run_inference.py input.csv `
  --direction 1 `
  --output-csv output_v24.csv
```

Outputs:

- `output_v24.csv`: absolute time, cropped raw count, and compensated count.
- `output_v24.json`: centers, count-conservation values, direction, and model
  parameters.

## Verification

```powershell
python .\v24_framework\verify_release.py
python -m pytest .\v24_framework\tests -q
```

`verify_release.py` checks the model hash, locked metadata, nonnegativity,
count conservation, and both direction-specific operators.

## Recalibrate And Reproduce

The release ships the script used for the external 1000-group selection and
evaluation:

```powershell
python -m pip install -r .\v24_framework\requirements-reproduction.txt

python .\v24_framework\run_direct_histogram_external_1000.py `
  --source-root "E:\path\to\50km_280Hz" `
  --target-psf .\v24_framework\models\physical_0km_target_psf.npz `
  --output-dir .\v24_framework\results\external_1000
```

The expected source structure is the same paired-direction histogram and
quality-table layout used by the original external `50 km / 280 Hz` run.
Recalibration uses the first 500 pairs; samples 501-1000 are the strict
held-out segment. Experimental histograms and generated result packages are
not stored in Git.

## Reported External Result

On the locked 1000-pair external `50 km / 280 Hz` evaluation:

| Metric | Before | V24 output |
|---|---:|---:|
| TDEV at 10 s, full 1000 | 4.098 ps | 2.380 ps |
| TDEV at 10 s, held-out 501-1000 | 4.036 ps | 2.485 ps |
| Median FWHM | 506.0 ps | 174.1 ps |
| Width reduction | 1.00x | 2.91x |
| Stability improvement | 1.00x | 1.72x |

The independent `1.6 ps` target was not reached and is not claimed.

## Files

```text
v24_framework/
  __init__.py
  direct_histogram_compensator.py
  public_compensated_histogram_operator.py
  run_inference.py
  run_direct_histogram_external_1000.py
  v24_common.py
  verify_release.py
  MODEL_CARD.md
  models/
    direct_histogram_model_v24.npz
    physical_0km_target_psf.npz
  tests/
    test_v24_compensator.py
```

The earlier `v17_framework` remains in the repository for audit and paper
provenance. New deployment should use `v24_framework`.
