# V24 Direct Histogram Dispersion Compensation

`v24_framework` is the standalone release of the stateless
histogram-to-histogram dispersion compensator. It now separates the recommended
independent-physics protocol from the earlier same-run empirical baseline.

The release does not use adjacent histograms, a clock-difference sequence, or
run-level post-processing. One histogram is independently transformed into one
nonnegative, count-preserving compensated histogram.

## Recommended Frozen Physics Protocol

Use two separate commands. The freeze command deliberately has no
`--source-root` or `--input-cache` argument, so evaluation histograms cannot be
used to construct the PSF or select hyperparameters.

```powershell
python .\v24_framework\freeze_blind_physics_protocol.py `
  --calibration-json .\independent_100hz\physics_calibration.json `
  --calibration-metrics .\independent_100hz\empirical_condition_metrics.csv `
  --length-km 50 `
  --bandwidth-nm 0.8 `
  --count-rate-hz 280 `
  --output-dir .\v24_framework\outputs\blind_physics_frozen
```

The command uses only independently acquired calibration statistics and
predeclared Poisson simulations. It writes `frozen_config.json`, the complete
simulation candidate table, and `frozen_config.sha256`. The selected RL
iteration count and center window cannot be overwritten in place.

After freezing, evaluate the external run exactly once:

```powershell
python .\v24_framework\run_blind_physics_external_1000.py `
  --frozen-config .\v24_framework\outputs\blind_physics_frozen\frozen_config.json `
  --source-root "E:\path\to\50km_280Hz" `
  --max-pairs 1000 `
  --output-dir .\v24_framework\outputs\blind_physics_external_1000
```

An existing input cache can be supplied with `--input-cache` instead of
`--source-root`. The cache is only a loader optimization and must contain
`local_histograms`, `coarse_center_abs_ps`, `quality_center_abs_ps`, and
`input_fwhm_gaussian_ps`.

The evaluation command verifies the frozen-config SHA-256 and the independent
physics-calibration SHA-256 before reading the external data. It refuses a
nonempty output directory and executes only the frozen configuration.

| Quantity | Frozen from |
|---|---|
| Direction-specific broad PSF | Independent 100 Hz physics calibration: dispersion and IRF |
| Direction-specific 0 km target PSF | The same independent physics model at `L=0` |
| RL iterations and center window | Predeclared physics/Poisson simulations only |
| Evaluation histogram | Used only as the current single-histogram input |
| Fisher residual template | Disabled unless a separate same-response calibration exists |
| Bounded or time-series correction | Disabled |

For the audited 50 km / 0.8 nm / 280 Hz locked replay, the independently
frozen configuration selected `R=512` and a `120 ps` center half-window. It
reduced mean FWHM from `505.97 ps` to `160.21 ps` and 10 s TDEV from `4.098 ps`
to `2.869 ps`. This is the honest reference-free code-path result and does not
reach `1.8 ps`. Because this run had been inspected before the protocol was
implemented, it is a code-enforced retrospective locked replay, not a pristine
prospective blind experiment. A new untouched acquisition is required for the
strongest paper claim.

## Legacy Same-Run Locked Model

The tracked `direct_histogram_model_v24.npz` is retained for reproducibility.
Its empirical broad PSF was estimated from pairs 1-500 of the same 50 km /
280 Hz run, so it must not be presented as an external or reference-free model.

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

## Legacy Same-Run Reproduction

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

## Reported Same-Run Calibration/Held-Out Baseline

The locked empirical-PSF model used pairs 1-500 of the `50 km / 280 Hz` run
for recalibration and pairs 501-1000 as its held-out segment. It is therefore a
same-run calibration/held-out baseline, not a reference-free external result:

| Metric | Before | V24 output |
|---|---:|---:|
| TDEV at 10 s, full 1000 | 4.098 ps | 2.380 ps |
| TDEV at 10 s, held-out 501-1000 | 4.036 ps | 2.485 ps |
| Median FWHM | 506.0 ps | 174.1 ps |
| Width reduction | 1.00x | 2.91x |
| Stability improvement | 1.00x | 1.72x |

The independent `1.6 ps` target was not reached and is not claimed. This
empirical model must not be used to support unknown-condition or zero-reference
generalization claims.

## Physics-Informed Extension

The locked operator above remains the reproducible `50 km / 280 Hz`
baseline. The optional `physics_informed` package extends v24 to unknown
length/bandwidth conditions without changing the saved baseline model.

The forward model represents the actual experiment as a CW C46-pumped,
cascaded SHG/type-0-SPDC PPLN source, an energy-anticorrelated C57/C35 pair,
two nominal Gaussian WSS intensity filters, single-mode-fiber spectral phase,
a fitted direction-specific timing IRF, and Poisson counting. Only effective
parameters identifiable from coincidence histograms are fitted.

Build the condition manifest and calibrate on measured histograms:

```powershell
python .\v24_framework\run_physics_calibration.py `
  --dataset-root "E:\path\to\100Hz_calibration_histograms" `
  --output-dir .\v24_framework\results\physics_calibration
```

Strict length and bandwidth holdouts can be selected without moving files:

```powershell
python .\v24_framework\run_physics_calibration.py `
  --calibration-layout channel_subdirectories `
  --calibration-layout pair_subdirectories `
  --holdout-length-km 125 `
  --holdout-bandwidth-nm 10
```

The command reads the 1 ps / 10 s histogram CSVs directly. Raw event
timestamps are not required. It supports all three layouts currently present
in the experiment archive: sequential directions in one folder,
channel-labelled direction folders, and `pair0`/`pair1` folders.

Outputs include a compact dataset manifest, measured aggregate profiles,
calibrated physical parameters, measured-versus-predicted condition metrics,
and a virtual length/bandwidth grid. Generated results remain ignored by Git.
See `PHYSICS_INFORMED.md` for model equations, assumptions, and validation
boundaries.

Apply the calibrated response manifold to one full-axis histogram:

```powershell
python .\v24_framework\run_physics_inference.py input.csv `
  --direction 1 `
  --calibration-json .\v24_framework\results\physics_calibration\physics_calibration.json `
  --output-csv output_physics_v24.csv
```

Inference uses only that histogram. Candidate responses that exceed the
configured time window are rejected instead of being silently wrapped by the
FFT. The JSON sidecar records the selected effective condition, fit
divergence, Fisher-information gate, iteration count, centers, widths, and
count conservation.

## Poisson/Fisher Residual Extension

The optional `PhysicsFisherResidualCorrector` adds a stateless center-residual
stage after physics RL. It calibrates direction-specific broad-response
templates offline, estimates the current raw-histogram center with a Poisson
location likelihood, applies a Fisher no-harm gate, and translates the
already-generated compensated histogram to the accepted center.

```text
one current raw histogram
  -> physics response and RL histogram reconstruction
  -> current-histogram Poisson template center
  -> Fisher-information no-harm gate
  -> count-preserving translation of the RL histogram
  -> one final compensated histogram
```

This stage still uses no adjacent histograms, run-level mean, time-series
filter, or bounded center correction. The model can be calibrated and used as:

```python
from v24_framework.physics_informed import (
    FisherResidualConfig,
    PhysicsFisherCompensationPipeline,
    PhysicsFisherResidualCorrector,
)

corrector = PhysicsFisherResidualCorrector.calibrate(
    calibration_histograms,          # shape: (2, samples, odd_bins)
    coarse_centers_ps,
    physics_alignment_centers_ps,
    calibration_is_independent=True,
    config=FisherResidualConfig(
        template_smoothing_sigma_bins=12.0,
        minimum_fisher_information_per_ps2=0.04,
    ),
)

result = corrector.align_compensated_histogram(
    raw_local_histogram,
    physics_rl_histogram,
    direction=1,
    coarse_center_ps=coarse_center_ps,
)

# Optional one-call deployment wrapper around PhysicsAdaptiveCompensator.
pipeline = PhysicsFisherCompensationPipeline(physics_operator, corrector)
result = pipeline.infer(raw_full_histogram, direction=1, absolute_time_ps=full_axis_ps)
final_histogram = result.compensated_counts
```

The earlier `2.365 ps` full-sequence and `2.490 ps` pairs-501-1000 values used
pairs 1-500 of that same 280 Hz run to construct the Poisson template. They are
retained only as a same-run diagnostic and are withdrawn from external-result
claims. The physics-generated-PSF run did not use 280 Hz histograms to build
its PSF, but it did use pairs 1-500 to choose the RL iteration checkpoint. Its
`155.9 ps` median FWHM and `2.827/2.841 ps` TDEV values are therefore post-hoc
PSF-independence diagnostics, not a fully frozen blind external result.

The frozen-physics scripts enforce that every PSF, iteration count, center
window, and threshold is selected before an evaluation input can be supplied.
A deployable Fisher residual stage still requires a separate acquisition with
the same broad-response state; without it, the residual is disabled rather
than fitted from the evaluation run.

See `FISHER_RESIDUAL_FLOW_CN.md` for the full equations, calibration/holdout
protocol, Fisher covariance interpretation, result table, and 1.8 ps claim
boundary.

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
  run_physics_calibration.py
  run_physics_inference.py
  freeze_blind_physics_protocol.py
  run_blind_physics_external_1000.py
  PHYSICS_INFORMED.md
  FISHER_RESIDUAL_FLOW_CN.md
  physics_informed/
    adaptive_compensator.py
    dataset.py
    fisher_residual.py
    forward_model.py
  MODEL_CARD.md
  models/
    direct_histogram_model_v24.npz
    physical_0km_target_psf.npz
  tests/
    test_v24_compensator.py
    test_physics_informed.py
```

The earlier `v17_framework` remains in the repository for audit and paper
provenance. New deployment should use `v24_framework`.
