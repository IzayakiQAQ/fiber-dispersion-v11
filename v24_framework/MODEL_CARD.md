# V24 Model Card

## Identity

- Release: `v24`
- Model file: `models/direct_histogram_model_v24.npz`
- SHA-256: `411db65754ae4ae8edfb04d0ac7850b2d3e4ae857ccd3458beb97cb9689e879f`
- Method: nonnegative Richardson-Lucy deconvolution followed by physical
  0 km target-PSF convolution
- Neural network: none
- Inference state: none

The NPZ contains the two direction-specific broad PSFs, the two
direction-specific target PSFs, and all locked scalar parameters.

## Input And Output

- Input: one nonnegative 2049-bin local histogram at `1 ps/bin`
- Direction: `1` or `2`
- Output: one nonnegative 2049-bin compensated histogram
- Count behavior: total count is restored after compensation
- Full-axis use: Gaussian coarse localization followed by a 2049-bin crop

The operator does not consume adjacent histograms or clock-series information.

## Locked Parameters

- RL iterations: `512`
- Background bins: `160` at each edge
- Ratio clip: `8.0`
- Latent floor: `1e-8`
- Center readout: background-subtracted center of mass within `+/-180 bins`
- Post-output bounded correction: disabled
- Legacy V17 `eta=0.67`, `blend_scale=1.2`, and `clip_fraction=0.095`: unused

## Calibration And Evaluation

The broad PSFs were calibrated independently for the two propagation
directions from the calibration segment of an external `50 km / 280 Hz`
dataset. The physical target PSFs were direction-specific corrected 0 km
responses. The first 500 pairs formed the calibration segment, and pairs
501-1000 formed the strict held-out segment.

| Metric | Before | V24 |
|---|---:|---:|
| Full-run TDEV at 10 s | 4.098 ps | 2.380 ps |
| Held-out TDEV at 10 s | 4.036 ps | 2.485 ps |
| Median FWHM | 506.0 ps | 174.1 ps |

## Intended Use

- Independent compensation of fixed-axis coincidence histograms
- Paper result reproduction under the documented acquisition format
- Streaming inference where each histogram must be processed without future or
  neighboring samples

## Limitations

- The external `1.6 ps` TDEV target was not reached.
- The saved broad PSFs are system- and condition-dependent. A materially
  different bandwidth, detector response, timing bin, or optical path should
  be recalibrated and independently validated.
- The release model assumes `1 ps/bin`.
- Width reduction does not by itself guarantee proportional TDEV reduction.
- Experimental source histograms are not included in Git; only the compact
  deployable model and physical target PSF are included.
