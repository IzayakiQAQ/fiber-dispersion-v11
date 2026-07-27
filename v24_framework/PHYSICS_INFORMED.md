# Physics-Informed Adaptive Dispersion Compensation

## Status

This package is an experimental extension of the locked v24 release. It does
not alter `direct_histogram_model_v24.npz` or the reported external result.
Its purpose is to calibrate a physical response manifold from measured
conditions and select a response from one independently observed histogram.

## Experimental Prior

- CW pump at 1540.56 nm (ITU C46).
- Cascaded SHG and type-0 SPDC in a PPLN module.
- Energy-anticorrelated photons selected at C57 and C35.
- Equal nominal Gaussian WSS intensity bandwidths.
- Physical spool length from the condition folder name.
- Corning single-mode fiber represented by fitted effective dispersion.
- Direction-specific combined detector, time-tagger, and electronics IRF.
- One histogram contains 10 s of integration at 1 ps/bin.
- The archive conditions default to 100 Hz coincidence rate.

The broad source-envelope prior is motivated by the measured approximately
60 nm spectrum in Zhang et al., *npj Quantum Information* 7, 123 (2021):
https://doi.org/10.1038/s41534-021-00462-7. The reference spectrum is a prior,
not a claim that the experiment has the paper's exact module parameters.

## Forward Model

Under the CW energy-conservation constraint, the model uses one frequency
detuning coordinate. The effective joint spectral amplitude is

```text
Phi(Omega) = Phi_source(Omega)
             sqrt(T_C57(Omega) T_C35(-Omega)).
```

Each `T` is a Gaussian intensity response with the nominal WSS FWHM. Fiber
propagation adds

```text
exp(i beta2 L Omega^2 / 2 + i beta3 L Omega^3 / 6).
```

The squared Fourier transform gives the ideal coincidence-time probability.
It is convolved with the fitted combined timing IRF and sampled as

```text
H_k ~ Poisson(N p_k + b_k).
```

The 0 km observations identify an effective convolution of source, detector,
time-tagger, and electronics responses. Coincidence histograms alone do not
separate those components, so the implementation deliberately fits a combined
IRF rather than reporting an unsupported detector-only jitter.

## Calibration And Holdout Rules

Calibration fits four identifiable first-stage parameters:

- effective fiber dispersion at 1550 nm;
- nominal WSS bandwidth scale;
- direction-1 combined IRF FWHM;
- direction-2 combined IRF FWHM.

The source FWHM and dispersion slope remain constrained priors. Higher-order
terms should be released only after the simpler model shows systematic shape
residuals. Splits are made by physical condition, never by randomly mixing
histograms from the same run.

The discovery layer also records the histogram-construction layout. Legacy
sequential flat folders, channel-labelled folders, and pair-labelled folders
are audited separately. A response family that fails the physical width model
is retained in the audit output but is not silently mixed into a selected
calibration layout.

Recommended strict tests are:

1. Calibrate on 0/25/50/75/100 km at 0.8 nm and hold out 125 km.
2. Calibrate on 0.2-8 nm and hold out 10 nm.
3. Use the length and bandwidth axes for calibration and test measured
   50 km / 5 nm and 50 km / 10 nm intersections.
4. Apply Poisson thinning to the same measured histogram for controlled
   low-count tests; keep real 280 Hz data as the count-rate external test.

## Online Compensation

`PhysicsAdaptiveCompensator` compares a single centered histogram with a
generated response bank, selects the closest physical response, and applies
RL deconvolution followed by the generated 0 km target response. It estimates
an effective condition; it does not require length or bandwidth metadata at
inference time.

A no-harm gate returns the input unchanged when signal count is too low, the
best response has excessive Jensen-Shannon divergence, or the predicted
translation Fisher-information gain is negligible. RL iterations scale with
signal count instead of remaining fixed at 512 for every condition.

The adaptive compensator's `fisher_gain` is a morphology-based ratio between
the generated broad and target response shapes. It is useful as a no-harm
gate, but it is not a claim that deterministic RL creates that amount of new
measurement Fisher information. RL output bins are correlated and must not be
treated as independent Poisson samples.

## Poisson/Fisher Residual Alignment

The optional `fisher_residual` module uses a direction-specific broad-response
template to estimate the current raw-histogram center with a Poisson location
likelihood. If signal count and translation Fisher information pass fixed
offline thresholds, the module translates the physics-RL histogram to that
center while preserving counts. Otherwise it returns the physics-RL center and
shape unchanged.

Calibration should use an independent or explicitly separated calibration
acquisition. Splitting the evaluation run into an early calibration segment and
a later held-out segment is useful only as a same-run diagnostic; it does not
qualify as reference-free external validation. The public calibration API
therefore requires `calibration_is_independent=True` and rejects a false value.

Fine template structure should also be audited with two non-overlapping halves
of the independent calibration acquisition. `cross_power_clock_crlb_ps` uses
their derivative cross product so finite-count texture is not mistaken for
repeatable Fisher information.

The complete procedure and the external `50 km / 280 Hz` audit are documented
in `FISHER_RESIDUAL_FLOW_CN.md`.

## Claim Boundary

Synthetic histograms validate morphology, estimator bias, and controlled
count-rate behavior. Long-term 1000-10000 s TDEV claims must still come from
ordered measured histogram sequences or from an explicitly validated dynamic
clock/environment model. Independent Poisson samples alone are not evidence
of long-term synchronization stability.
