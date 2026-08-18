from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import correlate, fftconvolve

from .config import FrozenConfig, file_sha256, load_frozen_config
from .neural_psf import NeuralPSFModel
from .physics import DirectionKernels, build_direction_kernels, fwhm_ps


def _clean(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(
            np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        ),
        0.0,
        None,
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    clean = _clean(values)
    total = float(np.sum(clean))
    if total <= 1e-15:
        result = np.zeros_like(clean)
        result[result.size // 2] = 1.0
        return result
    return clean / total


def edge_background(values: np.ndarray, edge_bins: int) -> float:
    clean = _clean(values)
    width = min(max(int(edge_bins), 1), max(clean.size // 4, 1))
    # For sparse Poisson bins the median is usually zero. The edge mean is the
    # maximum-likelihood estimate of a spatially uniform accidental floor.
    return float(np.mean(np.concatenate((clean[:width], clean[-width:]))))


def poisson_template_center_index(
    values: np.ndarray,
    template: np.ndarray,
    edge_bins: int,
    search_half_width: int = 240,
) -> float:
    """Profile-likelihood center from one histogram and one frozen PSF."""

    observed = _clean(values)
    probability = _normalize(template)
    if observed.shape != probability.shape:
        raise ValueError("values and template must have equal shape")
    background = max(edge_background(observed, edge_bins), 1e-6)
    signal_mass = float(np.sum(np.clip(observed - background, 0.0, None)))
    if signal_mass <= 1e-12:
        return local_center_of_mass(observed, search_half_width)
    expected = signal_mass * probability + background
    # The constant background log term is removed so zero-padded correlation
    # cannot create a false preference at the local-window boundary.
    log_weight = np.log(expected / background)
    scores = correlate(observed, log_weight, mode="same", method="fft")
    center = observed.size // 2
    left = max(1, center - int(search_half_width))
    right = min(observed.size - 1, center + int(search_half_width) + 1)
    peak = left + int(np.argmax(scores[left:right]))
    denominator = scores[peak - 1] - 2.0 * scores[peak] + scores[peak + 1]
    offset = 0.0
    if abs(float(denominator)) > 1e-15:
        offset = 0.5 * (scores[peak - 1] - scores[peak + 1]) / denominator
    return float(peak + np.clip(offset, -0.5, 0.5))


def local_center_of_mass(values: np.ndarray, half_window: int) -> float:
    clean = _clean(values)
    if not np.any(clean > 0.0):
        return float(clean.size // 2)
    peak = int(np.argmax(clean))
    left = max(0, peak - int(half_window))
    right = min(clean.size, peak + int(half_window) + 1)
    local = clean[left:right]
    background = edge_background(local, max(min(local.size // 8, 32), 1))
    weights = np.clip(local - background, 0.0, None)
    mass = float(np.sum(weights))
    if mass <= 1e-15:
        return float(peak)
    axis = np.arange(left, right, dtype=np.float64)
    return float(np.sum(axis * weights) / mass)


def gaussian_center_index(values: np.ndarray, half_window: int) -> float:
    """Fit the current output histogram only; no sequence correction is used."""

    clean = _clean(values)
    coarse = local_center_of_mass(clean, half_window)
    center = int(np.clip(round(coarse), 0, clean.size - 1))
    left = max(0, center - int(half_window))
    right = min(clean.size, center + int(half_window) + 1)
    x = np.arange(left, right, dtype=np.float64)
    y = clean[left:right]
    background = edge_background(clean, max(min(clean.size // 8, 160), 1))
    amplitude = max(float(np.max(y)) - background, 1e-12)
    sigma = max(float(half_window) / 3.0, 10.0)

    def model(
        xx: np.ndarray, height: float, location: float, width: float, floor: float
    ) -> np.ndarray:
        return floor + height * np.exp(-0.5 * np.square((xx - location) / width))

    try:
        fitted, _ = curve_fit(
            model,
            x,
            y,
            p0=(amplitude, coarse, sigma, background),
            bounds=(
                (0.0, coarse - half_window, 5.0, 0.0),
                (
                    max(float(np.max(y)) * 10.0, 1.0),
                    coarse + half_window,
                    max(float(half_window), 10.0),
                    max(float(np.max(y)), 1.0),
                ),
            ),
            sigma=np.sqrt(np.clip(y, 0.0, None) + 1.0),
            maxfev=4000,
        )
        return float(fitted[1])
    except (RuntimeError, ValueError, FloatingPointError):
        return coarse


def coarse_center_index(
    histogram: np.ndarray, smooth_sigma_bins: float, half_width_bins: int
) -> float:
    """Stateless smooth localization followed by a local signal centroid."""

    clean = _clean(histogram)
    if not np.any(clean > 0.0):
        return float(clean.size // 2)
    smooth = gaussian_filter1d(
        clean, max(float(smooth_sigma_bins), 0.0), mode="nearest"
    )
    peak = int(np.argmax(smooth))
    left = max(0, peak - int(half_width_bins))
    right = min(clean.size, peak + int(half_width_bins) + 1)
    local = clean[left:right]
    background = edge_background(local, max(min(local.size // 8, 120), 1))
    weights = np.clip(local - background, 0.0, None)
    mass = float(np.sum(weights))
    if mass <= 1e-15:
        return float(peak)
    axis = np.arange(left, right, dtype=np.float64)
    return float(np.sum(axis * weights) / mass)


@dataclass(frozen=True)
class CompensationResult:
    raw_local: np.ndarray
    compensated: np.ndarray
    time_ps: np.ndarray
    center_ps: float
    coarse_center_ps: float
    direction: int
    input_counts: float
    output_counts: float
    input_fwhm_ps: float
    output_fwhm_ps: float
    expected_fisher_gain: float
    config_sha256: str


class _RichardsonLucyOperator:
    def __init__(self, kernels: DirectionKernels, config: FrozenConfig) -> None:
        self.kernels = kernels
        self.settings = config.operator

    def infer(self, histogram: np.ndarray) -> np.ndarray:
        observed = _clean(histogram)
        if observed.ndim != 1 or observed.size != self.settings.kernel_bins:
            raise ValueError(
                f"Expected a {self.settings.kernel_bins}-bin local histogram"
            )
        total = float(np.sum(observed))
        if total <= 1e-15:
            return np.zeros_like(observed)
        background = edge_background(observed, self.settings.edge_bins)
        signal = np.clip(observed - background, 0.0, None)
        signal_mass = float(np.sum(signal))
        if signal_mass <= 1e-15:
            return observed.copy()
        probability = signal / signal_mass
        latent = np.clip(
            probability, self.settings.latent_floor_fraction, None
        )
        latent /= float(np.sum(latent))
        reverse_psf = self.kernels.broad[::-1]
        for _ in range(self.settings.rl_iterations):
            projection = fftconvolve(latent, self.kernels.broad, mode="same")
            ratio = probability / np.clip(projection, 1e-15, None)
            ratio = np.clip(ratio, 0.0, self.settings.ratio_clip)
            latent *= fftconvolve(ratio, reverse_psf, mode="same")
            latent = np.clip(
                latent, self.settings.latent_floor_fraction, None
            )
            latent /= max(float(np.sum(latent)), 1e-15)
        reconstructed = fftconvolve(latent, self.kernels.target, mode="same")
        output = _normalize(reconstructed) * signal_mass + background
        output *= total / max(float(np.sum(output)), 1e-15)
        return np.clip(output, 0.0, None)

    def infer_batch(self, histograms: np.ndarray) -> np.ndarray:
        """Vectorized form of the same frozen operator for blind batch audits."""

        observed = _clean(histograms)
        if observed.ndim != 2 or observed.shape[1] != self.settings.kernel_bins:
            raise ValueError(
                f"Expected a batch of {self.settings.kernel_bins}-bin histograms"
            )
        length = observed.shape[1]
        total = np.sum(observed, axis=1, keepdims=True)
        edge = min(self.settings.edge_bins, length // 4)
        background = np.mean(
            np.concatenate((observed[:, :edge], observed[:, -edge:]), axis=1),
            axis=1,
            keepdims=True,
        )
        signal = np.clip(observed - background, 0.0, None)
        signal_mass = np.sum(signal, axis=1, keepdims=True)
        probability = signal / np.clip(signal_mass, 1e-15, None)
        latent = np.clip(probability, self.settings.latent_floor_fraction, None)
        latent /= np.clip(np.sum(latent, axis=1, keepdims=True), 1e-15, None)
        fft_length = next_fast_len(2 * length - 1)
        start = (length - 1) // 2

        def convolve(values: np.ndarray, kernel_fft: np.ndarray) -> np.ndarray:
            full = irfft(
                rfft(values, n=fft_length, axis=-1, workers=-1)
                * kernel_fft[None, :],
                n=fft_length,
                axis=-1,
                workers=-1,
            )
            return full[:, start : start + length]

        broad_fft = rfft(self.kernels.broad, n=fft_length, workers=-1)
        reverse_fft = rfft(self.kernels.broad[::-1], n=fft_length, workers=-1)
        target_fft = rfft(self.kernels.target, n=fft_length, workers=-1)
        for _ in range(self.settings.rl_iterations):
            projection = convolve(latent, broad_fft)
            ratio = np.clip(
                probability / np.clip(projection, 1e-15, None),
                0.0,
                self.settings.ratio_clip,
            )
            latent *= convolve(ratio, reverse_fft)
            latent = np.clip(latent, self.settings.latent_floor_fraction, None)
            latent /= np.clip(np.sum(latent, axis=1, keepdims=True), 1e-15, None)
        reconstructed = np.clip(convolve(latent, target_fft), 0.0, None)
        reconstructed /= np.clip(
            np.sum(reconstructed, axis=1, keepdims=True), 1e-15, None
        )
        output = reconstructed * signal_mass + background
        output *= total / np.clip(np.sum(output, axis=1, keepdims=True), 1e-15, None)
        empty = np.ravel(total <= 1e-15)
        output[empty] = observed[empty]
        return np.clip(output, 0.0, None)


class V25Compensator:
    """Frozen neural-PSF, direction-specific, stateless compensator."""

    def __init__(self, config: FrozenConfig, neural_model: NeuralPSFModel) -> None:
        config.validate()
        self.config = config
        self.neural_model = neural_model
        self._kernels = {
            direction: build_direction_kernels(config, direction, neural_model)
            for direction in (1, 2)
        }
        self._operators = {
            direction: _RichardsonLucyOperator(self._kernels[direction], config)
            for direction in (1, 2)
        }

    @classmethod
    def from_frozen_json(
        cls, path: str | Path, require_hash: bool = True
    ) -> "V25Compensator":
        config_path = Path(path).resolve()
        config = load_frozen_config(config_path, require_hash=require_hash)
        model_path = Path(config.neural_psf_model)
        if not model_path.is_absolute():
            model_path = config_path.parent / model_path
        actual_hash = file_sha256(model_path)
        if actual_hash != config.neural_psf_sha256:
            raise ValueError(
                "Neural PSF SHA-256 mismatch: "
                f"expected {config.neural_psf_sha256}, got {actual_hash}"
            )
        return cls(config, NeuralPSFModel.load(model_path))

    def kernel_summary(self) -> dict[str, dict[str, float]]:
        return {
            f"direction{direction}": {
                "broad_fwhm_ps": kernels.broad_fwhm_ps,
                "target_fwhm_ps": kernels.target_fwhm_ps,
                "expected_fisher_gain": kernels.fisher_gain,
                "cropped_edge_mass": kernels.cropped_edge_mass,
                "broad_neural_width_scale": kernels.broad_neural_width_scale,
                "target_neural_width_scale": kernels.target_neural_width_scale,
                "broad_neural_confidence": kernels.broad_neural_confidence,
                "target_neural_confidence": kernels.target_neural_confidence,
            }
            for direction, kernels in self._kernels.items()
        }

    def infer_batch_local(self, histograms: np.ndarray, direction: int) -> np.ndarray:
        direction = int(direction)
        if direction not in (1, 2):
            raise ValueError("direction must be 1 or 2")
        return self._operators[direction].infer_batch(histograms)

    def direction_kernels(self, direction: int) -> DirectionKernels:
        direction = int(direction)
        if direction not in (1, 2):
            raise ValueError("direction must be 1 or 2")
        return self._kernels[direction]

    def infer_local(
        self,
        histogram: np.ndarray,
        direction: int,
        time_ps: np.ndarray | None = None,
    ) -> CompensationResult:
        direction = int(direction)
        if direction not in (1, 2):
            raise ValueError("direction must be 1 or 2")
        raw = _clean(histogram)
        settings = self.config.operator
        if raw.ndim != 1 or raw.size != settings.kernel_bins:
            raise ValueError(
                f"infer_local requires exactly {settings.kernel_bins} bins"
            )
        if time_ps is None:
            axis = (
                np.arange(settings.kernel_bins, dtype=np.float64)
                - settings.kernel_bins // 2
            ) * settings.bin_width_ps
        else:
            axis = np.asarray(time_ps, dtype=np.float64)
            if axis.shape != raw.shape:
                raise ValueError("time_ps and histogram must have equal shape")
            if not np.allclose(
                np.diff(axis), settings.bin_width_ps, rtol=0.0, atol=1e-6
            ):
                raise ValueError("time_ps spacing does not match bin_width_ps")
        coarse_index = coarse_center_index(
            raw,
            settings.localization_smooth_sigma_bins,
            settings.localization_half_width_bins,
        )
        compensated = self._operators[direction].infer(raw)
        center_index = poisson_template_center_index(
            compensated,
            self._kernels[direction].target,
            settings.edge_bins,
            max(settings.center_half_window_bins, 240),
        )
        index_axis = np.arange(raw.size, dtype=np.float64)
        center_ps = float(np.interp(center_index, index_axis, axis))
        coarse_center_ps = float(np.interp(coarse_index, index_axis, axis))
        input_signal = np.clip(
            raw - edge_background(raw, settings.edge_bins), 0.0, None
        )
        output_signal = np.clip(
            compensated - edge_background(compensated, settings.edge_bins),
            0.0,
            None,
        )
        return CompensationResult(
            raw_local=raw,
            compensated=compensated,
            time_ps=axis,
            center_ps=center_ps,
            coarse_center_ps=coarse_center_ps,
            direction=direction,
            input_counts=float(np.sum(raw)),
            output_counts=float(np.sum(compensated)),
            input_fwhm_ps=fwhm_ps(input_signal, settings.bin_width_ps),
            output_fwhm_ps=fwhm_ps(output_signal, settings.bin_width_ps),
            expected_fisher_gain=self._kernels[direction].fisher_gain,
            config_sha256=self.config.sha256(),
        )

    def infer_full(
        self,
        histogram: np.ndarray,
        direction: int,
        time_ps: np.ndarray | None = None,
    ) -> CompensationResult:
        """Localize a full fixed-axis histogram and return its local result."""

        observed = _clean(histogram)
        settings = self.config.operator
        if observed.ndim != 1 or observed.size < settings.kernel_bins:
            raise ValueError(
                f"Full histogram must contain at least {settings.kernel_bins} bins"
            )
        if observed.size == settings.kernel_bins:
            return self.infer_local(observed, direction, time_ps=time_ps)
        if time_ps is None:
            full_axis = np.arange(observed.size, dtype=np.float64) * settings.bin_width_ps
        else:
            full_axis = np.asarray(time_ps, dtype=np.float64)
            if full_axis.shape != observed.shape:
                raise ValueError("time_ps and histogram must have equal shape")
            if not np.allclose(
                np.diff(full_axis), settings.bin_width_ps, rtol=0.0, atol=1e-6
            ):
                raise ValueError("time_ps spacing does not match bin_width_ps")
        center_index = coarse_center_index(
            observed,
            settings.localization_smooth_sigma_bins,
            settings.localization_half_width_bins,
        )
        relative = np.arange(settings.kernel_bins, dtype=np.float64)
        relative -= settings.kernel_bins // 2
        sample_index = center_index + relative
        source_index = np.arange(observed.size, dtype=np.float64)
        local = np.interp(sample_index, source_index, observed, left=0.0, right=0.0)
        local_axis = np.interp(
            sample_index,
            source_index,
            full_axis,
            left=full_axis[0] + (sample_index[0] * settings.bin_width_ps),
            right=full_axis[-1]
            + ((sample_index[-1] - source_index[-1]) * settings.bin_width_ps),
        )
        return self.infer_local(local, direction, time_ps=local_axis)
