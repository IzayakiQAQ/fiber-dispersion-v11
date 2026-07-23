from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve


def _clean(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )


def normalize_probability(values: np.ndarray) -> np.ndarray:
    result = _clean(values)
    total = float(np.sum(result))
    if total <= 1e-12:
        result = np.zeros_like(result)
        result[result.size // 2] = 1.0
        return result
    return result / total


def edge_background(values: np.ndarray, edge_bins: int = 160) -> float:
    clean = _clean(values)
    width = min(max(int(edge_bins), 1), max(clean.size // 4, 1))
    edges = np.concatenate((clean[:width], clean[-width:]))
    return float(np.median(edges)) if edges.size else 0.0


def center_of_mass(values: np.ndarray, half_window: int = 180) -> float:
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
    if mass <= 1e-12:
        return float(peak)
    x = np.arange(left, right, dtype=np.float64)
    return float(np.sum(x * weights) / mass)


def fwhm_subbin(values: np.ndarray) -> float:
    clean = _clean(values)
    if clean.size < 3:
        return 0.0
    peak_idx = int(np.argmax(clean))
    peak = float(clean[peak_idx])
    if peak <= 1e-12:
        return 0.0
    half = 0.5 * peak
    left_candidates = np.flatnonzero(clean[: peak_idx + 1] < half)
    if left_candidates.size:
        i0 = int(left_candidates[-1])
        i1 = min(i0 + 1, peak_idx)
        denom = float(clean[i1] - clean[i0])
        left = float(i0) if abs(denom) <= 1e-12 else float(i0) + (half - float(clean[i0])) / denom
    else:
        left = 0.0
    right_candidates = np.flatnonzero(clean[peak_idx:] < half)
    if right_candidates.size:
        i1 = peak_idx + int(right_candidates[0])
        i0 = max(peak_idx, i1 - 1)
        denom = float(clean[i1] - clean[i0])
        right = float(i1) if abs(denom) <= 1e-12 else float(i0) + (half - float(clean[i0])) / denom
    else:
        right = float(clean.size - 1)
    return max(float(right - left), 0.0)


def gaussian_coarse_fit(
    histogram: np.ndarray,
    smooth_sigma: float = 24.0,
    fit_half_width: int = 800,
) -> tuple[float, float]:
    """Estimate center and sigma from one histogram without run-level state."""

    clean = _clean(histogram)
    if not np.any(clean > 0.0):
        return float(clean.size // 2), 0.0
    smooth = gaussian_filter1d(clean, max(float(smooth_sigma), 0.0), mode="nearest")
    peak = int(np.argmax(smooth))
    left = max(0, peak - int(fit_half_width))
    right = min(clean.size, peak + int(fit_half_width) + 1)
    x = np.arange(left, right, dtype=np.float64)
    y = clean[left:right]
    if y.size < 16:
        return float(peak), 0.0
    edge = min(120, max(y.size // 8, 1))
    background = float(np.median(np.concatenate((y[:edge], y[-edge:]))))
    signal = np.clip(y - background, 0.0, None)
    mass = float(np.sum(signal))
    moment_center = float(np.sum(x * signal) / mass) if mass > 1e-12 else float(peak)
    variance = float(np.sum((x - moment_center) ** 2 * signal) / mass) if mass > 1e-12 else 220.0**2
    sigma0 = float(np.clip(np.sqrt(max(variance, 1.0)), 40.0, 500.0))

    def gaussian_with_floor(xx: np.ndarray, amplitude: float, center: float, sigma: float, floor: float) -> np.ndarray:
        return floor + amplitude * np.exp(-0.5 * ((xx - center) / sigma) ** 2)

    try:
        params, _ = curve_fit(
            gaussian_with_floor,
            x,
            y,
            p0=(max(float(np.max(y)) - background, 1.0), moment_center, sigma0, max(background, 0.0)),
            bounds=(
                (0.0, float(left), 20.0, 0.0),
                (max(float(np.max(y)) * 5.0, 1.0), float(right - 1), 800.0, max(float(np.max(y)), 1.0)),
            ),
            maxfev=4000,
        )
        return float(params[1]), float(params[2])
    except (RuntimeError, ValueError, FloatingPointError):
        return moment_center, sigma0


def gaussian_coarse_center(
    histogram: np.ndarray,
    smooth_sigma: float = 24.0,
    fit_half_width: int = 800,
) -> float:
    """Estimate a coarse center from one histogram without run-level state."""

    center, _ = gaussian_coarse_fit(histogram, smooth_sigma=smooth_sigma, fit_half_width=fit_half_width)
    return center


@dataclass(frozen=True)
class DirectCompensatorConfig:
    iterations: int = 8
    ratio_clip: float = 8.0
    edge_bins: int = 160
    latent_floor_fraction: float = 1e-8


class PhysicalDirectHistogramCompensator:
    """Stateless nonnegative histogram-to-histogram dispersion compensator.

    The broad PSF and target 0 km PSF are fixed model parameters. Every call to
    ``infer`` uses only the supplied histogram and returns the compensated
    histogram itself. No clock-series state or post-output center correction is
    involved.
    """

    def __init__(
        self,
        broad_psf: np.ndarray,
        target_psf: np.ndarray,
        config: DirectCompensatorConfig | None = None,
    ) -> None:
        self.broad_psf = normalize_probability(broad_psf)
        self.target_psf = normalize_probability(target_psf)
        self.config = config or DirectCompensatorConfig()

    def infer(self, histogram: np.ndarray) -> np.ndarray:
        observed = _clean(histogram)
        if observed.ndim != 1:
            raise ValueError(f"Expected one 1-D histogram, got shape {observed.shape}")
        if observed.size != self.broad_psf.size or observed.size != self.target_psf.size:
            raise ValueError(
                "Histogram, broad PSF and target PSF must have the same length: "
                f"{observed.size}, {self.broad_psf.size}, {self.target_psf.size}"
            )
        total = float(np.sum(observed))
        if total <= 1e-12:
            return np.zeros_like(observed)

        background = edge_background(observed, self.config.edge_bins)
        signal = np.clip(observed - background, 0.0, None)
        signal_mass = float(np.sum(signal))
        if signal_mass <= 1e-12:
            return observed.copy()
        probability = signal / signal_mass

        # Starting from the observation is stable at low photon count. The
        # multiplicative RL update preserves nonnegativity by construction.
        latent = np.clip(probability, self.config.latent_floor_fraction, None)
        latent /= float(np.sum(latent))
        reverse_psf = self.broad_psf[::-1]
        ratio_clip = max(float(self.config.ratio_clip), 1.0)
        for _ in range(max(int(self.config.iterations), 0)):
            projection = fftconvolve(latent, self.broad_psf, mode="same")
            ratio = probability / np.clip(projection, 1e-12, None)
            ratio = np.clip(ratio, 0.0, ratio_clip)
            latent *= fftconvolve(ratio, reverse_psf, mode="same")
            latent = np.clip(latent, self.config.latent_floor_fraction, None)
            latent /= max(float(np.sum(latent)), 1e-12)

        reconstructed = fftconvolve(latent, self.target_psf, mode="same")
        reconstructed = normalize_probability(reconstructed)
        output = reconstructed * signal_mass + background
        # Numerical convolution/cropping can lose tiny edge mass. Correct the
        # area without changing the reconstructed shape or introducing negatives.
        output *= total / max(float(np.sum(output)), 1e-12)
        return np.clip(output, 0.0, None)

    def infer_full(self, histogram: np.ndarray) -> np.ndarray:
        """Compensate one fixed-axis histogram and return the same array shape."""

        observed = _clean(histogram)
        total = float(np.sum(observed))
        if total <= 1e-12:
            return np.zeros_like(observed)
        center = gaussian_coarse_center(observed)
        half_width = self.broad_psf.size // 2
        relative = np.arange(-half_width, half_width + 1, dtype=np.float64)
        source_grid = np.arange(observed.size, dtype=np.float64)
        local = np.interp(center + relative, source_grid, observed, left=0.0, right=0.0)
        local_output = self.infer(local)
        full_output = np.interp(source_grid, center + relative, local_output, left=0.0, right=0.0)
        full_output = np.clip(full_output, 0.0, None)
        full_output *= total / max(float(np.sum(full_output)), 1e-12)
        return full_output
