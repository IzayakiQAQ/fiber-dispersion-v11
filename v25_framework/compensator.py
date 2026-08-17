from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve

from .config import FrozenConfig, load_frozen_config
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
    return float(np.median(np.concatenate((clean[:width], clean[-width:]))))


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


class V25Compensator:
    """Frozen, direction-specific, stateless histogram compensator."""

    def __init__(self, config: FrozenConfig) -> None:
        config.validate()
        self.config = config
        self._kernels = {
            direction: build_direction_kernels(config, direction)
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
        return cls(load_frozen_config(path, require_hash=require_hash))

    def kernel_summary(self) -> dict[str, dict[str, float]]:
        return {
            f"direction{direction}": {
                "broad_fwhm_ps": kernels.broad_fwhm_ps,
                "target_fwhm_ps": kernels.target_fwhm_ps,
                "expected_fisher_gain": kernels.fisher_gain,
                "cropped_edge_mass": kernels.cropped_edge_mass,
            }
            for direction, kernels in self._kernels.items()
        }

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
        center_index = local_center_of_mass(
            compensated, settings.center_half_window_bins
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
