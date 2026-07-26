from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from ..direct_histogram_compensator import (
        DirectCompensatorConfig,
        PhysicalDirectHistogramCompensator,
        center_of_mass,
    )
except ImportError:
    from direct_histogram_compensator import (
        DirectCompensatorConfig,
        PhysicalDirectHistogramCompensator,
        center_of_mass,
    )
from .forward_model import PhysicsHistogramGenerator, fwhm_ps


@dataclass(frozen=True)
class AdaptiveCompensatorConfig:
    n_bins: int = 16385
    bin_width_ps: float = 1.0
    edge_bins: int = 512
    localization_smooth_sigma_bins: float = 256.0
    shape_smooth_sigma_bins: float = 16.0
    min_signal_counts: float = 100.0
    maximum_js_divergence: float = 0.22
    minimum_fisher_gain: float = 1.05
    minimum_iterations: int = 16
    maximum_iterations: int = 512
    reference_signal_counts: float = 1000.0
    ratio_clip: float = 8.0
    latent_floor_fraction: float = 1e-8


@dataclass(frozen=True)
class AdaptiveCompensationResult:
    compensated_counts: np.ndarray
    absolute_time_ps: np.ndarray
    center_ps: float
    inferred_length_km: float
    inferred_bandwidth_nm: float
    direction: int
    iterations: int
    js_divergence: float
    fisher_gain: float
    gated_to_identity: bool


@dataclass(frozen=True)
class _Prepared:
    counts: np.ndarray
    signal_probability: np.ndarray
    signal_counts: float
    background_per_bin: float
    absolute_time_ps: np.ndarray
    coarse_center_abs_ps: float


def _normalize(values: np.ndarray) -> np.ndarray:
    clean = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(np.sum(clean))
    if total <= 1e-300:
        return np.full(clean.size, 1.0 / clean.size, dtype=np.float64)
    return clean / total


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    p = _normalize(left)
    q = _normalize(right)
    midpoint = 0.5 * (p + q)
    p_mask = p > 0.0
    q_mask = q > 0.0
    p_term = np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
    q_term = np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    return float(0.5 * (p_term + q_term))


def _translation_fisher(probability: np.ndarray, bin_width_ps: float) -> float:
    p = _normalize(probability)
    derivative = np.gradient(p, float(bin_width_ps))
    return float(np.sum(np.square(derivative) / np.clip(p, 1e-15, None)))


def _poisson_edge_background(values: np.ndarray, edge_bins: int) -> float:
    clean = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    width = min(max(int(edge_bins), 1), max(clean.size // 4, 1))
    return float(np.mean(np.concatenate((clean[:width], clean[-width:]))))


class PhysicsAdaptiveCompensator:
    """Select a physical response from one histogram and compensate it.

    Length and bandwidth labels are not needed during inference. They index a
    calibrated response manifold and are returned as interpretable effective
    parameters. Ambiguous or unsupported inputs are passed through unchanged.
    """

    def __init__(
        self,
        generator: PhysicsHistogramGenerator,
        candidate_conditions: list[tuple[float, float]],
        config: AdaptiveCompensatorConfig | None = None,
    ) -> None:
        if not candidate_conditions:
            raise ValueError("candidate_conditions must not be empty")
        self.generator = generator
        self.candidate_conditions = sorted(
            {(float(length), float(bandwidth)) for length, bandwidth in candidate_conditions}
        )
        self.config = config or AdaptiveCompensatorConfig()
        if self.config.n_bins % 2 != 1:
            raise ValueError("Adaptive n_bins must be odd")
        self._bank: dict[tuple[int, float, float], tuple[np.ndarray, np.ndarray]] = {}

    def _responses(
        self, direction: int, length_km: float, bandwidth_nm: float
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (int(direction), float(length_km), float(bandwidth_nm))
        if key not in self._bank:
            _, broad = self.generator.probability(
                length_km,
                bandwidth_nm,
                direction,
                n_bins=self.config.n_bins,
                bin_width_ps=self.config.bin_width_ps,
            )
            _, target = self.generator.probability(
                0.0,
                bandwidth_nm,
                direction,
                n_bins=self.config.n_bins,
                bin_width_ps=self.config.bin_width_ps,
            )
            edge = max(int(0.05 * broad.size), 1)
            edge_mass = float(np.sum(broad[:edge]) + np.sum(broad[-edge:]))
            if edge_mass > 0.01:
                raise ValueError(
                    f"Candidate L={length_km:g} km, B={bandwidth_nm:g} nm exceeds "
                    f"the {self.config.n_bins}-bin adaptive window"
                )
            self._bank[key] = (broad, target)
        return self._bank[key]

    def _prepare(
        self, histogram: np.ndarray, absolute_time_ps: np.ndarray | None
    ) -> _Prepared:
        values = np.clip(np.asarray(histogram, dtype=np.float64), 0.0, None)
        if values.ndim != 1 or values.size < self.config.n_bins:
            raise ValueError(
                f"Histogram must be one-dimensional with at least {self.config.n_bins} bins"
            )
        if absolute_time_ps is None:
            axis = np.arange(values.size, dtype=np.float64) * self.config.bin_width_ps
        else:
            axis = np.asarray(absolute_time_ps, dtype=np.float64)
            if axis.shape != values.shape:
                raise ValueError("absolute_time_ps and histogram must have equal shape")
            if not np.allclose(
                np.diff(axis), self.config.bin_width_ps, rtol=0.0, atol=1e-6
            ):
                raise ValueError("Histogram axis spacing does not match bin_width_ps")

        background = _poisson_edge_background(values, self.config.edge_bins)
        smooth = gaussian_filter1d(
            values,
            self.config.localization_smooth_sigma_bins,
            mode="nearest",
        )
        peak = int(np.argmax(smooth))
        half = self.config.n_bins // 2
        left = max(0, peak - half)
        right = min(values.size, peak + half + 1)
        denoised = np.clip(
            gaussian_filter1d(
                values, self.config.shape_smooth_sigma_bins, mode="nearest"
            )
            - background,
            0.0,
            None,
        )
        local_signal = denoised[left:right]
        local_axis = axis[left:right]
        mass = float(np.sum(local_signal))
        if mass <= 0.0:
            center_abs = float(axis[peak])
        else:
            center_abs = float(np.sum(local_axis * local_signal) / mass)

        relative = (
            np.arange(self.config.n_bins, dtype=np.float64) - half
        ) * self.config.bin_width_ps
        local_counts = np.interp(center_abs + relative, axis, values, left=0.0, right=0.0)
        local_background = _poisson_edge_background(local_counts, self.config.edge_bins)
        local_signal = np.clip(
            gaussian_filter1d(
                local_counts, self.config.shape_smooth_sigma_bins, mode="nearest"
            )
            - local_background,
            0.0,
            None,
        )
        local_mass = max(
            float(np.sum(local_counts)) - local_background * local_counts.size,
            0.0,
        )
        return _Prepared(
            counts=local_counts,
            signal_probability=_normalize(local_signal),
            signal_counts=local_mass,
            background_per_bin=local_background,
            absolute_time_ps=center_abs + relative,
            coarse_center_abs_ps=center_abs,
        )

    def infer(
        self,
        histogram: np.ndarray,
        direction: int,
        absolute_time_ps: np.ndarray | None = None,
    ) -> AdaptiveCompensationResult:
        direction = int(direction)
        if direction not in (1, 2):
            raise ValueError("direction must be 1 or 2")
        prepared = self._prepare(histogram, absolute_time_ps)

        ranked: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
        for length_km, bandwidth_nm in self.candidate_conditions:
            try:
                broad, target = self._responses(direction, length_km, bandwidth_nm)
            except ValueError:
                continue
            broad_for_score = gaussian_filter1d(
                broad, self.config.shape_smooth_sigma_bins, mode="constant"
            )
            score = _jensen_shannon(prepared.signal_probability, broad_for_score)
            ranked.append((score, length_km, bandwidth_nm, broad, target))
        if not ranked:
            raise ValueError("No candidate physical responses fit the configured time window")
        ranked.sort(key=lambda item: item[0])
        score, length_km, bandwidth_nm, broad, target = ranked[0]

        observed_fisher = _translation_fisher(broad, self.config.bin_width_ps)
        target_fisher = _translation_fisher(target, self.config.bin_width_ps)
        fisher_gain = target_fisher / max(observed_fisher, 1e-15)
        gated = bool(
            prepared.signal_counts < self.config.min_signal_counts
            or score > self.config.maximum_js_divergence
            or fisher_gain < self.config.minimum_fisher_gain
        )

        count_scale = np.sqrt(
            prepared.signal_counts / max(self.config.reference_signal_counts, 1e-12)
        )
        iterations = int(
            np.clip(
                round(self.config.maximum_iterations * min(count_scale, 1.0)),
                self.config.minimum_iterations,
                self.config.maximum_iterations,
            )
        )
        if gated:
            output = prepared.counts.copy()
            iterations = 0
        else:
            operator = PhysicalDirectHistogramCompensator(
                broad,
                target,
                DirectCompensatorConfig(
                    iterations=iterations,
                    ratio_clip=self.config.ratio_clip,
                    edge_bins=self.config.edge_bins,
                    latent_floor_fraction=self.config.latent_floor_fraction,
                ),
            )
            signal_input = np.clip(
                prepared.counts - prepared.background_per_bin, 0.0, None
            )
            output = operator.infer(signal_input) + prepared.background_per_bin
            output *= float(np.sum(prepared.counts)) / max(float(np.sum(output)), 1e-12)

        output_background = _poisson_edge_background(output, self.config.edge_bins)
        output_for_center = np.clip(
            gaussian_filter1d(
                output, self.config.shape_smooth_sigma_bins, mode="nearest"
            )
            - output_background,
            0.0,
            None,
        )
        relative_center_bins = center_of_mass(
            output_for_center, half_window=max(180, self.config.n_bins // 16)
        )
        center_index_offset = relative_center_bins - float(self.config.n_bins // 2)
        center_ps = prepared.coarse_center_abs_ps + center_index_offset * self.config.bin_width_ps
        return AdaptiveCompensationResult(
            compensated_counts=output,
            absolute_time_ps=prepared.absolute_time_ps,
            center_ps=float(center_ps),
            inferred_length_km=length_km,
            inferred_bandwidth_nm=bandwidth_nm,
            direction=direction,
            iterations=iterations,
            js_divergence=score,
            fisher_gain=fisher_gain,
            gated_to_identity=gated,
        )

    def candidate_widths_ps(self, direction: int) -> list[dict[str, float]]:
        half = self.config.n_bins // 2
        time_ps = (
            np.arange(self.config.n_bins, dtype=np.float64) - half
        ) * self.config.bin_width_ps
        rows: list[dict[str, float]] = []
        for length_km, bandwidth_nm in self.candidate_conditions:
            broad, target = self._responses(direction, length_km, bandwidth_nm)
            rows.append(
                {
                    "length_km": length_km,
                    "bandwidth_nm": bandwidth_nm,
                    "broad_fwhm_ps": fwhm_ps(time_ps, broad),
                    "target_fwhm_ps": fwhm_ps(time_ps, target),
                }
            )
        return rows
