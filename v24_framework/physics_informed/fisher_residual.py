from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from ..direct_histogram_compensator import center_of_mass
except ImportError:
    from direct_histogram_compensator import center_of_mass

from .adaptive_compensator import (
    AdaptiveCompensationResult,
    PhysicsAdaptiveCompensator,
)


@dataclass(frozen=True)
class FisherResidualConfig:
    bin_width_ps: float = 1.0
    edge_bins: int = 160
    template_smoothing_sigma_bins: float = 12.0
    center_half_window_bins: int = 220
    minimum_signal_counts: float = 100.0
    minimum_fisher_information_per_ps2: float = 0.04
    maximum_newton_steps: int = 8
    maximum_newton_step_ps: float = 4.0
    maximum_residual_shift_ps: float = 100.0


@dataclass(frozen=True)
class PoissonCenterEstimate:
    center_ps: float
    residual_shift_ps: float
    fisher_information_per_ps2: float
    signal_counts: float
    background_per_bin: float
    gate_passed: bool


@dataclass(frozen=True)
class FisherResidualResult:
    compensated_counts: np.ndarray
    center_ps: float
    physics_center_ps: float
    poisson_center_ps: float
    applied_shift_ps: float
    fisher_information_per_ps2: float
    signal_counts: float
    gate_passed: bool


@dataclass(frozen=True)
class PhysicsFisherCompensationResult:
    compensated_counts: np.ndarray
    absolute_time_ps: np.ndarray
    center_ps: float
    direction: int
    physics_result: AdaptiveCompensationResult
    fisher_result: FisherResidualResult


def _clean(values: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(
            np.asarray(values, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
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


def _edge_mean(values: np.ndarray, edge_bins: int) -> float:
    clean = _clean(values)
    width = min(max(int(edge_bins), 1), max(clean.size // 4, 1))
    return float(np.mean(np.concatenate((clean[:width], clean[-width:]))))


def _direction_index(direction: int) -> int:
    value = int(direction)
    if value not in (1, 2):
        raise ValueError("direction must be 1 or 2")
    return value - 1


def cross_power_clock_crlb_ps(
    template_first: np.ndarray,
    template_second: np.ndarray,
    signal_counts: np.ndarray,
    background_per_bin: np.ndarray,
    bin_width_ps: float = 1.0,
) -> np.ndarray:
    """Estimate the two-way clock CRLB from independent template halves.

    The derivative cross product removes the positive Fisher bias caused by
    finite-count texture that appears in only one calibration half.
    """

    first = np.asarray(template_first, dtype=np.float64)
    second = np.asarray(template_second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != 2:
        raise ValueError("template_first and template_second must have shape (2, bins)")
    counts = np.asarray(signal_counts, dtype=np.float64)
    background = np.asarray(background_per_bin, dtype=np.float64)
    if counts.ndim == 1:
        counts = counts[:, None]
    if background.ndim == 1:
        background = background[:, None]
    if counts.shape != background.shape or counts.shape[0] != 2:
        raise ValueError("signal_counts and background_per_bin must have shape (2, samples)")

    information = np.zeros_like(counts)
    for direction in range(2):
        first_probability = _normalize(first[direction])
        second_probability = _normalize(second[direction])
        combined = _normalize(0.5 * (first_probability + second_probability))
        derivative_first = np.gradient(first_probability, float(bin_width_ps))
        derivative_second = np.gradient(second_probability, float(bin_width_ps))
        expected = (
            background[direction, :, None]
            + counts[direction, :, None] * combined[None, :]
        )
        information[direction] = np.sum(
            counts[direction, :, None] ** 2
            * (derivative_first * derivative_second)[None, :]
            / np.clip(expected, 1e-15, None),
            axis=-1,
        )
    valid = (information[0] > 0.0) & (information[1] > 0.0)
    result = np.full(counts.shape[1], np.inf, dtype=np.float64)
    result[valid] = np.sqrt(
        0.25
        * (
            1.0 / information[0, valid]
            + 1.0 / information[1, valid]
        )
    )
    return result


class PhysicsFisherResidualCorrector:
    """Stateless Poisson-center alignment for a physics-RL histogram.

    A direction-specific broad-response template is calibrated offline. At
    inference, the raw histogram supplies a Poisson maximum-likelihood center.
    When its Fisher information passes the no-harm gate, the already generated
    physics-RL histogram is translated to that center. No adjacent histogram,
    run-level mean, or bounded center update is used.
    """

    def __init__(
        self,
        templates: np.ndarray,
        config: FisherResidualConfig | None = None,
    ) -> None:
        values = np.asarray(templates, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 2:
            raise ValueError("templates must have shape (2, bins)")
        if values.shape[1] % 2 != 1:
            raise ValueError("template length must be odd")
        self.templates = np.stack([_normalize(values[0]), _normalize(values[1])])
        self.config = config or FisherResidualConfig()
        if self.config.bin_width_ps <= 0.0:
            raise ValueError("bin_width_ps must be positive")

    @classmethod
    def calibrate(
        cls,
        histograms: np.ndarray,
        coarse_centers_ps: np.ndarray,
        alignment_centers_ps: np.ndarray | None = None,
        config: FisherResidualConfig | None = None,
    ) -> "PhysicsFisherResidualCorrector":
        settings = config or FisherResidualConfig()
        values = np.asarray(histograms, dtype=np.float64)
        coarse = np.asarray(coarse_centers_ps, dtype=np.float64)
        alignment = coarse if alignment_centers_ps is None else np.asarray(
            alignment_centers_ps, dtype=np.float64
        )
        if values.ndim != 3 or values.shape[0] != 2:
            raise ValueError("histograms must have shape (2, samples, bins)")
        if coarse.shape != values.shape[:2] or alignment.shape != coarse.shape:
            raise ValueError("center arrays must have shape (2, samples)")
        if values.shape[-1] % 2 != 1:
            raise ValueError("histogram length must be odd")

        length = values.shape[-1]
        midpoint = length // 2
        relative_ps = (
            np.arange(length, dtype=np.float64) - midpoint
        ) * settings.bin_width_ps
        templates = np.zeros((2, length), dtype=np.float64)
        for direction in range(2):
            for sample in range(values.shape[1]):
                background = _edge_mean(values[direction, sample], settings.edge_bins)
                signal = np.clip(values[direction, sample] - background, 0.0, None)
                residual_ps = alignment[direction, sample] - coarse[direction, sample]
                templates[direction] += np.interp(
                    relative_ps + residual_ps,
                    relative_ps,
                    signal,
                    left=0.0,
                    right=0.0,
                )
            templates[direction] = gaussian_filter1d(
                templates[direction],
                settings.template_smoothing_sigma_bins,
                mode="nearest",
            )
        return cls(templates, settings)

    def estimate_center(
        self,
        histogram: np.ndarray,
        direction: int,
        coarse_center_ps: float,
    ) -> PoissonCenterEstimate:
        direction_index = _direction_index(direction)
        observed = _clean(histogram)
        template = self.templates[direction_index]
        if observed.shape != template.shape:
            raise ValueError("histogram and template must have equal shape")

        background = _edge_mean(observed, self.config.edge_bins)
        signal_counts = float(np.sum(np.clip(observed - background, 0.0, None)))
        relative_ps = (
            np.arange(observed.size, dtype=np.float64) - observed.size // 2
        ) * self.config.bin_width_ps
        derivative = np.gradient(template, self.config.bin_width_ps)
        shift_ps = 0.0
        information = 0.0
        for _ in range(max(int(self.config.maximum_newton_steps), 0)):
            shifted = np.interp(
                relative_ps - shift_ps,
                relative_ps,
                template,
                left=0.0,
                right=0.0,
            )
            shifted_derivative = -np.interp(
                relative_ps - shift_ps,
                relative_ps,
                derivative,
                left=0.0,
                right=0.0,
            )
            expected = np.clip(
                background + signal_counts * shifted, 1e-12, None
            )
            model_derivative = signal_counts * shifted_derivative
            score = float(np.sum((observed / expected - 1.0) * model_derivative))
            information = float(np.sum(model_derivative**2 / expected))
            if information <= 1e-15:
                break
            increment_ps = float(
                np.clip(
                    score / information,
                    -self.config.maximum_newton_step_ps,
                    self.config.maximum_newton_step_ps,
                )
            )
            shift_ps = float(
                np.clip(
                    shift_ps + increment_ps,
                    -self.config.maximum_residual_shift_ps,
                    self.config.maximum_residual_shift_ps,
                )
            )
            if abs(increment_ps) < 1e-4:
                break

        gate_passed = bool(
            signal_counts >= self.config.minimum_signal_counts
            and information >= self.config.minimum_fisher_information_per_ps2
        )
        return PoissonCenterEstimate(
            center_ps=float(coarse_center_ps + shift_ps),
            residual_shift_ps=shift_ps,
            fisher_information_per_ps2=information,
            signal_counts=signal_counts,
            background_per_bin=background,
            gate_passed=gate_passed,
        )

    def _compensated_center(
        self, compensated: np.ndarray, coarse_center_ps: float
    ) -> float:
        center_bins = center_of_mass(
            compensated,
            half_window=self.config.center_half_window_bins,
        )
        offset_bins = center_bins - compensated.size // 2
        return float(coarse_center_ps + offset_bins * self.config.bin_width_ps)

    def align_compensated_histogram(
        self,
        raw_histogram: np.ndarray,
        compensated_histogram: np.ndarray,
        direction: int,
        coarse_center_ps: float,
    ) -> FisherResidualResult:
        raw = _clean(raw_histogram)
        compensated = _clean(compensated_histogram)
        if raw.shape != compensated.shape or raw.shape != self.templates[0].shape:
            raise ValueError("raw, compensated, and template arrays must have equal shape")

        physics_center = self._compensated_center(compensated, coarse_center_ps)
        estimate = self.estimate_center(raw, direction, coarse_center_ps)
        desired_center = estimate.center_ps if estimate.gate_passed else physics_center
        applied_shift = desired_center - physics_center
        relative_ps = (
            np.arange(compensated.size, dtype=np.float64) - compensated.size // 2
        ) * self.config.bin_width_ps
        output = np.interp(
            relative_ps - applied_shift,
            relative_ps,
            compensated,
            left=0.0,
            right=0.0,
        )
        output = np.clip(output, 0.0, None)
        output *= float(np.sum(compensated)) / max(float(np.sum(output)), 1e-15)
        final_center = self._compensated_center(output, coarse_center_ps)
        return FisherResidualResult(
            compensated_counts=output,
            center_ps=final_center,
            physics_center_ps=physics_center,
            poisson_center_ps=estimate.center_ps,
            applied_shift_ps=applied_shift,
            fisher_information_per_ps2=estimate.fisher_information_per_ps2,
            signal_counts=estimate.signal_counts,
            gate_passed=estimate.gate_passed,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            template_direction1=self.templates[0],
            template_direction2=self.templates[1],
            bin_width_ps=np.asarray(self.config.bin_width_ps),
            edge_bins=np.asarray(self.config.edge_bins),
            template_smoothing_sigma_bins=np.asarray(
                self.config.template_smoothing_sigma_bins
            ),
            center_half_window_bins=np.asarray(self.config.center_half_window_bins),
            minimum_signal_counts=np.asarray(self.config.minimum_signal_counts),
            minimum_fisher_information_per_ps2=np.asarray(
                self.config.minimum_fisher_information_per_ps2
            ),
            maximum_newton_steps=np.asarray(self.config.maximum_newton_steps),
            maximum_newton_step_ps=np.asarray(self.config.maximum_newton_step_ps),
            maximum_residual_shift_ps=np.asarray(
                self.config.maximum_residual_shift_ps
            ),
            uses_adjacent_histograms=np.asarray(False),
            bounded_center_correction=np.asarray(False),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsFisherResidualCorrector":
        with np.load(Path(path), allow_pickle=False) as data:
            templates = np.stack(
                (data["template_direction1"], data["template_direction2"])
            )
            config = FisherResidualConfig(
                bin_width_ps=float(data["bin_width_ps"]),
                edge_bins=int(data["edge_bins"]),
                template_smoothing_sigma_bins=float(
                    data["template_smoothing_sigma_bins"]
                ),
                center_half_window_bins=int(data["center_half_window_bins"]),
                minimum_signal_counts=float(data["minimum_signal_counts"]),
                minimum_fisher_information_per_ps2=float(
                    data["minimum_fisher_information_per_ps2"]
                ),
                maximum_newton_steps=int(data["maximum_newton_steps"]),
                maximum_newton_step_ps=float(data["maximum_newton_step_ps"]),
                maximum_residual_shift_ps=float(data["maximum_residual_shift_ps"]),
            )
        return cls(templates, config)


class PhysicsFisherCompensationPipeline:
    """One-call full-histogram physics RL plus Fisher residual inference."""

    def __init__(
        self,
        physics_compensator: PhysicsAdaptiveCompensator,
        residual_corrector: PhysicsFisherResidualCorrector,
    ) -> None:
        if physics_compensator.config.n_bins != residual_corrector.templates.shape[1]:
            raise ValueError(
                "physics compensator and residual templates must use equal local lengths"
            )
        if not np.isclose(
            physics_compensator.config.bin_width_ps,
            residual_corrector.config.bin_width_ps,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "physics compensator and residual corrector must use equal bin widths"
            )
        self.physics_compensator = physics_compensator
        self.residual_corrector = residual_corrector

    def infer(
        self,
        histogram: np.ndarray,
        direction: int,
        absolute_time_ps: np.ndarray | None = None,
    ) -> PhysicsFisherCompensationResult:
        values = _clean(histogram)
        if values.ndim != 1:
            raise ValueError("histogram must be one-dimensional")
        if absolute_time_ps is None:
            full_axis = (
                np.arange(values.size, dtype=np.float64)
                * self.physics_compensator.config.bin_width_ps
            )
        else:
            full_axis = np.asarray(absolute_time_ps, dtype=np.float64)
            if full_axis.shape != values.shape:
                raise ValueError("absolute_time_ps and histogram must have equal shape")

        physics_result = self.physics_compensator.infer(
            values,
            direction=direction,
            absolute_time_ps=full_axis,
        )
        raw_local = np.interp(
            physics_result.absolute_time_ps,
            full_axis,
            values,
            left=0.0,
            right=0.0,
        )
        coarse_center_ps = float(
            physics_result.absolute_time_ps[physics_result.absolute_time_ps.size // 2]
        )
        fisher_result = self.residual_corrector.align_compensated_histogram(
            raw_local,
            physics_result.compensated_counts,
            direction=direction,
            coarse_center_ps=coarse_center_ps,
        )
        if physics_result.gated_to_identity and fisher_result.gate_passed:
            fisher_result = FisherResidualResult(
                compensated_counts=physics_result.compensated_counts.copy(),
                center_ps=fisher_result.physics_center_ps,
                physics_center_ps=fisher_result.physics_center_ps,
                poisson_center_ps=fisher_result.poisson_center_ps,
                applied_shift_ps=0.0,
                fisher_information_per_ps2=fisher_result.fisher_information_per_ps2,
                signal_counts=fisher_result.signal_counts,
                gate_passed=False,
            )
        return PhysicsFisherCompensationResult(
            compensated_counts=fisher_result.compensated_counts,
            absolute_time_ps=physics_result.absolute_time_ps,
            center_ps=fisher_result.center_ps,
            direction=int(direction),
            physics_result=physics_result,
            fisher_result=fisher_result,
        )
