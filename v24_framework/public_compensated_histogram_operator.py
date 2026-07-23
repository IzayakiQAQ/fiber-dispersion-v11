from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .direct_histogram_compensator import (
        DirectCompensatorConfig,
        PhysicalDirectHistogramCompensator,
        center_of_mass,
        gaussian_coarse_center,
    )
except ImportError:
    from direct_histogram_compensator import (
        DirectCompensatorConfig,
        PhysicalDirectHistogramCompensator,
        center_of_mass,
        gaussian_coarse_center,
    )


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = THIS_DIR / "models" / "direct_histogram_model_v24.npz"


@dataclass(frozen=True)
class PreparedHistogram:
    """One model-ready local histogram and its absolute coarse center."""

    counts: np.ndarray
    absolute_time_ps: np.ndarray
    coarse_center_abs_ps: float


class V24Compensator:
    """Locked, stateless v24 histogram-to-histogram compensator.

    The object loads the saved direction-specific PSFs once. Every inference
    call uses only the supplied histogram; no adjacent samples or run-level
    clock series are used.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        self.model_path = Path(model_path)
        with np.load(self.model_path, allow_pickle=False) as model:
            self.iterations = int(model["iterations"])
            self.half_width = int(model["half_width"])
            self.center_half_window = int(model["center_half_window"])
            self.edge_bins = int(model["edge_bins_per_side"])
            self.ratio_clip = float(model["ratio_clip"])
            self.latent_floor_fraction = float(model["latent_floor_fraction"])
            self.post_output_bounded_center_correction = bool(
                model["post_output_bounded_center_correction"]
            )
            self.legacy_v17_eta_blend_clip_used = bool(
                model["legacy_v17_eta_blend_clip_used"]
            )
            broad = {
                direction: np.asarray(
                    model[f"broad_psf_direction{direction}"], dtype=np.float64
                )
                for direction in (1, 2)
            }
            target = {
                direction: np.asarray(
                    model[f"target_psf_direction{direction}"], dtype=np.float64
                )
                for direction in (1, 2)
            }

        config = DirectCompensatorConfig(
            iterations=self.iterations,
            ratio_clip=self.ratio_clip,
            edge_bins=self.edge_bins,
            latent_floor_fraction=self.latent_floor_fraction,
        )
        self._operators = {
            direction: PhysicalDirectHistogramCompensator(
                broad[direction], target[direction], config
            )
            for direction in (1, 2)
        }
        self.local_length = 2 * self.half_width + 1

    @staticmethod
    def _direction(direction: int) -> int:
        value = int(direction)
        if value not in (1, 2):
            raise ValueError("direction must be 1 or 2")
        return value

    def infer_local(self, raw_local_histogram: np.ndarray, direction: int) -> np.ndarray:
        """Compensate one 2049-bin local histogram."""

        values = np.asarray(raw_local_histogram, dtype=np.float64)
        if values.ndim != 1 or values.size != self.local_length:
            raise ValueError(
                f"Expected one {self.local_length}-bin local histogram, got {values.shape}"
            )
        return self._operators[self._direction(direction)].infer(values)

    def prepare_full(
        self,
        raw_histogram: np.ndarray,
        absolute_time_ps: np.ndarray | None = None,
    ) -> PreparedHistogram:
        """Locate and crop a full fixed-axis histogram to the model window."""

        counts = np.clip(np.asarray(raw_histogram, dtype=np.float64), 0.0, None)
        if counts.ndim != 1 or counts.size < self.local_length:
            raise ValueError(
                f"Full histogram must be one-dimensional with at least {self.local_length} bins"
            )
        if absolute_time_ps is None:
            axis = np.arange(counts.size, dtype=np.float64)
        else:
            axis = np.asarray(absolute_time_ps, dtype=np.float64)
            if axis.shape != counts.shape:
                raise ValueError("absolute_time_ps and raw_histogram must have equal shape")
            spacing = np.diff(axis)
            if spacing.size and not np.allclose(spacing, 1.0, rtol=0.0, atol=1e-6):
                raise ValueError("v24 requires a uniform 1 ps histogram axis")

        center_index = gaussian_coarse_center(counts)
        coarse_center_abs_ps = float(
            np.interp(center_index, np.arange(axis.size, dtype=np.float64), axis)
        )
        relative = np.arange(-self.half_width, self.half_width + 1, dtype=np.float64)
        source_index = center_index + relative
        local = np.interp(
            source_index,
            np.arange(counts.size, dtype=np.float64),
            counts,
            left=0.0,
            right=0.0,
        )
        local_axis = coarse_center_abs_ps + relative
        return PreparedHistogram(
            counts=local,
            absolute_time_ps=local_axis,
            coarse_center_abs_ps=coarse_center_abs_ps,
        )

    def infer_full(
        self,
        raw_histogram: np.ndarray,
        direction: int,
        absolute_time_ps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Crop and compensate a full histogram.

        Returns the 2049-bin compensated histogram, its absolute 1 ps axis,
        and the final absolute center in ps.
        """

        prepared = self.prepare_full(raw_histogram, absolute_time_ps)
        output = self.infer_local(prepared.counts, direction)
        center = self.compensated_center(output, prepared.coarse_center_abs_ps)
        return output, prepared.absolute_time_ps, center

    def compensated_center(
        self,
        compensated_local_histogram: np.ndarray,
        coarse_center_abs_ps: float,
    ) -> float:
        return compensated_center_ps(
            compensated_local_histogram,
            coarse_center_abs_ps,
            center_half_window_bins=self.center_half_window,
        )


def compensated_center_ps(
    compensated_local_histogram: np.ndarray,
    coarse_center_abs_ps: float,
    center_half_window_bins: int = 180,
) -> float:
    """Read the final absolute center used by the v24 evaluation."""

    local = np.asarray(compensated_local_histogram, dtype=np.float64)
    relative_center = (
        center_of_mass(local, center_half_window_bins) - float(local.size // 2)
    )
    return float(coarse_center_abs_ps) + relative_center


def infer_with_saved_model(
    raw_local_histogram: np.ndarray,
    direction: int,
    model_path: str | Path = DEFAULT_MODEL,
) -> np.ndarray:
    """Convenience API for one local histogram.

    Reuse ``V24Compensator`` for a stream to avoid loading the model repeatedly.
    """

    return V24Compensator(model_path).infer_local(raw_local_histogram, direction)
