from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_SCHEMA = "v25-physics-residual-mlp-v1"


def condition_features(
    length_km: float, bandwidth_nm: float, direction: int
) -> np.ndarray:
    """Bounded physical features used by both training and deployment."""

    if int(direction) not in (1, 2):
        raise ValueError("direction must be 1 or 2")
    length = np.clip(float(length_km) / 125.0, 0.0, 2.0)
    log_bandwidth = np.log1p(max(float(bandwidth_nm), 0.0)) / np.log(11.0)
    direction_sign = -1.0 if int(direction) == 1 else 1.0
    return np.asarray(
        [length, log_bandwidth, direction_sign, length * log_bandwidth],
        dtype=np.float64,
    )


def dilate_probability(probability: np.ndarray, scale: float) -> np.ndarray:
    """Dilate a centered PSF without changing its center or probability mass."""

    source = np.clip(np.asarray(probability, dtype=np.float64), 0.0, None)
    total = float(np.sum(source))
    if source.ndim != 1 or source.size < 3 or total <= 1e-15:
        raise ValueError("probability must be a nonempty one-dimensional PSF")
    source /= total
    width_scale = float(np.clip(scale, 0.65, 1.50))
    center = 0.5 * (source.size - 1)
    axis = np.arange(source.size, dtype=np.float64) - center
    warped = np.interp(
        axis / width_scale,
        axis,
        source,
        left=0.0,
        right=0.0,
    ) / width_scale
    warped = np.clip(warped, 0.0, None)
    return warped / max(float(np.sum(warped)), 1e-15)


@dataclass(frozen=True)
class NeuralPSFPrediction:
    width_scale: float
    raw_width_scale: float
    confidence: float
    nearest_condition_distance: float


class NeuralPSFModel:
    """Small frozen MLP predicting a bounded residual on a physical PSF.

    The network predicts log(FWHM_measured / FWHM_physics). A coverage gate
    continuously returns the result to the physical model outside the measured
    condition manifold. It never consumes an evaluation histogram.
    """

    def __init__(
        self,
        weights: tuple[np.ndarray, ...],
        biases: tuple[np.ndarray, ...],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        training_conditions: np.ndarray,
        coverage_radius: float,
        max_abs_log_scale: float,
        metadata: dict[str, str | float | int] | None = None,
    ) -> None:
        if len(weights) != len(biases) or not weights:
            raise ValueError("weights and biases must describe at least one layer")
        self.weights = tuple(np.asarray(value, dtype=np.float64) for value in weights)
        self.biases = tuple(np.asarray(value, dtype=np.float64) for value in biases)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.training_conditions = np.asarray(training_conditions, dtype=np.float64)
        self.coverage_radius = float(coverage_radius)
        self.max_abs_log_scale = float(max_abs_log_scale)
        self.metadata = dict(metadata or {})
        if self.feature_mean.shape != (4,) or self.feature_scale.shape != (4,):
            raise ValueError("feature normalization must contain four values")
        if self.training_conditions.ndim != 2 or self.training_conditions.shape[1] != 3:
            raise ValueError("training_conditions must have columns L, B, direction")
        if self.coverage_radius <= 0.0 or self.max_abs_log_scale <= 0.0:
            raise ValueError("coverage and residual bounds must be positive")

    def _forward(self, features: np.ndarray) -> float:
        value = (features - self.feature_mean) / np.clip(
            self.feature_scale, 1e-12, None
        )
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight + bias
            if index + 1 < len(self.weights):
                value = np.tanh(value)
        return float(np.ravel(value)[0])

    def _distance(self, length_km: float, bandwidth_nm: float, direction: int) -> float:
        same_direction = self.training_conditions[
            self.training_conditions[:, 2] == float(direction)
        ]
        if same_direction.size == 0:
            return float("inf")
        length_distance = (same_direction[:, 0] - float(length_km)) / 125.0
        bandwidth_distance = (
            np.log1p(same_direction[:, 1]) - np.log1p(float(bandwidth_nm))
        ) / np.log(11.0)
        return float(np.min(np.hypot(length_distance, bandwidth_distance)))

    def predict(
        self, length_km: float, bandwidth_nm: float, direction: int
    ) -> NeuralPSFPrediction:
        features = condition_features(length_km, bandwidth_nm, direction)
        raw_log_scale = float(
            np.clip(
                self._forward(features),
                -self.max_abs_log_scale,
                self.max_abs_log_scale,
            )
        )
        distance = self._distance(length_km, bandwidth_nm, direction)
        confidence = (
            0.0
            if not np.isfinite(distance)
            else float(np.exp(-0.5 * np.square(distance / self.coverage_radius)))
        )
        raw_scale = float(np.exp(raw_log_scale))
        gated_scale = float(np.exp(confidence * raw_log_scale))
        return NeuralPSFPrediction(
            width_scale=gated_scale,
            raw_width_scale=raw_scale,
            confidence=confidence,
            nearest_condition_distance=distance,
        )

    def correct(
        self,
        probability: np.ndarray,
        length_km: float,
        bandwidth_nm: float,
        direction: int,
    ) -> tuple[np.ndarray, NeuralPSFPrediction]:
        prediction = self.predict(length_km, bandwidth_nm, direction)
        return dilate_probability(probability, prediction.width_scale), prediction

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "schema": np.asarray(MODEL_SCHEMA),
            "layer_count": np.asarray(len(self.weights), dtype=np.int64),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "training_conditions": self.training_conditions,
            "coverage_radius": np.asarray(self.coverage_radius),
            "max_abs_log_scale": np.asarray(self.max_abs_log_scale),
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = bias
        for key, value in sorted(self.metadata.items()):
            payload[f"meta_{key}"] = np.asarray(value)
        np.savez_compressed(output, **payload)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPSFModel":
        with np.load(Path(path), allow_pickle=False) as data:
            schema = str(np.asarray(data["schema"]).item())
            if schema != MODEL_SCHEMA:
                raise ValueError(f"Unsupported neural PSF schema: {schema!r}")
            count = int(np.asarray(data["layer_count"]).item())
            weights = tuple(np.asarray(data[f"weight_{index}"]) for index in range(count))
            biases = tuple(np.asarray(data[f"bias_{index}"]) for index in range(count))
            metadata = {
                key.removeprefix("meta_"): np.asarray(data[key]).item()
                for key in data.files
                if key.startswith("meta_")
            }
            return cls(
                weights=weights,
                biases=biases,
                feature_mean=np.asarray(data["feature_mean"]),
                feature_scale=np.asarray(data["feature_scale"]),
                training_conditions=np.asarray(data["training_conditions"]),
                coverage_radius=float(np.asarray(data["coverage_radius"]).item()),
                max_abs_log_scale=float(
                    np.asarray(data["max_abs_log_scale"]).item()
                ),
                metadata=metadata,
            )

    @classmethod
    def identity(cls) -> "NeuralPSFModel":
        return cls(
            weights=(np.zeros((4, 1), dtype=np.float64),),
            biases=(np.zeros(1, dtype=np.float64),),
            feature_mean=np.zeros(4, dtype=np.float64),
            feature_scale=np.ones(4, dtype=np.float64),
            training_conditions=np.asarray(
                [[0.0, 0.8, 1.0], [0.0, 0.8, 2.0]], dtype=np.float64
            ),
            coverage_radius=1.0,
            max_abs_log_scale=np.log(1.5),
            metadata={"kind": "identity"},
        )
