from __future__ import annotations

import json

import numpy as np
import pytest

from v25_framework import FrozenConfig, OperatorSettings, V25Compensator
from v25_framework.config import load_frozen_config, save_frozen_config
from v25_framework.neural_psf import NeuralPSFModel, dilate_probability
from v25_framework.physics import build_direction_kernels


@pytest.fixture(scope="module")
def config() -> FrozenConfig:
    return FrozenConfig(
        length_km=50.0,
        bandwidth_nm=0.8,
        calibration_sha256="0" * 64,
        neural_psf_model="neural_psf_model.npz",
        neural_psf_sha256="1" * 64,
        operator=OperatorSettings(rl_iterations=64),
    )


@pytest.fixture(scope="module")
def neural_model() -> NeuralPSFModel:
    return NeuralPSFModel.identity()


def test_physics_kernels_are_direction_specific_and_narrower_at_zero_km(
    config: FrozenConfig, neural_model: NeuralPSFModel
) -> None:
    first = build_direction_kernels(config, 1, neural_model)
    second = build_direction_kernels(config, 2, neural_model)
    assert np.isclose(np.sum(first.broad), 1.0)
    assert np.isclose(np.sum(first.target), 1.0)
    assert first.target_fwhm_ps < first.broad_fwhm_ps
    assert second.target_fwhm_ps < second.broad_fwhm_ps
    assert first.fisher_gain > 1.0
    assert second.fisher_gain > 1.0
    assert not np.allclose(first.broad, second.broad)


def test_stateless_rl_is_nonnegative_count_preserving_and_narrowing(
    config: FrozenConfig, neural_model: NeuralPSFModel
) -> None:
    kernels = build_direction_kernels(config, 1, neural_model)
    histogram = kernels.broad * 50_000.0 + 0.2
    operator = V25Compensator(config, neural_model)
    first = operator.infer_local(histogram, direction=1)
    second = operator.infer_local(histogram, direction=1)
    assert np.all(first.compensated >= 0.0)
    assert np.isclose(first.input_counts, first.output_counts, rtol=0.0, atol=1e-7)
    assert first.output_fwhm_ps < first.input_fwhm_ps
    assert np.array_equal(first.compensated, second.compensated)


def test_full_axis_localization_returns_an_absolute_center(
    config: FrozenConfig, neural_model: NeuralPSFModel
) -> None:
    kernels = build_direction_kernels(config, 2, neural_model)
    full = np.full(6001, 0.1, dtype=np.float64)
    start = 3000 - config.operator.kernel_bins // 2
    full[start : start + config.operator.kernel_bins] += kernels.broad * 20_000.0
    axis = (np.arange(full.size, dtype=np.float64) - 3000.0) * 1.0 + 1234.0
    result = V25Compensator(config, neural_model).infer_full(
        full, direction=2, time_ps=axis
    )
    assert result.compensated.size == config.operator.kernel_bins
    assert abs(result.center_ps - 1234.0) < 3.0


def test_frozen_config_hash_detects_tampering(
    config: FrozenConfig, tmp_path
) -> None:
    path, _ = save_frozen_config(config, tmp_path / "frozen.json")
    assert load_frozen_config(path) == config
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["length_km"] = 51.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_config(path)


def test_neural_psf_dilation_is_centered_normalized_and_bounded() -> None:
    axis = np.arange(401, dtype=np.float64) - 200.0
    probability = np.exp(-0.5 * np.square(axis / 40.0))
    probability /= np.sum(probability)
    widened = dilate_probability(probability, 1.25)
    assert np.isclose(np.sum(widened), 1.0)
    assert np.argmax(widened) == np.argmax(probability)
    assert np.sum(np.square(axis) * widened) > np.sum(np.square(axis) * probability)


def test_batch_rl_matches_single_histogram_operator(
    config: FrozenConfig, neural_model: NeuralPSFModel
) -> None:
    kernels = build_direction_kernels(config, 1, neural_model)
    histogram = kernels.broad * 20_000.0 + 0.15
    compensator = V25Compensator(config, neural_model)
    single = compensator.infer_local(histogram, 1).compensated
    batch = compensator.infer_batch_local(np.stack((histogram, histogram)), 1)
    assert np.allclose(batch[0], single, rtol=1e-10, atol=1e-10)
    assert np.array_equal(batch[0], batch[1])
