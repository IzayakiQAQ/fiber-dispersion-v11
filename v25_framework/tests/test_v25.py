from __future__ import annotations

import json

import numpy as np
import pytest

from v25_framework import FrozenConfig, OperatorSettings, V25Compensator
from v25_framework.config import load_frozen_config, save_frozen_config
from v25_framework.physics import build_direction_kernels


@pytest.fixture(scope="module")
def config() -> FrozenConfig:
    return FrozenConfig(
        length_km=50.0,
        bandwidth_nm=0.8,
        calibration_sha256="0" * 64,
        operator=OperatorSettings(rl_iterations=64),
    )


def test_physics_kernels_are_direction_specific_and_narrower_at_zero_km(
    config: FrozenConfig,
) -> None:
    first = build_direction_kernels(config, 1)
    second = build_direction_kernels(config, 2)
    assert np.isclose(np.sum(first.broad), 1.0)
    assert np.isclose(np.sum(first.target), 1.0)
    assert first.target_fwhm_ps < first.broad_fwhm_ps
    assert second.target_fwhm_ps < second.broad_fwhm_ps
    assert first.fisher_gain > 1.0
    assert second.fisher_gain > 1.0
    assert not np.allclose(first.broad, second.broad)


def test_stateless_rl_is_nonnegative_count_preserving_and_narrowing(
    config: FrozenConfig,
) -> None:
    kernels = build_direction_kernels(config, 1)
    histogram = kernels.broad * 50_000.0 + 0.2
    operator = V25Compensator(config)
    first = operator.infer_local(histogram, direction=1)
    second = operator.infer_local(histogram, direction=1)
    assert np.all(first.compensated >= 0.0)
    assert np.isclose(first.input_counts, first.output_counts, rtol=0.0, atol=1e-7)
    assert first.output_fwhm_ps < first.input_fwhm_ps
    assert np.array_equal(first.compensated, second.compensated)


def test_full_axis_localization_returns_an_absolute_center(config: FrozenConfig) -> None:
    kernels = build_direction_kernels(config, 2)
    full = np.full(6001, 0.1, dtype=np.float64)
    start = 3000 - config.operator.kernel_bins // 2
    full[start : start + config.operator.kernel_bins] += kernels.broad * 20_000.0
    axis = (np.arange(full.size, dtype=np.float64) - 3000.0) * 1.0 + 1234.0
    result = V25Compensator(config).infer_full(full, direction=2, time_ps=axis)
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
