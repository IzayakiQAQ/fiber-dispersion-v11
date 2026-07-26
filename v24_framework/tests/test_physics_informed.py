from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from v24_framework.physics_informed.adaptive_compensator import (
    AdaptiveCompensatorConfig,
    PhysicsAdaptiveCompensator,
)
from v24_framework.physics_informed.dataset import (
    discover_dataset,
    thin_histogram_counts,
)
from v24_framework.physics_informed.forward_model import (
    PhysicsHistogramGenerator,
    fwhm_ps,
    load_physics_parameters,
)


def _touch_histograms(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        (folder / f"hist_raw_{index:05d}.csv").touch()


def test_dataset_discovery_supports_all_three_layouts(tmp_path: Path) -> None:
    sequential = tmp_path / "0.8nm 0km"
    _touch_histograms(sequential, 8)

    channel_dirs = tmp_path / "5nm 25km"
    _touch_histograms(channel_dirs / "ch1_ch2_histograms_raw_1ps", 4)
    _touch_histograms(channel_dirs / "ch3_ch4_histograms_raw_1ps", 4)

    pair_dirs = tmp_path / "125km 0.8nm"
    _touch_histograms(pair_dirs / "pair0_histograms_raw_ps", 4)
    _touch_histograms(pair_dirs / "pair1_histograms_raw_ps", 4)

    conditions = discover_dataset(tmp_path)
    assert len(conditions) == 3
    lookup = {condition.name: condition for condition in conditions}
    assert lookup["0.8nm 0km"].pair_count == 4
    assert lookup["5nm 25km"].bandwidth_nm == 5.0
    assert lookup["125km 0.8nm"].length_km == 125.0
    for condition in conditions:
        assert set(condition.direction_paths) == {1, 2}


def test_forward_model_is_normalized_and_dispersion_broadens_peak() -> None:
    generator = PhysicsHistogramGenerator()
    time_ps, zero = generator.probability(0.0, 0.8, direction=1, n_bins=4097)
    _, fifty = generator.probability(50.0, 0.8, direction=1, n_bins=4097)
    assert np.isclose(np.sum(zero), 1.0)
    assert np.isclose(np.sum(fifty), 1.0)
    assert 150.0 < fwhm_ps(time_ps, zero) < 180.0
    assert fwhm_ps(time_ps, fifty) > 1.5 * fwhm_ps(time_ps, zero)


def test_poisson_sampling_is_reproducible_with_seed() -> None:
    generator = PhysicsHistogramGenerator()
    first = generator.sample(
        25.0,
        2.0,
        direction=2,
        signal_counts=1000.0,
        background_per_bin=0.01,
        n_bins=2049,
        rng=np.random.default_rng(7),
    )[1]
    second = generator.sample(
        25.0,
        2.0,
        direction=2,
        signal_counts=1000.0,
        background_per_bin=0.01,
        n_bins=2049,
        rng=np.random.default_rng(7),
    )[1]
    assert np.array_equal(first, second)


def test_adaptive_compensator_selects_synthetic_condition_and_preserves_counts() -> None:
    generator = PhysicsHistogramGenerator()
    full_time, full_counts = generator.expected_counts(
        50.0,
        0.8,
        direction=1,
        signal_counts=20000.0,
        background_per_bin=0.02,
        n_bins=4097,
    )
    absolute_axis = full_time + 33000.0
    operator = PhysicsAdaptiveCompensator(
        generator,
        [(0.0, 0.8), (25.0, 0.8), (50.0, 0.8)],
        AdaptiveCompensatorConfig(
            n_bins=2049,
            edge_bins=128,
            min_signal_counts=10.0,
            maximum_js_divergence=1.0,
            minimum_fisher_gain=1.01,
            minimum_iterations=1,
            maximum_iterations=4,
        ),
    )
    result = operator.infer(full_counts, direction=1, absolute_time_ps=absolute_axis)
    assert result.inferred_length_km == 50.0
    assert result.inferred_bandwidth_nm == 0.8
    assert not result.gated_to_identity
    assert np.all(result.compensated_counts >= 0.0)
    assert np.isclose(
        np.sum(result.compensated_counts),
        np.sum(np.interp(result.absolute_time_ps, absolute_axis, full_counts)),
        rtol=0.0,
        atol=1e-6,
    )


def test_low_signal_input_uses_no_harm_gate() -> None:
    generator = PhysicsHistogramGenerator()
    time_ps, counts = generator.expected_counts(
        25.0, 0.8, direction=1, signal_counts=20.0, n_bins=2049
    )
    operator = PhysicsAdaptiveCompensator(
        generator,
        [(25.0, 0.8)],
        AdaptiveCompensatorConfig(n_bins=2049, min_signal_counts=100.0),
    )
    result = operator.infer(counts, direction=1, absolute_time_ps=time_ps)
    assert result.gated_to_identity
    assert result.iterations == 0


def test_calibration_json_parameters_are_loadable(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "parameters": {
                    "dispersion_ps_nm_km_at_1550": 16.1,
                    "irf_fwhm_direction1_ps": 144.0,
                    "irf_fwhm_direction2_ps": 143.0,
                }
            }
        ),
        encoding="utf-8",
    )
    parameters = load_physics_parameters(path)
    assert parameters.dispersion_ps_nm_km_at_1550 == 16.1
    assert parameters.irf_fwhm_direction1_ps == 144.0


def test_binomial_count_thinning_is_reproducible_and_never_adds_events() -> None:
    counts = np.asarray([0.0, 1.0, 5.0, 20.0, 100.0])
    first = thin_histogram_counts(counts, 0.25, np.random.default_rng(11))
    second = thin_histogram_counts(counts, 0.25, np.random.default_rng(11))
    assert np.array_equal(first, second)
    assert np.all(first <= counts)
