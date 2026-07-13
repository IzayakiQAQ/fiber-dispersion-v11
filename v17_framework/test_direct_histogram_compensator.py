from __future__ import annotations

import numpy as np

from direct_histogram_compensator import (
    DirectCompensatorConfig,
    PhysicalDirectHistogramCompensator,
    center_of_mass,
    gaussian_coarse_center,
)
from public_compensated_histogram_operator import compensated_center_ps, compensated_histogram
from run_direct_histogram_external_1000 import infer_all


def gaussian(length: int, center: float, sigma: float) -> np.ndarray:
    x = np.arange(length, dtype=np.float64)
    values = np.exp(-0.5 * ((x - float(center)) / float(sigma)) ** 2)
    return values / float(np.sum(values))


def test_direct_output_is_nonnegative_and_count_preserving() -> None:
    length = 1025
    broad = gaussian(length, 512.0, 90.0)
    target = gaussian(length, 512.0, 20.0)
    observed = gaussian(length, 517.25, 90.0) * 3200.0 + 0.02
    model = PhysicalDirectHistogramCompensator(broad, target, DirectCompensatorConfig(iterations=5))
    output = model.infer(observed)
    assert output.shape == observed.shape
    assert np.all(output >= 0.0)
    assert np.isclose(np.sum(output), np.sum(observed), rtol=0.0, atol=1e-8)


def test_compensator_is_translation_equivariant_away_from_edges() -> None:
    length = 1025
    broad = gaussian(length, 512.0, 80.0)
    target = gaussian(length, 512.0, 18.0)
    model = PhysicalDirectHistogramCompensator(broad, target, DirectCompensatorConfig(iterations=6))
    output0 = model.infer(gaussian(length, 512.0, 80.0) * 3000.0)
    output7 = model.infer(gaussian(length, 519.0, 80.0) * 3000.0)
    assert abs((center_of_mass(output7) - center_of_mass(output0)) - 7.0) < 0.05


def test_empty_histogram_stays_empty() -> None:
    length = 257
    model = PhysicalDirectHistogramCompensator(
        gaussian(length, 128.0, 30.0),
        gaussian(length, 128.0, 8.0),
    )
    output = model.infer(np.zeros(length, dtype=np.float64))
    assert np.array_equal(output, np.zeros(length, dtype=np.float64))


def test_full_inference_finds_current_histogram_without_external_center() -> None:
    length = 4097
    broad_local = gaussian(1025, 512.0, 90.0)
    target_local = gaussian(1025, 512.0, 20.0)
    observed = gaussian(length, 2301.4, 90.0) * 3500.0
    model = PhysicalDirectHistogramCompensator(broad_local, target_local, DirectCompensatorConfig(iterations=4))
    output = model.infer_full(observed)
    assert abs(gaussian_coarse_center(observed) - 2301.4) < 0.05
    assert abs(center_of_mass(output) - 2301.4) < 0.05
    assert np.isclose(np.sum(output), np.sum(observed), rtol=0.0, atol=1e-8)


def test_batched_evaluation_matches_single_histogram_model() -> None:
    length = 1025
    broad = gaussian(length, 512.0, 82.0)
    target = gaussian(length, 512.0, 19.0)
    inputs = np.stack(
        [
            gaussian(length, 509.5, 82.0) * 2900.0 + 0.01,
            gaussian(length, 516.25, 82.0) * 3300.0 + 0.02,
        ]
    )
    all_inputs = np.stack((inputs, inputs))
    coarse = np.zeros((2, 2), dtype=np.float64)
    centers, _, outputs = infer_all(
        all_inputs,
        coarse,
        np.stack((broad, broad)),
        np.stack((target, target)),
        iterations=7,
        center_half_window=180,
        keep_histograms=True,
    )
    assert outputs is not None
    model = PhysicalDirectHistogramCompensator(broad, target, DirectCompensatorConfig(iterations=7))
    for index, histogram in enumerate(inputs):
        expected = model.infer(histogram)
        assert np.allclose(outputs[0, index], expected, rtol=2e-5, atol=2e-6)
        assert abs(centers[0, index] - (center_of_mass(expected) - 512.0)) < 1e-5


def test_public_operator_and_center_reader_match_final_pipeline() -> None:
    length = 1025
    broad = gaussian(length, 512.0, 81.0)
    target = gaussian(length, 512.0, 22.0)
    observed = gaussian(length, 515.75, 81.0) * 3100.0 + 0.015
    iterations = 9

    model = PhysicalDirectHistogramCompensator(
        broad,
        target,
        DirectCompensatorConfig(iterations=iterations, edge_bins=160),
    )
    expected = model.infer(observed)
    actual = compensated_histogram(observed, broad, target, iterations)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)

    coarse_abs_ps = 12500.25
    expected_center = coarse_abs_ps + center_of_mass(actual, 180) - float(length // 2)
    assert abs(compensated_center_ps(actual, coarse_abs_ps) - expected_center) < 1e-12
