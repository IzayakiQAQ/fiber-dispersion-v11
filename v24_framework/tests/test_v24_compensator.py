from __future__ import annotations

import numpy as np

from v24_framework.direct_histogram_compensator import center_of_mass
from v24_framework.public_compensated_histogram_operator import V24Compensator


def test_saved_model_metadata() -> None:
    operator = V24Compensator()
    assert operator.local_length == 2049
    assert operator.iterations == 512
    assert operator.edge_bins == 160
    assert operator.center_half_window == 180
    assert operator.ratio_clip == 8.0
    assert not operator.post_output_bounded_center_correction
    assert not operator.legacy_v17_eta_blend_clip_used


def test_each_direction_is_nonnegative_and_count_preserving() -> None:
    operator = V24Compensator()
    with np.load(operator.model_path, allow_pickle=False) as model:
        for direction in (1, 2):
            raw = np.asarray(
                model[f"broad_psf_direction{direction}"], dtype=np.float64
            ) * 3200.0
            target = np.asarray(
                model[f"target_psf_direction{direction}"], dtype=np.float64
            )
            output = operator.infer_local(raw, direction)
            assert np.all(output >= 0.0)
            assert np.isclose(np.sum(output), np.sum(raw), rtol=0.0, atol=1e-8)
            assert (
                abs(center_of_mass(output, 180) - center_of_mass(target, 180))
                < 0.05
            )


def test_full_histogram_is_located_and_cropped() -> None:
    operator = V24Compensator()
    full_axis = np.arange(65536, dtype=np.float64)
    x = np.arange(65536, dtype=np.float64)
    counts = np.exp(-0.5 * ((x - 32780.25) / 215.0) ** 2) * 8.0
    prepared = operator.prepare_full(counts, full_axis)
    assert prepared.counts.size == 2049
    assert abs(prepared.coarse_center_abs_ps - 32780.25) < 0.1
    output = operator.infer_local(prepared.counts, direction=1)
    assert np.isclose(np.sum(output), np.sum(prepared.counts), rtol=0.0, atol=1e-8)


def test_invalid_direction_is_rejected() -> None:
    operator = V24Compensator()
    with np.testing.assert_raises(ValueError):
        operator.infer_local(np.zeros(2049), direction=3)
