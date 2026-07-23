from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .direct_histogram_compensator import center_of_mass
    from .public_compensated_histogram_operator import DEFAULT_MODEL, V24Compensator
except ImportError:
    from direct_histogram_compensator import center_of_mass
    from public_compensated_histogram_operator import DEFAULT_MODEL, V24Compensator


EXPECTED_MODEL_SHA256 = "411db65754ae4ae8edfb04d0ac7850b2d3e4ae857ccd3458beb97cb9689e879f"


def main() -> None:
    model_bytes = Path(DEFAULT_MODEL).read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    assert model_sha256 == EXPECTED_MODEL_SHA256

    operator = V24Compensator(DEFAULT_MODEL)
    assert operator.iterations == 512
    assert operator.edge_bins == 160
    assert operator.center_half_window == 180
    assert not operator.post_output_bounded_center_correction
    assert not operator.legacy_v17_eta_blend_clip_used

    checks: list[dict[str, float | int]] = []
    with np.load(DEFAULT_MODEL, allow_pickle=False) as model:
        for direction in (1, 2):
            raw = np.asarray(
                model[f"broad_psf_direction{direction}"], dtype=np.float64
            ) * 3000.0
            target = np.asarray(
                model[f"target_psf_direction{direction}"], dtype=np.float64
            )
            output = operator.infer_local(raw, direction)
            count_error = abs(float(np.sum(output) - np.sum(raw)))
            center_error = abs(
                center_of_mass(output, operator.center_half_window)
                - center_of_mass(target, operator.center_half_window)
            )
            assert np.all(output >= 0.0)
            assert count_error < 1e-8
            assert center_error < 0.05
            checks.append(
                {
                    "direction": direction,
                    "count_error": count_error,
                    "target_center_error_bins": center_error,
                }
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "version": "v24",
                "model_sha256": model_sha256,
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
