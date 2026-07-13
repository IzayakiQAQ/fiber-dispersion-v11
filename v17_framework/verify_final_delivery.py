from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from public_compensated_histogram_operator import compensated_center_ps, infer_with_saved_model


THIS_DIR = Path(__file__).resolve().parent
RESULT_DIR = THIS_DIR / "results" / "v24_direct_histogram_external_1000_physical_0km"
PAPER_DIR = THIS_DIR / "results" / "paper_dcm100hz_vs_direct280hz"


def main() -> None:
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    comparison = json.loads((PAPER_DIR / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["selected_iterations"] == 512
    assert summary["edge_bins_per_side"] == 160
    assert not summary["legacy_v17_center_parameters_used"]
    assert comparison["software_operator"]["iterations"] == 512
    assert not comparison["software_operator"]["legacy_v17_eta_blend_clip_used"]

    with np.load(RESULT_DIR / "single_histogram_input_cache.npz", allow_pickle=False) as input_data:
        input_histograms = np.asarray(input_data["local_histograms"], dtype=np.float64)
        coarse_centers = np.asarray(input_data["coarse_center_abs_ps"], dtype=np.float64)
    with np.load(RESULT_DIR / "compensated_histograms_1000x2_local.npz", allow_pickle=False) as output_data:
        output_histograms = np.asarray(output_data["output_histograms"], dtype=np.float64)
        output_centers = np.asarray(output_data["output_center_abs_ps"], dtype=np.float64)

    histogram_errors: list[float] = []
    center_errors: list[float] = []
    for direction, index in ((1, 0), (1, 999), (2, 0), (2, 999)):
        actual = infer_with_saved_model(input_histograms[direction - 1, index], direction)
        expected = output_histograms[direction - 1, index]
        histogram_errors.append(float(np.max(np.abs(actual - expected))))
        actual_center = compensated_center_ps(actual, coarse_centers[direction - 1, index])
        center_errors.append(abs(actual_center - output_centers[direction - 1, index]))

    max_histogram_error = max(histogram_errors)
    max_center_error = max(center_errors)
    assert max_histogram_error < 1e-5
    assert max_center_error < 1e-5

    with (PAPER_DIR / "fig5_source_data_complete.csv").open(encoding="utf-8-sig", newline="") as handle:
        fig5_rows = list(csv.DictReader(handle))
    assert len(fig5_rows) == 2049
    assert float(fig5_rows[0]["relative_time_ps"]) == -1024.0
    assert float(fig5_rows[-1]["relative_time_ps"]) == 1024.0

    print(
        json.dumps(
            {
                "status": "passed",
                "max_histogram_abs_error": max_histogram_error,
                "max_center_error_ps": max_center_error,
                "fig5_source_rows": len(fig5_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
