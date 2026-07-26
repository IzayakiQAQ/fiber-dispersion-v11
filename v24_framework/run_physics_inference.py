from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .physics_informed.adaptive_compensator import (
        AdaptiveCompensatorConfig,
        PhysicsAdaptiveCompensator,
    )
    from .physics_informed.dataset import centered_profile, load_histogram_csv
    from .physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        fwhm_ps,
        load_physics_parameters,
    )
except ImportError:
    from physics_informed.adaptive_compensator import (
        AdaptiveCompensatorConfig,
        PhysicsAdaptiveCompensator,
    )
    from physics_informed.dataset import centered_profile, load_histogram_csv
    from physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        fwhm_ps,
        load_physics_parameters,
    )


def comma_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run physics-informed adaptive compensation on one histogram."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--direction", type=int, choices=(1, 2), required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--length-grid-km", default="0,25,50,75,100,125")
    parser.add_argument("--bandwidth-grid-nm", default="0.2,0.4,0.8,2,5,8,10")
    parser.add_argument("--n-bins", type=int, default=16385)
    parser.add_argument("--minimum-iterations", type=int, default=16)
    parser.add_argument("--maximum-iterations", type=int, default=512)
    return parser.parse_args()


def poisson_background(values: np.ndarray, edge_bins: int = 512) -> float:
    width = min(max(int(edge_bins), 1), max(values.size // 4, 1))
    return float(np.mean(np.concatenate((values[:width], values[-width:]))))


def probability_and_width(axis: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, float]:
    background = poisson_background(counts)
    signal = np.clip(counts - background, 0.0, None)
    probability = signal / max(float(np.sum(signal)), 1e-12)
    return probability, fwhm_ps(axis, signal)


def main() -> None:
    args = parse_args()
    axis, counts = load_histogram_csv(args.input_csv)
    parameters = load_physics_parameters(args.calibration_json)
    calibration_payload = json.loads(
        Path(args.calibration_json).read_text(encoding="utf-8-sig")
    )
    candidates = [
        (length, bandwidth)
        for length in comma_floats(args.length_grid_km)
        for bandwidth in comma_floats(args.bandwidth_grid_nm)
    ]
    operator = PhysicsAdaptiveCompensator(
        PhysicsHistogramGenerator(parameters),
        candidates,
        AdaptiveCompensatorConfig(
            n_bins=int(args.n_bins),
            minimum_iterations=int(args.minimum_iterations),
            maximum_iterations=int(args.maximum_iterations),
        ),
    )
    result = operator.infer(counts, args.direction, absolute_time_ps=axis)
    raw_local = np.interp(result.absolute_time_ps, axis, counts, left=0.0, right=0.0)
    expected_dispersion_width = (
        parameters.dispersion_ps_nm_km_at_1550
        * result.inferred_length_km
        * result.inferred_bandwidth_nm
    )
    raw_fit = centered_profile(
        args.input_csv,
        half_width_bins=int(args.n_bins) // 2,
        expected_dispersion_fwhm_ps=expected_dispersion_width,
    )
    output_probability, output_width = probability_and_width(
        result.absolute_time_ps, result.compensated_counts
    )
    output_center = result.center_ps

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_csv,
        np.column_stack(
            (result.absolute_time_ps, raw_local, result.compensated_counts)
        ),
        delimiter=",",
        header="absolute_time_ps,raw_count,compensated_count",
        comments="",
        fmt="%.10g",
    )
    output_json = Path(args.output_json) if args.output_json else output_csv.with_suffix(".json")
    payload = {
        "input_csv": str(Path(args.input_csv)),
        "direction": int(args.direction),
        "calibration_json": str(Path(args.calibration_json)),
        "calibration_deployment_status": calibration_payload.get("deployment_status"),
        "inferred_effective_length_km": result.inferred_length_km,
        "inferred_effective_bandwidth_nm": result.inferred_bandwidth_nm,
        "iterations": result.iterations,
        "gated_to_identity": result.gated_to_identity,
        "js_divergence": result.js_divergence,
        "ideal_model_fisher_gain": result.fisher_gain,
        "raw_center_ps": raw_fit.center_abs_ps,
        "compensated_center_ps": output_center,
        "operator_center_ps": result.center_ps,
        "raw_fwhm_ps": raw_fit.gaussian_fwhm_ps,
        "compensated_fwhm_ps": output_width,
        "input_local_count_sum": float(np.sum(raw_local)),
        "output_count_sum": float(np.sum(result.compensated_counts)),
        "post_output_bounded_center_correction": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
