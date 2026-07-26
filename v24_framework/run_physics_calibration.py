from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

try:
    from .physics_informed.dataset import (
        ConditionSeries,
        centered_profile,
        discover_dataset,
        evenly_spaced_records,
    )
    from .physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        PhysicsParameters,
        fwhm_ps,
    )
except ImportError:
    from physics_informed.dataset import (
        ConditionSeries,
        centered_profile,
        discover_dataset,
        evenly_spaced_records,
    )
    from physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        PhysicsParameters,
        fwhm_ps,
    )


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("E:/lzy/\u6d4b\u8bd5\u7ed3\u679c/\u8865\u507f\u6570\u636e")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the v24 physics-informed histogram generator from measured histograms."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=THIS_DIR / "results" / "physics_informed_calibration"
    )
    parser.add_argument("--max-samples-per-direction", type=int, default=32)
    parser.add_argument("--profile-half-width", type=int, default=8192)
    parser.add_argument("--smooth-sigma", type=float, default=6.0)
    parser.add_argument("--holdout-length-km", type=float, action="append", default=[])
    parser.add_argument("--holdout-bandwidth-nm", type=float, action="append", default=[])
    parser.add_argument(
        "--calibration-layout",
        choices=("sequential_flat", "channel_subdirectories", "pair_subdirectories"),
        action="append",
        default=[],
        help="Use only these acquisition layouts for fitting; other layouts remain audit holdouts.",
    )
    parser.add_argument("--max-optimizer-evaluations", type=int, default=120)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def condition_manifest(conditions: list[ConditionSeries]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for direction in (1, 2):
            paths = condition.direction_paths[direction]
            rows.append(
                {
                    "condition": condition.name,
                    "length_km": condition.length_km,
                    "bandwidth_nm": condition.bandwidth_nm,
                    "count_rate_hz": condition.count_rate_hz,
                    "layout": condition.layout,
                    "direction": direction,
                    "histogram_count": len(paths),
                    "first_histogram": str(paths[0]),
                    "last_histogram": str(paths[-1]),
                }
            )
    return rows


def aggregate_profile(
    condition: ConditionSeries,
    direction: int,
    max_samples: int,
    half_width: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    selected = evenly_spaced_records(condition.records(direction), max_samples)
    aggregate_counts = np.zeros(2 * int(half_width) + 1, dtype=np.float64)
    centers: list[float] = []
    fitted_widths: list[float] = []
    backgrounds: list[float] = []
    totals: list[float] = []
    expected_width = 17.0 * condition.length_km * condition.bandwidth_nm
    for record in selected:
        profile = centered_profile(
            record.path,
            half_width_bins=half_width,
            expected_dispersion_fwhm_ps=expected_width,
        )
        aggregate_counts += profile.counts
        centers.append(profile.center_abs_ps)
        fitted_widths.append(profile.gaussian_fwhm_ps)
        backgrounds.append(profile.background_per_bin)
        totals.append(profile.total_counts)
    time_ps = np.arange(-int(half_width), int(half_width) + 1, dtype=np.float64)
    aggregate = np.clip(aggregate_counts - float(np.sum(backgrounds)), 0.0, None)
    smoothed = gaussian_filter1d(aggregate, max(float(smooth_sigma), 0.0), mode="nearest")
    aggregate_halfmax = fwhm_ps(time_ps, smoothed)
    median_gaussian = float(np.median(fitted_widths))
    metrics = {
        "sample_count": float(len(selected)),
        "empirical_fwhm_ps": median_gaussian,
        "aggregate_halfmax_fwhm_ps": aggregate_halfmax,
        "median_individual_gaussian_fwhm_ps": median_gaussian,
        "mean_total_counts": float(np.mean(totals)),
        "mean_background_per_bin": float(np.mean(backgrounds)),
        "center_std_ps": float(np.std(centers)),
    }
    return time_ps, aggregate, metrics


def classify_split(
    length_km: float,
    bandwidth_nm: float,
    holdout_lengths: list[float],
    holdout_bandwidths: list[float],
    layout: str,
    calibration_layouts: list[str],
) -> bool:
    if calibration_layouts and layout not in calibration_layouts:
        return "layout_audit"
    if any(np.isclose(length_km, value) for value in holdout_lengths) or any(
        np.isclose(bandwidth_nm, value) for value in holdout_bandwidths
    ):
        return "condition_holdout"
    return "calibration"


def calibrated_parameters(
    observations: list[dict[str, Any]],
    initial: PhysicsParameters,
    max_nfev: int,
) -> tuple[PhysicsParameters, dict[str, Any]]:
    calibration_rows = [row for row in observations if row["split"] == "calibration"]
    if len(calibration_rows) < 4:
        raise ValueError("At least four condition-direction observations are required")

    def unpack(values: np.ndarray) -> PhysicsParameters:
        return replace(
            initial,
            dispersion_ps_nm_km_at_1550=float(values[0]),
            irf_fwhm_direction1_ps=float(values[1]),
            irf_fwhm_direction2_ps=float(values[2]),
        )

    def residuals(values: np.ndarray) -> np.ndarray:
        generator = PhysicsHistogramGenerator(unpack(values))
        residual: list[float] = []
        for row in calibration_rows:
            time_ps, probability = generator.probability(
                row["length_km"],
                row["bandwidth_nm"],
                row["direction"],
                n_bins=16385,
            )
            predicted = fwhm_ps(time_ps, probability)
            observed = max(float(row["empirical_fwhm_ps"]), 1.0)
            residual.append(np.log(max(predicted, 1.0) / observed))
        # The WSS bandwidth is nominally known, while D and a free bandwidth
        # scale are nearly degenerate from temporal widths alone. Keep the WSS
        # scale fixed and weakly regularize D around ordinary Corning SMF.
        residual.append((float(values[0]) - 17.0) / 4.0)
        return np.asarray(residual, dtype=np.float64)

    initial_vector = np.asarray(
        [
            initial.dispersion_ps_nm_km_at_1550,
            initial.irf_fwhm_direction1_ps,
            initial.irf_fwhm_direction2_ps,
        ],
        dtype=np.float64,
    )
    result = least_squares(
        residuals,
        initial_vector,
        bounds=(
            np.asarray([8.0, 20.0, 20.0]),
            np.asarray([28.0, 500.0, 500.0]),
        ),
        max_nfev=max(int(max_nfev), 1),
        x_scale="jac",
        loss="soft_l1",
        f_scale=0.15,
    )
    diagnostics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
    }
    return unpack(result.x), diagnostics


def add_predictions(
    observations: list[dict[str, Any]], parameters: PhysicsParameters
) -> list[dict[str, Any]]:
    generator = PhysicsHistogramGenerator(parameters)
    rows: list[dict[str, Any]] = []
    for observation in observations:
        time_ps, probability = generator.probability(
            observation["length_km"],
            observation["bandwidth_nm"],
            observation["direction"],
            n_bins=16385,
        )
        predicted = fwhm_ps(time_ps, probability)
        observed = float(observation["empirical_fwhm_ps"])
        rows.append(
            {
                **observation,
                "predicted_fwhm_ps": predicted,
                "width_error_ps": predicted - observed,
                "width_error_fraction": (predicted - observed) / max(observed, 1e-12),
                "physics_consistency_pass": abs(predicted - observed)
                / max(observed, 1e-12)
                <= 0.35,
            }
        )
    return rows


def virtual_grid(parameters: PhysicsParameters) -> list[dict[str, float]]:
    generator = PhysicsHistogramGenerator(parameters)
    rows: list[dict[str, float]] = []
    for length_km in (0.0, 25.0, 50.0, 75.0, 100.0, 125.0):
        for bandwidth_nm in (0.2, 0.4, 0.8, 2.0, 5.0, 8.0, 10.0, 20.0):
            for direction in (1, 2):
                approximate_fwhm = np.sqrt(
                    parameters.irf_fwhm_ps(direction) ** 2
                    + (parameters.dispersion_ps_nm_km_at_1550 * length_km * bandwidth_nm) ** 2
                )
                required = max(16384.0, 4.0 * approximate_fwhm)
                n_bins = int(2 ** np.ceil(np.log2(required))) + 1
                time_ps, probability = generator.probability(
                    length_km, bandwidth_nm, direction, n_bins=n_bins
                )
                rows.append(
                    {
                        "length_km": length_km,
                        "bandwidth_nm": bandwidth_nm,
                        "direction": float(direction),
                        "simulation_bins": float(n_bins),
                        "predicted_fwhm_ps": fwhm_ps(time_ps, probability),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = discover_dataset(args.dataset_root)
    write_csv(output_dir / "dataset_manifest.csv", condition_manifest(conditions))

    observations: list[dict[str, Any]] = []
    profile_payload: dict[str, np.ndarray] = {}
    for condition in conditions:
        split = classify_split(
            condition.length_km,
            condition.bandwidth_nm,
            args.holdout_length_km,
            args.holdout_bandwidth_nm,
            condition.layout,
            args.calibration_layout,
        )
        for direction in (1, 2):
            time_ps, profile, metrics = aggregate_profile(
                condition,
                direction,
                args.max_samples_per_direction,
                args.profile_half_width,
                args.smooth_sigma,
            )
            key = f"L{condition.length_km:g}_B{condition.bandwidth_nm:g}_D{direction}"
            profile_payload[f"{key}_time_ps"] = time_ps
            profile_payload[f"{key}_counts"] = profile
            observations.append(
                {
                    "condition": condition.name,
                    "length_km": condition.length_km,
                    "bandwidth_nm": condition.bandwidth_nm,
                    "count_rate_hz": condition.count_rate_hz,
                    "layout": condition.layout,
                    "direction": direction,
                    "split": split,
                    **metrics,
                }
            )
            print(
                f"aggregated {condition.name} direction {direction}: "
                f"FWHM={metrics['empirical_fwhm_ps']:.3f} ps",
                flush=True,
            )
    np.savez_compressed(output_dir / "empirical_profiles.npz", **profile_payload)
    write_csv(output_dir / "empirical_condition_metrics.csv", observations)

    parameters, optimizer = calibrated_parameters(
        observations,
        PhysicsParameters(),
        args.max_optimizer_evaluations,
    )
    evaluated = add_predictions(observations, parameters)
    write_csv(output_dir / "calibrated_condition_metrics.csv", evaluated)
    write_csv(output_dir / "virtual_condition_grid.csv", virtual_grid(parameters))

    calibration_errors = np.asarray(
        [row["width_error_fraction"] for row in evaluated if row["split"] == "calibration"],
        dtype=np.float64,
    )
    condition_holdout_errors = np.asarray(
        [
            row["width_error_fraction"]
            for row in evaluated
            if row["split"] == "condition_holdout"
        ],
        dtype=np.float64,
    )
    layout_audit_errors = np.asarray(
        [
            row["width_error_fraction"]
            for row in evaluated
            if row["split"] == "layout_audit"
        ],
        dtype=np.float64,
    )
    summary = {
        "dataset_root": str(Path(args.dataset_root)),
        "condition_count": len(conditions),
        "histograms_used_per_condition_direction": int(args.max_samples_per_direction),
        "profile_half_width_bins": int(args.profile_half_width),
        "model": {
            "source": "CW C46 pump, cascaded SHG/type-0 SPDC in PPLN",
            "channels": "C57/C35 energy-anticorrelated pair",
            "filters": "two nominal Gaussian WSS intensity responses",
            "fiber": "effective Corning single-mode dispersion model",
            "detector": "direction-specific fitted combined timing IRF",
            "observation": "Poisson signal plus measured edge background",
        },
        "parameters": asdict(parameters),
        "identifiability_constraint": (
            "filter_bandwidth_scale is fixed at 1.0 because temporal widths identify "
            "approximately D times bandwidth scale, not both independently"
        ),
        "optimizer": optimizer,
        "calibration_median_absolute_width_error_fraction": float(
            np.median(np.abs(calibration_errors))
        ),
        "condition_holdout_median_absolute_width_error_fraction": (
            float(np.median(np.abs(condition_holdout_errors)))
            if condition_holdout_errors.size
            else None
        ),
        "layout_audit_median_absolute_width_error_fraction": (
            float(np.median(np.abs(layout_audit_errors)))
            if layout_audit_errors.size
            else None
        ),
        "physics_consistent_observation_count": int(
            sum(bool(row["physics_consistency_pass"]) for row in evaluated)
        ),
        "physics_inconsistent_observation_count": int(
            sum(not bool(row["physics_consistency_pass"]) for row in evaluated)
        ),
        "calibration_inconsistent_observation_count": int(
            sum(
                row["split"] == "calibration"
                and not bool(row["physics_consistency_pass"])
                for row in evaluated
            )
        ),
        "consistency_by_layout": {
            layout: {
                "observation_count": sum(row["layout"] == layout for row in evaluated),
                "pass_count": sum(
                    row["layout"] == layout and bool(row["physics_consistency_pass"])
                    for row in evaluated
                ),
            }
            for layout in sorted({str(row["layout"]) for row in evaluated})
        },
        "deployment_status": "model_or_dataset_partition_requires_resolution_before_adaptive_deployment",
        "holdout_lengths_km": [float(value) for value in args.holdout_length_km],
        "holdout_bandwidths_nm": [float(value) for value in args.holdout_bandwidth_nm],
        "calibration_layouts": list(args.calibration_layout),
    }
    selected_rows = [row for row in evaluated if row["split"] != "layout_audit"]
    selected_inconsistent_fraction = sum(
        not bool(row["physics_consistency_pass"]) for row in selected_rows
    ) / max(
        len(selected_rows), 1
    )
    calibration_count = sum(row["split"] == "calibration" for row in evaluated)
    calibration_inconsistent_fraction = summary[
        "calibration_inconsistent_observation_count"
    ] / max(calibration_count, 1)
    if (
        float(np.median(np.abs(calibration_errors))) <= 0.2
        and calibration_inconsistent_fraction <= 0.2
    ):
        summary["deployment_status"] = (
            "calibrated_response_manifold_ready"
            if selected_inconsistent_fraction <= 0.2 and not layout_audit_errors.size
            else "ready_for_selected_calibration_layout_only_not_cross_layout"
        )
    write_json(output_dir / "physics_calibration.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
