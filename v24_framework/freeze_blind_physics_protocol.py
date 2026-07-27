from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

try:
    from .direct_histogram_compensator import (
        center_of_mass,
        fwhm_subbin,
        gaussian_coarse_fit,
    )
    from .physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        load_physics_parameters,
    )
    from .v24_common import split_csv, tdev_at_m, write_csv
except ImportError:
    from direct_histogram_compensator import (
        center_of_mass,
        fwhm_subbin,
        gaussian_coarse_fit,
    )
    from physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        load_physics_parameters,
    )
    from v24_common import split_csv, tdev_at_m, write_csv


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "blind_physics_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a physics-only compensation configuration without exposing an "
            "evaluation-data input."
        )
    )
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--calibration-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--length-km", type=float, default=50.0)
    parser.add_argument("--bandwidth-nm", type=float, default=0.8)
    parser.add_argument("--target-length-km", type=float, default=0.0)
    parser.add_argument("--count-rate-hz", type=float, default=280.0)
    parser.add_argument("--calibration-count-rate-hz", type=float, default=100.0)
    parser.add_argument("--integration-time-s", type=float, default=10.0)
    parser.add_argument("--samples-per-scenario", type=int, default=64)
    parser.add_argument("--seed", type=int, default=240280)
    parser.add_argument("--candidate-iterations", default="64,128,256,384,512")
    parser.add_argument("--candidate-center-windows", default="120,160,180,220,260")
    parser.add_argument("--target-fwhm-ps", type=float, default=165.0)
    parser.add_argument("--target-fwhm-tolerance-ps", type=float, default=20.0)
    return parser.parse_args()


def positive_integers(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in split_csv(value))
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("Candidate lists must contain positive integers")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_background_median(path: Path, count_rate_hz: float) -> float:
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "calibration":
                continue
            if row.get("layout") not in {"channel_subdirectories", "pair_subdirectories"}:
                continue
            if not np.isclose(float(row["count_rate_hz"]), count_rate_hz):
                continue
            values.append(float(row["mean_background_per_bin"]))
    if not values:
        raise ValueError(
            "No independent calibration background rows matched the declared rate"
        )
    return float(np.median(values))


def physical_psfs(
    calibration_json: Path,
    length: int,
    propagation_length_km: float,
    bandwidth_nm: float,
    target_length_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    generator = PhysicsHistogramGenerator(load_physics_parameters(calibration_json))
    broad = np.zeros((2, length), dtype=np.float64)
    target = np.zeros_like(broad)
    for direction in (1, 2):
        broad[direction - 1] = generator.probability(
            propagation_length_km, bandwidth_nm, direction, n_bins=length
        )[1]
        target[direction - 1] = generator.probability(
            target_length_km, bandwidth_nm, direction, n_bins=length
        )[1]
    return broad, target


def simulate_local_histograms(
    calibration_json: Path,
    scenarios: list[dict[str, float]],
    samples_per_scenario: int,
    seed: int,
    propagation_length_km: float,
    bandwidth_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    generator = PhysicsHistogramGenerator(load_physics_parameters(calibration_json))
    full_length = 4097
    local_length = 2049
    local_relative = np.arange(local_length, dtype=np.float64) - local_length // 2
    count = len(scenarios) * samples_per_scenario
    local = np.zeros((2, count, local_length), dtype=np.float32)
    coarse = np.zeros((2, count), dtype=np.float64)
    truth = np.zeros((2, count), dtype=np.float64)
    scenario_index = np.zeros(count, dtype=np.int64)

    for direction in (1, 2):
        time_ps, probability = generator.probability(
            propagation_length_km,
            bandwidth_nm,
            direction,
            n_bins=full_length,
        )
        for scenario_position, scenario in enumerate(scenarios):
            for within in range(samples_per_scenario):
                index = scenario_position * samples_per_scenario + within
                true_center = float(rng.uniform(-80.0, 80.0))
                shifted = np.interp(
                    time_ps - true_center,
                    time_ps,
                    probability,
                    left=0.0,
                    right=0.0,
                )
                expected = (
                    float(scenario["signal_counts"]) * shifted
                    + float(scenario["background_per_bin"])
                )
                observed = rng.poisson(expected).astype(np.float64)
                coarse_index, _ = gaussian_coarse_fit(observed)
                coarse_center = float(time_ps[0] + coarse_index)
                local[direction - 1, index] = np.interp(
                    coarse_center + local_relative,
                    time_ps,
                    observed,
                    left=0.0,
                    right=0.0,
                ).astype(np.float32)
                coarse[direction - 1, index] = coarse_center
                truth[direction - 1, index] = true_center
                scenario_index[index] = scenario_position
    return local, coarse, truth, scenario_index


def infer_checkpoints(
    histograms: np.ndarray,
    broad: np.ndarray,
    target: np.ndarray,
    checkpoints: tuple[int, ...],
    edge_bins: int,
) -> dict[int, np.ndarray]:
    directions, _, length = histograms.shape
    fft_length = next_fast_len(2 * length - 1)
    start = (length - 1) // 2
    checkpoint_set = set(checkpoints)
    results = {
        checkpoint: np.zeros_like(histograms, dtype=np.float32)
        for checkpoint in checkpoints
    }

    def same(values: np.ndarray, kernel_fft: np.ndarray) -> np.ndarray:
        full = irfft(
            rfft(values, n=fft_length, axis=-1, workers=-1)
            * kernel_fft[None, :],
            n=fft_length,
            axis=-1,
            workers=-1,
        )
        return full[:, start : start + length]

    for direction in range(directions):
        observed = np.clip(histograms[direction].astype(np.float32), 0.0, None)
        total = np.sum(observed, axis=-1, keepdims=True)
        background = np.mean(
            np.concatenate(
                (observed[:, :edge_bins], observed[:, -edge_bins:]), axis=-1
            ),
            axis=-1,
            keepdims=True,
        )
        signal = np.clip(observed - background, 0.0, None)
        signal_mass = np.sum(signal, axis=-1, keepdims=True)
        probability = signal / np.clip(signal_mass, 1e-12, None)
        latent = np.clip(probability, 1e-8, None)
        latent /= np.sum(latent, axis=-1, keepdims=True)
        broad_fft = rfft(broad[direction], n=fft_length, workers=-1)
        reverse_fft = rfft(broad[direction, ::-1], n=fft_length, workers=-1)
        target_fft = rfft(target[direction], n=fft_length, workers=-1)
        for iteration in range(1, max(checkpoints) + 1):
            projection = same(latent, broad_fft)
            ratio = np.clip(
                probability / np.clip(projection, 1e-12, None), 0.0, 8.0
            )
            latent *= same(ratio, reverse_fft)
            latent = np.clip(latent, 1e-8, None)
            latent /= np.sum(latent, axis=-1, keepdims=True)
            if iteration not in checkpoint_set:
                continue
            reconstructed = np.clip(same(latent, target_fft), 0.0, None)
            reconstructed /= np.clip(
                np.sum(reconstructed, axis=-1, keepdims=True), 1e-12, None
            )
            output = reconstructed * signal_mass + background
            output *= total / np.clip(np.sum(output, axis=-1, keepdims=True), 1e-12, None)
            results[iteration][direction] = output.astype(np.float32)
    return results


def candidate_metrics(
    outputs: np.ndarray,
    coarse: np.ndarray,
    truth: np.ndarray,
    scenario_index: np.ndarray,
    scenario_count: int,
    center_half_window: int,
) -> dict[str, float]:
    directions, count, length = outputs.shape
    centers = np.zeros((directions, count), dtype=np.float64)
    widths = np.zeros_like(centers)
    for direction in range(directions):
        for index in range(count):
            centers[direction, index] = coarse[direction, index] + (
                center_of_mass(outputs[direction, index], center_half_window)
                - length // 2
            )
            widths[direction, index] = fwhm_subbin(outputs[direction, index])
    center_error = centers - truth
    clock_error = 0.5 * (center_error[0] - center_error[1])
    scenario_tdev = []
    scenario_rmse = []
    for scenario in range(scenario_count):
        selected = clock_error[scenario_index == scenario]
        scenario_tdev.append(tdev_at_m(selected, 1))
        scenario_rmse.append(float(np.sqrt(np.mean(selected**2))))
    return {
        "simulated_tdev10_ps": tdev_at_m(clock_error, 1),
        "simulated_clock_rmse_ps": float(np.sqrt(np.mean(clock_error**2))),
        "worst_scenario_tdev10_ps": float(max(scenario_tdev)),
        "worst_scenario_clock_rmse_ps": float(max(scenario_rmse)),
        "output_fwhm_median_ps": float(np.median(widths)),
        "output_fwhm_p95_ps": float(np.percentile(widths, 95.0)),
    }


def main() -> None:
    args = parse_args()
    config_path = args.output_dir / "frozen_config.json"
    if config_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen configuration: {config_path}"
        )
    if args.count_rate_hz <= 0.0 or args.calibration_count_rate_hz <= 0.0:
        raise ValueError("Count rates must be positive")
    if args.length_km < 0.0 or args.target_length_km < 0.0:
        raise ValueError("Fiber lengths must be nonnegative")
    if args.bandwidth_nm <= 0.0 or args.integration_time_s <= 0.0:
        raise ValueError("Bandwidth and integration time must be positive")
    if args.samples_per_scenario < 3:
        raise ValueError("TDEV selection requires at least three samples per scenario")
    if args.target_fwhm_ps <= 0.0 or args.target_fwhm_tolerance_ps < 0.0:
        raise ValueError("Target FWHM must be positive and tolerance nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_background = calibration_background_median(
        args.calibration_metrics,
        args.calibration_count_rate_hz,
    )
    rate_ratio = args.count_rate_hz / args.calibration_count_rate_hz
    scenarios = []
    for signal_scale in (0.75, 1.0, 1.25):
        for background_power in (0.0, 1.0, 2.0):
            scenarios.append(
                {
                    "signal_counts": args.count_rate_hz
                    * args.integration_time_s
                    * signal_scale,
                    "background_per_bin": calibration_background
                    * rate_ratio**background_power,
                    "signal_scale": signal_scale,
                    "background_rate_power": background_power,
                }
            )

    local, coarse, truth, scenario_index = simulate_local_histograms(
        args.calibration_json,
        scenarios,
        args.samples_per_scenario,
        args.seed,
        args.length_km,
        args.bandwidth_nm,
    )
    broad, target = physical_psfs(
        args.calibration_json,
        local.shape[-1],
        args.length_km,
        args.bandwidth_nm,
        args.target_length_km,
    )
    checkpoints = positive_integers(args.candidate_iterations)
    windows = positive_integers(args.candidate_center_windows)
    all_outputs = infer_checkpoints(local, broad, target, checkpoints, edge_bins=160)
    rows: list[dict[str, Any]] = []
    for iteration in checkpoints:
        for window in windows:
            row: dict[str, Any] = {
                "iterations": iteration,
                "center_half_window_bins": window,
            }
            row.update(
                candidate_metrics(
                    all_outputs[iteration],
                    coarse,
                    truth,
                    scenario_index,
                    len(scenarios),
                    window,
                )
            )
            rows.append(row)

    width_low = args.target_fwhm_ps - args.target_fwhm_tolerance_ps
    width_high = args.target_fwhm_ps + args.target_fwhm_tolerance_ps
    width_qualified = [
        row
        for row in rows
        if width_low <= float(row["output_fwhm_median_ps"]) <= width_high
    ]
    candidates = width_qualified or rows
    selected = min(
        candidates,
        key=lambda row: (
            float(row["worst_scenario_tdev10_ps"]),
            float(row["simulated_tdev10_ps"]),
            int(row["iterations"]),
        ),
    )
    write_csv(args.output_dir / "physics_simulation_candidates.csv", rows)
    payload = {
        "protocol": "v24_blind_physics_freeze_before_evaluation",
        "selection_source": "independent_calibration_and_predeclared_physics_simulation",
        "forbidden_during_freeze": [
            "evaluation_histograms",
            "evaluation_quality_centers",
            "evaluation_fwhm",
            "evaluation_tdev",
        ],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_json": str(args.calibration_json.resolve()),
        "calibration_json_sha256": sha256_file(args.calibration_json),
        "calibration_metrics": str(args.calibration_metrics.resolve()),
        "calibration_metrics_sha256": sha256_file(args.calibration_metrics),
        "calibration_count_rate_hz": args.calibration_count_rate_hz,
        "independent_calibration_background_median_per_bin": calibration_background,
        "simulation": {
            "seed": args.seed,
            "samples_per_scenario": args.samples_per_scenario,
            "scenarios": scenarios,
            "candidate_iterations": list(checkpoints),
            "candidate_center_half_windows": list(windows),
            "selection_rule": (
                "minimum worst-scenario simulated TDEV inside the declared "
                "target-FWHM interval"
            ),
            "target_fwhm_ps": args.target_fwhm_ps,
            "target_fwhm_tolerance_ps": args.target_fwhm_tolerance_ps,
        },
        "inference": {
            "length_km": args.length_km,
            "bandwidth_nm": args.bandwidth_nm,
            "target_length_km": args.target_length_km,
            "count_rate_hz_metadata": args.count_rate_hz,
            "integration_time_s": args.integration_time_s,
            "bin_width_ps": 1.0,
            "local_bins": 2049,
            "edge_bins_per_side": 160,
            "background_estimator": "edge_mean",
            "rl_iterations": int(selected["iterations"]),
            "rl_ratio_clip": 8.0,
            "latent_floor_fraction": 1e-8,
            "center_estimator": "local_background_subtracted_center_of_mass",
            "center_half_window_bins": int(selected["center_half_window_bins"]),
            "fisher_residual_enabled": False,
            "bounded_center_correction": False,
            "uses_adjacent_histograms": False,
        },
        "selected_simulation_metrics": selected,
    }
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(config_path)
    (args.output_dir / "frozen_config.sha256").write_text(
        f"{digest}  {config_path.name}\n", encoding="ascii"
    )
    print(json.dumps({"config_sha256": digest, "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
