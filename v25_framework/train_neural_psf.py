from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from .config import PhysicsParameters
from .dataset import discover_conditions, gaussian_width_ps, sampled_paths
from .neural_psf import NeuralPSFModel, condition_features
from .physics import PhysicsHistogramGenerator, fwhm_ps


def _physics_width(
    generator: PhysicsHistogramGenerator,
    length_km: float,
    bandwidth_nm: float,
    direction: int,
) -> float:
    approximate = np.hypot(162.0, 18.0 * length_km * bandwidth_nm)
    required = int(max(8193, 4.0 * approximate + 1.0))
    power = int(np.ceil(np.log2(max(required - 1, 8192))))
    bins = min(2**power + 1, 65537)
    probability = generator.probability(
        length_km, bandwidth_nm, direction, bins, 1.0
    )
    return fwhm_ps(probability, 1.0)


def _fit_network(
    x_real: np.ndarray,
    y_real: np.ndarray,
    x_prior: np.ndarray,
    random_state: int,
) -> tuple[MLPRegressor, np.ndarray, np.ndarray]:
    mean = np.mean(np.vstack((x_real, x_prior)), axis=0)
    scale = np.std(np.vstack((x_real, x_prior)), axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    real_repeats = 6
    x_train = np.vstack((np.repeat(x_real, real_repeats, axis=0), x_prior))
    y_train = np.r_[np.repeat(y_real, real_repeats), np.zeros(x_prior.shape[0])]
    model = MLPRegressor(
        hidden_layer_sizes=(12, 8),
        activation="tanh",
        solver="lbfgs",
        alpha=0.08,
        max_iter=1500,
        random_state=int(random_state),
        tol=1e-8,
    )
    model.fit((x_train - mean) / scale, y_train)
    return model, mean, scale


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the V25 physical-residual neural PSF interpolator."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-direction", type=int, default=12)
    parser.add_argument("--minimum-width-ratio", type=float, default=0.65)
    parser.add_argument("--maximum-width-ratio", type=float, default=1.50)
    parser.add_argument("--maximum-relative-mad", type=float, default=0.35)
    parser.add_argument("--coverage-radius", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=2501)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    physics = PhysicsParameters()
    generator = PhysicsHistogramGenerator(physics)
    rows: list[dict[str, Any]] = []
    manifest = hashlib.sha256()

    for condition in discover_conditions(args.dataset_root):
        for direction in (1, 2):
            physical_width = _physics_width(
                generator,
                condition.length_km,
                condition.bandwidth_nm,
                direction,
            )
            widths: list[float] = []
            selected = sampled_paths(
                condition.direction_paths[direction], args.samples_per_direction
            )
            for path in selected:
                width = gaussian_width_ps(path, physical_width)
                if np.isfinite(width):
                    widths.append(float(width))
                stat = path.stat()
                manifest.update(
                    f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode(
                        "utf-8"
                    )
                )
            measured = float(np.median(widths)) if widths else float("nan")
            mad = (
                float(np.median(np.abs(np.asarray(widths) - measured)))
                if widths
                else float("inf")
            )
            ratio = measured / physical_width
            relative_mad = mad / max(measured, 1e-12)
            reasons: list[str] = []
            if measured < 110.0:
                reasons.append("below_detector_floor")
            if not args.minimum_width_ratio <= ratio <= args.maximum_width_ratio:
                reasons.append("physics_width_outlier")
            if relative_mad > args.maximum_relative_mad:
                reasons.append("unstable_width_fit")
            accepted = not reasons
            row = {
                "condition": condition.name,
                "length_km": condition.length_km,
                "bandwidth_nm": condition.bandwidth_nm,
                "direction": direction,
                "layout": condition.layout,
                "sample_count": len(widths),
                "measured_fwhm_ps": measured,
                "physical_fwhm_ps": physical_width,
                "width_ratio": ratio,
                "relative_mad": relative_mad,
                "accepted": int(accepted),
                "rejection_reason": ";".join(reasons),
            }
            rows.append(row)
            print(
                f"{condition.name} d{direction}: measured={measured:.2f} ps "
                f"physics={physical_width:.2f} ratio={ratio:.3f} "
                f"{'accepted' if accepted else row['rejection_reason']}",
                flush=True,
            )

    accepted_rows = [row for row in rows if bool(row["accepted"])]
    if len(accepted_rows) < 8:
        raise RuntimeError("Fewer than eight trustworthy condition-direction anchors")
    x_real = np.stack(
        [
            condition_features(row["length_km"], row["bandwidth_nm"], row["direction"])
            for row in accepted_rows
        ]
    )
    y_real = np.log(
        np.asarray([row["width_ratio"] for row in accepted_rows], dtype=np.float64)
    )
    prior_conditions = np.asarray(
        [
            (length, bandwidth, direction)
            for length in (0.0, 25.0, 50.0, 75.0, 100.0, 125.0)
            for bandwidth in (0.2, 0.4, 0.8, 2.0, 5.0, 8.0, 10.0)
            for direction in (1, 2)
        ],
        dtype=np.float64,
    )
    x_prior = np.stack(
        [condition_features(length, bandwidth, int(direction)) for length, bandwidth, direction in prior_conditions]
    )

    condition_keys = sorted(
        {(float(row["length_km"]), float(row["bandwidth_nm"])) for row in accepted_rows}
    )
    cv_errors: list[float] = []
    for fold, key in enumerate(condition_keys):
        test = np.asarray(
            [
                np.isclose(row["length_km"], key[0])
                and np.isclose(row["bandwidth_nm"], key[1])
                for row in accepted_rows
            ],
            dtype=bool,
        )
        if np.sum(~test) < 6:
            continue
        fold_model, fold_mean, fold_scale = _fit_network(
            x_real[~test], y_real[~test], x_prior, args.seed + fold + 1
        )
        predicted = fold_model.predict((x_real[test] - fold_mean) / fold_scale)
        cv_errors.extend(np.abs(np.exp(predicted - y_real[test]) - 1.0).tolist())

    estimator, feature_mean, feature_scale = _fit_network(
        x_real, y_real, x_prior, args.seed
    )
    training_conditions = np.asarray(
        [
            [row["length_km"], row["bandwidth_nm"], row["direction"]]
            for row in accepted_rows
        ],
        dtype=np.float64,
    )
    metadata: dict[str, str | float | int] = {
        "accepted_anchor_count": len(accepted_rows),
        "rejected_anchor_count": len(rows) - len(accepted_rows),
        "leave_condition_out_median_fraction_error": (
            float(np.median(cv_errors)) if cv_errors else float("nan")
        ),
        "training_manifest_sha256": manifest.hexdigest(),
        "seed": int(args.seed),
    }
    frozen = NeuralPSFModel(
        weights=tuple(np.asarray(value) for value in estimator.coefs_),
        biases=tuple(np.asarray(value) for value in estimator.intercepts_),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        training_conditions=training_conditions,
        coverage_radius=float(args.coverage_radius),
        max_abs_log_scale=float(np.log(1.5)),
        metadata=metadata,
    )
    model_path = frozen.save(output / "neural_psf_model.npz")
    _write_csv(output / "condition_audit.csv", rows)
    summary = {
        "schema": "v25-neural-psf-training-v1",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "model_path": str(model_path.resolve()),
        "model": "4-12-8-1 tanh MLP; output is bounded log-width residual",
        "physics_parameters": asdict(physics),
        "quality_gate": {
            "minimum_width_ps": 110.0,
            "minimum_width_ratio": args.minimum_width_ratio,
            "maximum_width_ratio": args.maximum_width_ratio,
            "maximum_relative_mad": args.maximum_relative_mad,
        },
        **metadata,
    }
    summary_path = output / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
