from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compensator import (
    V25Compensator,
    coarse_center_index,
    edge_background,
    poisson_template_center_index,
)
from .physics import fwhm_ps


SAMPLE_PERIOD_S = 10.0


def _load_quality(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _records(source_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for quality_path in source_root.rglob("singlepeak_peak_quality_gaussian.csv"):
        rows = _load_quality(quality_path)
        histogram_dirs = [
            path
            for path in quality_path.parent.iterdir()
            if path.is_dir() and "histograms_raw" in path.name
        ]
        if not rows or not histogram_dirs:
            continue
        pair = str(rows[0].get("pair", ""))
        direction = 1 if "ch3_ch1" in pair else 2 if "ch4_ch2" in pair else 0
        if direction == 0:
            continue
        files = sorted(histogram_dirs[0].glob("hist_raw_*.csv"))
        count = min(len(rows), len(files), 1000)
        result.append(
            {
                "direction": direction,
                "pair": pair,
                "files": files[:count],
                "quality": rows[:count],
            }
        )
    result.sort(key=lambda item: int(item["direction"]))
    if [item["direction"] for item in result[:2]] != [1, 2]:
        raise ValueError("Expected ch3_ch1 and ch4_ch2 fixed-axis data")
    return result[:2]


def _load_local(task: tuple[str, int, float, int]) -> tuple[float, float, np.ndarray]:
    path_text, kernel_bins, smooth_sigma, localization_half_width = task
    data = np.loadtxt(Path(path_text), delimiter=",", dtype=np.float32)
    axis0 = float(data[0, 0])
    counts = np.clip(np.nan_to_num(data[:, 1].astype(np.float64)), 0.0, None)
    center = coarse_center_index(counts, smooth_sigma, localization_half_width)
    relative = np.arange(kernel_bins, dtype=np.float64) - kernel_bins // 2
    local = np.interp(
        center + relative,
        np.arange(counts.size, dtype=np.float64),
        counts,
        left=0.0,
        right=0.0,
    )
    return axis0 + center, float(np.sum(counts)), local.astype(np.float32)


def _tdev(values: np.ndarray, m: int = 1) -> float:
    array = np.asarray(values, dtype=np.float64)
    difference = array[2 * m :] - 2.0 * array[m:-m] + array[: -2 * m]
    return float(np.sqrt(np.mean(np.square(difference)) / (6.0 * m * m)))


def _tdev_curve(values: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for m in (1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300):
        if values.size >= 2 * m + 1:
            result[str(int(m * SAMPLE_PERIOD_S))] = _tdev(values, m)
    return result


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blind 1000-group audit of the frozen V25 neural-PSF model."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    compensator = V25Compensator.from_frozen_json(args.frozen_config)
    settings = compensator.config.operator
    records = _records(Path(args.source_root))
    count = min(len(item["files"]) for item in records)
    cache_path = output / "input_cache.npz"
    if cache_path.exists() and not args.rebuild_cache:
        with np.load(cache_path, allow_pickle=False) as data:
            local = np.asarray(data["local_histograms"])
            coarse_abs = np.asarray(data["coarse_center_abs_ps"])
            total_counts = np.asarray(data["total_counts"])
            raw_centers = np.asarray(data["raw_quality_centers_ps"])
            input_widths = np.asarray(data["input_gaussian_fwhm_ps"])
    else:
        local = np.zeros((2, count, settings.kernel_bins), dtype=np.float32)
        coarse_abs = np.zeros((2, count), dtype=np.float64)
        total_counts = np.zeros((2, count), dtype=np.float64)
        raw_centers = np.zeros((2, count), dtype=np.float64)
        input_widths = np.zeros((2, count), dtype=np.float64)
        for direction_index, record in enumerate(records):
            tasks = [
                (
                    str(path),
                    settings.kernel_bins,
                    settings.localization_smooth_sigma_bins,
                    settings.localization_half_width_bins,
                )
                for path in record["files"][:count]
            ]
            with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
                for index, loaded in enumerate(pool.map(_load_local, tasks, chunksize=8)):
                    coarse_abs[direction_index, index], total_counts[direction_index, index], local[direction_index, index] = loaded
            for index, row in enumerate(record["quality"][:count]):
                raw_centers[direction_index, index] = float(row["center_hist_ps"])
                input_widths[direction_index, index] = 2.354820045 * float(row["sigma_ps"])
            print(f"loaded direction {direction_index + 1}: {count}", flush=True)
        np.savez_compressed(
            cache_path,
            local_histograms=local,
            coarse_center_abs_ps=coarse_abs,
            total_counts=total_counts,
            raw_quality_centers_ps=raw_centers,
            input_gaussian_fwhm_ps=input_widths,
        )

    compensated = np.zeros_like(local, dtype=np.float32)
    output_centers = np.zeros((2, count), dtype=np.float64)
    output_widths = np.zeros((2, count), dtype=np.float64)
    for direction in (1, 2):
        batch = compensator.infer_batch_local(local[direction - 1], direction)
        compensated[direction - 1] = batch.astype(np.float32)
        for index, histogram in enumerate(batch):
            center_index = poisson_template_center_index(
                histogram,
                compensator.direction_kernels(direction).target,
                settings.edge_bins,
                max(settings.center_half_window_bins, 240),
            )
            output_centers[direction - 1, index] = (
                coarse_abs[direction - 1, index]
                + center_index
                - settings.kernel_bins // 2
            )
            signal = np.clip(
                histogram - edge_background(histogram, settings.edge_bins),
                0.0,
                None,
            )
            output_widths[direction - 1, index] = fwhm_ps(
                signal, settings.bin_width_ps
            )
        print(f"inferred direction {direction}: {count}", flush=True)

    raw_clock = 0.5 * (raw_centers[0] - raw_centers[1])
    output_clock = 0.5 * (output_centers[0] - output_centers[1])
    clock_rows: list[dict[str, Any]] = []
    width_rows: list[dict[str, Any]] = []
    for index in range(count):
        clock_rows.append(
            {
                "index": index + 1,
                "time_s": (index + 1) * SAMPLE_PERIOD_S,
                "t1_before_ps": raw_centers[0, index],
                "t2_before_ps": raw_centers[1, index],
                "clock_before_ps": raw_clock[index],
                "t1_after_ps": output_centers[0, index],
                "t2_after_ps": output_centers[1, index],
                "clock_after_ps": output_clock[index],
            }
        )
        width_rows.append(
            {
                "index": index + 1,
                "W_in_direction1_ps": input_widths[0, index],
                "W_in_direction2_ps": input_widths[1, index],
                "W_in_mean_ps": float(np.mean(input_widths[:, index])),
                "W_out_direction1_ps": output_widths[0, index],
                "W_out_direction2_ps": output_widths[1, index],
                "W_out_mean_ps": float(np.mean(output_widths[:, index])),
                "reduction_factor": float(
                    np.mean(input_widths[:, index])
                    / max(float(np.mean(output_widths[:, index])), 1e-12)
                ),
            }
        )
    _write_csv(output / "clock_before_after_1000.csv", clock_rows)
    _write_csv(output / "width_1000.csv", width_rows)
    np.savez_compressed(
        output / "compensated_histograms_1000x2.npz",
        relative_time_ps=np.arange(settings.kernel_bins) - settings.kernel_bins // 2,
        raw_local_histograms=local,
        compensated_histograms=compensated,
        output_center_abs_ps=output_centers,
        output_fwhm_ps=output_widths,
    )

    time_s = np.arange(1, count + 1) * SAMPLE_PERIOD_S
    plt.figure(figsize=(11, 4.5))
    plt.plot(time_s, raw_clock - np.mean(raw_clock), label="Before", alpha=0.6)
    plt.plot(time_s, output_clock - np.mean(output_clock), label="After", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Clock difference after mean removal (ps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "clock_before_after_10000s.png", dpi=180)
    plt.close()

    selected = [0, min(499, count - 1), count - 1]
    relative = np.arange(settings.kernel_bins) - settings.kernel_bins // 2
    figure, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for row_index, sample_index in enumerate(selected):
        for direction_index in range(2):
            raw = local[direction_index, sample_index]
            after = compensated[direction_index, sample_index]
            axes[row_index, direction_index].plot(
                relative,
                raw / max(float(np.max(raw)), 1.0),
                label="Before",
                alpha=0.6,
            )
            axes[row_index, direction_index].plot(
                relative,
                after / max(float(np.max(after)), 1.0),
                label="After",
            )
            axes[row_index, direction_index].set_xlim(-700, 700)
            axes[row_index, direction_index].set_title(f"Group {sample_index + 1}, direction {direction_index + 1}")
        np.savetxt(
            output / f"representative_{sample_index + 1:04d}.csv",
            np.column_stack((relative, local[0, sample_index], compensated[0, sample_index], local[1, sample_index], compensated[1, sample_index])),
            delimiter=",",
            header="relative_time_ps,direction1_before,direction1_after,direction2_before,direction2_after",
            comments="",
        )
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output / "representative_histograms.png", dpi=180)
    plt.close(figure)

    area_error = np.max(
        np.abs(np.sum(compensated, axis=2) - np.sum(local, axis=2))
    )
    summary = {
        "framework": "v25_physics_informed_neural_psf",
        "evaluation_data_used_for_psf_or_parameter_selection": False,
        "count": count,
        "sample_period_s": SAMPLE_PERIOD_S,
        "duration_s": count * SAMPLE_PERIOD_S,
        "config_sha256": compensator.config.sha256(),
        "kernel_summary": compensator.kernel_summary(),
        "raw_tdev_curve_ps": _tdev_curve(raw_clock),
        "output_tdev_curve_ps": _tdev_curve(output_clock),
        "input_fwhm_ps": _summary(input_widths.ravel()),
        "output_fwhm_ps": _summary(output_widths.ravel()),
        "width_reduction_factor_median": float(
            np.median(np.mean(input_widths, axis=0) / np.mean(output_widths, axis=0))
        ),
        "raw_clock_ps": _summary(raw_clock),
        "output_clock_ps": _summary(output_clock),
        "maximum_count_area_error": float(area_error),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
