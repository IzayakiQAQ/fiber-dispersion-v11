from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len, rfft, irfft

from direct_histogram_compensator import (
    DirectCompensatorConfig,
    PhysicalDirectHistogramCompensator,
    center_of_mass,
    edge_background,
    fwhm_subbin,
    gaussian_coarse_center,
    normalize_probability,
)
from v17_common import SAMPLE_PERIOD_S, pair_dir_records, read_hist, summarize, tdev_at_m, tdev_curve, write_csv, write_json


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate a stateless direct histogram compensator.")
    parser.add_argument("--source-root", type=Path, default=Path("E:/lzy") / "测试结果" / "2026.6.29 50km 280Hz")
    parser.add_argument(
        "--target-psf",
        type=Path,
        default=THIS_DIR / "artifacts" / "physical_0km_target_162ps" / "target_psf.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "results" / "v24_direct_histogram_external_1000_physical_0km",
    )
    parser.add_argument("--half-width", type=int, default=1024)
    parser.add_argument("--calibration-count", type=int, default=500)
    parser.add_argument("--iterations", default="128,192,256,384,512")
    parser.add_argument("--center-half-window", type=int, default=180)
    parser.add_argument("--desired-output-fwhm", type=float, default=165.0)
    parser.add_argument("--output-fwhm-tolerance", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def load_one(task: tuple[str, int]) -> tuple[float, float, float, np.ndarray]:
    path, half_width = task
    first_x, counts = read_hist(Path(path))
    center_idx = gaussian_coarse_center(counts)
    relative = np.arange(-half_width, half_width + 1, dtype=np.float64)
    local = np.interp(
        center_idx + relative,
        np.arange(counts.size, dtype=np.float64),
        counts,
        left=0.0,
        right=0.0,
    )
    return float(first_x), float(center_idx), float(np.sum(counts)), local.astype(np.float32)


def build_input_cache(source_root: Path, cache_path: Path, half_width: int, workers: int) -> dict[str, np.ndarray]:
    records = pair_dir_records(source_root)[:2]
    count = min(len(record["hist_files"]) for record in records)
    length = 2 * half_width + 1
    locals_by_direction = np.zeros((2, count, length), dtype=np.float32)
    coarse_abs = np.zeros((2, count), dtype=np.float64)
    quality_abs = np.zeros((2, count), dtype=np.float64)
    input_fwhm = np.zeros((2, count), dtype=np.float64)
    total_count = np.zeros((2, count), dtype=np.float64)
    first_x_values = np.zeros((2, count), dtype=np.float64)
    pair_names: list[str] = []

    for direction_index, record in enumerate(records):
        pair_names.append(str(record["pair"]))
        files = list(record["hist_files"])[:count]
        quality_rows = list(record["quality_rows"])[:count]
        tasks = [(str(path), half_width) for path in files]
        with ProcessPoolExecutor(max_workers=max(int(workers), 1)) as executor:
            results = executor.map(load_one, tasks, chunksize=8)
            for index, (result, quality) in enumerate(zip(results, quality_rows)):
                first_x, center_idx, full_count, local = result
                locals_by_direction[direction_index, index] = local
                first_x_values[direction_index, index] = first_x
                coarse_abs[direction_index, index] = first_x + center_idx
                quality_abs[direction_index, index] = float(quality["center_hist_ps"])
                sigma = float(quality.get("sigma_ps", np.nan))
                input_fwhm[direction_index, index] = 2.354820045 * sigma if np.isfinite(sigma) else np.nan
                total_count[direction_index, index] = full_count
                if (index + 1) % 100 == 0:
                    print(f"load direction={direction_index + 1} histograms={index + 1}/{count}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        local_histograms=locals_by_direction,
        coarse_center_abs_ps=coarse_abs,
        quality_center_abs_ps=quality_abs,
        input_fwhm_gaussian_ps=input_fwhm,
        total_count=total_count,
        first_x_ps=first_x_values,
        pair_names=np.asarray(pair_names),
    )
    return {
        "local_histograms": locals_by_direction,
        "coarse_center_abs_ps": coarse_abs,
        "quality_center_abs_ps": quality_abs,
        "input_fwhm_gaussian_ps": input_fwhm,
        "total_count": total_count,
        "first_x_ps": first_x_values,
        "pair_names": np.asarray(pair_names),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def signal_probability(histogram: np.ndarray) -> np.ndarray:
    background = edge_background(histogram, 160)
    return normalize_probability(np.clip(np.asarray(histogram, dtype=np.float64) - background, 0.0, None))


def build_broad_psfs(locals_by_direction: np.ndarray, calibration_count: int) -> np.ndarray:
    broad = np.zeros((2, locals_by_direction.shape[-1]), dtype=np.float64)
    for direction in range(2):
        for histogram in locals_by_direction[direction, :calibration_count]:
            broad[direction] += signal_probability(histogram)
        broad[direction] = normalize_probability(broad[direction])
    return broad


def embed_target_psfs(path: Path, output_length: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        source = [
            np.asarray(data["direction1_probability"], dtype=np.float64),
            np.asarray(data["direction2_probability"], dtype=np.float64),
        ]
    targets = np.zeros((2, output_length), dtype=np.float64)
    center = output_length // 2
    for direction, probability in enumerate(source):
        probability = normalize_probability(probability)
        half = probability.size // 2
        targets[direction, center - half : center + half + 1] = probability
        targets[direction] = normalize_probability(targets[direction])
    return targets


def infer_all(
    locals_by_direction: np.ndarray,
    coarse_abs: np.ndarray,
    broad_psfs: np.ndarray,
    target_psfs: np.ndarray,
    iterations: int,
    center_half_window: int,
    keep_histograms: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    directions, count, length = locals_by_direction.shape
    centers = np.zeros((directions, count), dtype=np.float64)
    widths = np.zeros((directions, count), dtype=np.float64)
    outputs = np.zeros_like(locals_by_direction, dtype=np.float32) if keep_histograms else None

    def batch_same_convolution(values: np.ndarray, kernel_fft: np.ndarray, fft_length: int) -> np.ndarray:
        full = irfft(rfft(values, n=fft_length, axis=-1, workers=-1) * kernel_fft[None, :], n=fft_length, axis=-1, workers=-1)
        start = (length - 1) // 2
        return full[:, start : start + length]

    fft_length = next_fast_len(2 * length - 1)
    for direction in range(directions):
        observed = np.clip(np.asarray(locals_by_direction[direction], dtype=np.float64), 0.0, None)
        total = np.sum(observed, axis=-1, keepdims=True)
        edge_width = min(160, max(length // 4, 1))
        background = np.median(
            np.concatenate((observed[:, :edge_width], observed[:, -edge_width:]), axis=-1),
            axis=-1,
            keepdims=True,
        )
        signal = np.clip(observed - background, 0.0, None)
        signal_mass = np.sum(signal, axis=-1, keepdims=True)
        probability = signal / np.clip(signal_mass, 1e-12, None)
        latent = np.clip(probability, 1e-8, None)
        latent /= np.clip(np.sum(latent, axis=-1, keepdims=True), 1e-12, None)
        broad_fft = rfft(broad_psfs[direction], n=fft_length, workers=-1)
        reverse_fft = rfft(broad_psfs[direction, ::-1], n=fft_length, workers=-1)
        target_fft = rfft(target_psfs[direction], n=fft_length, workers=-1)
        for _ in range(max(int(iterations), 0)):
            projection = batch_same_convolution(latent, broad_fft, fft_length)
            ratio = np.clip(probability / np.clip(projection, 1e-12, None), 0.0, 8.0)
            latent *= batch_same_convolution(ratio, reverse_fft, fft_length)
            latent = np.clip(latent, 1e-8, None)
            latent /= np.clip(np.sum(latent, axis=-1, keepdims=True), 1e-12, None)
        reconstructed = np.clip(batch_same_convolution(latent, target_fft, fft_length), 0.0, None)
        reconstructed /= np.clip(np.sum(reconstructed, axis=-1, keepdims=True), 1e-12, None)
        direction_output = reconstructed * signal_mass + background
        direction_output *= total / np.clip(np.sum(direction_output, axis=-1, keepdims=True), 1e-12, None)
        for index in range(count):
            output = direction_output[index]
            relative_center = center_of_mass(output, center_half_window) - float(length // 2)
            centers[direction, index] = float(coarse_abs[direction, index]) + relative_center
            widths[direction, index] = fwhm_subbin(output)
            if outputs is not None:
                outputs[direction, index] = output.astype(np.float32)
    return centers, widths, outputs


def infer_all_cuda(
    locals_by_direction: np.ndarray,
    coarse_abs: np.ndarray,
    broad_psfs: np.ndarray,
    target_psfs: np.ndarray,
    iterations: int,
    center_half_window: int,
    keep_histograms: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    directions, count, length = locals_by_direction.shape
    centers = np.zeros((directions, count), dtype=np.float64)
    widths = np.zeros((directions, count), dtype=np.float64)
    outputs = np.zeros_like(locals_by_direction, dtype=np.float32) if keep_histograms else None
    fft_length = next_fast_len(2 * length - 1)
    start = (length - 1) // 2
    device = torch.device("cuda")

    def batch_same(values: Any, kernel_fft: Any) -> Any:
        full = torch.fft.irfft(torch.fft.rfft(values, n=fft_length, dim=-1) * kernel_fft[None, :], n=fft_length, dim=-1)
        return full[:, start : start + length]

    for direction in range(directions):
        observed_np = np.clip(np.asarray(locals_by_direction[direction], dtype=np.float32), 0.0, None)
        total_np = np.sum(observed_np, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        edge_width = min(160, max(length // 4, 1))
        background_np = np.median(
            np.concatenate((observed_np[:, :edge_width], observed_np[:, -edge_width:]), axis=-1),
            axis=-1,
            keepdims=True,
        ).astype(np.float32)
        signal_np = np.clip(observed_np - background_np, 0.0, None)
        signal_mass_np = np.sum(signal_np, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        probability_np = signal_np / np.clip(signal_mass_np, 1e-12, None)

        probability = torch.as_tensor(probability_np, device=device)
        signal_mass = torch.as_tensor(signal_mass_np, device=device)
        background = torch.as_tensor(background_np, device=device)
        total = torch.as_tensor(total_np, device=device)
        latent = torch.clamp(probability, min=1e-8)
        latent = latent / torch.clamp(torch.sum(latent, dim=-1, keepdim=True), min=1e-12)
        broad = torch.as_tensor(broad_psfs[direction].astype(np.float32), device=device)
        target = torch.as_tensor(target_psfs[direction].astype(np.float32), device=device)
        broad_fft = torch.fft.rfft(broad, n=fft_length)
        reverse_fft = torch.fft.rfft(torch.flip(broad, dims=(-1,)), n=fft_length)
        target_fft = torch.fft.rfft(target, n=fft_length)
        for _ in range(max(int(iterations), 0)):
            projection = batch_same(latent, broad_fft)
            ratio = torch.clamp(probability / torch.clamp(projection, min=1e-12), min=0.0, max=8.0)
            latent = latent * batch_same(ratio, reverse_fft)
            latent = torch.clamp(latent, min=1e-8)
            latent = latent / torch.clamp(torch.sum(latent, dim=-1, keepdim=True), min=1e-12)
        reconstructed = torch.clamp(batch_same(latent, target_fft), min=0.0)
        reconstructed = reconstructed / torch.clamp(torch.sum(reconstructed, dim=-1, keepdim=True), min=1e-12)
        direction_output = reconstructed * signal_mass + background
        direction_output = direction_output * total / torch.clamp(torch.sum(direction_output, dim=-1, keepdim=True), min=1e-12)
        direction_output_np = direction_output.cpu().numpy()
        for index in range(count):
            output = direction_output_np[index]
            relative_center = center_of_mass(output, center_half_window) - float(length // 2)
            centers[direction, index] = float(coarse_abs[direction, index]) + relative_center
            widths[direction, index] = fwhm_subbin(output)
            if outputs is not None:
                outputs[direction, index] = output
    return centers, widths, outputs


def pair_clock(centers: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(centers[0], dtype=np.float64) - np.asarray(centers[1], dtype=np.float64))


def split_tdev(clock: np.ndarray, calibration_count: int) -> dict[str, float]:
    return {
        "calibration_tdev10_ps": tdev_at_m(clock[:calibration_count], 1),
        "heldout_tdev10_ps": tdev_at_m(clock[calibration_count:], 1),
        "full_tdev10_ps": tdev_at_m(clock, 1),
    }


def select_candidate(
    rows: list[dict[str, Any]],
    desired_output_fwhm: float,
    output_fwhm_tolerance: float,
) -> dict[str, Any]:
    lower = float(desired_output_fwhm) - abs(float(output_fwhm_tolerance))
    upper = float(desired_output_fwhm) + abs(float(output_fwhm_tolerance))
    width_qualified = [
        row
        for row in rows
        if lower <= float(row["calibration_output_fwhm_median_ps"]) <= upper
    ]
    if width_qualified:
        candidates = width_qualified
    else:
        nearest_error = min(
            abs(float(row["calibration_output_fwhm_median_ps"]) - float(desired_output_fwhm))
            for row in rows
        )
        candidates = [
            row
            for row in rows
            if abs(float(row["calibration_output_fwhm_median_ps"]) - float(desired_output_fwhm))
            <= nearest_error + 1e-9
        ]
    return min(candidates, key=lambda row: float(row["calibration_tdev10_ps"]))


def save_clock_rows(output_dir: Path, raw_centers: np.ndarray, coarse_centers: np.ndarray, output_centers: np.ndarray) -> None:
    raw_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for index in range(output_centers.shape[1]):
        raw_t1 = float(raw_centers[0, index])
        raw_t2 = float(raw_centers[1, index])
        comp_t1 = float(output_centers[0, index])
        comp_t2 = float(output_centers[1, index])
        raw_rows.append(
            {
                "index": index + 1,
                "time_s": (index + 1) * SAMPLE_PERIOD_S,
                "t1_ps": raw_t1,
                "t2_ps": raw_t2,
                "t0_ps": 0.5 * (raw_t1 - raw_t2),
                "model_internal_coarse_t1_ps": float(coarse_centers[0, index]),
                "model_internal_coarse_t2_ps": float(coarse_centers[1, index]),
            }
        )
        output_rows.append(
            {
                "index": index + 1,
                "time_s": (index + 1) * SAMPLE_PERIOD_S,
                "t1_ps": comp_t1,
                "t2_ps": comp_t2,
                "t0_ps": 0.5 * (comp_t1 - comp_t2),
            }
        )
    write_csv(output_dir / "clock_before_t1_t2_t0.csv", raw_rows)
    write_csv(output_dir / "clock_after_from_output_histograms_t1_t2_t0.csv", output_rows)


def save_width_rows(output_dir: Path, input_fwhm: np.ndarray, output_fwhm: np.ndarray) -> None:
    rows: list[dict[str, Any]] = []
    for index in range(output_fwhm.shape[1]):
        rows.append(
            {
                "index": index + 1,
                "W_in_direction1_ps": float(input_fwhm[0, index]),
                "W_in_direction2_ps": float(input_fwhm[1, index]),
                "W_in_pair_mean_ps": float(np.mean(input_fwhm[:, index])),
                "W_tpl_direction1_ps": float(output_fwhm[0, index]),
                "W_tpl_direction2_ps": float(output_fwhm[1, index]),
                "W_tpl_pair_mean_ps": float(np.mean(output_fwhm[:, index])),
                "width_reduction_factor": float(np.mean(input_fwhm[:, index]) / max(np.mean(output_fwhm[:, index]), 1e-12)),
            }
        )
    write_csv(output_dir / "width_1000_W_in_vs_W_tpl.csv", rows)


def save_figures(
    output_dir: Path,
    locals_by_direction: np.ndarray,
    output_histograms: np.ndarray,
    raw_clock: np.ndarray,
    output_clock: np.ndarray,
) -> None:
    time = np.arange(1, raw_clock.size + 1, dtype=np.float64) * SAMPLE_PERIOD_S
    plt.figure(figsize=(12, 5))
    plt.plot(time, raw_clock - np.mean(raw_clock), label="before", alpha=0.65, linewidth=1.0)
    plt.plot(time, output_clock - np.mean(output_clock), label="after direct histogram output", alpha=0.8, linewidth=1.0)
    plt.xlabel("time (s)")
    plt.ylabel("clock difference after mean removal (ps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "clock_difference_before_after.png", dpi=180)
    plt.close()

    relative = np.arange(-(locals_by_direction.shape[-1] // 2), locals_by_direction.shape[-1] // 2 + 1)
    selected = [0, min(499, raw_clock.size - 1), raw_clock.size - 1]
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 9), sharex=True)
    for row_index, sample_index in enumerate(selected):
        columns = [relative]
        header = ["relative_time_ps"]
        for direction in range(2):
            raw = np.asarray(locals_by_direction[direction, sample_index], dtype=np.float64)
            comp = np.asarray(output_histograms[direction, sample_index], dtype=np.float64)
            raw_scaled = raw / max(float(np.max(raw)), 1e-12)
            comp_scaled = comp / max(float(np.max(comp)), 1e-12)
            axes[row_index, direction].plot(relative, raw_scaled, label="before", alpha=0.6)
            axes[row_index, direction].plot(relative, comp_scaled, label="after", alpha=0.9)
            axes[row_index, direction].set_xlim(-800, 800)
            axes[row_index, direction].set_title(f"histogram {sample_index + 1}, direction {direction + 1}")
            columns.extend([raw, comp])
            header.extend([f"direction{direction + 1}_before", f"direction{direction + 1}_after"])
        np.savetxt(
            output_dir / f"representative_histogram_{sample_index + 1:04d}.csv",
            np.column_stack(columns),
            delimiter=",",
            header=",".join(header),
            comments="",
            fmt="%.8g",
        )
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("relative time (ps)")
    axes[-1, 1].set_xlabel("relative time (ps)")
    fig.tight_layout()
    fig.savefig(output_dir / "representative_histograms_before_after.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "single_histogram_input_cache.npz"
    if cache_path.exists() and not args.rebuild_cache:
        data = load_cache(cache_path)
    else:
        data = build_input_cache(Path(args.source_root), cache_path, int(args.half_width), int(args.workers))

    locals_by_direction = np.asarray(data["local_histograms"], dtype=np.float32)
    coarse_abs = np.asarray(data["coarse_center_abs_ps"], dtype=np.float64)
    quality_abs = np.asarray(data["quality_center_abs_ps"], dtype=np.float64)
    input_fwhm = np.asarray(data["input_fwhm_gaussian_ps"], dtype=np.float64)
    count = locals_by_direction.shape[1]
    calibration_count = min(max(int(args.calibration_count), 16), count - 3)
    broad_psfs = build_broad_psfs(locals_by_direction, calibration_count)
    target_psfs = embed_target_psfs(Path(args.target_psf), locals_by_direction.shape[-1])

    selected_device = str(args.device)
    if selected_device == "auto":
        try:
            import torch

            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            selected_device = "cpu"
    inference_function = infer_all_cuda if selected_device == "cuda" else infer_all
    print(f"inference_device={selected_device}", flush=True)

    raw_clock = pair_clock(quality_abs)
    coarse_clock = pair_clock(coarse_abs)
    candidate_rows: list[dict[str, Any]] = []
    for iterations in [int(value.strip()) for value in str(args.iterations).split(",") if value.strip()]:
        centers, widths, _ = inference_function(
            locals_by_direction,
            coarse_abs,
            broad_psfs,
            target_psfs,
            iterations,
            int(args.center_half_window),
            keep_histograms=False,
        )
        clock = pair_clock(centers)
        metrics = split_tdev(clock, calibration_count)
        row = {
            "iterations": iterations,
            **metrics,
            "calibration_output_fwhm_median_ps": float(np.median(widths[:, :calibration_count])),
            "heldout_output_fwhm_median_ps": float(np.median(widths[:, calibration_count:])),
            "full_output_fwhm_median_ps": float(np.median(widths)),
        }
        candidate_rows.append(row)
        print(row, flush=True)
    selected = select_candidate(
        candidate_rows,
        desired_output_fwhm=float(args.desired_output_fwhm),
        output_fwhm_tolerance=float(args.output_fwhm_tolerance),
    )
    selected_iterations = int(selected["iterations"])
    output_centers, output_fwhm, output_histograms = inference_function(
        locals_by_direction,
        coarse_abs,
        broad_psfs,
        target_psfs,
        selected_iterations,
        int(args.center_half_window),
        keep_histograms=True,
    )
    assert output_histograms is not None
    output_clock = pair_clock(output_centers)

    model_path = output_dir / "direct_histogram_model.npz"
    np.savez_compressed(
        model_path,
        broad_psf_direction1=broad_psfs[0].astype(np.float32),
        broad_psf_direction2=broad_psfs[1].astype(np.float32),
        target_psf_direction1=target_psfs[0].astype(np.float32),
        target_psf_direction2=target_psfs[1].astype(np.float32),
        iterations=np.asarray(selected_iterations, dtype=np.int32),
        edge_bins_per_side=np.asarray(160, dtype=np.int32),
        ratio_clip=np.asarray(8.0, dtype=np.float32),
        latent_floor_fraction=np.asarray(1e-8, dtype=np.float32),
        broad_psf_is_direction_specific=np.asarray(True),
        target_psf_is_direction_specific=np.asarray(True),
        post_output_bounded_center_correction=np.asarray(False),
        legacy_v17_eta_blend_clip_used=np.asarray(False),
        half_width=np.asarray(int(args.half_width), dtype=np.int32),
        center_half_window=np.asarray(int(args.center_half_window), dtype=np.int32),
    )
    np.savez_compressed(
        output_dir / "compensated_histograms_1000x2_local.npz",
        relative_time_ps=np.arange(-int(args.half_width), int(args.half_width) + 1, dtype=np.float32),
        output_histograms=output_histograms,
        output_center_abs_ps=output_centers,
        output_fwhm_ps=output_fwhm,
    )
    write_csv(output_dir / "iteration_calibration_and_heldout.csv", candidate_rows)
    save_clock_rows(output_dir, quality_abs, coarse_abs, output_centers)
    save_width_rows(output_dir, input_fwhm, output_fwhm)
    save_figures(output_dir, locals_by_direction, output_histograms, raw_clock, output_clock)

    raw_metrics = split_tdev(raw_clock, calibration_count)
    coarse_metrics = split_tdev(coarse_clock, calibration_count)
    output_metrics = split_tdev(output_clock, calibration_count)
    area_error = np.abs(
        np.sum(output_histograms, axis=-1, dtype=np.float64)
        - np.sum(locals_by_direction, axis=-1, dtype=np.float64)
    )
    summary = {
        "framework": "v19_direct_histogram_compensator",
        "deployment_contract": "one fixed-axis histogram -> stateless model -> one compensated histogram",
        "uses_adjacent_histograms": False,
        "uses_run_level_statistics_at_inference": False,
        "post_output_center_correction": False,
        "edge_bins_per_side": 160,
        "broad_psf_is_direction_specific": True,
        "output_center_estimator": "raw Gaussian coarse center + compensated local background-subtracted center of mass within +/-180 bins",
        "count": int(count),
        "calibration_count": int(calibration_count),
        "strict_heldout_count": int(count - calibration_count),
        "selected_iterations": selected_iterations,
        "desired_output_fwhm_ps": float(args.desired_output_fwhm),
        "output_fwhm_tolerance_ps": float(args.output_fwhm_tolerance),
        "target_psf": str(Path(args.target_psf)),
        "evaluation_device": selected_device,
        "raw_quality_clock": raw_metrics,
        "model_internal_coarse_clock": coarse_metrics,
        "direct_output_clock": output_metrics,
        "raw_tdev_curve_ps": tdev_curve(raw_clock),
        "direct_output_tdev_curve_ps": tdev_curve(output_clock),
        "raw_clock_ps": summarize(raw_clock),
        "direct_output_clock_ps": summarize(output_clock),
        "input_fwhm_gaussian_ps": summarize(input_fwhm.ravel()),
        "direct_output_fwhm_ps": summarize(output_fwhm.ravel()),
        "width_reduction_factor_median": float(np.median(input_fwhm / np.clip(output_fwhm, 1e-12, None))),
        "coarse_center_vs_cached_quality_error_ps": summarize((coarse_abs - quality_abs).ravel()),
        "maximum_local_count_conservation_error": float(np.max(area_error)),
        "candidate_results": candidate_rows,
        "outputs": {
            "model": str(model_path),
            "clock_before": str(output_dir / "clock_before_t1_t2_t0.csv"),
            "clock_after": str(output_dir / "clock_after_from_output_histograms_t1_t2_t0.csv"),
            "width_1000": str(output_dir / "width_1000_W_in_vs_W_tpl.csv"),
            "histograms_1000x2": str(output_dir / "compensated_histograms_1000x2_local.npz"),
            "representative_figure": str(output_dir / "representative_histograms_before_after.png"),
            "clock_figure": str(output_dir / "clock_difference_before_after.png"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
