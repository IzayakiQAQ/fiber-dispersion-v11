from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

try:
    from .direct_histogram_compensator import (
        center_of_mass,
        fwhm_subbin,
        gaussian_coarse_center,
    )
    from .physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        load_physics_parameters,
    )
    from .v24_common import (
        pair_dir_records,
        read_hist,
        summarize,
        tdev_at_m,
        tdev_curve,
        write_csv,
        write_json,
    )
except ImportError:
    from direct_histogram_compensator import (
        center_of_mass,
        fwhm_subbin,
        gaussian_coarse_center,
    )
    from physics_informed.forward_model import (
        PhysicsHistogramGenerator,
        load_physics_parameters,
    )
    from v24_common import (
        pair_dir_records,
        read_hist,
        summarize,
        tdev_at_m,
        tdev_curve,
        write_csv,
        write_json,
    )


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "blind_physics_evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one already-frozen v24 physics configuration."
    )
    parser.add_argument("--frozen-config", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-root", type=Path)
    inputs.add_argument("--input-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_config(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    sha_path = path.with_suffix(".sha256")
    expected = sha_path.read_text(encoding="ascii").split()[0]
    if digest != expected:
        raise ValueError("Frozen configuration hash mismatch")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol") != "v24_blind_physics_freeze_before_evaluation":
        raise ValueError("Configuration does not carry the blind-freeze protocol")
    if config.get("selection_source") != (
        "independent_calibration_and_predeclared_physics_simulation"
    ):
        raise ValueError("Configuration was not frozen by the blind protocol")
    inference = config["inference"]
    if inference.get("fisher_residual_enabled") is not False:
        raise ValueError("Blind run must not use a same-run Fisher residual template")
    if inference.get("bounded_center_correction") is not False:
        raise ValueError("Blind run must not use bounded center correction")
    if inference.get("uses_adjacent_histograms") is not False:
        raise ValueError("Blind run must process each histogram independently")
    return config, digest


def physical_psfs(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    calibration_path = Path(config["calibration_json"])
    if sha256_file(calibration_path) != config["calibration_json_sha256"]:
        raise ValueError("Physics calibration changed after freeze")
    inference = config["inference"]
    generator = PhysicsHistogramGenerator(load_physics_parameters(calibration_path))
    length = int(inference["local_bins"])
    broad = np.zeros((2, length), dtype=np.float64)
    target = np.zeros_like(broad)
    for direction in (1, 2):
        broad[direction - 1] = generator.probability(
            float(inference["length_km"]),
            float(inference["bandwidth_nm"]),
            direction,
            n_bins=length,
        )[1]
        target[direction - 1] = generator.probability(
            float(inference["target_length_km"]),
            float(inference["bandwidth_nm"]),
            direction,
            n_bins=length,
        )[1]
    return broad, target


def load_one_histogram(task: tuple[str, int]) -> tuple[float, float, np.ndarray]:
    path, half_width = task
    first_x, counts = read_hist(Path(path))
    center_index = gaussian_coarse_center(counts)
    relative = np.arange(-half_width, half_width + 1, dtype=np.float64)
    local = np.interp(
        center_index + relative,
        np.arange(counts.size, dtype=np.float64),
        counts,
        left=0.0,
        right=0.0,
    )
    return float(first_x), float(center_index), local.astype(np.float32)


def build_input_data(
    source_root: Path,
    local_bins: int,
    workers: int,
    max_pairs: int,
) -> dict[str, np.ndarray]:
    records = pair_dir_records(source_root)[:2]
    count = min(max_pairs, *(len(record["hist_files"]) for record in records))
    half_width = local_bins // 2
    histograms = np.zeros((2, count, local_bins), dtype=np.float32)
    coarse = np.zeros((2, count), dtype=np.float64)
    quality = np.zeros((2, count), dtype=np.float64)
    widths = np.zeros((2, count), dtype=np.float64)
    pair_names: list[str] = []
    for direction_index, record in enumerate(records):
        pair_names.append(str(record["pair"]))
        tasks = [
            (str(path), half_width)
            for path in list(record["hist_files"])[:count]
        ]
        quality_rows = list(record["quality_rows"])[:count]
        with ProcessPoolExecutor(max_workers=max(int(workers), 1)) as executor:
            loaded = executor.map(load_one_histogram, tasks, chunksize=8)
            for index, (values, row) in enumerate(zip(loaded, quality_rows)):
                first_x, center_index, local = values
                histograms[direction_index, index] = local
                coarse[direction_index, index] = first_x + center_index
                quality[direction_index, index] = float(row["center_hist_ps"])
                sigma = float(row.get("sigma_ps", np.nan))
                widths[direction_index, index] = (
                    2.354820045 * sigma if np.isfinite(sigma) else np.nan
                )
    return {
        "local_histograms": histograms,
        "coarse_center_abs_ps": coarse,
        "quality_center_abs_ps": quality,
        "input_fwhm_gaussian_ps": widths,
        "pair_names": np.asarray(pair_names),
    }


def load_input_cache(path: Path, max_pairs: int) -> dict[str, np.ndarray]:
    required = {
        "local_histograms",
        "coarse_center_abs_ps",
        "quality_center_abs_ps",
        "input_fwhm_gaussian_ps",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Input cache is missing keys: {sorted(missing)}")
        return {
            key: np.asarray(data[key])[:, :max_pairs]
            for key in required
        }


def infer(
    histograms: np.ndarray,
    coarse: np.ndarray,
    broad: np.ndarray,
    target: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions, count, length = histograms.shape
    fft_length = next_fast_len(2 * length - 1)
    start = (length - 1) // 2
    edge_bins = int(config["edge_bins_per_side"])
    iterations = int(config["rl_iterations"])
    ratio_clip = float(config["rl_ratio_clip"])
    latent_floor = float(config["latent_floor_fraction"])
    center_window = int(config["center_half_window_bins"])
    outputs = np.zeros_like(histograms, dtype=np.float32)
    centers = np.zeros((directions, count), dtype=np.float64)
    widths = np.zeros_like(centers)

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
        latent = np.clip(probability, latent_floor, None)
        latent /= np.sum(latent, axis=-1, keepdims=True)
        broad_fft = rfft(broad[direction], n=fft_length, workers=-1)
        reverse_fft = rfft(broad[direction, ::-1], n=fft_length, workers=-1)
        target_fft = rfft(target[direction], n=fft_length, workers=-1)
        for _ in range(iterations):
            projection = same(latent, broad_fft)
            ratio = np.clip(
                probability / np.clip(projection, 1e-12, None),
                0.0,
                ratio_clip,
            )
            latent *= same(ratio, reverse_fft)
            latent = np.clip(latent, latent_floor, None)
            latent /= np.sum(latent, axis=-1, keepdims=True)
        reconstructed = np.clip(same(latent, target_fft), 0.0, None)
        reconstructed /= np.clip(
            np.sum(reconstructed, axis=-1, keepdims=True), 1e-12, None
        )
        output = reconstructed * signal_mass + background
        output *= total / np.clip(np.sum(output, axis=-1, keepdims=True), 1e-12, None)
        outputs[direction] = output.astype(np.float32)
        for index in range(count):
            centers[direction, index] = coarse[direction, index] + (
                center_of_mass(output[index], center_window) - length // 2
            )
            widths[direction, index] = fwhm_subbin(output[index])
    return outputs, centers, widths


def clock(centers: np.ndarray) -> np.ndarray:
    return 0.5 * (centers[0] - centers[1])


def main() -> None:
    args = parse_args()
    frozen, config_hash = verify_frozen_config(args.frozen_config)
    inference_config = frozen["inference"]
    # Build both PSFs and verify the independent calibration hash before any
    # evaluation histogram, quality center, or width is opened.
    broad, target = physical_psfs(frozen)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation: {args.output_dir}"
        )
    if args.max_pairs < 3:
        raise ValueError("At least three paired samples are required")
    if args.input_cache is not None:
        data = load_input_cache(args.input_cache, args.max_pairs)
        input_source = str(args.input_cache.resolve())
        input_source_sha256 = sha256_file(args.input_cache)
    else:
        data = build_input_data(
            args.source_root,
            int(inference_config["local_bins"]),
            args.workers,
            args.max_pairs,
        )
        input_source = str(args.source_root.resolve())
        input_source_sha256 = None
    histograms = np.asarray(data["local_histograms"], dtype=np.float32)
    coarse = np.asarray(data["coarse_center_abs_ps"], dtype=np.float64)
    # These two arrays are report-only. Neither is passed to infer().
    raw_centers = np.asarray(data["quality_center_abs_ps"], dtype=np.float64)
    input_widths = np.asarray(data["input_fwhm_gaussian_ps"], dtype=np.float64)
    if histograms.ndim != 3 or histograms.shape[0] != 2:
        raise ValueError("Expected paired histograms with shape (2, samples, bins)")
    if histograms.shape[-1] != int(inference_config["local_bins"]):
        raise ValueError("Evaluation histogram length does not match frozen config")

    outputs, output_centers, output_widths = infer(
        histograms,
        coarse,
        broad,
        target,
        inference_config,
    )
    raw_clock = clock(raw_centers)
    output_clock = clock(output_centers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clock_rows = []
    width_rows = []
    for index in range(raw_clock.size):
        clock_rows.append(
            {
                "index": index + 1,
                "time_s": (index + 1) * float(inference_config["integration_time_s"]),
                "raw_t1_ps": float(raw_centers[0, index]),
                "raw_t2_ps": float(raw_centers[1, index]),
                "raw_clock_ps": float(raw_clock[index]),
                "output_t1_ps": float(output_centers[0, index]),
                "output_t2_ps": float(output_centers[1, index]),
                "output_clock_ps": float(output_clock[index]),
            }
        )
        width_rows.append(
            {
                "index": index + 1,
                "W_in_direction1_ps": float(input_widths[0, index]),
                "W_in_direction2_ps": float(input_widths[1, index]),
                "W_out_direction1_ps": float(output_widths[0, index]),
                "W_out_direction2_ps": float(output_widths[1, index]),
            }
        )
    write_csv(args.output_dir / "clock_before_after_1000.csv", clock_rows)
    write_csv(args.output_dir / "width_1000_W_in_vs_W_out.csv", width_rows)
    raw_curve = tdev_curve(raw_clock)
    output_curve = tdev_curve(output_clock)
    tdev_rows = [
        {
            "tau_s": float(tau),
            "raw_tdev_ps": float(raw_curve[tau]),
            "output_tdev_ps": float(output_curve[tau]),
            "improvement_factor": float(raw_curve[tau] / output_curve[tau]),
        }
        for tau in raw_curve
        if tau in output_curve
    ]
    write_csv(args.output_dir / "tdev_before_after.csv", tdev_rows)
    np.savez_compressed(
        args.output_dir / "blind_output_histograms_1000.npz",
        output_histograms=outputs,
        output_centers_ps=output_centers,
        output_fwhm_ps=output_widths,
    )

    time_s = np.arange(1, raw_clock.size + 1) * float(
        inference_config["integration_time_s"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    axes[0].plot(time_s, raw_clock - np.mean(raw_clock), label="before", alpha=0.7)
    axes[0].plot(
        time_s,
        output_clock - np.mean(output_clock),
        label="frozen blind physics output",
        alpha=0.8,
    )
    axes[0].set_ylabel("Clock difference (ps)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].loglog(
        [row["tau_s"] for row in tdev_rows],
        [row["raw_tdev_ps"] for row in tdev_rows],
        "o-",
        label="before",
    )
    axes[1].loglog(
        [row["tau_s"] for row in tdev_rows],
        [row["output_tdev_ps"] for row in tdev_rows],
        "s-",
        label="frozen blind physics output",
    )
    axes[1].set_xlabel("Averaging time (s)")
    axes[1].set_ylabel("TDEV (ps)")
    axes[1].grid(which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "clock_and_tdev_blind.png", dpi=180)
    plt.close(fig)

    summary = {
        "protocol": "v24_frozen_physics_locked_replay",
        "frozen_config_sha256": config_hash,
        "input_source": input_source,
        "input_source_sha256": input_source_sha256,
        "frozen_inference": inference_config,
        "count": int(raw_clock.size),
        "raw_fwhm_ps": summarize(input_widths.reshape(-1)),
        "output_fwhm_ps": summarize(output_widths.reshape(-1)),
        "raw_tdev10_full_ps": tdev_at_m(raw_clock, 1),
        "output_tdev10_full_ps": tdev_at_m(output_clock, 1),
        "raw_tdev10_first500_ps": tdev_at_m(raw_clock[:500], 1),
        "output_tdev10_first500_ps": tdev_at_m(output_clock[:500], 1),
        "raw_tdev10_last500_ps": tdev_at_m(raw_clock[500:], 1),
        "output_tdev10_last500_ps": tdev_at_m(output_clock[500:], 1),
        "count_conservation_max_abs": float(
            np.max(
                np.abs(
                    np.sum(outputs, axis=-1)
                    - np.sum(histograms, axis=-1)
                )
            )
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
