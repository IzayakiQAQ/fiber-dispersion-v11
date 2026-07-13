from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from direct_histogram_compensator import fwhm_subbin, gaussian_coarse_fit
from v17_common import read_hist, sample_local, summarize, tdev_at_m, write_csv, write_json


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DCM_ROOT = Path("E:/lzy") / "测试结果" / "2025.7.24 50km 24h 0.8nm 100Hz 色散补偿模块"
DEFAULT_DCM_HIST_DIR = (
    DEFAULT_DCM_ROOT
    / "单峰全程_13_42_1ps完整横坐标"
    / "segment1_20250724_2145_ch13_ch42_fullaxis"
    / "ch1_ch3_histograms_raw_1ps"
)
DEFAULT_SOFTWARE_ROOT = THIS_DIR / "results" / "v24_direct_histogram_external_1000_physical_0km"

COLORS = {
    "raw": "#6B7280",
    "software": "#007C83",
    "hardware": "#D97706",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready DCM hardware versus direct software compensation results.")
    parser.add_argument("--dcm-root", type=Path, default=DEFAULT_DCM_ROOT)
    parser.add_argument("--dcm-hist-dir", type=Path, default=DEFAULT_DCM_HIST_DIR)
    parser.add_argument("--software-root", type=Path, default=DEFAULT_SOFTWARE_ROOT)
    parser.add_argument(
        "--software-input-cache",
        type=Path,
        default=DEFAULT_SOFTWARE_ROOT / "single_histogram_input_cache.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "results" / "paper_dcm100hz_vs_direct280hz",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rebuild-dcm-cache", action="store_true")
    return parser.parse_args()


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fit_dcm_histogram(path_text: str) -> dict[str, float]:
    _, counts = read_hist(Path(path_text))
    center, sigma = gaussian_coarse_fit(counts)
    return {
        "center_ps": float(center),
        "sigma_ps": float(sigma),
        "fwhm_ps": 2.354820045 * float(sigma),
        "total_count": float(np.sum(counts)),
        "peak_count": float(np.max(counts)),
    }


def normalized_aligned_dcm_histogram(task: tuple[str, float, int]) -> np.ndarray:
    path_text, center_ps, half_width = task
    first_x, counts = read_hist(Path(path_text))
    local = np.clip(sample_local(counts, float(center_ps) - float(first_x), int(half_width)), 0.0, None)
    edge = min(80, max(local.size // 8, 1))
    background = float(np.median(np.concatenate((local[:edge], local[-edge:]))))
    signal = np.clip(local - background, 0.0, None)
    return signal / max(float(np.sum(signal)), 1e-12)


def load_or_build_dcm_histograms(
    dcm_hist_dir: Path,
    cache_path: Path,
    workers: int,
    rebuild: bool,
) -> list[dict[str, Any]]:
    if cache_path.exists() and not rebuild:
        rows = read_dict_csv(cache_path)
        return [
            {
                "index": int(row["index"]),
                "histogram_path": row["histogram_path"],
                "center_ps": float(row["center_ps"]),
                "sigma_ps": float(row["sigma_ps"]),
                "fwhm_ps": float(row["fwhm_ps"]),
                "total_count": float(row["total_count"]),
                "peak_count": float(row["peak_count"]),
            }
            for row in rows
        ]
    files = sorted(dcm_hist_dir.glob("hist_raw_*.csv"))
    if not files:
        raise ValueError(f"No DCM histograms found under {dcm_hist_dir}")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        for index, (path, result) in enumerate(
            zip(files, executor.map(fit_dcm_histogram, [str(path) for path in files], chunksize=8)),
            start=1,
        ):
            rows.append({"index": index, "histogram_path": str(path), **result})
            if index % 100 == 0:
                print(f"DCM histograms fitted: {index}/{len(files)}", flush=True)
    write_csv(cache_path, rows)
    return rows


def load_or_build_dcm_average(
    histogram_rows: list[dict[str, Any]],
    cache_path: Path,
    workers: int,
    rebuild: bool,
    half_width: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_relative = np.asarray(data["relative_time_ps"], dtype=np.float64)
            cached_probability = np.asarray(data["probability"], dtype=np.float64)
        if cached_relative.size == 2 * int(half_width) + 1:
            return cached_relative, cached_probability
    tasks = [
        (str(row["histogram_path"]), float(row["center_ps"]), int(half_width))
        for row in histogram_rows
    ]
    total = np.zeros(2 * half_width + 1, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        for probability in executor.map(normalized_aligned_dcm_histogram, tasks, chunksize=8):
            total += np.asarray(probability, dtype=np.float64)
    probability = total / max(float(np.sum(total)), 1e-12)
    relative = np.arange(-half_width, half_width + 1, dtype=np.float64)
    np.savez_compressed(cache_path, relative_time_ps=relative, probability=probability)
    return relative, probability


def aligned_average_from_arrays(histograms: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(histograms, dtype=np.float64)
    center_values = np.asarray(centers, dtype=np.float64)
    if values.shape[:-1] != center_values.shape:
        raise ValueError(f"Histogram and center shapes do not match: {values.shape}, {center_values.shape}")
    length = values.shape[-1]
    half_width = length // 2
    relative = np.arange(-half_width, half_width + 1, dtype=np.float64)
    axis = np.arange(length, dtype=np.float64)
    total = np.zeros(length, dtype=np.float64)
    for histogram, center in zip(values.reshape(-1, length), center_values.ravel()):
        aligned = np.interp(float(center) + relative, axis, histogram, left=0.0, right=0.0)
        edge = min(160, max(aligned.size // 8, 1))
        background = float(np.median(np.concatenate((aligned[:edge], aligned[-edge:]))))
        signal = np.clip(aligned - background, 0.0, None)
        total += signal / max(float(np.sum(signal)), 1e-12)
    probability = total / max(float(np.sum(total)), 1e-12)
    return relative, probability


def tdev_curve_rows(series: dict[str, np.ndarray], sample_period_s: float = 10.0) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    length = min(values.size for values in series.values())
    for m in (1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300):
        if length < 2 * m + 1:
            continue
        row: dict[str, float] = {"tau_s": sample_period_s * m}
        for name, values in series.items():
            row[name] = tdev_at_m(values, m)
        rows.append(row)
    return rows


def metric_row(
    condition_id: str,
    label: str,
    method: str,
    event_rate_hz: float,
    clock: np.ndarray,
    widths: np.ndarray,
    raw_tdev: float,
    raw_width: float,
) -> dict[str, Any]:
    tdev10 = tdev_at_m(clock, 1)
    width_summary = summarize(widths)
    return {
        "condition_id": condition_id,
        "paper_label": label,
        "method": method,
        "measurement_status": "measured",
        "nominal_rate_hz": float(event_rate_hz),
        "accumulation_time_s": 10.0,
        "sample_count": int(clock.size),
        "tdev_10s_ps": tdev10,
        "clock_std_ps": float(np.std(clock)),
        "fwhm_mean_ps": width_summary["mean"],
        "fwhm_std_ps": width_summary["std"],
        "fwhm_p05_ps": width_summary["p05"],
        "fwhm_median_ps": width_summary["median"],
        "fwhm_p95_ps": width_summary["p95"],
        "stability_improvement_vs_raw": raw_tdev / max(tdev10, 1e-12),
        "width_reduction_vs_raw": raw_width / max(width_summary["median"], 1e-12),
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_clock_series(output_dir: Path, time_s: np.ndarray, clocks: dict[str, np.ndarray]) -> None:
    labels = [
        ("software_raw", "50 km input, 280 Hz", COLORS["raw"]),
        ("software_direct", "Direct software output, 280 Hz", COLORS["software"]),
        ("hardware_dcm", "Hardware DCM, 100 Hz", COLORS["hardware"]),
    ]
    demeaned = {key: values - float(np.mean(values)) for key, values in clocks.items()}
    bound = max(float(np.percentile(np.abs(values), 99.5)) for values in demeaned.values()) * 1.08
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.4), sharex=True, sharey=True)
    for axis, (key, label, color) in zip(axes, labels):
        axis.plot(time_s, demeaned[key], color=color, linewidth=0.75)
        axis.axhline(0.0, color="#111827", linewidth=0.55, alpha=0.55)
        axis.set_ylim(-bound, bound)
        axis.text(0.015, 0.88, label, transform=axis.transAxes, fontsize=10, fontweight="bold")
        axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.55)
    axes[1].set_ylabel("Clock difference after mean removal (ps)")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(h_pad=0.25)
    save_figure(fig, output_dir, "fig1_clock_series")


def plot_tdev(output_dir: Path, rows: list[dict[str, float]], rate_scale: float) -> None:
    tau = np.asarray([row["tau_s"] for row in rows])
    fig, axis = plt.subplots(figsize=(7.6, 5.2))
    axis.loglog(tau, [row["software_raw_tdev_ps"] for row in rows], "o-", color=COLORS["raw"], label="50 km input, 280 Hz")
    axis.loglog(tau, [row["software_direct_tdev_ps"] for row in rows], "s-", color=COLORS["software"], label="Direct software output, 280 Hz")
    dcm = np.asarray([row["hardware_dcm_tdev_ps"] for row in rows])
    axis.loglog(tau, dcm, "^-", color=COLORS["hardware"], label="Hardware DCM, 100 Hz")
    axis.loglog(
        tau,
        dcm * rate_scale,
        "--",
        color=COLORS["hardware"],
        linewidth=1.3,
        label="Hardware DCM projected to 280 Hz",
    )
    axis.set_xlabel(r"Averaging time $\tau$ (s)")
    axis.set_ylabel("TDEV (ps)")
    axis.grid(True, which="both", color="#D1D5DB", linewidth=0.5, alpha=0.6)
    axis.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.94, fontsize=9, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig2_tdev_comparison")


def plot_fwhm(output_dir: Path, widths: dict[str, np.ndarray]) -> None:
    fig, axis = plt.subplots(figsize=(7.8, 5.2))
    values = [widths["software_raw"], widths["software_direct"], widths["hardware_dcm"]]
    box = axis.boxplot(
        values,
        tick_labels=["50 km input\n280 Hz", "Direct software\n280 Hz", "Hardware DCM\n100 Hz"],
        patch_artist=True,
        showfliers=False,
        widths=0.58,
    )
    for patch, color in zip(box["boxes"], (COLORS["raw"], COLORS["software"], COLORS["hardware"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.4)
    axis.set_yscale("log")
    axis.set_ylabel("FWHM (ps, log scale)")
    axis.grid(True, axis="y", which="both", color="#D1D5DB", linewidth=0.5, alpha=0.65)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig3_fwhm_comparison")


def plot_width_vs_tdev(
    output_dir: Path,
    main_rows: list[dict[str, Any]],
    projected_tdev: float,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    colors = (COLORS["raw"], COLORS["software"], COLORS["hardware"])
    offsets = ((-155, 9), (9, 8), (9, -18))
    for row, color, offset in zip(main_rows, colors, offsets):
        axis.scatter(row["fwhm_median_ps"], row["tdev_10s_ps"], s=72, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        axis.annotate(row["paper_label"], (row["fwhm_median_ps"], row["tdev_10s_ps"]), xytext=offset, textcoords="offset points", fontsize=9)
    dcm_row = main_rows[2]
    axis.scatter(
        dcm_row["fwhm_median_ps"],
        projected_tdev,
        s=76,
        facecolors="none",
        edgecolors=COLORS["hardware"],
        linewidth=1.5,
        zorder=3,
    )
    axis.annotate("Hardware DCM, projected 280 Hz", (dcm_row["fwhm_median_ps"], projected_tdev), xytext=(9, 8), textcoords="offset points", fontsize=9)
    axis.set_xscale("log")
    axis.set_xlim(45.0, 850.0)
    axis.set_ylim(max(1.05, projected_tdev - 0.15), 4.3)
    axis.set_xlabel("Median FWHM (ps, log scale)")
    axis.set_ylabel("TDEV at 10 s (ps)")
    axis.grid(True, which="both", color="#D1D5DB", linewidth=0.5, alpha=0.65)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig4_width_vs_tdev")


def plot_representative_histograms(
    output_dir: Path,
    software_x: np.ndarray,
    software_raw: np.ndarray,
    software_direct: np.ndarray,
    dcm_relative: np.ndarray,
    dcm_probability: np.ndarray,
) -> None:
    full_x = np.arange(
        max(float(np.min(software_x)), float(np.min(dcm_relative))),
        min(float(np.max(software_x)), float(np.max(dcm_relative))) + 1.0,
        1.0,
    )
    full_probabilities = {
        "software_input_aligned_probability": np.interp(full_x, software_x, software_raw, left=0.0, right=0.0),
        "software_output_aligned_probability": np.interp(full_x, software_x, software_direct, left=0.0, right=0.0),
        "hardware_dcm_ch1_ch3_aligned_probability": np.interp(full_x, dcm_relative, dcm_probability, left=0.0, right=0.0),
    }
    full_peak_normalized = {
        key.replace("_aligned_probability", "_peak_normalized"): values / max(float(np.max(values)), 1e-12)
        for key, values in full_probabilities.items()
    }
    write_csv(
        output_dir / "fig5_source_data_complete.csv",
        [
            {
                "relative_time_ps": float(full_x[index]),
                **{key: float(values[index]) for key, values in full_probabilities.items()},
                **{key: float(values[index]) for key, values in full_peak_normalized.items()},
            }
            for index in range(full_x.size)
        ],
    )

    common_x = np.arange(-800.0, 801.0, 1.0)
    curves = {
        "software_raw_probability": np.interp(common_x, software_x, software_raw, left=0.0, right=0.0),
        "software_direct_probability": np.interp(common_x, software_x, software_direct, left=0.0, right=0.0),
        "hardware_dcm_aligned_average_probability": np.interp(common_x, dcm_relative, dcm_probability, left=0.0, right=0.0),
    }
    for key, values in curves.items():
        curves[key] = values / max(float(np.max(values)), 1e-12)
    plot_rows = [
        {"relative_time_ps": float(common_x[index]), **{key: float(values[index]) for key, values in curves.items()}}
        for index in range(common_x.size)
    ]
    write_csv(output_dir / "fig5_plot_data.csv", plot_rows)
    write_csv(output_dir / "paper_representative_histograms.csv", plot_rows)
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(common_x, curves["software_raw_probability"], color=COLORS["raw"], linewidth=1.5, label="50 km input aligned average, 280 Hz")
    axis.plot(common_x, curves["software_direct_probability"], color=COLORS["software"], linewidth=1.8, label="Direct software aligned average, 280 Hz")
    axis.plot(common_x, curves["hardware_dcm_aligned_average_probability"], color=COLORS["hardware"], linewidth=1.8, label="Hardware DCM ch1-ch3 aligned average, 100 Hz")
    axis.set_xlim(-650.0, 650.0)
    axis.set_xlabel("Relative time (ps)")
    axis.set_ylabel("Normalized counts")
    axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.6)
    axis.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.94, fontsize=9, loc="lower left")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5_representative_histograms")


def main() -> None:
    args = parse_args()
    dcm_root = Path(args.dcm_root)
    dcm_hist_dir = Path(args.dcm_hist_dir)
    software_root = Path(args.software_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dcm_clock_rows = read_dict_csv(dcm_root / "0km.csv")
    dcm_t1 = np.asarray([float(row["ch1-ch4"]) for row in dcm_clock_rows], dtype=np.float64)
    dcm_t2 = np.asarray([float(row["ch2-ch3"]) for row in dcm_clock_rows], dtype=np.float64)
    dcm_clock = np.asarray([float(row["clock correction"]) for row in dcm_clock_rows], dtype=np.float64)
    dcm_time = np.asarray([float(row["time"]) for row in dcm_clock_rows], dtype=np.float64)
    dcm_formula_error = np.max(np.abs(0.5 * (dcm_t1 - dcm_t2) - dcm_clock))

    before_rows = read_dict_csv(software_root / "clock_before_t1_t2_t0.csv")
    after_rows = read_dict_csv(software_root / "clock_after_from_output_histograms_t1_t2_t0.csv")
    width_rows = read_dict_csv(software_root / "width_1000_W_in_vs_W_tpl.csv")
    software_raw_clock = np.asarray([float(row["t0_ps"]) for row in before_rows], dtype=np.float64)
    software_direct_clock = np.asarray([float(row["t0_ps"]) for row in after_rows], dtype=np.float64)
    software_time = np.asarray([float(row["time_s"]) for row in before_rows], dtype=np.float64)
    software_input_width = np.asarray([float(row["W_in_pair_mean_ps"]) for row in width_rows], dtype=np.float64)
    software_output_width = np.asarray([float(row["W_tpl_pair_mean_ps"]) for row in width_rows], dtype=np.float64)

    count = min(
        dcm_clock.size,
        software_raw_clock.size,
        software_direct_clock.size,
        software_input_width.size,
        software_output_width.size,
    )
    if count != 1000:
        raise ValueError(f"The paper comparison expects 1000 aligned samples, found {count}")
    dcm_clock = dcm_clock[:count]
    dcm_time = dcm_time[:count]
    software_raw_clock = software_raw_clock[:count]
    software_direct_clock = software_direct_clock[:count]
    software_time = software_time[:count]
    software_input_width = software_input_width[:count]
    software_output_width = software_output_width[:count]
    if not np.allclose(dcm_time, software_time, rtol=0.0, atol=1e-9):
        raise ValueError("Software and DCM time axes are not aligned")

    dcm_hist_rows = load_or_build_dcm_histograms(
        dcm_hist_dir,
        output_dir / "dcm_histogram_fits_1000.csv",
        int(args.workers),
        bool(args.rebuild_dcm_cache),
    )
    if len(dcm_hist_rows) != count:
        raise ValueError(f"Expected {count} DCM histograms, found {len(dcm_hist_rows)}")
    dcm_width = np.asarray([float(row["fwhm_ps"]) for row in dcm_hist_rows], dtype=np.float64)
    dcm_counts = np.asarray([float(row["total_count"]) for row in dcm_hist_rows], dtype=np.float64)
    dcm_relative, dcm_average = load_or_build_dcm_average(
        dcm_hist_rows,
        output_dir / "dcm_aligned_average_psf.npz",
        int(args.workers),
        bool(args.rebuild_dcm_cache),
    )
    average_center, average_sigma = gaussian_coarse_fit(dcm_average, smooth_sigma=2.0, fit_half_width=240)
    dcm_average_gaussian_fwhm = 2.354820045 * float(average_sigma)
    dcm_average_halfmax_fwhm = fwhm_subbin(dcm_average)
    write_csv(
        output_dir / "dcm_aligned_average_histogram.csv",
        [
            {"relative_time_ps": float(dcm_relative[index]), "probability": float(dcm_average[index])}
            for index in range(dcm_relative.size)
        ],
    )
    with np.load(Path(args.software_input_cache), allow_pickle=False) as input_data:
        software_input_histograms = np.asarray(input_data["local_histograms"], dtype=np.float32)
        software_coarse_abs = np.asarray(input_data["coarse_center_abs_ps"], dtype=np.float64)
        software_total_count = np.asarray(input_data["total_count"], dtype=np.float64)
    with np.load(software_root / "compensated_histograms_1000x2_local.npz", allow_pickle=False) as output_data:
        software_output_histograms = np.asarray(output_data["output_histograms"], dtype=np.float32)
        software_output_center_abs = np.asarray(output_data["output_center_abs_ps"], dtype=np.float64)
    software_center_index = software_input_histograms.shape[-1] // 2
    software_input_relative_centers = np.full(software_coarse_abs.shape, float(software_center_index))
    software_output_relative_centers = float(software_center_index) + software_output_center_abs - software_coarse_abs
    software_average_x, software_input_average = aligned_average_from_arrays(
        software_input_histograms,
        software_input_relative_centers,
    )
    _, software_output_average = aligned_average_from_arrays(
        software_output_histograms,
        software_output_relative_centers,
    )
    _, software_input_average_sigma = gaussian_coarse_fit(software_input_average, smooth_sigma=2.0, fit_half_width=800)
    _, software_output_average_sigma = gaussian_coarse_fit(software_output_average, smooth_sigma=2.0, fit_half_width=240)

    clocks = {
        "software_raw": software_raw_clock,
        "software_direct": software_direct_clock,
        "hardware_dcm": dcm_clock,
    }
    widths = {
        "software_raw": software_input_width,
        "software_direct": software_output_width,
        "hardware_dcm": dcm_width,
    }
    raw_tdev = tdev_at_m(software_raw_clock, 1)
    raw_width = float(np.median(software_input_width))
    main_rows = [
        metric_row(
            "software_raw_280hz",
            "50 km input, 280 Hz",
            "No software compensation",
            280.0,
            software_raw_clock,
            software_input_width,
            raw_tdev,
            raw_width,
        ),
        metric_row(
            "software_direct_280hz",
            "Direct software output, 280 Hz",
            "Single-histogram Fisher-decoded software compensation",
            280.0,
            software_direct_clock,
            software_output_width,
            raw_tdev,
            raw_width,
        ),
        metric_row(
            "hardware_dcm_100hz",
            "Hardware DCM, 100 Hz",
            "Optical dispersion compensation module",
            100.0,
            dcm_clock,
            dcm_width,
            raw_tdev,
            raw_width,
        ),
    ]
    hardware_fwhm_diagnostic = {
        "dataset_id": "hardware_dcm_same_condition_separate_acquisition",
        "measurement_status": "operator-provided hardware DCM histogram run",
        "data_linkage": "FWHM uses the specified ch1-ch3 run; TDEV uses the validated 0km.csv series; condition-level comparison",
        "histogram_direction": "ch1-ch3",
        "histogram_source": str(dcm_hist_dir),
        "histogram_count": int(dcm_width.size),
        "fwhm_mean_ps": float(np.mean(dcm_width)),
        "fwhm_std_ps": float(np.std(dcm_width)),
        "fwhm_p05_ps": float(np.percentile(dcm_width, 5)),
        "fwhm_median_ps": float(np.median(dcm_width)),
        "fwhm_p95_ps": float(np.percentile(dcm_width, 95)),
        "aligned_average_gaussian_fwhm_ps": dcm_average_gaussian_fwhm,
        "aligned_average_halfmax_fwhm_ps": dcm_average_halfmax_fwhm,
    }
    paper_main_rows = [dict(row) for row in main_rows]
    for row in paper_main_rows[:2]:
        row["fwhm_data_status"] = "matched to the reported software clock series"
    paper_main_rows[2]["fwhm_data_status"] = "operator-provided ch1-ch3 hardware histogram run; condition-level comparison"
    rate_scale = float(np.sqrt(100.0 / 280.0))
    projected_tdev = float(main_rows[2]["tdev_10s_ps"]) * rate_scale
    projection_row = {
        "condition_id": "hardware_dcm_projected_280hz",
        "paper_label": "Hardware DCM projected to 280 Hz",
        "measurement_status": "shot-noise projection, not measured",
        "source_nominal_rate_hz": 100.0,
        "projected_nominal_rate_hz": 280.0,
        "scaling_rule": "TDEV_280 = TDEV_100 * sqrt(100/280)",
        "projection_assumption": "Detected counts scale linearly with nominal rate and all other noise terms remain unchanged.",
        "measured_tdev_10s_ps_at_100hz": float(main_rows[2]["tdev_10s_ps"]),
        "projected_tdev_10s_ps_at_280hz": projected_tdev,
        "hardware_dcm_fwhm_median_ps": float(main_rows[2]["fwhm_median_ps"]),
    }

    tdev_rows = tdev_curve_rows(
        {
            "software_raw_tdev_ps": software_raw_clock,
            "software_direct_tdev_ps": software_direct_clock,
            "hardware_dcm_tdev_ps": dcm_clock,
        }
    )
    for row in tdev_rows:
        row["hardware_dcm_projected_280hz_tdev_ps"] = row["hardware_dcm_tdev_ps"] * rate_scale
    clock_rows = [
        {
            "index": index + 1,
            "time_s": float(software_time[index]),
            "software_raw_t0_ps": float(software_raw_clock[index]),
            "software_raw_t0_demeaned_ps": float(software_raw_clock[index] - np.mean(software_raw_clock)),
            "software_direct_t0_ps": float(software_direct_clock[index]),
            "software_direct_t0_demeaned_ps": float(software_direct_clock[index] - np.mean(software_direct_clock)),
            "hardware_dcm_t0_ps": float(dcm_clock[index]),
            "hardware_dcm_t0_demeaned_ps": float(dcm_clock[index] - np.mean(dcm_clock)),
        }
        for index in range(count)
    ]
    fwhm_rows = [
        {
            "index": index + 1,
            "software_input_W_in_ps": float(software_input_width[index]),
            "software_output_W_tpl_ps": float(software_output_width[index]),
            "hardware_dcm_fwhm_ps": float(dcm_width[index]),
            "hardware_dcm_total_count": float(dcm_counts[index]),
            "hardware_dcm_data_status": "operator-provided ch1-ch3 run; condition-level comparison",
        }
        for index in range(count)
    ]
    write_csv(output_dir / "paper_table_main_metrics.csv", paper_main_rows)
    write_csv(output_dir / "paper_table_hardware_fwhm_condition.csv", [hardware_fwhm_diagnostic])
    write_csv(
        output_dir / "paper_table_unmatched_hardware_fwhm_diagnostic.csv",
        [{"status": "superseded", "reason": "operator confirmed the histogram run used the same experimental condition"}],
    )
    write_csv(output_dir / "paper_table_rate_normalized_projection.csv", [projection_row])
    write_csv(output_dir / "paper_clock_series_1000.csv", clock_rows)
    write_csv(output_dir / "paper_tdev_curve.csv", tdev_rows)
    write_csv(output_dir / "paper_fwhm_distribution_1000.csv", fwhm_rows)

    plot_clock_series(output_dir, software_time, clocks)
    plot_tdev(output_dir, tdev_rows, rate_scale)
    plot_fwhm(output_dir, widths)
    plot_width_vs_tdev(output_dir, main_rows, projected_tdev)
    plot_representative_histograms(
        output_dir,
        software_average_x,
        software_input_average,
        software_output_average,
        dcm_relative,
        dcm_average,
    )

    comparison = {
        "software_operator": {
            "type": "Richardson-Lucy deconvolution followed by physical 0 km target-PSF convolution",
            "iterations": 512,
            "edge_bins_per_side": 160,
            "direction_specific_broad_psf": True,
            "direction_specific_target_psf": True,
            "output_center_estimator": "raw Gaussian coarse center + compensated local background-subtracted center of mass within +/-180 bins",
            "post_output_bounded_center_correction": False,
            "legacy_v17_eta_blend_clip_used": False,
        },
        "software_vs_hardware": {
            "software_direct_tdev_10s_ps": float(main_rows[1]["tdev_10s_ps"]),
            "hardware_dcm_measured_tdev_10s_ps": float(main_rows[2]["tdev_10s_ps"]),
            "software_to_hardware_tdev_ratio": float(main_rows[1]["tdev_10s_ps"] / main_rows[2]["tdev_10s_ps"]),
            "hardware_fwhm_comparison_status": "operator-provided ch1-ch3 run; condition-level comparison to validated TDEV",
        },
        "hardware_histogram_measurement": {
            "use_for_paper_main_comparison": True,
            "direction": "ch1-ch3",
            "data_linkage": "same hardware condition; FWHM and TDEV are not sample-wise paired",
            "per_histogram_median_fwhm_ps": float(main_rows[2]["fwhm_median_ps"]),
            "aligned_average_gaussian_fwhm_ps": dcm_average_gaussian_fwhm,
            "aligned_average_halfmax_fwhm_ps": dcm_average_halfmax_fwhm,
            "histogram_directory": str(dcm_hist_dir),
            "histogram_directory_last_write_time": datetime.fromtimestamp(dcm_hist_dir.stat().st_mtime).isoformat(),
            "clock_csv_last_write_time": datetime.fromtimestamp((dcm_root / "0km.csv").stat().st_mtime).isoformat(),
        },
        "rate_normalized_projection": projection_row,
        "validation": {
            "sample_count": count,
            "sample_period_s": 10.0,
            "dcm_clock_formula_max_error_ps": float(dcm_formula_error),
            "dcm_histogram_count": len(dcm_hist_rows),
            "dcm_total_count": summarize(dcm_counts),
            "software_input_total_count": summarize(software_total_count.ravel()),
            "software_input_aligned_average_gaussian_fwhm_ps": 2.354820045 * float(software_input_average_sigma),
            "software_direct_aligned_average_gaussian_fwhm_ps": 2.354820045 * float(software_output_average_sigma),
        },
        "sources": {
            "dcm_root": str(dcm_root),
            "dcm_histogram_directory": str(dcm_hist_dir),
            "dcm_histogram_direction": "ch1-ch3",
            "dcm_clock_csv": str(dcm_root / "0km.csv"),
            "software_root": str(software_root),
        },
    }
    write_json(output_dir / "comparison_summary.json", comparison)

    markdown = f"""# 50 km software compensation versus hardware DCM

> Hardware FWHM uses the operator-provided 1000-histogram ch1-ch3 run. Hardware TDEV uses the validated `0km.csv` series. They represent the same hardware condition but are not sample-wise paired.

## Software operator

The final method uses 512 Richardson-Lucy iterations, 160 background bins per edge, and independent broad/target PSFs for the two directions. Its output center is the raw Gaussian coarse center plus the background-subtracted local center of mass within +/-180 bins. No bounded center correction is applied. The legacy V17 constants 0.67, 1.2, and 0.095 are not used by this operator and belong only to the legacy baseline or Supplement.

## Main measured result

| Condition | Rate | TDEV at 10 s | Median FWHM | Stability gain vs raw | Width reduction vs raw |
|---|---:|---:|---:|---:|---:|
| 50 km input | 280 Hz | {main_rows[0]['tdev_10s_ps']:.3f} ps | {main_rows[0]['fwhm_median_ps']:.1f} ps | 1.00x | 1.00x |
| Direct software output | 280 Hz | {main_rows[1]['tdev_10s_ps']:.3f} ps | {main_rows[1]['fwhm_median_ps']:.1f} ps | {main_rows[1]['stability_improvement_vs_raw']:.2f}x | {main_rows[1]['width_reduction_vs_raw']:.2f}x |
| Hardware DCM | 100 Hz | {main_rows[2]['tdev_10s_ps']:.3f} ps | {main_rows[2]['fwhm_median_ps']:.1f} ps | {main_rows[2]['stability_improvement_vs_raw']:.2f}x | {main_rows[2]['width_reduction_vs_raw']:.2f}x |

The hardware width is measured from the specified ch1-ch3 histogram run. Under a pure shot-noise scaling assumption, the hardware TDEV projects from {main_rows[2]['tdev_10s_ps']:.3f} ps at 100 Hz to {projected_tdev:.3f} ps at 280 Hz. This projection is not a measured data point.

## Paper interpretation

1. Direct software compensation reduces median width from {main_rows[0]['fwhm_median_ps']:.1f} ps to {main_rows[1]['fwhm_median_ps']:.1f} ps and TDEV from {main_rows[0]['tdev_10s_ps']:.3f} ps to {main_rows[1]['tdev_10s_ps']:.3f} ps.
2. Hardware DCM gives a TDEV of {main_rows[2]['tdev_10s_ps']:.3f} ps at 100 Hz and a ch1-ch3 median FWHM of {main_rows[2]['fwhm_median_ps']:.1f} ps.
3. Hardware FWHM and TDEV are condition-level measurements and must not be described as sample-wise paired values.
4. The two measurements were acquired on different dates and at different event rates. The main table keeps measured values separate; the 280 Hz hardware value is only a rate-normalized projection.
5. The median accumulated count is {np.median(dcm_counts):.0f} for the hardware histograms and {np.median(software_total_count):.0f} for the software-input histograms. Therefore, nominal-rate scaling is not a count-matched experimental comparison.
"""
    (output_dir / "PAPER_COMPARISON_README.md").write_text(markdown, encoding="utf-8")
    markdown_cn = f"""# 50 km 软件补偿与硬件色散补偿模块结果对比

> 硬件 FWHM 使用操作者指定的1000张 ch1-ch3 完整横轴直方图；硬件 TDEV 使用已验证的 `0km.csv`。二者对应相同硬件条件，但不是逐样本配对数据。

## 软件方法复现口径

- 单张直方图算子：Richardson-Lucy 反卷积后与物理 0 km 目标 PSF 卷积，最终 `R = 512`。
- 背景：左右边缘各160 bin的合并中位数；两个传播方向分别保存独立的 broad/target PSF。
- 输出中心：原始 Gaussian 粗中心加补偿后局部直方图在 +/-180 bin 内的背景扣除质心；不存在输出后的 bounded center correction。
- `eta=0.67`、`blend_scale=1.2`、`clip_fraction=0.095` 仅属于旧 V17 Fisher-score 中心修正，不进入最终 RL 算子；若保留则放入 legacy baseline 或 Supplement。

## 论文主表建议

| 条件 | 标称速率 | TDEV@10 s | 中位 FWHM | 相对补偿前稳定性提升 | 相对补偿前宽度压缩 |
|---|---:|---:|---:|---:|---:|
| 50 km 补偿前 | 280 Hz | {main_rows[0]['tdev_10s_ps']:.3f} ps | {main_rows[0]['fwhm_median_ps']:.1f} ps | 1.00 倍 | 1.00 倍 |
| 单直方图软件直接输出 | 280 Hz | {main_rows[1]['tdev_10s_ps']:.3f} ps | {main_rows[1]['fwhm_median_ps']:.1f} ps | {main_rows[1]['stability_improvement_vs_raw']:.2f} 倍 | {main_rows[1]['width_reduction_vs_raw']:.2f} 倍 |
| 硬件色散补偿模块 | 100 Hz | {main_rows[2]['tdev_10s_ps']:.3f} ps | {main_rows[2]['fwhm_median_ps']:.1f} ps | {main_rows[2]['stability_improvement_vs_raw']:.2f} 倍 | {main_rows[2]['width_reduction_vs_raw']:.2f} 倍 |

## 可直接用于正文的结论

1. 软件方法将 FWHM 从 {main_rows[0]['fwhm_median_ps']:.1f} ps 压缩至 {main_rows[1]['fwhm_median_ps']:.1f} ps，压缩 {main_rows[1]['width_reduction_vs_raw']:.2f} 倍；同时将 TDEV@10 s 从 {main_rows[0]['tdev_10s_ps']:.3f} ps 降低至 {main_rows[1]['tdev_10s_ps']:.3f} ps，稳定性提升 {main_rows[1]['stability_improvement_vs_raw']:.2f} 倍。
2. 硬件色散补偿模块在 100 Hz 下的 TDEV@10 s 为 {main_rows[2]['tdev_10s_ps']:.3f} ps；指定 ch1-ch3 直方图的中位 FWHM 为 {main_rows[2]['fwhm_median_ps']:.1f} ps，宽度压缩 {main_rows[2]['width_reduction_vs_raw']:.2f} 倍。
3. 硬件 FWHM 与 TDEV 作为相同实验条件下的条件级结果进入比较，正文中不应表述成1000组逐样本一一配对。
4. 若仅假设探测计数随标称速率线性增加、其他噪声不变，则硬件结果从 100 Hz 投影到 280 Hz 为 {projected_tdev:.3f} ps。该值不是实测结果，只应在图中使用虚线或空心点。
5. 两批实验采集日期和标称速率不同，论文主结论应以三组实测值为主，速率归一化投影只能作为讨论项。
6. 硬件模块直方图中位累计计数为 {np.median(dcm_counts):.0f}，软件外部输入为 {np.median(software_total_count):.0f}。因此，按 100/280 Hz 得到的 1.273 ps 不是同计数条件下的严格比较，不能替代 2.130 ps 实测结果。
"""
    (output_dir / "PAPER_COMPARISON_CN.md").write_text(markdown_cn, encoding="utf-8")
    print(comparison, flush=True)


if __name__ == "__main__":
    main()
