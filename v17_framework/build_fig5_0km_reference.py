from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from direct_histogram_compensator import fwhm_subbin, gaussian_coarse_fit, normalize_probability


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_BANK = THIS_DIR / "artifacts" / "template_bank_0km_20260305_full" / "template_bank.npz"
DEFAULT_TEMPLATE_MANIFEST = THIS_DIR / "artifacts" / "template_bank_0km_20260305_full" / "template_manifest.json"
DEFAULT_TARGET_PSF = THIS_DIR / "artifacts" / "physical_0km_target_162ps" / "target_psf.npz"
DEFAULT_OUTPUT_DIR = THIS_DIR / "results" / "fig5_0km_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Fig. 5-style aligned 0 km reference package.")
    parser.add_argument("--template-bank", type=Path, default=DEFAULT_TEMPLATE_BANK)
    parser.add_argument("--template-manifest", type=Path, default=DEFAULT_TEMPLATE_MANIFEST)
    parser.add_argument("--target-psf", type=Path, default=DEFAULT_TARGET_PSF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-half-width-ps", type=int, default=800)
    return parser.parse_args()


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    length = {np.asarray(values).size for values in columns.values()}
    if len(length) != 1:
        raise ValueError("All output columns must have equal length")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for index in range(next(iter(length))):
            writer.writerow({name: float(np.asarray(values)[index]) for name, values in columns.items()})


def select_raw_template_indices(manifest: dict[str, Any]) -> dict[int, int]:
    selected: dict[int, int] = {}
    for direction in (1, 2):
        candidates = [
            row
            for row in manifest["templates"]
            if int(row["direction"]) == direction and float(row["smooth_sigma"]) == 0.0
        ]
        row = min(candidates, key=lambda item: (float(item["background_frac"]), int(item["index"])))
        selected[direction] = int(row["index"])
    return selected


def peak_normalized(values: np.ndarray) -> np.ndarray:
    clean = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    return clean / max(float(np.max(clean)), 1e-12)


def width_metrics(values: np.ndarray) -> dict[str, float]:
    probability = normalize_probability(values)
    _, sigma = gaussian_coarse_fit(probability, smooth_sigma=1.2, fit_half_width=600)
    return {
        "halfmax_fwhm_ps": float(fwhm_subbin(probability)),
        "gaussian_fwhm_ps": float(2.354820045 * sigma),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.template_manifest).read_text(encoding="utf-8"))
    indices = select_raw_template_indices(manifest)
    measured_x = np.arange(-int(manifest["half_width"]), int(manifest["half_width"]) + 1, dtype=np.float64)
    with np.load(args.template_bank, allow_pickle=False) as bank:
        measured_d1 = normalize_probability(bank[f"prob_{indices[1]:04d}"])
        measured_d2 = normalize_probability(bank[f"prob_{indices[2]:04d}"])

    with np.load(args.target_psf, allow_pickle=False) as target:
        target_x = np.asarray(target["relative_time_ps"], dtype=np.float64)
        target_d1_local = normalize_probability(target["direction1_probability"])
        target_d2_local = normalize_probability(target["direction2_probability"])
    target_d1 = normalize_probability(np.interp(measured_x, target_x, target_d1_local, left=0.0, right=0.0))
    target_d2 = normalize_probability(np.interp(measured_x, target_x, target_d2_local, left=0.0, right=0.0))

    measured_average = normalize_probability(0.5 * (measured_d1 + measured_d2))
    target_average = normalize_probability(0.5 * (target_d1 + target_d2))
    probabilities = {
        "zero_km_measured_direction1_probability": measured_d1,
        "zero_km_measured_direction2_probability": measured_d2,
        "zero_km_measured_two_direction_average_probability": measured_average,
        "zero_km_physical_target_direction1_probability": target_d1,
        "zero_km_physical_target_direction2_probability": target_d2,
        "zero_km_physical_target_two_direction_average_probability": target_average,
    }
    full_columns = {"relative_time_ps": measured_x, **probabilities}
    full_columns.update({name.replace("_probability", "_peak_normalized"): peak_normalized(values) for name, values in probabilities.items()})
    write_csv(output_dir / "fig5_0km_source_data_complete.csv", full_columns)

    plot_x = np.arange(-int(args.plot_half_width_ps), int(args.plot_half_width_ps) + 1, dtype=np.float64)
    plot_columns = {"relative_time_ps": plot_x}
    for name, values in probabilities.items():
        plot_columns[name.replace("_probability", "_peak_normalized")] = peak_normalized(
            np.interp(plot_x, measured_x, values, left=0.0, right=0.0)
        )
    write_csv(output_dir / "fig5_0km_plot_data.csv", plot_columns)

    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(plot_x, plot_columns["zero_km_measured_direction1_peak_normalized"], color="#7C3AED", linewidth=1.0, alpha=0.55, label="0 km measured, direction 1")
    axis.plot(plot_x, plot_columns["zero_km_measured_direction2_peak_normalized"], color="#2563EB", linewidth=1.0, alpha=0.55, label="0 km measured, direction 2")
    axis.plot(plot_x, plot_columns["zero_km_measured_two_direction_average_peak_normalized"], color="#111827", linewidth=1.8, label="0 km measured, aligned average")
    axis.plot(plot_x, plot_columns["zero_km_physical_target_two_direction_average_peak_normalized"], color="#D97706", linewidth=1.8, linestyle="--", label="Corrected physical 0 km target")
    axis.set_xlim(-450.0, 450.0)
    axis.set_xlabel("Relative time (ps)")
    axis.set_ylabel("Peak-normalized counts")
    axis.grid(True, color="#D1D5DB", linewidth=0.5, alpha=0.65)
    axis.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_0km_reference.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "fig5_0km_reference.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source_dataset": manifest["source_root"],
        "source_histogram_count_per_direction": 8640,
        "source_template_indices": [indices[1], indices[2]],
        "measured_direction1": width_metrics(measured_d1),
        "measured_direction2": width_metrics(measured_d2),
        "measured_two_direction_aligned_average": width_metrics(measured_average),
        "corrected_target_direction1": width_metrics(target_d1),
        "corrected_target_direction2": width_metrics(target_d2),
        "corrected_target_two_direction_average": width_metrics(target_average),
        "note": "Measured 0 km aligned average and corrected physical target are reported as distinct curves.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# Fig.5 风格 0 km 参考峰形

- 实测来源：`{manifest['source_root']}`，每个方向8640张直方图逐峰对齐后的平均模板。
- 实测方向1 FWHM（半高宽法）：{summary['measured_direction1']['halfmax_fwhm_ps']:.2f} ps。
- 实测方向2 FWHM（半高宽法）：{summary['measured_direction2']['halfmax_fwhm_ps']:.2f} ps。
- 实测双方向对齐平均 FWHM（半高宽法）：{summary['measured_two_direction_aligned_average']['halfmax_fwhm_ps']:.2f} ps。
- 物理校正目标双方向平均 FWHM（半高宽法）：{summary['corrected_target_two_direction_average']['halfmax_fwhm_ps']:.2f} ps。

`fig5_0km_source_data_complete.csv` 是 -1400 ps 到 +1400 ps 的完整概率及峰值归一化源数据；`fig5_0km_plot_data.csv` 是 -800 ps 到 +800 ps 的作图数据。实测0 km曲线与物理校正目标曲线分列保存，不能把后者表述成独立实测结果。
"""
    (output_dir / "RESULT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
