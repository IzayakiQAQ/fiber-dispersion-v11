from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from likelihood_reader import TemplateBank, V17ReaderConfig, read_one
from v17_common import (
    SAMPLE_PERIOD_S,
    pair_dir_records,
    read_hist,
    sample_local,
    summarize,
    tdev_at_m,
    tdev_curve,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V17 self-calibrating likelihood reader on external 50km/280Hz data.")
    parser.add_argument("--source-root", type=Path, default=Path("E:/lzy") / "\u6d4b\u8bd5\u7ed3\u679c" / "2026.6.29 50km 280Hz")
    parser.add_argument("--template-manifest", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "template_bank_v1" / "template_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "v17_self_calibrated_20260629_50km_280hz")
    parser.add_argument("--max-histograms", type=int, default=0)
    parser.add_argument("--target-template-fraction", type=float, default=0.67)
    parser.add_argument("--width-penalty", type=float, default=4.0)
    parser.add_argument("--blend-scale", type=float, default=1.2)
    parser.add_argument("--clip-fraction", type=float, default=0.095)
    parser.add_argument("--background-prior", type=float, default=0.001)
    return parser.parse_args()


def load_external_rows(source_root: Path, bank: TemplateBank, args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    records = pair_dir_records(source_root)[:2]
    by_pair: dict[str, list[dict[str, Any]]] = {}
    max_hist = int(args.max_histograms)
    for record in records:
        files = list(record["hist_files"])
        quality_rows = list(record["quality_rows"])
        if max_hist > 0:
            files = files[:max_hist]
            quality_rows = quality_rows[:max_hist]
        rows: list[dict[str, Any]] = []
        for idx, (hist_path, quality) in enumerate(zip(files, quality_rows), start=1):
            first_x, counts = read_hist(hist_path)
            center_abs = float(quality["center_hist_ps"])
            center_idx = center_abs - float(first_x)
            local = sample_local(counts, center_idx, bank.half_width)
            sigma = float(quality.get("sigma_ps", np.nan))
            rows.append(
                {
                    "pair": str(record["pair"]),
                    "direction": int(record["direction"]),
                    "index": int(idx),
                    "time_s": float(idx) * SAMPLE_PERIOD_S,
                    "hist_path": str(hist_path),
                    "first_x_ps": float(first_x),
                    "quality_center_abs_ps": float(center_abs),
                    "quality_center_idx": float(center_idx),
                    "input_sigma_ps": sigma,
                    "input_fwhm_gaussian_ps": 2.354820045 * sigma if np.isfinite(sigma) else np.nan,
                    "total_count": float(np.sum(counts)),
                    "local_count": float(np.sum(local)),
                    "local": local,
                }
            )
        by_pair[str(record["pair"])] = rows
    return by_pair


def apply_reader(rows_by_pair: dict[str, list[dict[str, Any]]], bank: TemplateBank, config: V17ReaderConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for rows in rows_by_pair.values():
        for row in rows:
            input_fwhm = float(row["input_fwhm_gaussian_ps"])
            result = read_one(np.asarray(row["local"], dtype=np.float64), int(row["direction"]), input_fwhm, bank, config)
            row["v17_center_abs_ps"] = float(row["quality_center_abs_ps"]) + float(result["v17_delta_ps"])
            row.update(result)
            details.append({key: value for key, value in row.items() if key != "local"})
    pair_names = list(rows_by_pair)
    p1 = rows_by_pair[pair_names[0]]
    p2 = rows_by_pair[pair_names[1]]
    n = min(len(p1), len(p2))
    raw_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    for i in range(n):
        raw1 = float(p1[i]["quality_center_abs_ps"])
        raw2 = float(p2[i]["quality_center_abs_ps"])
        c1 = float(p1[i]["v17_center_abs_ps"])
        c2 = float(p2[i]["v17_center_abs_ps"])
        raw_rows.append({"time": (i + 1) * SAMPLE_PERIOD_S, "t1": raw1, "t2": raw2, "t0": 0.5 * (raw1 - raw2)})
        comp_rows.append({"time": (i + 1) * SAMPLE_PERIOD_S, "t1": c1, "t2": c2, "t0": 0.5 * (c1 - c2)})
    return raw_rows, comp_rows, details


def save_example(output_dir: Path, rows_by_pair: dict[str, list[dict[str, Any]]], bank: TemplateBank) -> None:
    pair_names = list(rows_by_pair)
    if len(pair_names) < 2 or not rows_by_pair[pair_names[0]] or not rows_by_pair[pair_names[1]]:
        return
    x = np.arange(-bank.half_width, bank.half_width + 1, dtype=np.float64)
    plt.figure(figsize=(12, 5))
    data_cols = [x]
    header = ["relative_time_ps"]
    for pair_name in pair_names[:2]:
        row = rows_by_pair[pair_name][0]
        local = np.asarray(row["local"], dtype=np.float64)
        selected = [
            item
            for item in bank.templates
            if int(item["direction"]) == int(row["direction"])
            and str(item["label"]) == str(row["v17_selected_label"])
            and abs(float(item["smooth_sigma"]) - float(row["v17_selected_smooth"])) < 1e-9
            and abs(float(item["background_frac"]) - float(row["v17_selected_background"])) < 1e-12
        ]
        template_scaled = np.zeros_like(local)
        if selected:
            prob, _ = bank.prob_grad(selected[0])
            template_scaled = prob * (max(float(np.max(local)), 1e-12) / max(float(np.max(prob)), 1e-12))
        plt.plot(x, local, label=f"{pair_name} raw local", alpha=0.55)
        plt.plot(x, template_scaled, label=f"{pair_name} selected template", alpha=0.9)
        data_cols.extend([local, template_scaled])
        header.extend([f"{pair_name}_raw_local", f"{pair_name}_selected_template"])
    plt.xlim(-900, 900)
    plt.xlabel("relative time ps")
    plt.ylabel("counts / scaled probability")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "processed_example_histogram_00001.png", dpi=160)
    plt.close()
    np.savetxt(
        output_dir / "processed_example_histogram_00001.csv",
        np.column_stack(data_cols),
        fmt="%.6f",
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = TemplateBank(Path(args.template_manifest))
    config = V17ReaderConfig(
        target_template_fraction=float(args.target_template_fraction),
        width_penalty=float(args.width_penalty),
        blend_scale=float(args.blend_scale),
        clip_fraction=float(args.clip_fraction),
        background_prior=float(args.background_prior),
    )
    rows_by_pair = load_external_rows(Path(args.source_root), bank, args)
    raw_rows, comp_rows, details = apply_reader(rows_by_pair, bank, config)
    raw_t0 = [float(row["t0"]) for row in raw_rows]
    comp_t0 = [float(row["t0"]) for row in comp_rows]
    write_csv(output_dir / "raw_time_t1_t2_t0_quality.csv", raw_rows)
    write_csv(output_dir / "time_t1_t2_t0_four_columns.csv", comp_rows)
    write_csv(output_dir / "pair_detail.csv", details)
    save_example(output_dir, rows_by_pair, bank)
    summary = {
        "framework": "v17",
        "variant": "self_calibrated_effective_dispersion_likelihood_reader",
        "source": str(Path(args.source_root)),
        "template_manifest": str(Path(args.template_manifest)),
        "count": int(len(comp_rows)),
        "config": {
            "target_template_fraction": float(config.target_template_fraction),
            "width_penalty": float(config.width_penalty),
            "blend_scale": float(config.blend_scale),
            "clip_fraction": float(config.clip_fraction),
            "background_prior": float(config.background_prior),
        },
        "raw_tdev10_ps": tdev_at_m(raw_t0, 1),
        "v17_tdev10_ps": tdev_at_m(comp_t0, 1),
        "raw_tdev_curve_ps": tdev_curve(raw_t0),
        "v17_tdev_curve_ps": tdev_curve(comp_t0),
        "raw_clock_ps": summarize(raw_t0),
        "v17_clock_ps": summarize(comp_t0),
        "input_fwhm_gaussian_ps": summarize([float(row["input_fwhm_gaussian_ps"]) for row in details]),
        "v17_delta_ps": summarize([float(row["v17_delta_ps"]) for row in details]),
        "v17_raw_delta_ps": summarize([float(row["v17_raw_delta_ps"]) for row in details]),
        "v17_blend": summarize([float(row["v17_blend"]) for row in details]),
        "v17_clip_ps": summarize([float(row["v17_clip_ps"]) for row in details]),
        "v17_template_fwhm_ps": summarize([float(row["v17_template_fwhm_ps"]) for row in details]),
        "selected_labels": sorted({str(row["v17_selected_label"]) for row in details}),
        "selected_smooth_values": sorted({float(row["v17_selected_smooth"]) for row in details}),
        "selected_background_values": sorted({float(row["v17_selected_background"]) for row in details}),
        "outputs": {
            "raw_four_column_csv": str(output_dir / "raw_time_t1_t2_t0_quality.csv"),
            "four_column_csv": str(output_dir / "time_t1_t2_t0_four_columns.csv"),
            "detail_csv": str(output_dir / "pair_detail.csv"),
            "example_csv": str(output_dir / "processed_example_histogram_00001.csv"),
            "example_png": str(output_dir / "processed_example_histogram_00001.png"),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
