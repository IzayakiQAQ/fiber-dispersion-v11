from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v17_common import fwhm_np, normalize_prob, read_json, write_json


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a corrected physical 0 km target PSF.")
    parser.add_argument(
        "--template-manifest",
        type=Path,
        default=THIS_DIR / "artifacts" / "template_bank_0km_20260305_full" / "template_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR / "artifacts" / "physical_0km_target_162ps",
    )
    parser.add_argument("--direction1-fwhm-ps", type=float, default=165.81005)
    parser.add_argument("--direction2-fwhm-ps", type=float, default=159.88495)
    parser.add_argument("--half-width-ps", type=int, default=512)
    return parser.parse_args()


def raw_template_row(manifest: dict[str, Any], direction: int) -> dict[str, Any]:
    candidates = [
        row
        for row in manifest["templates"]
        if int(row["direction"]) == int(direction) and float(row["smooth_sigma"]) == 0.0
    ]
    if not candidates:
        raise ValueError(f"No unsmoothed template for direction {direction}")
    return min(candidates, key=lambda row: float(row["background_frac"]))


def scale_width(probability: np.ndarray, target_fwhm: float) -> np.ndarray:
    source = normalize_prob(np.asarray(probability, dtype=np.float64), 0.0)
    source_fwhm = float(fwhm_np(source))
    if source_fwhm <= 0.0:
        raise ValueError("Source template has zero FWHM")
    half_width = source.size // 2
    x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    source_x = x * source_fwhm / float(target_fwhm)
    scaled = np.interp(source_x, x, source, left=0.0, right=0.0)
    return normalize_prob(scaled, 0.0)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.template_manifest)
    manifest = read_json(manifest_path)
    bank_path = Path(manifest["template_bank_npz"])
    if not bank_path.is_absolute():
        bank_path = manifest_path.parent / bank_path.name

    rows = [raw_template_row(manifest, direction) for direction in (1, 2)]
    with np.load(bank_path, allow_pickle=False) as bank:
        source = [np.asarray(bank[f"prob_{int(row['index']):04d}"], dtype=np.float64) for row in rows]

    target_widths = [float(args.direction1_fwhm_ps), float(args.direction2_fwhm_ps)]
    full_targets = [scale_width(probability, width) for probability, width in zip(source, target_widths)]
    source_half_width = full_targets[0].size // 2
    half_width = min(max(int(args.half_width_ps), 1), source_half_width)
    targets = [
        normalize_prob(probability[source_half_width - half_width : source_half_width + half_width + 1], 0.0)
        for probability in full_targets
    ]
    relative_time = np.arange(-half_width, half_width + 1, dtype=np.float64)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "target_psf.npz"
    np.savez_compressed(
        output_npz,
        relative_time_ps=relative_time,
        direction1_probability=targets[0].astype(np.float32),
        direction2_probability=targets[1].astype(np.float32),
    )
    summary = {
        "variant": "corrected_physical_0km_target_psf",
        "invalid_target_excluded": "E:/lzy/测试结果/补偿数据/0.8nm 0km (about 52 ps FWHM)",
        "source_manifest": str(manifest_path),
        "source_dataset": str(manifest["source_root"]),
        "source_template_indices": [int(row["index"]) for row in rows],
        "source_fwhm_ps": [float(fwhm_np(probability)) for probability in source],
        "target_reference": "2025.5.14 1TDC 24h 0km 100Hz fwhm.csv medians",
        "direction1_fwhm_ps": float(fwhm_np(targets[0])),
        "direction2_fwhm_ps": float(fwhm_np(targets[1])),
        "mean_fwhm_ps": 0.5 * (float(fwhm_np(targets[0])) + float(fwhm_np(targets[1]))),
        "output_npz": str(output_npz),
    }
    write_json(output_dir / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
