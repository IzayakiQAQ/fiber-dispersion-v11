from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from v17_common import (
    DEFAULT_MANIFEST,
    NUM_PAIRS,
    TARGET_CENTER,
    fwhm_np,
    normalize_prob,
    read_json,
    resolve_manifest_path,
    sample_local,
    split_csv,
    split_floats,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V17 template bank from centered V10 tensors.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "template_bank_v1")
    parser.add_argument("--template-labels", type=str, default="d025km_bw0p8nm,d050km_bw0p8nm,d075km_bw0p8nm,d100km_bw0p8nm")
    parser.add_argument("--smooth-sigmas", type=str, default="20,40,60,80,100,120,160,200")
    parser.add_argument("--background-fracs", type=str, default="0.0003,0.0005,0.001,0.003,0.005,0.01")
    parser.add_argument("--half-width", type=int, default=1400)
    parser.add_argument("--max-pairs", type=int, default=1024)
    return parser.parse_args()


def load_tensor(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu").float().numpy().astype(np.float64)


def build_raw_template(tensor_dir: Path, direction: int, half_width: int, max_pairs: int) -> tuple[np.ndarray, int]:
    ids = range(1, int(max_pairs) + 1) if int(direction) == 1 else range(NUM_PAIRS + 1, NUM_PAIRS + int(max_pairs) + 1)
    acc = np.zeros(2 * int(half_width) + 1, dtype=np.float64)
    count = 0
    for hist_id in ids:
        path = tensor_dir / f"hist_{int(hist_id):05d}.pt"
        if not path.exists():
            continue
        acc += sample_local(load_tensor(path), TARGET_CENTER, int(half_width))
        count += 1
    if count <= 0:
        raise FileNotFoundError(f"No hist tensors found in {tensor_dir} for direction={direction}")
    return np.clip(acc / float(count), 0.0, None), count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(Path(args.manifest))
    by_label = {str(item["label"]): item for item in manifest.get("datasets", [])}
    labels = split_csv(str(args.template_labels))
    smooth_sigmas = split_floats(str(args.smooth_sigmas))
    backgrounds = split_floats(str(args.background_fracs))
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for label in labels:
        if label not in by_label:
            raise KeyError(f"Template label {label!r} not found in {args.manifest}")
        dataset = by_label[label]
        tensor_dir = resolve_manifest_path(str(dataset["tensor_dir"]))
        usable = min(int(dataset.get("converted_pairs", 0)), int(args.max_pairs))
        for direction in (1, 2):
            raw_template, count = build_raw_template(tensor_dir, direction, int(args.half_width), usable)
            raw_fwhm = fwhm_np(raw_template)
            for sigma in smooth_sigmas:
                smoothed = gaussian_filter1d(raw_template, sigma=float(sigma), mode="nearest") if float(sigma) > 0.0 else raw_template.copy()
                for bg in backgrounds:
                    prob = normalize_prob(smoothed, float(bg))
                    grad = np.gradient(prob)
                    idx = len(rows)
                    arrays[f"prob_{idx:04d}"] = prob.astype(np.float32)
                    arrays[f"grad_{idx:04d}"] = grad.astype(np.float32)
                    rows.append(
                        {
                            "index": idx,
                            "label": str(label),
                            "direction": int(direction),
                            "distance_km": float(dataset.get("distance_km", 0.0)),
                            "bandwidth_nm": float(dataset.get("bandwidth_nm", 0.0)),
                            "smooth_sigma": float(sigma),
                            "background_frac": float(bg),
                            "template_count": int(count),
                            "raw_template_fwhm_ps": float(raw_fwhm),
                            "template_fwhm_ps": float(fwhm_np(prob)),
                            "fisher_per_count": float(np.sum((grad * grad) / np.clip(prob, 1e-12, None))),
                        }
                    )
    np.savez_compressed(output_dir / "template_bank.npz", **arrays)
    bank_manifest = {
        "framework": "v17",
        "variant": "effective_dispersion_template_bank",
        "source_manifest": str(Path(args.manifest)),
        "template_labels": labels,
        "smooth_sigmas": smooth_sigmas,
        "background_fracs": backgrounds,
        "half_width": int(args.half_width),
        "max_pairs": int(args.max_pairs),
        "template_count": len(rows),
        "templates": rows,
        "template_bank_npz": str(output_dir / "template_bank.npz"),
    }
    write_json(output_dir / "template_manifest.json", bank_manifest)
    print(f"Wrote {len(rows)} templates to {output_dir}")


if __name__ == "__main__":
    main()
