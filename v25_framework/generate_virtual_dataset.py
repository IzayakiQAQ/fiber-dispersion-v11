from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import PhysicsParameters
from .neural_psf import NeuralPSFModel
from .physics import PhysicsHistogramGenerator, fwhm_ps


def _floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _shift(probability: np.ndarray, shift_bins: float) -> np.ndarray:
    axis = np.arange(probability.size, dtype=np.float64)
    shifted = np.interp(
        axis - float(shift_bins), axis, probability, left=0.0, right=0.0
    )
    return shifted / max(float(np.sum(shifted)), 1e-15)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate virtual coincidence histograms from a frozen Neural-PSF."
    )
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--neural-model", type=Path, required=True)
    parser.add_argument("--lengths-km", default="0,25,50,75,100,125")
    parser.add_argument("--bandwidths-nm", default="0.2,0.4,0.8,2,5,8,10,20")
    parser.add_argument("--count-rates-hz", default="50,100,280")
    parser.add_argument("--integration-s", type=float, default=10.0)
    parser.add_argument("--histograms-per-condition", type=int, default=32)
    parser.add_argument("--bins", type=int, default=8193)
    parser.add_argument("--bin-width-ps", type=float, default=1.0)
    parser.add_argument("--background-per-bin", type=float, default=0.05)
    parser.add_argument("--center-jitter-ps", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=2502)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bins < 257 or args.bins % 2 != 1:
        raise ValueError("bins must be an odd integer of at least 257")
    summary = json.loads(args.training_summary.read_text(encoding="utf-8-sig"))
    physics = PhysicsParameters.from_mapping(summary["physics_parameters"])
    generator = PhysicsHistogramGenerator(physics)
    network = NeuralPSFModel.load(args.neural_model)
    rng = np.random.default_rng(args.seed)
    lengths = _floats(args.lengths_km)
    bandwidths = _floats(args.bandwidths_nm)
    rates = _floats(args.count_rates_hz)
    histograms: list[np.ndarray] = []
    labels: list[list[float]] = []
    condition_rows: list[dict[str, float]] = []

    for length in lengths:
        for bandwidth in bandwidths:
            for direction in (1, 2):
                physical = generator.probability(
                    length,
                    bandwidth,
                    direction,
                    args.bins,
                    args.bin_width_ps,
                )
                probability, prediction = network.correct(
                    physical, length, bandwidth, direction
                )
                edge_mass = float(np.sum(probability[:32]) + np.sum(probability[-32:]))
                if edge_mass > 0.01:
                    raise ValueError(
                        f"Virtual PSF L={length:g}, B={bandwidth:g}, d={direction} "
                        "does not fit the selected bin window"
                    )
                condition_rows.append(
                    {
                        "length_km": length,
                        "bandwidth_nm": bandwidth,
                        "direction": float(direction),
                        "psf_fwhm_ps": fwhm_ps(probability, args.bin_width_ps),
                        "neural_width_scale": prediction.width_scale,
                        "neural_confidence": prediction.confidence,
                    }
                )
                for rate in rates:
                    signal_counts = max(rate * args.integration_s, 0.0)
                    for _ in range(max(args.histograms_per_condition, 1)):
                        center_ps = rng.uniform(
                            -abs(args.center_jitter_ps), abs(args.center_jitter_ps)
                        )
                        shifted = _shift(
                            probability, center_ps / args.bin_width_ps
                        )
                        expectation = (
                            signal_counts * shifted + args.background_per_bin
                        )
                        histograms.append(rng.poisson(expectation).astype(np.float32))
                        labels.append(
                            [length, bandwidth, rate, float(direction), center_ps]
                        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        histograms=np.stack(histograms),
        labels=np.asarray(labels, dtype=np.float64),
        label_columns=np.asarray(
            [
                "length_km",
                "bandwidth_nm",
                "count_rate_hz",
                "direction",
                "true_center_ps",
            ]
        ),
        bin_width_ps=np.asarray(args.bin_width_ps),
        seed=np.asarray(args.seed),
    )
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "v25-neural-psf-virtual-dataset-v1",
                "output": str(output.resolve()),
                "histogram_count": len(histograms),
                "bins": args.bins,
                "integration_s": args.integration_s,
                "background_per_bin": args.background_per_bin,
                "conditions": condition_rows,
                "evaluation_histograms_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output.resolve())
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
