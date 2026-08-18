from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from .compensator import V25Compensator
from .config import (
    FrozenConfig,
    OperatorSettings,
    PhysicsParameters,
    file_sha256,
    save_frozen_config,
)


def load_independent_parameters(path: str | Path) -> PhysicsParameters:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    values = payload.get("physics_parameters", payload.get("parameters", payload))
    if not isinstance(values, dict):
        raise ValueError("Calibration JSON must contain an object of physics parameters")
    return PhysicsParameters.from_mapping(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze an independently calibrated V25 physics configuration."
    )
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--neural-model", required=True)
    parser.add_argument("--length-km", type=float, required=True)
    parser.add_argument("--bandwidth-nm", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--kernel-bins", type=int, default=2049)
    parser.add_argument("--synthesis-bins", type=int, default=8193)
    parser.add_argument("--center-half-window", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calibration_path = Path(args.calibration_json).resolve()
    neural_source = Path(args.neural_model).resolve()
    parameters = load_independent_parameters(calibration_path)
    settings = replace(
        OperatorSettings(),
        rl_iterations=args.iterations,
        kernel_bins=args.kernel_bins,
        synthesis_bins=args.synthesis_bins,
        center_half_window_bins=args.center_half_window,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    neural_target = output_path.parent / "neural_psf_model.npz"
    if neural_source != neural_target:
        shutil.copyfile(neural_source, neural_target)
    config = FrozenConfig(
        length_km=args.length_km,
        bandwidth_nm=args.bandwidth_nm,
        calibration_sha256=file_sha256(calibration_path),
        neural_psf_model=neural_target.name,
        neural_psf_sha256=file_sha256(neural_target),
        physics=parameters,
        operator=settings,
    )
    from .neural_psf import NeuralPSFModel

    compensator = V25Compensator(config, NeuralPSFModel.load(neural_target))
    output, sidecar = save_frozen_config(config, output_path)
    print(
        json.dumps(
            {
                "config": str(output.resolve()),
                "hash_sidecar": str(sidecar.resolve()),
                "config_sha256": config.sha256(),
                "calibration_sha256": config.calibration_sha256,
                "neural_psf_sha256": config.neural_psf_sha256,
                "kernel_summary": compensator.kernel_summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
