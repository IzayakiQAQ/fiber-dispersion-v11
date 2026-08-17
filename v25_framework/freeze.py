from __future__ import annotations

import argparse
import json
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
    values = payload.get("parameters", payload)
    if not isinstance(values, dict):
        raise ValueError("Calibration JSON must contain an object of physics parameters")
    return PhysicsParameters.from_mapping(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze an independently calibrated V25 physics configuration."
    )
    parser.add_argument("--calibration-json", required=True)
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
    parameters = load_independent_parameters(calibration_path)
    settings = replace(
        OperatorSettings(),
        rl_iterations=args.iterations,
        kernel_bins=args.kernel_bins,
        synthesis_bins=args.synthesis_bins,
        center_half_window_bins=args.center_half_window,
    )
    config = FrozenConfig(
        length_km=args.length_km,
        bandwidth_nm=args.bandwidth_nm,
        calibration_sha256=file_sha256(calibration_path),
        physics=parameters,
        operator=settings,
    )
    compensator = V25Compensator(config)
    output, sidecar = save_frozen_config(config, args.output)
    print(
        json.dumps(
            {
                "config": str(output.resolve()),
                "hash_sidecar": str(sidecar.resolve()),
                "config_sha256": config.sha256(),
                "calibration_sha256": config.calibration_sha256,
                "kernel_summary": compensator.kernel_summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
