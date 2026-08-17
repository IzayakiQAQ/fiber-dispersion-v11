from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .compensator import V25Compensator


def load_histogram_csv(path: str | Path) -> tuple[np.ndarray | None, np.ndarray]:
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            try:
                numeric = [float(value.strip()) for value in row[:2]]
            except ValueError:
                continue
            rows.append(numeric)
    if not rows:
        raise ValueError(f"No numeric histogram rows found in {path}")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("CSV rows must consistently contain one or two columns")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] == 1:
        return None, data[:, 0]
    return data[:, 0], data[:, 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen V25 compensation on one histogram CSV."
    )
    parser.add_argument("input_csv")
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--direction", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--allow-unhashed-config",
        action="store_true",
        help="Development only: skip the frozen-config SHA-256 sidecar check.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    axis, counts = load_histogram_csv(args.input_csv)
    compensator = V25Compensator.from_frozen_json(
        args.frozen_config, require_hash=not args.allow_unhashed_config
    )
    result = compensator.infer_full(counts, args.direction, time_ps=axis)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_ps", "raw_local_count", "compensated_count"))
        writer.writerows(
            zip(result.time_ps, result.raw_local, result.compensated, strict=True)
        )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "direction": result.direction,
                "center_ps": result.center_ps,
                "coarse_center_ps": result.coarse_center_ps,
                "input_counts": result.input_counts,
                "output_counts": result.output_counts,
                "input_fwhm_ps": result.input_fwhm_ps,
                "output_fwhm_ps": result.output_fwhm_ps,
                "expected_fisher_gain": result.expected_fisher_gain,
                "config_sha256": result.config_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path.resolve())
    print(metadata_path.resolve())


if __name__ == "__main__":
    main()
