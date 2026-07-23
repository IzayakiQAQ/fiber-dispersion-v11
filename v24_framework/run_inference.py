from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .public_compensated_histogram_operator import DEFAULT_MODEL, V24Compensator
except ImportError:
    from public_compensated_histogram_operator import DEFAULT_MODEL, V24Compensator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compensate one local or full fixed-axis histogram with v24."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--direction", type=int, required=True, choices=(1, 2))
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def read_histogram(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.reader(handle):
            if not raw:
                continue
            try:
                if len(raw) == 1:
                    rows.append((float(len(rows)), float(raw[0])))
                else:
                    rows.append((float(raw[0]), float(raw[1])))
            except ValueError:
                if rows:
                    raise
    if not rows:
        raise ValueError(f"No numeric histogram rows found in {path}")
    values = np.asarray(rows, dtype=np.float64)
    return values[:, 0], np.clip(values[:, 1], 0.0, None)


def write_histogram(
    path: Path,
    axis: np.ndarray,
    raw: np.ndarray,
    compensated: np.ndarray,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_ps", "raw_local_count", "compensated_count"))
        writer.writerows(zip(axis, raw, compensated))


def main() -> None:
    args = parse_args()
    input_axis, counts = read_histogram(args.input_csv)
    operator = V24Compensator(args.model)

    if counts.size == operator.local_length:
        spacing = np.diff(input_axis)
        if spacing.size and not np.allclose(spacing, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("v24 requires a uniform 1 ps histogram axis")
        raw_local = counts
        output_axis = input_axis
        coarse_center = float(input_axis[counts.size // 2])
        compensated = operator.infer_local(raw_local, args.direction)
        final_center = operator.compensated_center(compensated, coarse_center)
    else:
        prepared = operator.prepare_full(counts, input_axis)
        raw_local = prepared.counts
        output_axis = prepared.absolute_time_ps
        coarse_center = prepared.coarse_center_abs_ps
        compensated = operator.infer_local(raw_local, args.direction)
        final_center = operator.compensated_center(compensated, coarse_center)

    output_csv = args.output_csv or args.input_csv.with_name(
        f"{args.input_csv.stem}_v24_compensated.csv"
    )
    summary_json = args.summary_json or output_csv.with_suffix(".json")
    write_histogram(output_csv, output_axis, raw_local, compensated)
    summary = {
        "version": "v24",
        "direction": int(args.direction),
        "input_csv": str(args.input_csv),
        "model": str(Path(args.model)),
        "input_bin_count": int(counts.size),
        "output_bin_count": int(compensated.size),
        "coarse_center_abs_ps": coarse_center,
        "compensated_center_abs_ps": final_center,
        "input_local_total_count": float(np.sum(raw_local)),
        "output_total_count": float(np.sum(compensated)),
        "iterations": operator.iterations,
        "edge_bins_per_side": operator.edge_bins,
        "center_half_window_bins": operator.center_half_window,
    }
    Path(summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
