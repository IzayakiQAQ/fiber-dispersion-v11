from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_PERIOD_S = 10.0
TARGET_CENTER = 32768.0
INPUT_DIM = 65536
NUM_PAIRS = 8640

THIS_DIR = Path(__file__).resolve().parent
REPOSITORIES_ROOT = THIS_DIR.parents[1]
FRAMEWORK_ROOT = REPOSITORIES_ROOT / "fiber-dispersion-framework"
V11_DIR = THIS_DIR.parents[0] / "v11_framework"
DEFAULT_MANIFEST = FRAMEWORK_ROOT / "v10_framework" / "artifacts" / "new_compensation_data_tensor_centered_1024" / "manifest.json"


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def split_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_manifest_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return FRAMEWORK_ROOT / path


def summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {key: 0.0 for key in ("mean", "std", "min", "p05", "median", "p95", "max")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def tdev_at_m(values: list[float] | np.ndarray, m: int = 1) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    m = int(m)
    if m < 1 or arr.size < 2 * m + 1:
        return 0.0
    d2 = arr[2 * m :] - 2.0 * arr[m:-m] + arr[: -2 * m]
    return float(np.sqrt(np.mean(d2 * d2) / (6.0 * float(m * m))))


def tdev_curve(values: list[float] | np.ndarray, sample_period_s: float = SAMPLE_PERIOD_S) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in (1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size >= 2 * m + 1:
            out[str(int(round(float(sample_period_s) * m)))] = tdev_at_m(arr, m)
    return out


def sample_local(values: np.ndarray, center_idx: float, half_width: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.arange(-int(half_width), int(half_width) + 1, dtype=np.float64)
    src_x = float(center_idx) + grid
    base_x = np.arange(values.size, dtype=np.float64)
    return np.interp(src_x, base_x, values, left=0.0, right=0.0).astype(np.float64)


def normalize_prob(values: np.ndarray, background_frac: float = 0.0) -> np.ndarray:
    arr = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = float(np.sum(arr))
    if total <= 1e-12:
        arr = np.ones_like(arr, dtype=np.float64)
    else:
        arr = arr / total
    bg = float(np.clip(background_frac, 0.0, 0.5))
    if bg > 0.0:
        arr = (1.0 - bg) * arr + bg / float(arr.size)
    return arr / max(float(np.sum(arr)), 1e-12)


def fwhm_np(values: np.ndarray) -> float:
    arr = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if arr.size == 0:
        return 0.0
    peak = float(np.max(arr))
    if peak <= 1e-12:
        return 0.0
    idx = np.flatnonzero(arr >= 0.5 * peak)
    if idx.size == 0:
        return 0.0
    return float(idx[-1] - idx[0])


def read_hist(path: Path) -> tuple[float, np.ndarray]:
    data = np.loadtxt(str(path), delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    first_x = float(data[0, 0])
    counts = np.nan_to_num(data[:, 1].astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if counts.size != INPUT_DIM:
        out = np.zeros(INPUT_DIM, dtype=np.float32)
        if counts.size > INPUT_DIM:
            start = (counts.size - INPUT_DIM) // 2
            out[:] = counts[start : start + INPUT_DIM]
        else:
            start = (INPUT_DIM - counts.size) // 2
            out[start : start + counts.size] = counts
        counts = out
    return first_x, counts.astype(np.float64, copy=False)


def load_quality_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = np.nan
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def pair_dir_records(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for quality_path in source_root.rglob("singlepeak_peak_quality_gaussian.csv"):
        work_dir = quality_path.parent
        rows = load_quality_rows(quality_path)
        if not rows:
            continue
        pair = str(rows[0].get("pair", work_dir.name))
        hist_dirs = sorted([p for p in work_dir.iterdir() if p.is_dir() and "histograms_raw" in p.name])
        if not hist_dirs:
            continue
        hist_files = sorted(hist_dirs[0].glob("hist_raw_*.csv"))
        count = min(len(hist_files), len(rows))
        records.append(
            {
                "pair": pair,
                "direction": 1 if "ch3_ch1" in pair else 2 if "ch4_ch2" in pair else len(records) + 1,
                "work_dir": work_dir,
                "hist_dir": hist_dirs[0],
                "hist_files": hist_files[:count],
                "quality_rows": rows[:count],
            }
        )
    if len(records) < 2:
        raise ValueError(f"Expected at least two pair directories under {source_root}, found {len(records)}")
    return sorted(records, key=lambda item: int(item["direction"]))
