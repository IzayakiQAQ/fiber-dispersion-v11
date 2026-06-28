from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configs. Install with: pip install pyyaml") from exc
    data = yaml.safe_load(text)
    return dict(data or {})


@dataclass(frozen=True)
class PairSplit:
    train_pairs: tuple[int, ...]
    val_pairs: tuple[int, ...]
    test_pairs: tuple[int, ...]
    policy: str = "blocked_nonoverlap_v1"

    def as_manifest(self) -> dict[str, object]:
        return {
            "split_policy": self.policy,
            "train_pairs": list(self.train_pairs),
            "val_pairs": list(self.val_pairs),
            "test_pairs": list(self.test_pairs),
            "train_count": len(self.train_pairs),
            "val_count": len(self.val_pairs),
            "test_count": len(self.test_pairs),
        }


def expand_windows(windows: Iterable[tuple[int, int]]) -> tuple[int, ...]:
    pairs: list[int] = []
    for start, end in windows:
        if int(start) < 1 or int(end) < int(start):
            raise ValueError(f"Invalid pair window: {(start, end)!r}")
        pairs.extend(range(int(start), int(end) + 1))
    return tuple(sorted(set(pairs)))


def default_blocked_nonoverlap_split(num_pairs: int = 8640) -> PairSplit:
    total = int(num_pairs)
    if total < 3:
        raise ValueError("num_pairs must be at least 3")
    train_windows: list[tuple[int, int]] = []
    val_windows: list[tuple[int, int]] = []
    test_windows: list[tuple[int, int]] = []
    num_blocks = max(1, total // 1024)
    for block_idx in range(num_blocks):
        block_start = 1 + block_idx * 1024
        block_end = total if block_idx == num_blocks - 1 else min(block_start + 1023, total)
        train_end = min(block_start + 511, block_end)
        val_start = train_end + 1
        val_end = min(block_start + 767, block_end)
        test_start = val_end + 1
        if block_start <= train_end:
            train_windows.append((block_start, train_end))
        if val_start <= val_end:
            val_windows.append((val_start, val_end))
        if test_start <= block_end:
            test_windows.append((test_start, block_end))
    return PairSplit(
        train_pairs=expand_windows(train_windows),
        val_pairs=expand_windows(val_windows),
        test_pairs=expand_windows(test_windows),
    )


def split_from_config(config: Mapping[str, Any]) -> PairSplit:
    data = dict(config.get("data", {})) if isinstance(config.get("data", {}), Mapping) else {}
    policy = str(data.get("split_policy", "blocked_nonoverlap_v1"))
    if policy != "blocked_nonoverlap_v1":
        raise ValueError(f"Unsupported split policy: {policy!r}")
    return default_blocked_nonoverlap_split(num_pairs=int(config.get("num_pairs", 8640)))


def pair_selection_summary(indices: list[int]) -> dict[str, Any]:
    pairs = sorted(int(idx) + 1 for idx in indices)
    if not pairs:
        return {"count": 0, "windows": []}
    windows: list[list[int]] = []
    start = prev = pairs[0]
    for pair in pairs[1:]:
        if pair == prev + 1:
            prev = pair
            continue
        windows.append([start, prev])
        start = prev = pair
    windows.append([start, prev])
    return {"count": len(pairs), "first": pairs[0], "last": pairs[-1], "windows": windows}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def safe_norm_batch(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    peak = torch.amax(x.reshape(x.shape[0], -1), dim=1).reshape(-1, 1)
    return torch.where(peak > 1e-8, x / (peak + 1e-8), torch.zeros_like(x))


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def choose_target(
    dataset: dict[str, Any],
    targets_by_bandwidth: dict[float, dict[str, Any]],
    fixed_target: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    if policy == "bandwidth_if_available":
        return targets_by_bandwidth.get(round(float(dataset["bandwidth_nm"]), 6), fixed_target)
    return fixed_target


def scenarios_from_manifest(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_cfg = dict(config.get("data", {}))
    manifest_path = args.manifest or Path(str(data_cfg.get("scenario_manifest", "")))
    if not str(manifest_path).strip():
        raise ValueError("Set --manifest or data.scenario_manifest")
    manifest = load_manifest(Path(manifest_path))
    datasets = list(manifest.get("datasets", []))
    targets = [dataset for dataset in datasets if str(dataset.get("role")) == "target"]
    inputs = [dataset for dataset in datasets if str(dataset.get("role")) != "target"]
    if not targets:
        raise ValueError("No target datasets found in manifest")

    target_label = args.target_label or str(data_cfg.get("target_label", "d000km_bw0p8nm"))
    targets_by_label = {str(dataset["label"]): dataset for dataset in targets}
    fixed_target = targets_by_label.get(target_label)
    if fixed_target is None:
        raise ValueError(f"Target label {target_label!r} not found. Available targets: {sorted(targets_by_label)}")
    targets_by_bandwidth = {round(float(dataset["bandwidth_nm"]), 6): dataset for dataset in targets}
    target_policy = args.target_policy or str(data_cfg.get("target_policy", "fixed"))
    include = set(split_csv(args.include_labels or data_cfg.get("include_labels", "")))
    exclude = set(split_csv(args.exclude_labels or data_cfg.get("exclude_labels", "")))

    scenarios: list[dict[str, Any]] = []
    for dataset in inputs:
        label = str(dataset["label"])
        if include and label not in include:
            continue
        if label in exclude:
            continue
        target = choose_target(dataset, targets_by_bandwidth, fixed_target, target_policy)
        input_dir = Path(str(dataset["tensor_dir"]))
        target_dir = Path(str(target["tensor_dir"]))
        if not input_dir.is_dir() or not target_dir.is_dir():
            continue
        usable_pairs = min(int(dataset.get("converted_pairs", 0)), int(target.get("converted_pairs", 0)))
        if usable_pairs <= 0:
            continue
        scenarios.append(
            {
                "label": label,
                "input_dir": str(input_dir),
                "input_shift_csv": str(dataset.get("shift_csv", "")),
                "target_label": str(target["label"]),
                "target_dir": str(target_dir),
                "target_shift_csv": str(target.get("shift_csv", "")),
                "distance_km": float(dataset["distance_km"]),
                "bandwidth_nm": float(dataset["bandwidth_nm"]),
                "usable_pairs": int(usable_pairs),
            }
        )
    if not scenarios:
        raise ValueError("No usable scenarios selected")
    return scenarios, manifest


DEFAULT_M_VALUES = (1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300)


def _clean_1d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def tdev_at_tau0(values: Sequence[float] | np.ndarray) -> float:
    return tdev_at_m(values, 1)


def tdev_at_m(values: Sequence[float] | np.ndarray, m: int) -> float:
    arr = _clean_1d(values)
    m = int(m)
    if m < 1 or arr.size < 2 * m + 1:
        return 0.0
    d2 = arr[2 * m :] - 2.0 * arr[m:-m] + arr[: -2 * m]
    return float(np.sqrt(np.mean(d2 * d2) / (6.0 * float(m * m))))


def tdev_curve(
    values: Sequence[float] | np.ndarray,
    sample_period_s: float = 10.0,
    m_values: Iterable[int] | None = None,
) -> dict[str, float]:
    arr = _clean_1d(values)
    curve: dict[str, float] = {}
    for m in DEFAULT_M_VALUES if m_values is None else m_values:
        m_int = int(m)
        if m_int >= 1 and arr.size >= 2 * m_int + 1:
            tau = int(round(float(sample_period_s) * m_int))
            curve[str(tau)] = tdev_at_m(arr, m_int)
    return curve


@dataclass
class PeakMetrics:
    peak_index: float
    argmax_index: int
    baseline: float
    peak_value: float
    amplitude: float
    area: float
    fwhm_samples: float
    equivalent_width_samples: float
    concentration_50: float
    concentration_200: float
    concentration_500: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_array(y: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def estimate_baseline(y: np.ndarray, quantile: float = 0.1) -> float:
    arr = _safe_array(y)
    return float(np.quantile(arr, float(np.clip(quantile, 0.0, 1.0))))


def fit_peak_quadratic(y: np.ndarray) -> float:
    arr = _safe_array(y)
    k = int(np.argmax(arr))
    if k <= 0 or k >= arr.size - 1:
        return float(k)
    y0, y1, y2 = arr[k - 1], arr[k], arr[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    offset = 0.5 * (y0 - y2) / denom
    return float(k + float(np.clip(offset, -1.0, 1.0)))


def fit_peak_window_centroid(y: np.ndarray, half_width: int = 50, seed_center: float | None = None) -> float:
    arr = _safe_array(y)
    center = fit_peak_quadratic(arr) if seed_center is None else float(seed_center)
    half_width = max(int(half_width), 1)
    left = max(int(math.floor(center - half_width)), 0)
    right = min(int(math.ceil(center + half_width)) + 1, arr.size)
    local = np.clip(arr[left:right], a_min=0.0, a_max=None)
    mass = float(np.sum(local))
    if mass <= 1e-12:
        return float(np.argmax(arr))
    x = np.arange(left, right, dtype=np.float64)
    return float(np.sum(x * local) / mass)


def fwhm_samples(y: np.ndarray, baseline: float | None = None) -> float:
    arr = _safe_array(y)
    base = estimate_baseline(arr) if baseline is None else float(baseline)
    peak = float(np.max(arr))
    amp = max(peak - base, 0.0)
    if amp <= 1e-12:
        return 0.0
    level = base + 0.5 * amp
    idx = np.flatnonzero(arr >= level)
    if idx.size == 0:
        return 0.0
    return float(idx[-1] - idx[0])


def concentration_ratio(y: np.ndarray, half_width: int, center: float | None = None) -> float:
    arr = _safe_array(y)
    c = fit_peak_quadratic(arr) if center is None else float(center)
    left = max(int(math.floor(c - half_width)), 0)
    right = min(int(math.ceil(c + half_width)) + 1, arr.size)
    total = float(np.sum(np.clip(arr, a_min=0.0, a_max=None)))
    if total <= 1e-12:
        return 0.0
    local = float(np.sum(np.clip(arr[left:right], a_min=0.0, a_max=None)))
    return local / total


def compute_peak_metrics(y: np.ndarray | list[float]) -> PeakMetrics:
    arr = _safe_array(y)
    peak_idx = fit_peak_quadratic(arr)
    argmax_idx = int(np.argmax(arr))
    baseline = estimate_baseline(arr)
    peak_value = float(np.max(arr))
    amplitude = max(peak_value - baseline, 0.0)
    area = float(np.sum(np.clip(arr - baseline, a_min=0.0, a_max=None)))
    eq_width = area / amplitude if amplitude > 1e-12 else 0.0
    return PeakMetrics(
        peak_index=float(peak_idx),
        argmax_index=argmax_idx,
        baseline=float(baseline),
        peak_value=peak_value,
        amplitude=float(amplitude),
        area=area,
        fwhm_samples=fwhm_samples(arr, baseline=baseline),
        equivalent_width_samples=float(eq_width),
        concentration_50=concentration_ratio(arr, 50, center=peak_idx),
        concentration_200=concentration_ratio(arr, 200, center=peak_idx),
        concentration_500=concentration_ratio(arr, 500, center=peak_idx),
    )
