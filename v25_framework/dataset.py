from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


_LENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*km", re.IGNORECASE)
_BANDWIDTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*nm", re.IGNORECASE)
_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


@dataclass(frozen=True)
class ConditionSeries:
    name: str
    root: Path
    length_km: float
    bandwidth_nm: float
    layout: str
    direction_paths: dict[int, tuple[Path, ...]]


def _histogram_files(folder: Path) -> tuple[Path, ...]:
    def index(path: Path) -> int:
        match = _INDEX_RE.search(path.name)
        return int(match.group(1)) if match else -1

    return tuple(sorted(folder.glob("hist_raw_*.csv"), key=index))


def _direction_from_name(name: str) -> int | None:
    lower = name.lower()
    if any(token in lower for token in ("pair0", "ch1_ch2", "ch3_ch1")):
        return 1
    if any(token in lower for token in ("pair1", "ch3_ch4", "ch4_ch2")):
        return 2
    return None


def _directions(root: Path) -> tuple[dict[int, tuple[Path, ...]], str]:
    flat = _histogram_files(root)
    if flat:
        if len(flat) % 2:
            raise ValueError(f"Flat condition must contain two equal directions: {root}")
        half = len(flat) // 2
        return {1: flat[:half], 2: flat[half:]}, "sequential_flat"
    candidates = [
        (child, _histogram_files(child))
        for child in sorted(root.iterdir(), key=lambda path: path.name)
        if child.is_dir() and _histogram_files(child)
    ]
    if len(candidates) != 2:
        raise ValueError(f"Expected two histogram directions under {root}")
    assigned: dict[int, tuple[Path, ...]] = {}
    for child, files in candidates:
        direction = _direction_from_name(child.name)
        if direction is not None:
            assigned[direction] = files
    unassigned = [item for item in candidates if _direction_from_name(item[0].name) is None]
    for direction, (_, files) in zip(
        (value for value in (1, 2) if value not in assigned), unassigned
    ):
        assigned[direction] = files
    if set(assigned) != {1, 2}:
        raise ValueError(f"Could not assign directions under {root}")
    layout = (
        "pair_subdirectories"
        if all("pair" in child.name.lower() for child, _ in candidates)
        else "channel_subdirectories"
    )
    return assigned, layout


def discover_conditions(root: str | Path) -> list[ConditionSeries]:
    dataset_root = Path(root)
    result: list[ConditionSeries] = []
    for folder in sorted(dataset_root.iterdir(), key=lambda path: path.name):
        if not folder.is_dir():
            continue
        length_match = _LENGTH_RE.search(folder.name)
        bandwidth_match = _BANDWIDTH_RE.search(folder.name)
        if length_match is None or bandwidth_match is None:
            continue
        direction_paths, layout = _directions(folder)
        result.append(
            ConditionSeries(
                name=folder.name,
                root=folder,
                length_km=float(length_match.group(1)),
                bandwidth_nm=float(bandwidth_match.group(1)),
                layout=layout,
                direction_paths=direction_paths,
            )
        )
    if not result:
        raise ValueError(f"No conditions found under {dataset_root}")
    return result


def load_histogram(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(Path(path), delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected two-column histogram: {path}")
    axis = np.asarray(data[:, 0], dtype=np.float64)
    counts = np.clip(np.nan_to_num(data[:, 1], nan=0.0), 0.0, None)
    if not np.allclose(np.diff(axis), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"Histogram axis must use 1 ps bins: {path}")
    return axis, counts


def gaussian_width_ps(path: str | Path, expected_fwhm_ps: float) -> float:
    """Measure one histogram with a background-aware, rebinned Gaussian fit."""

    axis, counts = load_histogram(path)
    edge = min(512, max(counts.size // 8, 1))
    background = float(np.mean(np.r_[counts[:edge], counts[-edge:]]))
    window = int(np.clip(max(1200.0, 1.5 * expected_fwhm_ps), 1200, 12000))
    cumulative = np.cumsum(np.r_[0.0, counts])
    rolling = cumulative[window:] - cumulative[:-window]
    left = int(np.argmax(rolling))
    search = slice(max(0, left - window), min(counts.size, left + 2 * window))
    signal = np.clip(counts[search] - background, 0.0, None)
    search_axis = axis[search]
    mass = float(np.sum(signal))
    center0 = (
        float(np.sum(search_axis * signal) / mass)
        if mass > 0.0
        else float(axis[left + window // 2])
    )
    factor = 5
    usable = counts.size // factor * factor
    x = axis[:usable].reshape(-1, factor).mean(axis=1)
    y = counts[:usable].reshape(-1, factor).sum(axis=1)
    keep = np.abs(x - center0) <= min(max(1200.0, 1.75 * expected_fwhm_ps), 8192)
    x = x[keep]
    y = y[keep]

    def model(
        xx: np.ndarray, amplitude: float, center: float, sigma: float, floor: float
    ) -> np.ndarray:
        return floor + amplitude * np.exp(-0.5 * np.square((xx - center) / sigma))

    sigma0 = np.clip(max(expected_fwhm_ps, 162.0) / 2.354820045, 20.0, 4000.0)
    try:
        fitted, _ = curve_fit(
            model,
            x,
            y,
            p0=(max(float(np.max(y)) - background * factor, 1.0), center0, sigma0, background * factor),
            bounds=(
                (0.0, center0 - window, 10.0, 0.0),
                (max(float(np.max(y)) * 10.0, 1.0), center0 + window, 8000.0, max(float(np.max(y)), 1.0)),
            ),
            sigma=np.sqrt(np.clip(y, 0.0, None) + 1.0),
            maxfev=10000,
        )
        return float(2.354820045 * fitted[2])
    except (RuntimeError, ValueError, FloatingPointError):
        return float("nan")


def sampled_paths(paths: tuple[Path, ...], count: int) -> tuple[Path, ...]:
    take = min(max(int(count), 1), len(paths))
    indices = np.unique(np.linspace(0, len(paths) - 1, take, dtype=np.int64))
    return tuple(paths[int(index)] for index in indices)
