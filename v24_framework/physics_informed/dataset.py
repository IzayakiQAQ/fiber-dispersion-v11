from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


_LENGTH_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*km", re.IGNORECASE)
_BANDWIDTH_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*nm", re.IGNORECASE)
_RATE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*hz", re.IGNORECASE)
_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


@dataclass(frozen=True)
class HistogramRecord:
    condition_name: str
    length_km: float
    bandwidth_nm: float
    count_rate_hz: float
    direction: int
    index: int
    path: Path


@dataclass(frozen=True)
class ConditionSeries:
    name: str
    root: Path
    length_km: float
    bandwidth_nm: float
    count_rate_hz: float
    layout: str
    direction_paths: dict[int, tuple[Path, ...]]

    @property
    def pair_count(self) -> int:
        return min(len(paths) for paths in self.direction_paths.values())

    def records(self, direction: int) -> tuple[HistogramRecord, ...]:
        value = int(direction)
        if value not in self.direction_paths:
            raise ValueError(f"Condition {self.name!r} has no direction {value}")
        return tuple(
            HistogramRecord(
                condition_name=self.name,
                length_km=self.length_km,
                bandwidth_nm=self.bandwidth_nm,
                count_rate_hz=self.count_rate_hz,
                direction=value,
                index=index,
                path=path,
            )
            for index, path in enumerate(self.direction_paths[value], start=1)
        )


@dataclass(frozen=True)
class CenteredProfile:
    relative_time_ps: np.ndarray
    counts: np.ndarray
    center_abs_ps: float
    gaussian_fwhm_ps: float
    background_per_bin: float
    total_counts: float


def _number(pattern: re.Pattern[str], text: str, label: str) -> float:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not parse {label} from condition folder {text!r}")
    return float(match.group("value"))


def _histogram_index(path: Path) -> int:
    match = _INDEX_RE.search(path.name)
    if match is None:
        raise ValueError(f"Histogram file has no numeric index: {path}")
    return int(match.group(1))


def _histogram_files(folder: Path) -> tuple[Path, ...]:
    paths = list(folder.glob("hist_raw_*.csv"))
    paths.sort(key=_histogram_index)
    return tuple(paths)


def _known_direction(folder_name: str) -> int | None:
    name = folder_name.lower()
    direction1 = ("pair0", "ch1_ch2", "ch3_ch1")
    direction2 = ("pair1", "ch3_ch4", "ch4_ch2")
    if any(token in name for token in direction1):
        return 1
    if any(token in name for token in direction2):
        return 2
    return None


def _discover_directions(
    condition_root: Path,
) -> tuple[dict[int, tuple[Path, ...]], str]:
    root_files = _histogram_files(condition_root)
    if root_files:
        if len(root_files) % 2:
            raise ValueError(
                f"Sequential two-direction folder must contain an even number of files: {condition_root}"
            )
        half = len(root_files) // 2
        return {1: root_files[:half], 2: root_files[half:]}, "sequential_flat"

    candidates: list[tuple[Path, tuple[Path, ...]]] = []
    for child in sorted((path for path in condition_root.iterdir() if path.is_dir()), key=lambda p: p.name):
        files = _histogram_files(child)
        if files:
            candidates.append((child, files))
    if len(candidates) != 2:
        raise ValueError(
            f"Expected exactly two histogram directions under {condition_root}, found {len(candidates)}"
        )

    assigned: dict[int, tuple[Path, ...]] = {}
    unknown: list[tuple[Path, tuple[Path, ...]]] = []
    for folder, files in candidates:
        direction = _known_direction(folder.name)
        if direction is None or direction in assigned:
            unknown.append((folder, files))
        else:
            assigned[direction] = files
    for direction, (_, files) in zip((value for value in (1, 2) if value not in assigned), unknown):
        assigned[direction] = files
    if set(assigned) != {1, 2}:
        raise ValueError(f"Could not assign two directions under {condition_root}")
    layout = (
        "pair_subdirectories"
        if all("pair" in folder.name.lower() for folder, _ in candidates)
        else "channel_subdirectories"
    )
    return assigned, layout


def discover_dataset(root: str | Path, default_count_rate_hz: float = 100.0) -> list[ConditionSeries]:
    """Discover every length/bandwidth condition in the compensation dataset."""

    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    conditions: list[ConditionSeries] = []
    for condition_root in sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda p: p.name):
        try:
            length_km = _number(_LENGTH_RE, condition_root.name, "fiber length")
            bandwidth_nm = _number(_BANDWIDTH_RE, condition_root.name, "filter bandwidth")
        except ValueError:
            continue
        rate_match = _RATE_RE.search(condition_root.name)
        count_rate_hz = (
            float(rate_match.group("value")) if rate_match is not None else float(default_count_rate_hz)
        )
        direction_paths, layout = _discover_directions(condition_root)
        if len(direction_paths[1]) != len(direction_paths[2]):
            raise ValueError(
                f"Direction counts differ for {condition_root.name}: "
                f"{len(direction_paths[1])} and {len(direction_paths[2])}"
            )
        conditions.append(
            ConditionSeries(
                name=condition_root.name,
                root=condition_root,
                length_km=length_km,
                bandwidth_nm=bandwidth_nm,
                count_rate_hz=count_rate_hz,
                layout=layout,
                direction_paths=direction_paths,
            )
        )
    if not conditions:
        raise ValueError(f"No length/bandwidth conditions found under {dataset_root}")
    return conditions


def load_histogram_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(Path(path), delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a two-column histogram CSV: {path}")
    axis = np.asarray(data[:, 0], dtype=np.float64)
    counts = np.clip(np.nan_to_num(data[:, 1], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    if axis.size < 3:
        raise ValueError(f"Histogram is too short: {path}")
    spacing = np.diff(axis)
    if not np.allclose(spacing, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"Histogram must have a uniform 1 ps axis: {path}")
    return axis, counts


def centered_profile(
    path: str | Path,
    half_width_bins: int = 8192,
    background_edge_bins: int = 512,
    expected_dispersion_fwhm_ps: float = 0.0,
    fit_rebin_bins: int = 5,
) -> CenteredProfile:
    """Read and align one histogram with an integrated-window Gaussian fit."""

    axis, counts = load_histogram_csv(path)
    edge = min(max(int(background_edge_bins), 1), max(counts.size // 8, 1))
    # At 100 Hz the per-bin accidental level is often below one count, so the
    # edge median collapses to zero. The edge mean is the Poisson MLE here.
    background = float(np.mean(np.concatenate((counts[:edge], counts[-edge:]))))
    expected_width = max(float(expected_dispersion_fwhm_ps), 0.0)
    localization_width = int(
        np.clip(max(1200.0, 1.25 * expected_width), 1200.0, 12000.0)
    )
    localization_width = min(localization_width, counts.size - 1)
    cumulative = np.cumsum(np.r_[0.0, counts])
    rolling = cumulative[localization_width:] - cumulative[:-localization_width]
    best_left = int(np.argmax(rolling))
    search_left = max(0, best_left - localization_width)
    search_right = min(counts.size, best_left + 2 * localization_width)
    search_axis = axis[search_left:search_right]
    search_signal = np.clip(counts[search_left:search_right] - background, 0.0, None)
    search_mass = float(np.sum(search_signal))
    coarse_center = (
        float(np.sum(search_axis * search_signal) / search_mass)
        if search_mass > 0.0
        else float(axis[best_left + localization_width // 2])
    )

    factor = max(int(fit_rebin_bins), 1)
    usable = (counts.size // factor) * factor
    fit_axis = axis[:usable].reshape(-1, factor).mean(axis=1)
    fit_counts = counts[:usable].reshape(-1, factor).sum(axis=1)
    fit_half_width = min(
        max(1200.0, 1.75 * expected_width), float(half_width_bins)
    )
    keep = np.abs(fit_axis - coarse_center) <= fit_half_width
    x = fit_axis[keep]
    y = fit_counts[keep]

    def gaussian_floor(
        xx: np.ndarray, amplitude: float, center: float, sigma: float, floor: float
    ) -> np.ndarray:
        return floor + amplitude * np.exp(-0.5 * np.square((xx - center) / sigma))

    rebinned_background = background * factor
    amplitude0 = max(float(np.max(y)) - rebinned_background, 1.0)
    total_width0 = np.sqrt(162.0**2 + expected_width**2)
    sigma0 = np.clip(total_width0 / 2.354820045, 20.0, fit_half_width / 2.0)
    fitted_center = coarse_center
    fitted_sigma = float(sigma0)
    try:
        parameters, _ = curve_fit(
            gaussian_floor,
            x,
            y,
            p0=(amplitude0, coarse_center, sigma0, max(rebinned_background, 0.0)),
            bounds=(
                (0.0, coarse_center - localization_width, 10.0, 0.0),
                (
                    max(float(np.max(y)) * 10.0, 1.0),
                    coarse_center + localization_width,
                    max(fit_half_width, 20.0),
                    max(float(np.max(y)), 1.0),
                ),
            ),
            sigma=np.sqrt(np.clip(y, 0.0, None) + 1.0),
            absolute_sigma=False,
            maxfev=10000,
        )
        fitted_center = float(parameters[1])
        fitted_sigma = float(parameters[2])
    except (RuntimeError, ValueError, FloatingPointError):
        pass

    half_width = int(half_width_bins)
    relative = np.arange(-half_width, half_width + 1, dtype=np.float64)
    aligned = np.interp(fitted_center + relative, axis, counts, left=background, right=background)
    return CenteredProfile(
        relative_time_ps=relative,
        counts=aligned,
        center_abs_ps=fitted_center,
        gaussian_fwhm_ps=2.354820045 * fitted_sigma,
        background_per_bin=background,
        total_counts=float(np.sum(counts)),
    )


def evenly_spaced_records(records: tuple[HistogramRecord, ...], count: int) -> tuple[HistogramRecord, ...]:
    if not records:
        return ()
    take = min(max(int(count), 1), len(records))
    indices = np.linspace(0, len(records) - 1, take, dtype=np.int64)
    return tuple(records[int(index)] for index in np.unique(indices))


def thin_histogram_counts(
    counts: np.ndarray,
    keep_fraction: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Independently retain detected events for a controlled count-rate test."""

    fraction = float(keep_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("keep_fraction must be between 0 and 1")
    integer_counts = np.rint(
        np.clip(np.asarray(counts, dtype=np.float64), 0.0, None)
    ).astype(np.int64)
    generator = rng or np.random.default_rng()
    return generator.binomial(integer_counts, fraction).astype(np.float64)
