from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from direct_histogram_compensator import center_of_mass


EDGE_BINS_PER_SIDE = 160
CENTER_HALF_WINDOW_BINS = 180


DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "results"
    / "v24_direct_histogram_external_1000_physical_0km"
    / "direct_histogram_model.npz"
)


def compensated_histogram(
    raw_histogram: np.ndarray,
    broad_psf: np.ndarray,
    target_0km_psf: np.ndarray,
    iterations: int,
    edge_bins: int = EDGE_BINS_PER_SIDE,
) -> np.ndarray:
    """Return one compensated histogram; no neighboring samples are used."""

    observed = np.clip(np.asarray(raw_histogram, dtype=np.float64), 0.0, None)
    broad = np.clip(np.asarray(broad_psf, dtype=np.float64), 0.0, None)
    target = np.clip(np.asarray(target_0km_psf, dtype=np.float64), 0.0, None)
    broad /= max(float(np.sum(broad)), 1e-12)
    target /= max(float(np.sum(target)), 1e-12)

    total_count = float(np.sum(observed))
    edge = min(max(int(edge_bins), 1), max(observed.size // 4, 1))
    background = float(np.median(np.concatenate((observed[:edge], observed[-edge:]))))
    signal = np.clip(observed - background, 0.0, None)
    signal_count = float(np.sum(signal))
    measured = signal / max(signal_count, 1e-12)

    latent = np.clip(measured, 1e-8, None)
    latent /= max(float(np.sum(latent)), 1e-12)
    for _ in range(max(int(iterations), 0)):
        predicted = fftconvolve(latent, broad, mode="same")
        ratio = np.clip(measured / np.clip(predicted, 1e-12, None), 0.0, 8.0)
        latent *= fftconvolve(ratio, broad[::-1], mode="same")
        latent = np.clip(latent, 1e-8, None)
        latent /= max(float(np.sum(latent)), 1e-12)

    compensated = np.clip(fftconvolve(latent, target, mode="same"), 0.0, None)
    compensated /= max(float(np.sum(compensated)), 1e-12)
    compensated = compensated * signal_count + background
    compensated *= total_count / max(float(np.sum(compensated)), 1e-12)
    return np.clip(compensated, 0.0, None)


def compensated_center_ps(
    compensated_local_histogram: np.ndarray,
    coarse_center_abs_ps: float,
    center_half_window_bins: int = CENTER_HALF_WINDOW_BINS,
) -> float:
    """Read the final absolute center exactly as used in the 1000-group evaluation."""

    local = np.asarray(compensated_local_histogram, dtype=np.float64)
    relative_center = center_of_mass(local, center_half_window_bins) - float(local.size // 2)
    return float(coarse_center_abs_ps) + relative_center


def infer_with_saved_model(
    raw_histogram: np.ndarray,
    direction: int,
    model_path: Path = DEFAULT_MODEL,
) -> np.ndarray:
    """Load the final v24 operator parameters and compensate one local histogram."""

    if direction not in (1, 2):
        raise ValueError("direction must be 1 or 2")
    with np.load(model_path, allow_pickle=False) as model:
        broad = np.asarray(model[f"broad_psf_direction{direction}"], dtype=np.float64)
        target = np.asarray(model[f"target_psf_direction{direction}"], dtype=np.float64)
        iterations = int(model["iterations"])
        edge_bins = int(model["edge_bins_per_side"]) if "edge_bins_per_side" in model else EDGE_BINS_PER_SIDE
    if np.asarray(raw_histogram).size != broad.size:
        raise ValueError(f"Expected {broad.size} bins, got {np.asarray(raw_histogram).size}")
    return compensated_histogram(raw_histogram, broad, target, iterations, edge_bins=edge_bins)
