from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import FrozenConfig, PhysicsParameters


SPEED_OF_LIGHT_NM_PER_PS = 299_792.458
GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493


def _normalize(values: np.ndarray) -> np.ndarray:
    clean = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(np.sum(clean))
    if total <= 1e-300:
        raise ValueError("A physical response has zero probability mass")
    return clean / total


def fwhm_ps(values: np.ndarray, bin_width_ps: float = 1.0) -> float:
    y = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if y.ndim != 1 or y.size < 3:
        raise ValueError("values must be a one-dimensional response")
    peak = int(np.argmax(y))
    half = 0.5 * float(y[peak])
    if half <= 0.0:
        return 0.0
    left_candidates = np.flatnonzero(y[:peak] < half)
    right_candidates = np.flatnonzero(y[peak + 1 :] < half)
    left = float(left_candidates[-1]) if left_candidates.size else 0.0
    right = (
        float(peak + 1 + right_candidates[0])
        if right_candidates.size
        else float(y.size - 1)
    )
    return max(right - left, 0.0) * float(bin_width_ps)


def translation_fisher(probability: np.ndarray, bin_width_ps: float) -> float:
    p = _normalize(probability)
    derivative = np.gradient(p, float(bin_width_ps))
    return float(np.sum(np.square(derivative) / np.clip(p, 1e-15, None)))


class PhysicsHistogramGenerator:
    """CW-SPDC, Gaussian WSS, SMF dispersion and timing-IRF model."""

    def __init__(self, parameters: PhysicsParameters) -> None:
        self.parameters = parameters

    @staticmethod
    def wavelength_nm(frequency_thz: float) -> float:
        return SPEED_OF_LIGHT_NM_PER_PS / float(frequency_thz)

    def travelling_wavelength_nm(self, direction: int) -> float:
        if direction == 1:
            return self.wavelength_nm(self.parameters.channel_c57_frequency_thz)
        if direction == 2:
            return self.wavelength_nm(self.parameters.channel_c35_frequency_thz)
        raise ValueError("direction must be 1 or 2")

    def dispersion_ps_nm_km(self, wavelength_nm: float) -> float:
        p = self.parameters
        return float(
            p.dispersion_ps_nm_km_at_1550
            + p.dispersion_slope_ps_nm2_km * (float(wavelength_nm) - 1550.0)
        )

    def beta2_ps2_km(self, wavelength_nm: float) -> float:
        wavelength = float(wavelength_nm)
        return -(wavelength**2) * self.dispersion_ps_nm_km(wavelength) / (
            2.0 * np.pi * SPEED_OF_LIGHT_NM_PER_PS
        )

    @staticmethod
    def _gaussian_intensity(offset_nm: np.ndarray, fwhm_nm: float) -> np.ndarray:
        width = max(float(fwhm_nm), 1e-12)
        return np.exp(-4.0 * np.log(2.0) * np.square(offset_nm / width))

    def joint_spectral_amplitude(
        self, detuning_thz: np.ndarray, filter_bandwidth_nm: float
    ) -> np.ndarray:
        p = self.parameters
        detuning = np.asarray(detuning_thz, dtype=np.float64)
        lambda_c57 = SPEED_OF_LIGHT_NM_PER_PS / (
            p.channel_c57_frequency_thz + detuning
        )
        lambda_c35 = SPEED_OF_LIGHT_NM_PER_PS / (
            p.channel_c35_frequency_thz - detuning
        )
        center_c57 = self.wavelength_nm(p.channel_c57_frequency_thz)
        center_c35 = self.wavelength_nm(p.channel_c35_frequency_thz)
        offset_c57 = lambda_c57 - center_c57
        offset_c35 = lambda_c35 - center_c35
        bandwidth = max(
            float(filter_bandwidth_nm) * p.filter_bandwidth_scale, 1e-6
        )
        source = self._gaussian_intensity(offset_c57, p.source_fwhm_nm)
        filter_c57 = self._gaussian_intensity(offset_c57, bandwidth)
        filter_c35 = self._gaussian_intensity(offset_c35, bandwidth)
        return np.sqrt(source * filter_c57 * filter_c35)

    def probability(
        self,
        length_km: float,
        bandwidth_nm: float,
        direction: int,
        n_bins: int,
        bin_width_ps: float,
    ) -> np.ndarray:
        if n_bins < 33 or n_bins % 2 != 1:
            raise ValueError("n_bins must be an odd integer of at least 33")
        frequency_thz = np.fft.fftfreq(n_bins, d=float(bin_width_ps))
        omega_rad_ps = 2.0 * np.pi * frequency_thz
        amplitude = self.joint_spectral_amplitude(frequency_thz, bandwidth_nm)
        wavelength_nm = self.travelling_wavelength_nm(direction)
        phase = 0.5 * self.beta2_ps2_km(wavelength_nm) * float(
            length_km
        ) * np.square(omega_rad_ps)
        phase += (
            self.parameters.beta3_ps3_km
            * float(length_km)
            / 6.0
            * np.power(omega_rad_ps, 3)
        )
        temporal_amplitude = np.fft.fftshift(
            np.fft.ifft(amplitude * np.exp(1j * phase))
        )
        probability = np.square(np.abs(temporal_amplitude))
        sigma_bins = (
            self.parameters.irf_fwhm_ps(direction)
            / GAUSSIAN_FWHM_TO_SIGMA
            / float(bin_width_ps)
        )
        probability = gaussian_filter1d(probability, sigma_bins, mode="constant")
        return _normalize(probability)


@dataclass(frozen=True)
class DirectionKernels:
    broad: np.ndarray
    target: np.ndarray
    broad_fwhm_ps: float
    target_fwhm_ps: float
    fisher_gain: float
    cropped_edge_mass: float


def build_direction_kernels(config: FrozenConfig, direction: int) -> DirectionKernels:
    """Build deterministic broad and 0 km kernels from a frozen config."""

    config.validate()
    generator = PhysicsHistogramGenerator(config.physics)
    settings = config.operator
    broad_full = generator.probability(
        config.length_km,
        config.bandwidth_nm,
        direction,
        settings.synthesis_bins,
        settings.bin_width_ps,
    )
    target_full = generator.probability(
        0.0,
        config.bandwidth_nm,
        direction,
        settings.synthesis_bins,
        settings.bin_width_ps,
    )
    half = settings.kernel_bins // 2
    center = settings.synthesis_bins // 2
    selection = slice(center - half, center + half + 1)
    broad = _normalize(broad_full[selection])
    target = _normalize(target_full[selection])
    edge = min(settings.edge_bins, settings.kernel_bins // 4)
    cropped_edge_mass = float(
        np.sum(broad[:edge]) + np.sum(broad[-edge:])
    )
    broad_fisher = translation_fisher(broad, settings.bin_width_ps)
    target_fisher = translation_fisher(target, settings.bin_width_ps)
    fisher_gain = target_fisher / max(broad_fisher, 1e-15)
    if cropped_edge_mass > 0.01:
        raise ValueError(
            "The broad response exceeds the local kernel window; increase kernel_bins"
        )
    if fisher_gain <= 1.0:
        raise ValueError("The frozen target does not improve translation Fisher information")
    return DirectionKernels(
        broad=broad,
        target=target,
        broad_fwhm_ps=fwhm_ps(broad, settings.bin_width_ps),
        target_fwhm_ps=fwhm_ps(target, settings.bin_width_ps),
        fisher_gain=fisher_gain,
        cropped_edge_mass=cropped_edge_mass,
    )
