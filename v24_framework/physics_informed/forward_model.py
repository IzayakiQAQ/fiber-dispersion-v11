from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


SPEED_OF_LIGHT_NM_PER_PS = 299_792.458
GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493


def itu_wavelength_nm(channel_frequency_thz: float) -> float:
    return SPEED_OF_LIGHT_NM_PER_PS / float(channel_frequency_thz)


@dataclass(frozen=True)
class PhysicsParameters:
    """Effective source, link and timing-response parameters.

    The source envelope is a prior from the cascaded SHG/SPDC PPLN source. The
    experiment-specific WSS filters, fiber dispersion and combined timing IRF
    determine the generated coincidence profile.
    """

    pump_wavelength_nm: float = 1540.56
    channel_c57_wavelength_nm: float = itu_wavelength_nm(195.7)
    channel_c35_wavelength_nm: float = itu_wavelength_nm(193.5)
    source_fwhm_nm: float = 60.0
    filter_bandwidth_scale: float = 1.0
    dispersion_ps_nm_km_at_1550: float = 17.0
    dispersion_slope_ps_nm2_km: float = 0.058
    beta3_ps3_km: float = 0.0
    irf_fwhm_direction1_ps: float = 162.0
    irf_fwhm_direction2_ps: float = 162.0

    def irf_fwhm_ps(self, direction: int) -> float:
        if int(direction) == 1:
            return float(self.irf_fwhm_direction1_ps)
        if int(direction) == 2:
            return float(self.irf_fwhm_direction2_ps)
        raise ValueError("direction must be 1 or 2")

    def travelling_wavelength_nm(self, direction: int) -> float:
        if int(direction) == 1:
            return float(self.channel_c57_wavelength_nm)
        if int(direction) == 2:
            return float(self.channel_c35_wavelength_nm)
        raise ValueError("direction must be 1 or 2")


class PhysicsHistogramGenerator:
    """Generate coincidence histograms for the CW PPLN/WSS/SMF experiment."""

    def __init__(self, parameters: PhysicsParameters | None = None) -> None:
        self.parameters = parameters or PhysicsParameters()

    def with_parameters(self, **updates: float) -> "PhysicsHistogramGenerator":
        return PhysicsHistogramGenerator(replace(self.parameters, **updates))

    def dispersion_ps_nm_km(self, wavelength_nm: float) -> float:
        p = self.parameters
        return float(
            p.dispersion_ps_nm_km_at_1550
            + p.dispersion_slope_ps_nm2_km * (float(wavelength_nm) - 1550.0)
        )

    def beta2_ps2_km(self, wavelength_nm: float) -> float:
        wavelength = float(wavelength_nm)
        dispersion = self.dispersion_ps_nm_km(wavelength)
        return -(wavelength * wavelength) * dispersion / (
            2.0 * np.pi * SPEED_OF_LIGHT_NM_PER_PS
        )

    @staticmethod
    def _gaussian_intensity(offset: np.ndarray, fwhm: float) -> np.ndarray:
        width = max(float(fwhm), 1e-12)
        return np.exp(-4.0 * np.log(2.0) * np.square(offset / width))

    def joint_spectral_amplitude(
        self,
        detuning_thz: np.ndarray,
        filter_bandwidth_nm: float,
    ) -> np.ndarray:
        """Return the energy-anticorrelated one-dimensional JSA slice."""

        p = self.parameters
        detuning = np.asarray(detuning_thz, dtype=np.float64)
        nu_c57 = SPEED_OF_LIGHT_NM_PER_PS / p.channel_c57_wavelength_nm
        nu_c35 = SPEED_OF_LIGHT_NM_PER_PS / p.channel_c35_wavelength_nm
        lambda_c57 = SPEED_OF_LIGHT_NM_PER_PS / (nu_c57 + detuning)
        lambda_c35 = SPEED_OF_LIGHT_NM_PER_PS / (nu_c35 - detuning)
        offset_c57 = lambda_c57 - p.channel_c57_wavelength_nm
        offset_c35 = lambda_c35 - p.channel_c35_wavelength_nm

        bandwidth = max(float(filter_bandwidth_nm) * p.filter_bandwidth_scale, 1e-6)
        source_intensity = self._gaussian_intensity(offset_c57, p.source_fwhm_nm)
        filter_c57 = self._gaussian_intensity(offset_c57, bandwidth)
        filter_c35 = self._gaussian_intensity(offset_c35, bandwidth)
        # WSS bandwidths describe intensity FWHM. The biphoton amplitude sees
        # the square root of each channel transmission.
        return np.sqrt(source_intensity * filter_c57 * filter_c35)

    def probability(
        self,
        length_km: float,
        filter_bandwidth_nm: float,
        direction: int,
        n_bins: int = 16385,
        bin_width_ps: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return relative time and normalized coincidence probability."""

        size = int(n_bins)
        if size < 33 or size % 2 != 1:
            raise ValueError("n_bins must be an odd integer of at least 33")
        spacing = float(bin_width_ps)
        if spacing <= 0.0:
            raise ValueError("bin_width_ps must be positive")
        direction = int(direction)
        wavelength = self.parameters.travelling_wavelength_nm(direction)
        frequency_thz = np.fft.fftfreq(size, d=spacing)
        omega_rad_ps = 2.0 * np.pi * frequency_thz
        amplitude = self.joint_spectral_amplitude(frequency_thz, filter_bandwidth_nm)

        beta2 = self.beta2_ps2_km(wavelength)
        beta3 = float(self.parameters.beta3_ps3_km)
        length = float(length_km)
        phase = (
            0.5 * beta2 * length * np.square(omega_rad_ps)
            + (beta3 * length / 6.0) * np.power(omega_rad_ps, 3)
        )
        temporal_amplitude = np.fft.fftshift(np.fft.ifft(amplitude * np.exp(1j * phase)))
        probability = np.square(np.abs(temporal_amplitude))

        irf_sigma_bins = (
            self.parameters.irf_fwhm_ps(direction) / GAUSSIAN_FWHM_TO_SIGMA / spacing
        )
        if irf_sigma_bins > 1e-9:
            probability = gaussian_filter1d(probability, irf_sigma_bins, mode="constant")
        probability = np.clip(probability, 0.0, None)
        probability /= max(float(np.sum(probability)), 1e-300)
        time_ps = (np.arange(size, dtype=np.float64) - size // 2) * spacing
        return time_ps, probability

    def expected_counts(
        self,
        length_km: float,
        filter_bandwidth_nm: float,
        direction: int,
        signal_counts: float,
        background_per_bin: float = 0.0,
        n_bins: int = 16385,
        bin_width_ps: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_ps, probability = self.probability(
            length_km,
            filter_bandwidth_nm,
            direction,
            n_bins=n_bins,
            bin_width_ps=bin_width_ps,
        )
        expected = probability * max(float(signal_counts), 0.0) + max(
            float(background_per_bin), 0.0
        )
        return time_ps, expected

    def sample(
        self,
        length_km: float,
        filter_bandwidth_nm: float,
        direction: int,
        signal_counts: float,
        background_per_bin: float = 0.0,
        n_bins: int = 16385,
        bin_width_ps: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        time_ps, expected = self.expected_counts(
            length_km,
            filter_bandwidth_nm,
            direction,
            signal_counts,
            background_per_bin,
            n_bins=n_bins,
            bin_width_ps=bin_width_ps,
        )
        generator = rng or np.random.default_rng()
        return time_ps, generator.poisson(expected).astype(np.float64)


def fwhm_ps(time_ps: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(time_ps, dtype=np.float64)
    y = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    if x.shape != y.shape or y.size < 3:
        raise ValueError("time_ps and values must have equal one-dimensional shape")
    peak_index = int(np.argmax(y))
    half = 0.5 * float(y[peak_index])
    if half <= 0.0:
        return 0.0

    left_below = np.flatnonzero(y[:peak_index] < half)
    if left_below.size:
        i0 = int(left_below[-1])
        i1 = i0 + 1
        left = float(np.interp(half, y[i0 : i1 + 1], x[i0 : i1 + 1]))
    else:
        left = float(x[0])

    right_below = np.flatnonzero(y[peak_index + 1 :] < half)
    if right_below.size:
        i1 = peak_index + 1 + int(right_below[0])
        i0 = i1 - 1
        right = float(np.interp(half, y[i0 : i1 + 1][::-1], x[i0 : i1 + 1][::-1]))
    else:
        right = float(x[-1])
    return max(right - left, 0.0)


def load_physics_parameters(path: str | Path) -> PhysicsParameters:
    """Load parameters from a calibration JSON or a direct parameter JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    values = payload.get("parameters", payload)
    allowed = {field.name for field in fields(PhysicsParameters)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown physics parameter fields: {sorted(unknown)}")
    return PhysicsParameters(**{key: float(value) for key, value in values.items()})
