from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "fiber-dispersion-v25"


@dataclass(frozen=True)
class PhysicsParameters:
    """Effective parameters identified from independent calibration."""

    pump_wavelength_nm: float = 1540.56
    channel_c57_frequency_thz: float = 195.7
    channel_c35_frequency_thz: float = 193.5
    source_fwhm_nm: float = 60.0
    filter_bandwidth_scale: float = 1.0
    dispersion_ps_nm_km_at_1550: float = 17.0
    dispersion_slope_ps_nm2_km: float = 0.058
    beta3_ps3_km: float = 0.0
    irf_fwhm_direction1_ps: float = 162.0
    irf_fwhm_direction2_ps: float = 162.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PhysicsParameters":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown physics parameter fields: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})

    def validate(self) -> None:
        if self.pump_wavelength_nm <= 0.0:
            raise ValueError("pump_wavelength_nm must be positive")
        if (
            self.channel_c57_frequency_thz <= 0.0
            or self.channel_c35_frequency_thz <= 0.0
        ):
            raise ValueError("channel frequencies must be positive")
        if self.source_fwhm_nm <= 0.0:
            raise ValueError("source_fwhm_nm must be positive")
        if self.filter_bandwidth_scale <= 0.0:
            raise ValueError("filter_bandwidth_scale must be positive")
        if (
            self.irf_fwhm_direction1_ps <= 0.0
            or self.irf_fwhm_direction2_ps <= 0.0
        ):
            raise ValueError("direction IRF widths must be positive")

    def irf_fwhm_ps(self, direction: int) -> float:
        if direction == 1:
            return float(self.irf_fwhm_direction1_ps)
        if direction == 2:
            return float(self.irf_fwhm_direction2_ps)
        raise ValueError("direction must be 1 or 2")


@dataclass(frozen=True)
class OperatorSettings:
    """Frozen numerical settings; evaluation data must not modify them."""

    kernel_bins: int = 2049
    synthesis_bins: int = 8193
    bin_width_ps: float = 1.0
    rl_iterations: int = 512
    edge_bins: int = 160
    ratio_clip: float = 8.0
    latent_floor_fraction: float = 1e-8
    localization_smooth_sigma_bins: float = 24.0
    localization_half_width_bins: int = 800
    center_half_window_bins: int = 180

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OperatorSettings":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown operator setting fields: {sorted(unknown)}")
        return cls(**values)

    def validate(self) -> None:
        if self.kernel_bins < 33 or self.kernel_bins % 2 != 1:
            raise ValueError("kernel_bins must be an odd integer of at least 33")
        if self.synthesis_bins < self.kernel_bins or self.synthesis_bins % 2 != 1:
            raise ValueError("synthesis_bins must be odd and no smaller than kernel_bins")
        if self.bin_width_ps <= 0.0:
            raise ValueError("bin_width_ps must be positive")
        if self.rl_iterations < 0:
            raise ValueError("rl_iterations must be nonnegative")
        if not 1 <= self.edge_bins <= self.kernel_bins // 4:
            raise ValueError("edge_bins must be between 1 and kernel_bins/4")
        if self.ratio_clip < 1.0:
            raise ValueError("ratio_clip must be at least 1")
        if not 0.0 < self.latent_floor_fraction < 1.0:
            raise ValueError("latent_floor_fraction must be in (0, 1)")
        if not 1 <= self.center_half_window_bins <= self.kernel_bins // 2:
            raise ValueError("center_half_window_bins is outside the local histogram")


@dataclass(frozen=True)
class FrozenConfig:
    """Complete deterministic V25 deployment configuration."""

    length_km: float
    bandwidth_nm: float
    calibration_sha256: str
    physics: PhysicsParameters = PhysicsParameters()
    operator: OperatorSettings = OperatorSettings()
    schema: str = SCHEMA
    provenance: str = "independent_physics_calibration"

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"Unsupported schema: {self.schema!r}")
        if self.provenance != "independent_physics_calibration":
            raise ValueError("V25 requires independent_physics_calibration provenance")
        if self.length_km < 0.0:
            raise ValueError("length_km must be nonnegative")
        if self.bandwidth_nm <= 0.0:
            raise ValueError("bandwidth_nm must be positive")
        if not self.calibration_sha256:
            raise ValueError("calibration_sha256 must identify the independent calibration")
        self.physics.validate()
        self.operator.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown frozen config fields: {sorted(unknown)}")
        result = cls(
            length_km=float(payload["length_km"]),
            bandwidth_nm=float(payload["bandwidth_nm"]),
            calibration_sha256=str(payload["calibration_sha256"]),
            physics=PhysicsParameters.from_mapping(payload.get("physics", {})),
            operator=OperatorSettings.from_mapping(payload.get("operator", {})),
            schema=str(payload.get("schema", SCHEMA)),
            provenance=str(payload.get("provenance", "independent_physics_calibration")),
        )
        result.validate()
        return result

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_frozen_config(config: FrozenConfig, path: str | Path) -> tuple[Path, Path]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = Path(f"{output}.sha256")
    sidecar.write_text(f"{config.sha256()}  {output.name}\n", encoding="ascii")
    return output, sidecar


def load_frozen_config(path: str | Path, require_hash: bool = True) -> FrozenConfig:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    config = FrozenConfig.from_dict(payload)
    sidecar = Path(f"{source}.sha256")
    if require_hash:
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing frozen-config hash sidecar: {sidecar}")
        expected = sidecar.read_text(encoding="ascii").split()[0].strip().lower()
        actual = config.sha256()
        if expected != actual:
            raise ValueError(
                f"Frozen config SHA-256 mismatch: expected {expected}, got {actual}"
            )
    return config
