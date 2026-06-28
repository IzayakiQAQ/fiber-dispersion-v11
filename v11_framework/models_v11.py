from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    current = int(tensor.shape[-1])
    if current == int(target_length):
        return tensor
    diff = current - int(target_length)
    left = diff // 2
    right = diff - left
    return tensor[..., left : current - right]


class GVDEstimatorV165(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class DynamicPhysicalDispersionLayerV165(nn.Module):
    def __init__(self, input_length: int, physics_mode: str = "train") -> None:
        super().__init__()
        self.input_length = int(input_length)
        freqs = torch.fft.fftfreq(self.input_length)
        mode = str(physics_mode).strip().lower()
        if mode == "inference":
            omega = freqs * 2.0 * np.pi
            self.scale = 5000.0
        else:
            omega = freqs * 2.0
            self.scale = 10000.0
        self.register_buffer("omega_squared", omega**2)
        mask = torch.zeros(self.input_length)
        cutoff = max(int(self.input_length * 0.1), 1)
        mask[:cutoff] = 1.0
        mask[-cutoff:] = 1.0
        self.register_buffer("freq_mask", mask)

    def forward(self, x_intensity: torch.Tensor, gvd_val: torch.Tensor) -> torch.Tensor:
        x_amp = torch.sqrt(torch.clamp(x_intensity, min=0.0) + 1e-10)
        x_complex = x_amp.to(torch.complex64)
        x_freq = torch.fft.fft(x_complex, dim=-1)
        x_freq = x_freq * self.freq_mask.reshape(1, -1)
        phase = gvd_val * self.omega_squared.reshape(1, -1) * self.scale
        h = torch.complex(torch.cos(phase), torch.sin(phase))
        y_complex = torch.fft.ifft(x_freq * h, dim=-1)
        return y_complex.abs() ** 2


class AdaptiveHistogramUNetV165(nn.Module):
    def __init__(self, input_dim: int = 65536, physics_mode: str = "train") -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.estimator = GVDEstimatorV165()
        self.dispersion = DynamicPhysicalDispersionLayerV165(input_length=self.input_dim, physics_mode=physics_mode)
        self.encoder1 = nn.Sequential(nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv1d(16, 32, 15, stride=2, padding=7), nn.ReLU())
        self.encoder3 = nn.Sequential(nn.Conv1d(32, 64, 15, stride=2, padding=7), nn.ReLU())
        self.middle = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.ReLU())
        self.decoder3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear"), nn.Conv1d(64, 32, 15, padding=7), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear"), nn.Conv1d(64, 16, 15, padding=7), nn.ReLU())
        self.decoder1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="linear"), nn.Conv1d(32, 1, 15, padding=7))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_gvd = self.estimator(x)
        x_phys = self.dispersion(x, pred_gvd)
        x_in = x_phys.unsqueeze(1)
        e1 = self.encoder1(x_in)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        mid = self.middle(e3)
        d3 = self.decoder3(mid)
        d3 = center_crop(d3, e2.shape[-1])
        d2 = self.decoder2(torch.cat([d3, e2], dim=1))
        d2 = center_crop(d2, e1.shape[-1])
        residual = self.decoder1(torch.cat([d2, e1], dim=1)).squeeze(1)
        return F.relu(x_phys + residual), pred_gvd


def load_v165_teacher(path: str | Path, input_dim: int = 65536, physics_mode: str = "train") -> AdaptiveHistogramUNetV165:
    model = AdaptiveHistogramUNetV165(input_dim=input_dim, physics_mode=physics_mode)
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def low_order_stats(x: torch.Tensor) -> torch.Tensor:
    x_pos = torch.clamp(x, min=0.0)
    peak = x_pos.amax(dim=-1, keepdim=True)
    area = x_pos.sum(dim=-1, keepdim=True)
    mean = x_pos.mean(dim=-1, keepdim=True)
    centered = x_pos - mean
    rms = torch.sqrt(torch.clamp((centered * centered).mean(dim=-1, keepdim=True), min=0.0) + 1e-8)
    return torch.cat([peak, torch.log1p(area), mean, rms], dim=1)


def fourier_shift_1d(x: torch.Tensor, shift_samples: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(shift_samples):
        shift_samples = torch.tensor(shift_samples, device=x.device, dtype=torch.float32)
    shift = shift_samples.to(device=x.device, dtype=torch.float32).reshape(-1, 1)
    length = int(x.shape[-1])
    freqs = torch.fft.fftfreq(length, device=x.device)
    phase = torch.exp(-2j * torch.pi * freqs.reshape(1, -1) * shift).to(torch.complex64)
    spectrum = torch.fft.fft(x.to(torch.complex64), dim=-1)
    shifted = torch.fft.ifft(spectrum * phase, dim=-1).real
    return shifted.to(dtype=x.dtype)


class GaussianWidthGuard1D(nn.Module):
    """Differentiable symmetric broadening guard for teacher-compensated peaks."""

    def __init__(
        self,
        sigma_init: float = 8.0,
        sigma_min: float = 1.0,
        sigma_max: float = 40.0,
        mix_init: float = 0.75,
        mix_min: float = 0.0,
        mix_max: float = 1.0,
        kernel_radius: int = 96,
        area_preserve: bool = True,
    ) -> None:
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.mix_min = float(mix_min)
        self.mix_max = float(mix_max)
        self.kernel_radius = int(kernel_radius)
        self.area_preserve = bool(area_preserve)
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        if self.mix_max <= self.mix_min:
            raise ValueError("mix_max must be greater than mix_min")
        self.sigma_logit = nn.Parameter(torch.tensor(self._inverse_bounded(sigma_init, self.sigma_min, self.sigma_max)))
        self.mix_logit = nn.Parameter(torch.tensor(self._inverse_bounded(mix_init, self.mix_min, self.mix_max)))

    @staticmethod
    def _inverse_bounded(value: float, low: float, high: float) -> float:
        unit = (float(value) - float(low)) / (float(high) - float(low))
        unit = min(max(unit, 1e-5), 1.0 - 1e-5)
        return float(torch.logit(torch.tensor(unit, dtype=torch.float32)).item())

    def sigma(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma_logit)

    def mix(self) -> torch.Tensor:
        return self.mix_min + (self.mix_max - self.mix_min) * torch.sigmoid(self.mix_logit)

    def kernel(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        radius = self.kernel_radius
        x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        sigma = torch.clamp(self.sigma().to(device=device, dtype=dtype), min=1e-3)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / torch.clamp(kernel.sum(), min=1e-8)
        return kernel.reshape(1, 1, -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.clamp(x, min=0.0)
        kernel = self.kernel(dtype=x.dtype, device=x.device)
        padded = F.pad(x.unsqueeze(1), (self.kernel_radius, self.kernel_radius), mode="replicate")
        smoothed = F.conv1d(padded, kernel).squeeze(1)
        mix = self.mix().to(device=x.device, dtype=x.dtype)
        out = (1.0 - mix) * x + mix * smoothed
        out = torch.clamp(out, min=0.0)
        if self.area_preserve:
            area_before = x.sum(dim=-1, keepdim=True)
            area_after = out.sum(dim=-1, keepdim=True)
            out = out * (area_before / torch.clamp(area_after, min=1e-8))
        return torch.clamp(out, min=0.0), {"sigma": self.sigma(), "mix": self.mix()}


class V11TeacherWidthGuard(nn.Module):
    def __init__(
        self,
        teacher_checkpoint: str | Path,
        input_dim: int = 65536,
        physics_mode: str = "train",
        sigma_init: float = 8.0,
        sigma_min: float = 1.0,
        sigma_max: float = 40.0,
        mix_init: float = 0.75,
        mix_min: float = 0.0,
        mix_max: float = 1.0,
        kernel_radius: int = 96,
        area_preserve: bool = True,
    ) -> None:
        super().__init__()
        self.teacher = load_v165_teacher(teacher_checkpoint, input_dim=input_dim, physics_mode=physics_mode)
        self.guard = GaussianWidthGuard1D(
            sigma_init=sigma_init,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            mix_init=mix_init,
            mix_min=mix_min,
            mix_max=mix_max,
            kernel_radius=kernel_radius,
            area_preserve=area_preserve,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.teacher.eval()
        with torch.no_grad():
            teacher_out, gvd = self.teacher(x)
        out, guard_info = self.guard(teacher_out)
        return out, {
            "teacher": teacher_out,
            "gvd": gvd.reshape(-1),
            "sigma": guard_info["sigma"].reshape(()),
            "mix": guard_info["mix"].reshape(()),
        }


class ShiftCorrectionHead1D(nn.Module):
    def __init__(
        self,
        max_abs_shift: float = 4.0,
        channels: int = 24,
        hidden: int = 48,
    ) -> None:
        super().__init__()
        self.max_abs_shift = float(max_abs_shift)
        c = int(channels)
        self.features = nn.Sequential(
            nn.Conv1d(3, c, kernel_size=31, stride=8, padding=15),
            nn.GroupNorm(1, c),
            nn.SiLU(),
            nn.Conv1d(c, c * 2, kernel_size=31, stride=8, padding=15),
            nn.GroupNorm(1, c * 2),
            nn.SiLU(),
            nn.Conv1d(c * 2, c * 2, kernel_size=15, stride=4, padding=7),
            nn.GroupNorm(1, c * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(c * 2 + 12, int(hidden)), nn.SiLU(), nn.Linear(int(hidden), 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_input: torch.Tensor, teacher: torch.Tensor, guarded: torch.Tensor) -> torch.Tensor:
        x_input = torch.clamp(x_input, min=0.0)
        teacher = torch.clamp(teacher, min=0.0)
        guarded = torch.clamp(guarded, min=0.0)
        stacked = torch.stack([x_input, teacher, guarded], dim=1)
        features = self.features(stacked)
        stats = torch.cat([low_order_stats(x_input), low_order_stats(teacher), low_order_stats(guarded)], dim=1)
        raw = self.head(torch.cat([features, stats], dim=1)).reshape(-1)
        return self.max_abs_shift * torch.tanh(raw)


class V11TeacherWidthShift(nn.Module):
    def __init__(
        self,
        teacher_checkpoint: str | Path,
        input_dim: int = 65536,
        physics_mode: str = "train",
        sigma_init: float = 22.0,
        sigma_min: float = 1.0,
        sigma_max: float = 40.0,
        mix_init: float = 0.925,
        mix_min: float = 0.0,
        mix_max: float = 1.0,
        kernel_radius: int = 96,
        area_preserve: bool = True,
        max_abs_shift: float = 4.0,
        shift_channels: int = 24,
        shift_hidden: int = 48,
        freeze_guard: bool = True,
    ) -> None:
        super().__init__()
        self.teacher = load_v165_teacher(teacher_checkpoint, input_dim=input_dim, physics_mode=physics_mode)
        self.guard = GaussianWidthGuard1D(
            sigma_init=sigma_init,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            mix_init=mix_init,
            mix_min=mix_min,
            mix_max=mix_max,
            kernel_radius=kernel_radius,
            area_preserve=area_preserve,
        )
        self.shift_head = ShiftCorrectionHead1D(
            max_abs_shift=max_abs_shift,
            channels=shift_channels,
            hidden=shift_hidden,
        )
        if bool(freeze_guard):
            for parameter in self.guard.parameters():
                parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.teacher.eval()
        with torch.no_grad():
            teacher_out, gvd = self.teacher(x)
        guarded, guard_info = self.guard(teacher_out)
        delta_shift = self.shift_head(x, teacher_out.detach(), guarded)
        shifted = torch.clamp(fourier_shift_1d(guarded, delta_shift), min=0.0)
        area_before = guarded.sum(dim=1, keepdim=True)
        area_after = shifted.sum(dim=1, keepdim=True)
        shifted = shifted * (area_before / torch.clamp(area_after, min=1e-8))
        shifted = torch.clamp(shifted, min=0.0)
        return shifted, {
            "teacher": teacher_out,
            "guarded": guarded,
            "gvd": gvd.reshape(-1),
            "sigma": guard_info["sigma"].reshape(()),
            "mix": guard_info["mix"].reshape(()),
            "delta_shift": delta_shift.reshape(-1),
        }


def trainable_guard_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def checkpoint_state(model: V11TeacherWidthGuard) -> dict[str, Any]:
    out = {
        "guard_state": model.guard.state_dict(),
        "sigma": float(model.guard.sigma().detach().cpu().item()),
        "mix": float(model.guard.mix().detach().cpu().item()),
    }
    if hasattr(model, "shift_head"):
        out["shift_head_state"] = model.shift_head.state_dict()
        out["max_abs_shift"] = float(model.shift_head.max_abs_shift)
    return out
