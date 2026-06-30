from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v17_common import fwhm_np, read_json


@dataclass
class V17ReaderConfig:
    target_template_fraction: float = 0.67
    width_penalty: float = 4.0
    likelihood_temperature: float = 25.0
    max_candidates: int = 12
    width_preselect: int = 48
    blend_scale: float = 1.2
    blend_min: float = 0.25
    blend_max: float = 3.0
    clip_fraction: float = 0.095
    clip_min_ps: float = 8.0
    clip_max_ps: float = 64.0
    background_prior: float = 0.001
    background_penalty: float = 0.05


class TemplateBank:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = read_json(self.manifest_path)
        npz_path = Path(str(self.manifest.get("template_bank_npz", self.manifest_path.with_name("template_bank.npz"))))
        if not npz_path.is_absolute():
            npz_path = self.manifest_path.parent / npz_path.name
        self.arrays = np.load(npz_path)
        self.templates = list(self.manifest["templates"])

    @property
    def half_width(self) -> int:
        return int(self.manifest["half_width"])

    def records_for_direction(self, direction: int) -> list[dict[str, Any]]:
        return [row for row in self.templates if int(row["direction"]) == int(direction)]

    def prob_grad(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        idx = int(row["index"])
        return self.arrays[f"prob_{idx:04d}"].astype(np.float64), self.arrays[f"grad_{idx:04d}"].astype(np.float64)


def poisson_ll(y: np.ndarray, prob: np.ndarray) -> float:
    yy = np.clip(np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = float(np.sum(yy))
    if total <= 1e-12:
        return 0.0
    mu = total * np.clip(prob, 1e-12, None)
    return float(np.sum(yy * np.log(mu) - mu))


def fisher_score_delta(y: np.ndarray, prob: np.ndarray, grad: np.ndarray) -> tuple[float, float, float]:
    yy = np.clip(np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    total = float(np.sum(yy))
    if total <= 1e-12:
        return 0.0, 0.0, 0.0
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, None)
    g = np.asarray(grad, dtype=np.float64)
    mu = total * p
    fisher = total * float(np.sum((g * g) / p))
    if fisher <= 1e-12:
        return 0.0, 0.0, poisson_ll(yy, p)
    score = -float(np.sum((yy - mu) * g / p))
    return score / fisher, fisher, poisson_ll(yy, p)


def estimate_input_width(local: np.ndarray, gaussian_fwhm_ps: float | None) -> float:
    if gaussian_fwhm_ps is not None and np.isfinite(float(gaussian_fwhm_ps)) and float(gaussian_fwhm_ps) > 0.0:
        return float(gaussian_fwhm_ps)
    return float(fwhm_np(local))


def read_one(
    local: np.ndarray,
    direction: int,
    input_fwhm_ps: float,
    bank: TemplateBank,
    config: V17ReaderConfig,
) -> dict[str, Any]:
    target_width = max(float(input_fwhm_ps) * float(config.target_template_fraction), 1.0)
    row_candidates: list[tuple[float, dict[str, Any]]] = []
    for row in bank.records_for_direction(int(direction)):
        template_fwhm = max(float(row["template_fwhm_ps"]), 1.0)
        width_error = float(np.log(template_fwhm / target_width))
        bg_error = float(np.log(max(float(row["background_frac"]), 1e-8) / max(float(config.background_prior), 1e-8)))
        pre_score = float(config.width_penalty) * width_error * width_error + float(config.background_penalty) * bg_error * bg_error
        row_candidates.append((pre_score, row))
    row_candidates = sorted(row_candidates, key=lambda item: item[0])[: max(1, int(config.width_preselect))]
    candidates: list[dict[str, Any]] = []
    for _, row in row_candidates:
        template_fwhm = max(float(row["template_fwhm_ps"]), 1.0)
        width_error = float(np.log(template_fwhm / target_width))
        bg_error = float(np.log(max(float(row["background_frac"]), 1e-8) / max(float(config.background_prior), 1e-8)))
        prob, grad = bank.prob_grad(row)
        raw_delta, fisher, ll = fisher_score_delta(local, prob, grad)
        objective = float(ll) / max(float(np.sum(local)), 1.0)
        objective -= float(config.width_penalty) * width_error * width_error
        objective -= float(config.background_penalty) * bg_error * bg_error
        candidates.append(
            {
                "raw_delta_ps": float(raw_delta),
                "fisher": float(fisher),
                "ll": float(ll),
                "objective": float(objective),
                "width_error": float(width_error),
                "target_template_fwhm_ps": float(target_width),
                "label": str(row["label"]),
                "direction": int(row["direction"]),
                "distance_km": float(row["distance_km"]),
                "bandwidth_nm": float(row["bandwidth_nm"]),
                "smooth_sigma": float(row["smooth_sigma"]),
                "background_frac": float(row["background_frac"]),
                "raw_template_fwhm_ps": float(row["raw_template_fwhm_ps"]),
                "template_fwhm_ps": float(row["template_fwhm_ps"]),
            }
        )
    if not candidates:
        raise ValueError(f"No V17 template candidates for direction={direction}")
    candidates = sorted(candidates, key=lambda item: float(item["objective"]), reverse=True)
    keep = candidates[: max(1, int(config.max_candidates))]
    scores = np.asarray([float(item["objective"]) for item in keep], dtype=np.float64)
    weights = np.exp(np.clip((scores - float(np.max(scores))) / max(float(config.likelihood_temperature), 1e-6), -60.0, 0.0))
    weights = weights / max(float(np.sum(weights)), 1e-12)
    raw_delta = float(np.sum(weights * np.asarray([float(item["raw_delta_ps"]) for item in keep], dtype=np.float64)))
    fisher = float(np.sum(weights * np.asarray([float(item["fisher"]) for item in keep], dtype=np.float64)))
    template_fwhm = float(np.sum(weights * np.asarray([float(item["template_fwhm_ps"]) for item in keep], dtype=np.float64)))
    blend = float(config.blend_scale) * float(input_fwhm_ps) / max(template_fwhm, 1.0)
    blend = float(np.clip(blend, float(config.blend_min), float(config.blend_max)))
    clip_ps = float(np.clip(float(config.clip_fraction) * float(input_fwhm_ps), float(config.clip_min_ps), float(config.clip_max_ps)))
    delta = float(np.clip(blend * raw_delta, -clip_ps, clip_ps))
    best = dict(keep[0])
    best.update(
        {
            "v17_raw_delta_ps": raw_delta,
            "v17_delta_ps": delta,
            "v17_blend": blend,
            "v17_clip_ps": clip_ps,
            "v17_fisher": fisher,
            "v17_template_fwhm_ps": template_fwhm,
            "v17_candidate_count": int(len(candidates)),
            "v17_kept_candidates": int(len(keep)),
            "v17_weight_top": float(weights[0]),
            "v17_selected_label": str(best["label"]),
            "v17_selected_smooth": float(best["smooth_sigma"]),
            "v17_selected_background": float(best["background_frac"]),
            "v17_target_template_fwhm_ps": float(target_width),
        }
    )
    return best
