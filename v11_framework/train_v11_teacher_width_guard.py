from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V11_DIR = PROJECT_ROOT / "v11_framework"
for path in (PROJECT_ROOT, V11_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models_v11 import V11TeacherWidthGuard, checkpoint_state, trainable_guard_parameters
from v11_utils import (
    compute_peak_metrics,
    fit_peak_quadratic,
    fit_peak_window_centroid,
    load_config,
    pair_selection_summary,
    resolve_device,
    safe_norm_batch,
    scenarios_from_manifest,
    split_from_config,
    tdev_at_tau0,
    tdev_curve,
)


class V11PairDataset(Dataset):
    def __init__(self, scenario: dict[str, Any], pair_ids: list[int], num_pairs: int = 8640) -> None:
        self.scenario = dict(scenario)
        self.pair_ids = [int(pair) for pair in pair_ids]
        self.num_pairs = int(num_pairs)

    def __len__(self) -> int:
        return len(self.pair_ids)

    @staticmethod
    def _safe_norm(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        peak = torch.max(x)
        if float(peak.detach().cpu().item()) > 1e-8:
            return x / (peak + 1e-8)
        return torch.zeros_like(x)

    def _load_pair(self, directory: str, pair_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(directory)
        id1 = int(pair_id)
        id2 = id1 + self.num_pairs
        return (
            self._safe_norm(torch.load(path / f"hist_{id1:05d}.pt", map_location="cpu")),
            self._safe_norm(torch.load(path / f"hist_{id2:05d}.pt", map_location="cpu")),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair_id = self.pair_ids[index]
        target1, target2 = self._load_pair(str(self.scenario["target_dir"]), pair_id)
        input1, input2 = self._load_pair(str(self.scenario["input_dir"]), pair_id)
        return target1, input1, target2, input2, torch.tensor(pair_id, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V11 teacher-width-guard on V10 centered scenario tensors.")
    parser.add_argument("--config", type=Path, default=Path("v11_framework/configs/v11_50km_0p8_teacher_width_guard.yaml"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--max-eval-pairs", type=int, default=0)
    parser.add_argument("--eval-splits", type=str, default="val,test")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--include-labels", type=str, default="")
    parser.add_argument("--exclude-labels", type=str, default="")
    parser.add_argument("--target-label", type=str, default="")
    parser.add_argument("--target-policy", choices=["fixed", "bandwidth_if_available"], default="")
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _cfg_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(config.get(name, {}))


def split_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def limited_pairs(pairs: tuple[int, ...], usable_pairs: int, limit: int) -> list[int]:
    selected = [int(pair) for pair in pairs if 1 <= int(pair) <= int(usable_pairs)]
    if int(limit) > 0:
        selected = selected[: int(limit)]
    return selected


def find_checkpoint(root: str | Path, filename: str) -> Path:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    matches = sorted(root_path.rglob(str(filename)), key=lambda path: str(path))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {root_path}")
    return matches[0]


def load_shift_csv(path: str | Path) -> dict[int, float]:
    shifts: dict[int, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                shifts[int(row.get("index", "0"))] = float(row.get("shift", "0"))
            except (TypeError, ValueError):
                continue
    return shifts


def safe_load_tensor_np(path: Path) -> np.ndarray:
    arr = torch.load(path, map_location="cpu").float().numpy().astype(np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(arr)) if arr.size else 0.0
    return arr / (peak + 1e-8) if peak > 1e-8 else np.zeros_like(arr)


def fit_center(values: np.ndarray, method: str) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if str(method) == "argmax":
        return float(np.argmax(arr))
    if str(method) == "quadratic":
        return float(fit_peak_quadratic(arr))
    return float(fit_peak_window_centroid(arr, half_width=50))


def fwhm(values: np.ndarray) -> float:
    return float(compute_peak_metrics(values).fwhm_samples)


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else 0.0


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def local_mask(length: int, centers: torch.Tensor, half_window: int, like: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(length, device=like.device, dtype=like.dtype).reshape(1, -1)
    center = centers.to(device=like.device, dtype=like.dtype).reshape(-1, 1)
    return (torch.abs(idx - center) <= float(half_window)).to(dtype=like.dtype)


def local_moment(x: torch.Tensor, centers: torch.Tensor, half_window: int) -> dict[str, torch.Tensor]:
    x_pos = torch.clamp(x, min=0.0)
    mask = local_mask(x.shape[-1], centers, half_window, x_pos)
    idx = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).reshape(1, -1)
    weights = x_pos * mask
    mass = weights.sum(dim=1)
    center = (weights * idx).sum(dim=1) / (mass + 1e-8)
    var = (weights * (idx - center.reshape(-1, 1)) ** 2).sum(dim=1) / (mass + 1e-8)
    width = torch.sqrt(torch.clamp(var, min=0.0) + 1e-8)
    return {"center": center, "width": width, "area": mass}


def masked_l1_mse(out: torch.Tensor, target: torch.Tensor, centers: torch.Tensor, half_window: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = local_mask(out.shape[-1], centers, half_window, out)
    denom = torch.clamp(mask.sum(), min=1.0)
    diff = (out - target) * mask
    return torch.sum(torch.abs(diff)) / denom, torch.sum(diff * diff) / denom


def single_loss(out: torch.Tensor, target: torch.Tensor, teacher: torch.Tensor, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = _cfg_section(config, "loss")
    target_peak = torch.argmax(torch.clamp(target, min=0.0), dim=1).detach().to(dtype=out.dtype)
    teacher_peak = torch.argmax(torch.clamp(teacher, min=0.0), dim=1).detach().to(dtype=out.dtype)

    window_half = int(loss_cfg.get("window_half", 220))
    width_half = int(loss_cfg.get("width_half", 500))
    center_norm = float(loss_cfg.get("center_norm", 100.0))
    width_norm = float(loss_cfg.get("width_norm", 100.0))
    width_min_ratio = float(loss_cfg.get("width_min_ratio", 0.90))
    width_max_ratio = float(loss_cfg.get("width_max_ratio", 1.18))

    rec_l1, rec_mse = masked_l1_mse(out, target, target_peak, window_half)
    out_target_stats = local_moment(out, target_peak, width_half)
    target_stats = local_moment(target, target_peak, width_half)
    out_teacher_stats = local_moment(out, teacher_peak, width_half)
    teacher_stats = local_moment(teacher, teacher_peak, width_half)

    width = out_target_stats["width"]
    target_width = target_stats["width"].detach()
    width_match = torch.mean(((width - target_width) / width_norm) ** 2)
    width_band = torch.mean(
        (F.relu(width_min_ratio * target_width - width) / width_norm) ** 2
        + (F.relu(width - width_max_ratio * target_width) / width_norm) ** 2
    )
    center_loss = torch.mean(((out_target_stats["center"] - target_stats["center"].detach()) / center_norm) ** 2)
    teacher_center_loss = torch.mean(((out_teacher_stats["center"] - teacher_stats["center"].detach()) / center_norm) ** 2)
    area_loss = torch.mean(
        ((out_target_stats["area"] - target_stats["area"].detach()) / torch.clamp(target_stats["area"].detach(), min=1.0)) ** 2
    )
    teacher_delta = torch.mean(((out - teacher) * local_mask(out.shape[-1], teacher_peak, width_half, out)) ** 2)

    total = (
        float(loss_cfg.get("w_window_l1", 10.0)) * rec_l1
        + float(loss_cfg.get("w_window_mse", 20.0)) * rec_mse
        + float(loss_cfg.get("w_width_match", 120.0)) * width_match
        + float(loss_cfg.get("w_width_band", 240.0)) * width_band
        + float(loss_cfg.get("w_center", 15.0)) * center_loss
        + float(loss_cfg.get("w_teacher_center", 60.0)) * teacher_center_loss
        + float(loss_cfg.get("w_area", 0.25)) * area_loss
        + float(loss_cfg.get("w_teacher_delta", 0.02)) * teacher_delta
    )
    return total, {
        "rec_l1": rec_l1,
        "rec_mse": rec_mse,
        "width_match": width_match,
        "width_band": width_band,
        "center": center_loss,
        "teacher_center": teacher_center_loss,
        "area": area_loss,
        "teacher_delta": teacher_delta,
        "out_width": torch.mean(width.detach()),
        "target_width": torch.mean(target_width.detach()),
    }


def make_models(config: dict[str, Any], device: torch.device) -> tuple[dict[int, V11TeacherWidthGuard], dict[int, Path]]:
    model_cfg = _cfg_section(config, "model")
    teacher_cfg = _cfg_section(config, "teacher")
    teacher_root = Path(str(teacher_cfg.get("root", "checkpoints/v165_teacher")))
    teacher_paths = {
        1: find_checkpoint(teacher_root, str(teacher_cfg.get("dir1", "adaptivecheck_compensator_dir1.pt"))),
        2: find_checkpoint(teacher_root, str(teacher_cfg.get("dir2", "adaptivecheck_compensator_dir2.pt"))),
    }
    models = {
        direction: V11TeacherWidthGuard(
            teacher_checkpoint=teacher_paths[direction],
            input_dim=int(config.get("input_dim", 65536)),
            physics_mode=str(teacher_cfg.get("physics_mode", "train")),
            sigma_init=float(model_cfg.get("sigma_init", 8.0)),
            sigma_min=float(model_cfg.get("sigma_min", 1.0)),
            sigma_max=float(model_cfg.get("sigma_max", 40.0)),
            mix_init=float(model_cfg.get("mix_init", 0.75)),
            mix_min=float(model_cfg.get("mix_min", 0.0)),
            mix_max=float(model_cfg.get("mix_max", 1.0)),
            kernel_radius=int(model_cfg.get("kernel_radius", 96)),
            area_preserve=bool(model_cfg.get("area_preserve", True)),
        ).to(device)
        for direction in (1, 2)
    }
    return models, teacher_paths


def train_epoch(
    models: dict[int, V11TeacherWidthGuard],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    log_every_batches: int,
    epoch: int,
) -> dict[str, float]:
    for model in models.values():
        model.train()
    sums: dict[str, float] = {}
    batches = 0
    for batch_idx, batch in enumerate(loader, start=1):
        target1, input1, target2, input2, _pair_id = batch
        target1 = safe_norm_batch(target1.to(device))
        input1 = safe_norm_batch(input1.to(device))
        target2 = safe_norm_batch(target2.to(device))
        input2 = safe_norm_batch(input2.to(device))
        out1, aux1 = models[1](input1)
        out2, aux2 = models[2](input2)
        loss1, parts1 = single_loss(out1, target1, aux1["teacher"], config)
        loss2, parts2 = single_loss(out2, target2, aux2["teacher"], config)
        loss = 0.5 * (loss1 + loss2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batches += 1
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu().item())
        for key in parts1:
            value = 0.5 * (parts1[key] + parts2[key])
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
        if log_every_batches > 0 and batch_idx % int(log_every_batches) == 0:
            print(
                f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={float(loss.detach().cpu().item()):.6g} "
                f"sigma1={models[1].guard.sigma().item():.3f} mix1={models[1].guard.mix().item():.3f} "
                f"sigma2={models[2].guard.sigma().item():.3f} mix2={models[2].guard.mix().item():.3f}",
                flush=True,
            )
    out = {key: value / max(batches, 1) for key, value in sums.items()}
    out.update(
        {
            "sigma1": float(models[1].guard.sigma().detach().cpu().item()),
            "mix1": float(models[1].guard.mix().detach().cpu().item()),
            "sigma2": float(models[2].guard.sigma().detach().cpu().item()),
            "mix2": float(models[2].guard.mix().detach().cpu().item()),
        }
    )
    return out


@torch.no_grad()
def eval_loss(models: dict[int, V11TeacherWidthGuard], loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    for model in models.values():
        model.eval()
    sums: dict[str, float] = {}
    batches = 0
    for batch in loader:
        target1, input1, target2, input2, _pair_id = batch
        target1 = safe_norm_batch(target1.to(device))
        input1 = safe_norm_batch(input1.to(device))
        target2 = safe_norm_batch(target2.to(device))
        input2 = safe_norm_batch(input2.to(device))
        out1, aux1 = models[1](input1)
        out2, aux2 = models[2](input2)
        loss1, parts1 = single_loss(out1, target1, aux1["teacher"], config)
        loss2, parts2 = single_loss(out2, target2, aux2["teacher"], config)
        loss = 0.5 * (loss1 + loss2)
        batches += 1
        sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu().item())
        for key in parts1:
            value = 0.5 * (parts1[key] + parts2[key])
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
    out = {key: value / max(batches, 1) for key, value in sums.items()}
    out.update(
        {
            "sigma1": float(models[1].guard.sigma().detach().cpu().item()),
            "mix1": float(models[1].guard.mix().detach().cpu().item()),
            "sigma2": float(models[2].guard.sigma().detach().cpu().item()),
            "mix2": float(models[2].guard.mix().detach().cpu().item()),
        }
    )
    return out


def evaluate_split(
    models: dict[int, V11TeacherWidthGuard],
    scenario: dict[str, Any],
    split_name: str,
    pair_ids: list[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    eval_cfg = _cfg_section(config, "eval")
    center_method = str(eval_cfg.get("center_method", "window_centroid"))
    sample_period_s = float(eval_cfg.get("sample_period_s", 10.0))
    batch_size = int(eval_cfg.get("batch_size", 64))
    num_pairs = int(config.get("num_pairs", 8640))
    input_dir = Path(str(scenario["input_dir"]))
    target_dir = Path(str(scenario["target_dir"]))
    input_shifts = load_shift_csv(str(scenario["input_shift_csv"]))
    target_shifts = load_shift_csv(str(scenario["target_shift_csv"]))

    input_clock: list[float] = []
    comp_clock: list[float] = []
    target_clock: list[float] = []
    input_fwhm: list[float] = []
    comp_fwhm: list[float] = []
    target_fwhm: list[float] = []
    comp_input_fwhm_ratios: list[float] = []
    comp_target_fwhm_ratios: list[float] = []
    teacher_fwhm: list[float] = []
    teacher_clock: list[float] = []

    for model in models.values():
        model.eval()
    for start in range(0, len(pair_ids), batch_size):
        ids1 = [int(pair) for pair in pair_ids[start : start + batch_size]]
        ids2 = [pair + num_pairs for pair in ids1]
        x1_np = np.stack([safe_load_tensor_np(input_dir / f"hist_{idx:05d}.pt") for idx in ids1], axis=0)
        x2_np = np.stack([safe_load_tensor_np(input_dir / f"hist_{idx:05d}.pt") for idx in ids2], axis=0)
        t1_np = np.stack([safe_load_tensor_np(target_dir / f"hist_{idx:05d}.pt") for idx in ids1], axis=0)
        t2_np = np.stack([safe_load_tensor_np(target_dir / f"hist_{idx:05d}.pt") for idx in ids2], axis=0)
        x1 = torch.tensor(x1_np, dtype=torch.float32, device=device)
        x2 = torch.tensor(x2_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            y1, aux1 = models[1](x1)
            y2, aux2 = models[2](x2)
        y1_np = y1.detach().cpu().numpy().astype(np.float64)
        y2_np = y2.detach().cpu().numpy().astype(np.float64)
        teacher1_np = aux1["teacher"].detach().cpu().numpy().astype(np.float64)
        teacher2_np = aux2["teacher"].detach().cpu().numpy().astype(np.float64)

        for row_idx, pair_id in enumerate(ids1):
            pair_id2 = pair_id + num_pairs
            shift1 = float(input_shifts.get(pair_id, 0.0))
            shift2 = float(input_shifts.get(pair_id2, 0.0))
            target_shift1 = float(target_shifts.get(pair_id, 0.0))
            target_shift2 = float(target_shifts.get(pair_id2, 0.0))

            in_c1 = fit_center(x1_np[row_idx], center_method) - shift1
            in_c2 = fit_center(x2_np[row_idx], center_method) - shift2
            out_c1 = fit_center(y1_np[row_idx], center_method) - shift1
            out_c2 = fit_center(y2_np[row_idx], center_method) - shift2
            teacher_c1 = fit_center(teacher1_np[row_idx], center_method) - shift1
            teacher_c2 = fit_center(teacher2_np[row_idx], center_method) - shift2
            tgt_c1 = fit_center(t1_np[row_idx], center_method) - target_shift1
            tgt_c2 = fit_center(t2_np[row_idx], center_method) - target_shift2
            input_clock.append(0.5 * (in_c1 - in_c2))
            comp_clock.append(0.5 * (out_c1 - out_c2))
            teacher_clock.append(0.5 * (teacher_c1 - teacher_c2))
            target_clock.append(0.5 * (tgt_c1 - tgt_c2))

            in_f = 0.5 * (fwhm(x1_np[row_idx]) + fwhm(x2_np[row_idx]))
            out_f = 0.5 * (fwhm(y1_np[row_idx]) + fwhm(y2_np[row_idx]))
            teacher_f = 0.5 * (fwhm(teacher1_np[row_idx]) + fwhm(teacher2_np[row_idx]))
            tgt_f = 0.5 * (fwhm(t1_np[row_idx]) + fwhm(t2_np[row_idx]))
            input_fwhm.append(in_f)
            comp_fwhm.append(out_f)
            teacher_fwhm.append(teacher_f)
            target_fwhm.append(tgt_f)
            comp_input_fwhm_ratios.append(safe_ratio(out_f, in_f))
            comp_target_fwhm_ratios.append(safe_ratio(out_f, tgt_f))

    input_arr = np.asarray(input_clock, dtype=np.float64)
    comp_arr = np.asarray(comp_clock, dtype=np.float64)
    teacher_arr = np.asarray(teacher_clock, dtype=np.float64)
    target_arr = np.asarray(target_clock, dtype=np.float64)
    input_tdev = tdev_at_tau0(input_arr)
    comp_tdev = tdev_at_tau0(comp_arr)
    teacher_tdev = tdev_at_tau0(teacher_arr)
    target_tdev = tdev_at_tau0(target_arr)
    return {
        "scenario": str(scenario["label"]),
        "split": str(split_name),
        "target_label": str(scenario["target_label"]),
        "distance_km": float(scenario["distance_km"]),
        "bandwidth_nm": float(scenario["bandwidth_nm"]),
        "count": float(len(pair_ids)),
        "pair_summary": pair_selection_summary([pair - 1 for pair in pair_ids]),
        "center_method": center_method,
        "input_tdev10": input_tdev,
        "teacher_tdev10": teacher_tdev,
        "comp_tdev10": comp_tdev,
        "target_tdev10": target_tdev,
        "teacher_input_tdev_ratio": safe_ratio(teacher_tdev, input_tdev),
        "comp_input_tdev_ratio": safe_ratio(comp_tdev, input_tdev),
        "comp_target_tdev_ratio": safe_ratio(comp_tdev, target_tdev),
        "teacher_std": std_or_zero(teacher_clock),
        "comp_std": std_or_zero(comp_clock),
        "target_std": std_or_zero(target_clock),
        "input_fwhm": mean_or_zero(input_fwhm),
        "teacher_fwhm": mean_or_zero(teacher_fwhm),
        "comp_fwhm": mean_or_zero(comp_fwhm),
        "target_fwhm": mean_or_zero(target_fwhm),
        "comp_input_fwhm_ratio": mean_or_zero(comp_input_fwhm_ratios),
        "comp_target_fwhm_ratio": mean_or_zero(comp_target_fwhm_ratios),
        "sigma1": float(models[1].guard.sigma().detach().cpu().item()),
        "mix1": float(models[1].guard.mix().detach().cpu().item()),
        "sigma2": float(models[2].guard.sigma().detach().cpu().item()),
        "mix2": float(models[2].guard.mix().detach().cpu().item()),
        "input_tdev_curve": tdev_curve(input_arr, sample_period_s=sample_period_s),
        "teacher_tdev_curve": tdev_curve(teacher_arr, sample_period_s=sample_period_s),
        "comp_tdev_curve": tdev_curve(comp_arr, sample_period_s=sample_period_s),
        "target_tdev_curve": tdev_curve(target_arr, sample_period_s=sample_period_s),
    }


def write_train_log(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: V11TeacherWidthGuard, optimizer: torch.optim.Optimizer, config: dict[str, Any], direction: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "guard": checkpoint_state(model),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "direction": int(direction),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = _cfg_section(config, "train")
    save_dir = Path(str(train_cfg.get("save_dir", "v11_framework/artifacts/v11_teacher_width_guard")))
    seed = int(args.seed if args.seed >= 0 else train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)

    scenarios, source_manifest = scenarios_from_manifest(config, args)
    if len(scenarios) != 1:
        raise ValueError(f"V11 teacher-width-guard currently expects one scenario, got {len(scenarios)}")
    scenario = scenarios[0]
    usable_pairs = int(scenario["usable_pairs"])
    split = split_from_config(config)
    train_pairs = limited_pairs(split.train_pairs, usable_pairs, int(args.max_train_pairs))
    val_pairs = limited_pairs(split.val_pairs, usable_pairs, int(args.max_val_pairs))
    if not train_pairs:
        train_pairs = list(range(1, min(usable_pairs, int(args.max_train_pairs) if args.max_train_pairs > 0 else usable_pairs) + 1))
    if not val_pairs:
        val_pairs = train_pairs[: min(len(train_pairs), 64)]

    batch_size = int(args.batch_size if args.batch_size > 0 else train_cfg.get("batch_size", 16))
    num_workers = int(args.num_workers if args.num_workers >= 0 else train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        V11PairDataset(scenario, train_pairs, num_pairs=int(config.get("num_pairs", 8640))),
        batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle", True)),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        V11PairDataset(scenario, val_pairs, num_pairs=int(config.get("num_pairs", 8640))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    models, teacher_paths = make_models(config, device)
    params = trainable_guard_parameters(models[1]) + trainable_guard_parameters(models[2])
    optimizer = torch.optim.AdamW(params, lr=float(args.lr if args.lr > 0 else train_cfg.get("lr", 2.0e-2)))
    epochs = int(args.epochs if args.epochs > 0 else train_cfg.get("epochs", 8))

    dry_report = {
        "framework": "v11",
        "variant": str(config.get("variant", "teacher_width_guard")),
        "scenario": scenario,
        "teacher_paths": {str(key): str(value) for key, value in teacher_paths.items()},
        "train_pairs": pair_selection_summary([pair - 1 for pair in train_pairs]),
        "val_pairs": pair_selection_summary([pair - 1 for pair in val_pairs]),
        "device": str(device),
        "save_dir": str(save_dir),
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, ensure_ascii=False))
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config_used.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"=== V11 teacher-width-guard train start: device={device}, scenario={scenario['label']}, "
        f"train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)}, batch_size={batch_size}, epochs={epochs} ===",
        flush=True,
    )
    history: list[dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(models, train_loader, optimizer, config, device, int(args.log_every_batches), epoch)
        val_stats = eval_loss(models, val_loader, config, device)
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in val_stats.items()})
        history.append(row)
        for direction in (1, 2):
            save_checkpoint(save_dir / f"v11_teacher_width_guard_latest_dir{direction}.pt", models[direction], optimizer, config, direction)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            for direction in (1, 2):
                save_checkpoint(save_dir / f"v11_teacher_width_guard_valbest_dir{direction}.pt", models[direction], optimizer, config, direction)
        print(
            f"epoch={epoch} train_loss={train_stats['loss']:.6g} val_loss={val_stats['loss']:.6g} "
            f"sigma1={val_stats['sigma1']:.3f} mix1={val_stats['mix1']:.3f} "
            f"sigma2={val_stats['sigma2']:.3f} mix2={val_stats['mix2']:.3f}",
            flush=True,
        )

    write_train_log(save_dir / "train_log.csv", history)
    split_names = split_csv(args.eval_splits)
    eval_limit = int(args.max_eval_pairs)
    eval_rows: list[dict[str, Any]] = []
    for split_name in split_names:
        if split_name == "train":
            pairs = limited_pairs(split.train_pairs, usable_pairs, eval_limit)
        elif split_name == "val":
            pairs = limited_pairs(split.val_pairs, usable_pairs, eval_limit)
        elif split_name == "test":
            pairs = limited_pairs(split.test_pairs, usable_pairs, eval_limit)
        elif split_name == "all":
            pairs = list(range(1, usable_pairs + 1))
            if eval_limit > 0:
                pairs = pairs[:eval_limit]
        else:
            raise ValueError(f"Unsupported eval split: {split_name!r}")
        if not pairs:
            continue
        result = evaluate_split(models, scenario, split_name, pairs, config, device)
        eval_rows.append(result)
        print(
            f"eval split={split_name} count={result['count']:.0f} "
            f"teacher_tdev={result['teacher_tdev10']:.4f} comp_tdev={result['comp_tdev10']:.4f} "
            f"target_tdev={result['target_tdev10']:.4f} comp_fwhm={result['comp_fwhm']:.3f} "
            f"target_fwhm={result['target_fwhm']:.3f}",
            flush=True,
        )

    summary = {
        "framework": "v11",
        "variant": str(config.get("variant", "teacher_width_guard")),
        "config": str(args.config),
        "source_manifest_output_root": source_manifest.get("output_root", ""),
        "scenario": scenario,
        "teacher_paths": {str(key): str(value) for key, value in teacher_paths.items()},
        "device": str(device),
        "best_val_loss": best_val,
        "train_pairs": pair_selection_summary([pair - 1 for pair in train_pairs]),
        "val_pairs": pair_selection_summary([pair - 1 for pair in val_pairs]),
        "history_last": history[-1] if history else {},
        "results": eval_rows,
        "checkpoints": {
            "dir1": str(save_dir / "v11_teacher_width_guard_valbest_dir1.pt"),
            "dir2": str(save_dir / "v11_teacher_width_guard_valbest_dir2.pt"),
        },
    }
    output = save_dir / "v11_teacher_width_guard_eval_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": eval_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
