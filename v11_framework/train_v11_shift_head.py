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
from torch.utils.data import DataLoader, Dataset

from models_v11 import V11TeacherWidthShift, checkpoint_state
from train_v11_teacher_width_guard import (
    evaluate_split,
    find_checkpoint,
    limited_pairs,
    load_shift_csv,
    local_moment,
    pair_selection_summary,
    safe_norm_batch,
    split_csv,
    write_train_log,
)
from v11_utils import load_config, resolve_device, scenarios_from_manifest, split_from_config


class V11PairShiftDataset(Dataset):
    def __init__(self, scenario: dict[str, Any], pair_ids: list[int], num_pairs: int = 8640) -> None:
        self.scenario = dict(scenario)
        self.pair_ids = [int(pair) for pair in pair_ids]
        self.num_pairs = int(num_pairs)
        self.input_shifts = load_shift_csv(str(self.scenario["input_shift_csv"]))
        self.target_shifts = load_shift_csv(str(self.scenario["target_shift_csv"]))

    def __len__(self) -> int:
        return len(self.pair_ids)

    @staticmethod
    def _safe_norm(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        peak = torch.max(x)
        if float(peak.detach().cpu().item()) > 1e-8:
            return x / (peak + 1e-8)
        return torch.zeros_like(x)

    def _load_tensor(self, directory: str, hist_id: int) -> torch.Tensor:
        return self._safe_norm(torch.load(Path(directory) / f"hist_{int(hist_id):05d}.pt", map_location="cpu"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        pair_id = self.pair_ids[index]
        id1 = int(pair_id)
        id2 = id1 + self.num_pairs
        target1 = self._load_tensor(str(self.scenario["target_dir"]), id1)
        input1 = self._load_tensor(str(self.scenario["input_dir"]), id1)
        target2 = self._load_tensor(str(self.scenario["target_dir"]), id2)
        input2 = self._load_tensor(str(self.scenario["input_dir"]), id2)
        return (
            target1,
            input1,
            target2,
            input2,
            torch.tensor(pair_id, dtype=torch.float32),
            torch.tensor(float(self.input_shifts.get(id1, 0.0)), dtype=torch.float32),
            torch.tensor(float(self.input_shifts.get(id2, 0.0)), dtype=torch.float32),
            torch.tensor(float(self.target_shifts.get(id1, 0.0)), dtype=torch.float32),
            torch.tensor(float(self.target_shifts.get(id2, 0.0)), dtype=torch.float32),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V11 single-histogram shift correction head.")
    parser.add_argument("--config", type=Path, default=Path("v11_framework/configs/v11_50km_0p8_shift_head.yaml"))
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
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cfg_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(config.get(name, {}))


def tdev_surrogate_torch(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 3:
        return values.new_zeros(())
    d2 = values[2:] - 2.0 * values[1:-1] + values[:-2]
    return torch.mean(d2 * d2) / 6.0


def make_models(config: dict[str, Any], device: torch.device) -> tuple[dict[int, V11TeacherWidthShift], dict[int, Path]]:
    model_cfg = cfg_section(config, "model")
    teacher_cfg = cfg_section(config, "teacher")
    teacher_root = Path(str(teacher_cfg.get("root", "checkpoints/v165_teacher")))
    teacher_paths = {
        1: find_checkpoint(teacher_root, str(teacher_cfg.get("dir1", "adaptivecheck_compensator_dir1.pt"))),
        2: find_checkpoint(teacher_root, str(teacher_cfg.get("dir2", "adaptivecheck_compensator_dir2.pt"))),
    }
    models = {
        direction: V11TeacherWidthShift(
            teacher_checkpoint=teacher_paths[direction],
            input_dim=int(config.get("input_dim", 65536)),
            physics_mode=str(teacher_cfg.get("physics_mode", "train")),
            sigma_init=float(model_cfg.get("sigma_init", 22.0)),
            sigma_min=float(model_cfg.get("sigma_min", 1.0)),
            sigma_max=float(model_cfg.get("sigma_max", 40.0)),
            mix_init=float(model_cfg.get("mix_init", 0.925)),
            mix_min=float(model_cfg.get("mix_min", 0.0)),
            mix_max=float(model_cfg.get("mix_max", 1.0)),
            kernel_radius=int(model_cfg.get("kernel_radius", 96)),
            area_preserve=bool(model_cfg.get("area_preserve", True)),
            max_abs_shift=float(model_cfg.get("max_abs_shift", 4.0)),
            shift_channels=int(model_cfg.get("shift_channels", 24)),
            shift_hidden=int(model_cfg.get("shift_hidden", 48)),
            freeze_guard=bool(model_cfg.get("freeze_guard", True)),
        ).to(device)
        for direction in (1, 2)
    }
    warm_cfg = cfg_section(config, "warm_start")
    guard_dir = Path(str(warm_cfg.get("guard_dir", "")))
    if str(guard_dir).strip():
        for direction in (1, 2):
            checkpoint_path = guard_dir / f"v11_teacher_width_guard_valbest_dir{direction}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state = checkpoint.get("model_state", checkpoint)
            missing, unexpected = models[direction].load_state_dict(state, strict=False)
            missing = [key for key in missing if not key.startswith("shift_head.")]
            if missing or unexpected:
                print(
                    f"warm-start direction={direction} missing={missing[:4]} unexpected={unexpected[:4]}",
                    flush=True,
                )
    return models, teacher_paths


def clock_loss(
    out1: torch.Tensor,
    out2: torch.Tensor,
    target1: torch.Tensor,
    target2: torch.Tensor,
    aux1: dict[str, torch.Tensor],
    aux2: dict[str, torch.Tensor],
    pair_id: torch.Tensor,
    input_shift1: torch.Tensor,
    input_shift2: torch.Tensor,
    target_shift1: torch.Tensor,
    target_shift2: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg_section(config, "loss")
    center_half = int(loss_cfg.get("clock_center_half", 180))
    center_norm = float(loss_cfg.get("center_norm", 100.0))
    target_tdev_ps = float(loss_cfg.get("target_tdev_ps", 2.0))
    guarded1 = aux1["guarded"].detach()
    guarded2 = aux2["guarded"].detach()
    seed1 = torch.argmax(torch.clamp(guarded1, min=0.0), dim=1).to(dtype=out1.dtype)
    seed2 = torch.argmax(torch.clamp(guarded2, min=0.0), dim=1).to(dtype=out2.dtype)
    target_seed1 = torch.argmax(torch.clamp(target1, min=0.0), dim=1).to(dtype=target1.dtype)
    target_seed2 = torch.argmax(torch.clamp(target2, min=0.0), dim=1).to(dtype=target2.dtype)
    center1 = local_moment(out1, seed1, center_half)["center"]
    center2 = local_moment(out2, seed2, center_half)["center"]
    guarded_center1 = local_moment(guarded1, seed1, center_half)["center"].detach()
    guarded_center2 = local_moment(guarded2, seed2, center_half)["center"].detach()
    target_center1 = local_moment(target1, target_seed1, center_half)["center"].detach()
    target_center2 = local_moment(target2, target_seed2, center_half)["center"].detach()
    pred_clock = 0.5 * ((center1 - input_shift1) - (center2 - input_shift2))
    guarded_clock = 0.5 * ((guarded_center1 - input_shift1) - (guarded_center2 - input_shift2))
    target_clock = 0.5 * ((target_center1 - target_shift1) - (target_center2 - target_shift2))

    order = torch.argsort(pair_id)
    pred = pred_clock[order]
    target = target_clock[order]
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    clock_mse = torch.mean(((pred_centered - target_centered) / center_norm) ** 2)
    pred_tdev = torch.sqrt(torch.clamp(tdev_surrogate_torch(pred / center_norm), min=0.0) + 1e-12)
    target_tdev = torch.sqrt(torch.clamp(tdev_surrogate_torch(target / center_norm), min=0.0) + 1e-12)
    abs_target = pred.new_tensor(target_tdev_ps / center_norm)
    tdev_abs = torch.relu(pred_tdev - abs_target) ** 2
    tdev_match = torch.relu(pred_tdev - target_tdev * float(loss_cfg.get("target_tdev_ratio_margin", 1.25))) ** 2

    shift1 = aux1["delta_shift"]
    shift2 = aux2["delta_shift"]
    residual_clock = (target_clock - guarded_clock).detach()
    pseudo_pair_shift = residual_clock - torch.mean(residual_clock)
    pseudo_clip = float(loss_cfg.get("pseudo_shift_clip_ps", 6.0))
    pseudo_pair_shift = torch.clamp(pseudo_pair_shift, min=-pseudo_clip, max=pseudo_clip)
    shift_norm = max(float(loss_cfg.get("shift_norm", 4.0)), 1e-6)
    pseudo_shift = torch.mean(((shift1 - pseudo_pair_shift) / shift_norm) ** 2 + ((shift2 + pseudo_pair_shift) / shift_norm) ** 2)
    shift_l2 = torch.mean(shift1 * shift1 + shift2 * shift2)
    shift_mean = torch.mean(0.5 * (shift1 + shift2))
    if shift1.numel() >= 3:
        shift_smooth = tdev_surrogate_torch(shift1[order]) + tdev_surrogate_torch(shift2[order])
    else:
        shift_smooth = shift_l2.new_zeros(())
    guarded_mse = torch.mean((out1 - guarded1) ** 2 + (out2 - guarded2) ** 2)

    total = (
        float(loss_cfg.get("w_clock_mse", 600.0)) * clock_mse
        + float(loss_cfg.get("w_tdev_abs", 5000.0)) * tdev_abs
        + float(loss_cfg.get("w_tdev_match", 1500.0)) * tdev_match
        + float(loss_cfg.get("w_pseudo_shift", 0.0)) * pseudo_shift
        + float(loss_cfg.get("w_shift_l2", 0.01)) * shift_l2
        + float(loss_cfg.get("w_shift_mean", 0.5)) * shift_mean * shift_mean
        + float(loss_cfg.get("w_shift_smooth", 0.5)) * shift_smooth
        + float(loss_cfg.get("w_guarded_mse", 2.0)) * guarded_mse
    )
    return total, {
        "clock_mse": clock_mse,
        "tdev_abs": tdev_abs,
        "tdev_match": tdev_match,
        "pseudo_shift": pseudo_shift,
        "pred_tdev_ps_batch": pred_tdev * center_norm,
        "target_tdev_ps_batch": target_tdev * center_norm,
        "pseudo_pair_shift_std": torch.std(pseudo_pair_shift) if pseudo_pair_shift.numel() > 1 else pseudo_pair_shift.new_zeros(()),
        "shift_l2": shift_l2,
        "shift_mean": shift_mean,
        "shift_smooth": shift_smooth,
        "guarded_mse": guarded_mse,
        "shift1_mean": torch.mean(shift1),
        "shift2_mean": torch.mean(shift2),
        "shift1_std": torch.std(shift1) if shift1.numel() > 1 else shift1.new_zeros(()),
        "shift2_std": torch.std(shift2) if shift2.numel() > 1 else shift2.new_zeros(()),
    }


def run_epoch(
    models: dict[int, V11TeacherWidthShift],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    device: torch.device,
    train: bool,
    epoch: int,
    log_every_batches: int,
) -> dict[str, float]:
    for model in models.values():
        model.train(mode=train)
    sums: dict[str, float] = {}
    batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            (
                target1,
                input1,
                target2,
                input2,
                pair_id,
                input_shift1,
                input_shift2,
                target_shift1,
                target_shift2,
            ) = batch
            target1 = safe_norm_batch(target1.to(device))
            input1 = safe_norm_batch(input1.to(device))
            target2 = safe_norm_batch(target2.to(device))
            input2 = safe_norm_batch(input2.to(device))
            pair_id = pair_id.to(device=device, dtype=torch.float32)
            input_shift1 = input_shift1.to(device=device, dtype=torch.float32)
            input_shift2 = input_shift2.to(device=device, dtype=torch.float32)
            target_shift1 = target_shift1.to(device=device, dtype=torch.float32)
            target_shift2 = target_shift2.to(device=device, dtype=torch.float32)
            out1, aux1 = models[1](input1)
            out2, aux2 = models[2](input2)
            loss, parts = clock_loss(
                out1,
                out2,
                target1,
                target2,
                aux1,
                aux2,
                pair_id,
                input_shift1,
                input_shift2,
                target_shift1,
                target_shift2,
                config,
            )
            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required in train mode")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batches += 1
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu().item())
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
            if train and log_every_batches > 0 and batch_idx % int(log_every_batches) == 0:
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(loader)} loss={float(loss.detach().cpu().item()):.6g} "
                    f"batch_tdev={float(parts['pred_tdev_ps_batch'].detach().cpu().item()):.3f} "
                    f"shift1={float(parts['shift1_mean'].detach().cpu().item()):.3f} "
                    f"shift2={float(parts['shift2_mean'].detach().cpu().item()):.3f}",
                    flush=True,
                )
    return {key: value / max(batches, 1) for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    model: V11TeacherWidthShift,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    direction: int,
) -> None:
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
    train_cfg = cfg_section(config, "train")
    save_dir = Path(str(train_cfg.get("save_dir", "v11_framework/artifacts/v11_50km_0p8_shift_head_v1")))
    seed = int(args.seed if args.seed >= 0 else train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    scenarios, source_manifest = scenarios_from_manifest(config, args)
    if len(scenarios) != 1:
        raise ValueError(f"V11 shift head currently expects one scenario, got {len(scenarios)}")
    scenario = scenarios[0]
    usable_pairs = int(scenario["usable_pairs"])
    split = split_from_config(config)
    train_pairs = limited_pairs(split.train_pairs, usable_pairs, int(args.max_train_pairs))
    val_pairs = limited_pairs(split.val_pairs, usable_pairs, int(args.max_val_pairs))
    batch_size = int(args.batch_size if args.batch_size > 0 else train_cfg.get("batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers >= 0 else train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        V11PairShiftDataset(scenario, train_pairs, num_pairs=int(config.get("num_pairs", 8640))),
        batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle", False)),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        V11PairShiftDataset(scenario, val_pairs, num_pairs=int(config.get("num_pairs", 8640))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    models, teacher_paths = make_models(config, device)
    params = [parameter for model in models.values() for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr if args.lr > 0 else train_cfg.get("lr", 1.0e-3)))
    epochs = int(args.epochs if args.epochs > 0 else train_cfg.get("epochs", 12))
    dry_report = {
        "framework": "v11",
        "variant": str(config.get("variant", "shift_head")),
        "scenario": scenario,
        "teacher_paths": {str(key): str(value) for key, value in teacher_paths.items()},
        "train_pairs": pair_selection_summary([pair - 1 for pair in train_pairs]),
        "val_pairs": pair_selection_summary([pair - 1 for pair in val_pairs]),
        "trainable_parameters": int(sum(parameter.numel() for parameter in params)),
        "device": str(device),
        "save_dir": str(save_dir),
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, ensure_ascii=False))
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config_used.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"=== V11 shift-head train start: device={device}, scenario={scenario['label']}, "
        f"train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)}, batch_size={batch_size}, "
        f"epochs={epochs}, trainable={dry_report['trainable_parameters']} ===",
        flush=True,
    )
    history: list[dict[str, float]] = []
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(models, train_loader, optimizer, config, device, True, epoch, int(args.log_every_batches))
        val_stats = run_epoch(models, val_loader, None, config, device, False, epoch, 0)
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_stats.items()})
        row.update({f"val_{key}": value for key, value in val_stats.items()})
        history.append(row)
        for direction in (1, 2):
            save_checkpoint(save_dir / f"v11_shift_head_latest_dir{direction}.pt", models[direction], optimizer, config, direction)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            for direction in (1, 2):
                save_checkpoint(save_dir / f"v11_shift_head_valbest_dir{direction}.pt", models[direction], optimizer, config, direction)
        print(
            f"epoch={epoch} train_loss={train_stats['loss']:.6g} val_loss={val_stats['loss']:.6g} "
            f"val_batch_tdev={val_stats['pred_tdev_ps_batch']:.3f} "
            f"shift1={val_stats['shift1_mean']:.3f}/{val_stats['shift1_std']:.3f} "
            f"shift2={val_stats['shift2_mean']:.3f}/{val_stats['shift2_std']:.3f}",
            flush=True,
        )

    write_train_log(save_dir / "train_log.csv", history)
    eval_rows: list[dict[str, Any]] = []
    for split_name in split_csv(args.eval_splits):
        if split_name == "train":
            pairs = limited_pairs(split.train_pairs, usable_pairs, int(args.max_eval_pairs))
        elif split_name == "val":
            pairs = limited_pairs(split.val_pairs, usable_pairs, int(args.max_eval_pairs))
        elif split_name == "test":
            pairs = limited_pairs(split.test_pairs, usable_pairs, int(args.max_eval_pairs))
        else:
            raise ValueError(f"Unsupported eval split: {split_name!r}")
        result = evaluate_split(models, scenario, split_name, pairs, config, device)
        eval_rows.append(result)
        print(
            f"eval split={split_name} count={result['count']:.0f} comp_tdev={result['comp_tdev10']:.4f} "
            f"target_tdev={result['target_tdev10']:.4f} comp_fwhm={result['comp_fwhm']:.3f} "
            f"target_fwhm={result['target_fwhm']:.3f}",
            flush=True,
        )
    summary = {
        "framework": "v11",
        "variant": str(config.get("variant", "shift_head")),
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
            "dir1": str(save_dir / "v11_shift_head_valbest_dir1.pt"),
            "dir2": str(save_dir / "v11_shift_head_valbest_dir2.pt"),
        },
    }
    output = save_dir / "v11_shift_head_eval_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": eval_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
