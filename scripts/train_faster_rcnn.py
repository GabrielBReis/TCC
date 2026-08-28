#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from tcc_pipeline.config import load_config, model_run_dir, project_root_from_config, resolve_path
from tcc_pipeline.datasets import CocoDetectionTorchDataset, detection_collate
from tcc_pipeline.tracking import (
    log_directory_if_enabled,
    log_metrics_if_enabled,
    log_table_if_enabled,
    save_metadata,
    tracked_run,
)


def build_model(pretrained_path: Path, num_classes_with_bg: int, min_size: int):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=min_size, max_size=min_size)
    payload = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


def compute_loss(model, loader, device, amp_enabled=False):
    model.train()
    total, n = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                losses = model(images, targets)
                loss = sum(losses.values())
            total += float(loss.item())
            n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_ROOT / "configs" / "project.yaml"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    m = cfg["models"]["faster_rcnn"].copy()
    if args.name:
        m["name"] = args.name
    device = torch.device(m.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    seed = int(cfg["project"].get("seed", 42))
    torch.manual_seed(seed)

    aug = m.get("augmentation", {})
    hflip_prob = float(aug.get("horizontal_flip", 0.5)) if aug.get("enabled", False) else 0.0
    train_ds = CocoDetectionTorchDataset(
        resolve_path(root, cfg["paths"]["train_images"]),
        resolve_path(root, cfg["paths"]["train_annotations"]),
        train=True,
        hflip_prob=hflip_prob,
    )
    val_ds = CocoDetectionTorchDataset(
        resolve_path(root, cfg["paths"]["val_images"]), resolve_path(root, cfg["paths"]["val_annotations"]), train=False
    )
    if train_ds.id2label != val_ds.id2label:
        raise ValueError("Mapeamento de classes diferente entre treino e validação.")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(m["batch"]),
        shuffle=True,
        num_workers=int(m["workers"]),
        collate_fn=detection_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(m["batch"]) // 2),
        shuffle=False,
        num_workers=int(m["workers"]),
        collate_fn=detection_collate,
        pin_memory=True,
    )

    model = build_model(resolve_path(root, m["pretrained"]), len(train_ds.id2label) + 1, int(m["imgsz"])).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(m["lr"]),
        momentum=float(m["momentum"]),
        weight_decay=float(m["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(m["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(m.get("amp", True) and device.type == "cuda"))

    run_dir = model_run_dir(root, cfg, "faster_rcnn", m["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"idx0_to_coco_category_id": train_ds.idx0_to_cat, "id2label": train_ds.id2label}
    save_metadata(run_dir / "class_mapping.json", mapping)
    params = {"model": "fasterrcnn_resnet50_fpn", **m, "num_classes": len(train_ds.id2label)}
    save_metadata(run_dir / "run_config.json", params)

    best_val = float("inf")
    bad_epochs = 0
    history = []
    with tracked_run(cfg, m["name"], run_dir, params, model_key="faster_rcnn"):
        for epoch in range(1, int(m["epochs"]) + 1):
            model.train()
            total = 0.0
            t0 = time.time()
            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{m['epochs']}")
            for images, targets in bar:
                images = [x.to(device, non_blocking=True) for x in images]
                targets = [
                    {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in t.items()}
                    for t in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.item())
                bar.set_postfix(loss=f"{loss.item():.4f}")
            scheduler.step()
            train_loss = total / max(len(train_loader), 1)
            val_loss = compute_loss(model, val_loader, device, scaler.is_enabled())
            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            }
            history.append(rec)
            log_metrics_if_enabled(cfg, {k: v for k, v in rec.items() if k != "epoch"}, step=epoch)
            print(rec)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "mapping": mapping, "config": params},
                run_dir / "last.pth",
            )
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "mapping": mapping, "config": params},
                    run_dir / "best.pth",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= int(m.get("patience", 10)):
                    print("Early stopping por val_loss.")
                    break
        log_table_if_enabled(cfg, history, "tables/training_history.json")
        log_directory_if_enabled(cfg, run_dir, "training_outputs")
    print("Melhor checkpoint:", run_dir / "best.pth")


if __name__ == "__main__":
    main()
