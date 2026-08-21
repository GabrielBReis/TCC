#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse

from tcc_pipeline.config import load_config, project_root_from_config, resolve_path
from tcc_pipeline.tracking import log_artifact_if_enabled, save_metadata, tracked_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    m = cfg["models"]["yolo"].copy()
    if args.name:
        m["name"] = args.name
    from ultralytics import YOLO

    runs_dir = resolve_path(root, cfg["paths"]["runs_dir"])
    run_dir = runs_dir / "yolo" / m["name"]
    yolo_yaml = resolve_path(root, cfg["paths"]["yolo_dataset_dir"]) / "dataset.yaml"
    weights = resolve_path(root, m["pretrained"])
    params = {"model": "yolo11n", **m, "dataset": str(yolo_yaml)}
    save_metadata(run_dir / "run_config.json", params)

    with tracked_run(cfg, m["name"], run_dir, params):
        model = YOLO(str(weights))
        aug = m.get("augmentation", {})
        augmentation_enabled = bool(aug.get("enabled", False))
        model.train(
            data=str(yolo_yaml),
            project=str(runs_dir / "yolo"),
            name=m["name"],
            exist_ok=True,
            epochs=int(m["epochs"]),
            imgsz=int(m["imgsz"]),
            batch=m["batch"],
            workers=int(m["workers"]),
            device=m["device"],
            optimizer=m.get("optimizer", "auto"),
            lr0=float(m.get("lr0", 0.01)),
            lrf=float(m.get("lrf", 0.01)),
            weight_decay=float(m.get("weight_decay", 0.0005)),
            patience=int(m.get("patience", 20)),
            seed=int(cfg["project"].get("seed", 42)),
            deterministic=True,
            plots=True,
            verbose=True,
            # Baseline controlado: as augmentations ficam desligadas por padrão.
            # Experimentos dedicados devem habilitá-las explicitamente no YAML.
            hsv_h=0.015 if augmentation_enabled else 0.0,
            hsv_s=0.7 if augmentation_enabled else 0.0,
            hsv_v=0.4 if augmentation_enabled else 0.0,
            degrees=0.0,
            translate=0.1 if augmentation_enabled else 0.0,
            scale=0.5 if augmentation_enabled else 0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=float(aug.get("horizontal_flip", 0.5)) if augmentation_enabled else 0.0,
            mosaic=1.0 if augmentation_enabled else 0.0,
            mixup=0.0,
            copy_paste=0.0,
        )
        for p in (run_dir / "results.csv", run_dir / "results.png"):
            log_artifact_if_enabled(cfg, p)
    print("Treino concluído:", run_dir)
    print("Melhor checkpoint esperado:", run_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
