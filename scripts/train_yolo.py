#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import csv
from pathlib import Path

from tcc_pipeline.config import load_config, model_run_dir, project_root_from_config, resolve_path
from tcc_pipeline.tracking import (
    log_directory_if_enabled,
    log_metrics_if_enabled,
    log_table_if_enabled,
    save_metadata,
    tracked_run,
)


def retain_best_checkpoint(run_dir: Path) -> Path:
    """Mantém somente best.pt e remove last/checkpoints periódicos."""
    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"O treinamento não produziu o checkpoint esperado: {best}")
    for checkpoint in weights_dir.iterdir():
        if checkpoint.is_file() and checkpoint != best:
            checkpoint.unlink()
    return best


def resolve_model_source(root: Path, value: str | Path) -> str:
    """Resolve pesos locais, mas preserva nomes de arquiteturas do Ultralytics."""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    local = (root / candidate).resolve()
    return str(local) if local.exists() else str(value)


def train_without_ultralytics_mlflow(model, **kwargs):
    """Evita autolog duplicado do Ultralytics; o projeto controla o MLflow."""
    from ultralytics.utils import SETTINGS

    previous = SETTINGS.get("mlflow", False)
    # Ignora a persistência do SettingsManager: a mudança vale somente para
    # este processo e é restaurada mesmo se o treinamento falhar.
    dict.__setitem__(SETTINGS, "mlflow", False)
    try:
        return model.train(**kwargs)
    finally:
        dict.__setitem__(SETTINGS, "mlflow", previous)


def log_yolo_results(cfg, run_dir):
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    for index, row in enumerate(rows, start=1):
        step = int(float(row.get("epoch", index)))
        log_metrics_if_enabled(cfg, {key: value for key, value in row.items() if key != "epoch"}, step=step)
    log_table_if_enabled(cfg, rows, "tables/training_history.json")
    log_directory_if_enabled(cfg, run_dir, "training_outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_ROOT / "configs" / "project.yaml"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    m = cfg["models"]["yolo"].copy()
    if args.name:
        m["name"] = args.name
    from ultralytics import YOLO

    run_dir = model_run_dir(root, cfg, "yolo", m["name"])
    yolo_yaml = resolve_path(root, cfg["paths"]["yolo_dataset_dir"]) / "dataset.yaml"
    weights = resolve_model_source(root, m["pretrained"])
    params = {"model": "yolo11n", **m, "dataset": str(yolo_yaml)}
    save_metadata(run_dir / "run_config.json", params)

    with tracked_run(cfg, m["name"], run_dir, params, model_key="yolo"):
        model = YOLO(str(weights))
        aug = m.get("augmentation", {})
        augmentation_enabled = bool(aug.get("enabled", False))
        train_without_ultralytics_mlflow(
            model,
            data=str(yolo_yaml),
            project=str(run_dir.parent),
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
            momentum=float(m.get("momentum", 0.937)),
            weight_decay=float(m.get("weight_decay", 0.0005)),
            warmup_epochs=float(m.get("warmup_epochs", 3.0)),
            cos_lr=bool(m.get("cos_lr", False)),
            multi_scale=m.get("multi_scale", 0.0),
            patience=int(m.get("patience", 20)),
            amp=bool(m.get("amp", True)),
            save=True,
            save_period=int(m.get("save_period", -1)),
            seed=int(cfg["project"].get("seed", 42)),
            deterministic=True,
            plots=True,
            verbose=True,
            # Baseline controlado: as augmentations ficam desligadas por padrão.
            # Experimentos dedicados devem habilitá-las explicitamente no YAML.
            hsv_h=float(aug.get("hue", 0.0)) if augmentation_enabled else 0.0,
            hsv_s=float(aug.get("saturation", 0.0)) if augmentation_enabled else 0.0,
            hsv_v=float(aug.get("brightness", 0.0)) if augmentation_enabled else 0.0,
            degrees=float(aug.get("rotation_degrees", 0.0)) if augmentation_enabled else 0.0,
            translate=float(aug.get("translate", 0.0)) if augmentation_enabled else 0.0,
            scale=float(aug.get("scale", 0.0)) if augmentation_enabled else 0.0,
            shear=float(aug.get("shear_degrees", 0.0)) if augmentation_enabled else 0.0,
            perspective=float(aug.get("perspective", 0.0)) if augmentation_enabled else 0.0,
            flipud=float(aug.get("vertical_flip", 0.0)) if augmentation_enabled else 0.0,
            fliplr=float(aug.get("horizontal_flip", 0.0)) if augmentation_enabled else 0.0,
            mosaic=float(aug.get("mosaic", 0.0)) if augmentation_enabled else 0.0,
            close_mosaic=int(aug.get("close_mosaic", 0)) if augmentation_enabled else 0,
            mixup=float(aug.get("mixup", 0.0)) if augmentation_enabled else 0.0,
            copy_paste=float(aug.get("copy_paste", 0.0)) if augmentation_enabled else 0.0,
            copy_paste_mode=str(aug.get("copy_paste_mode", "flip")),
        )
        best_checkpoint = retain_best_checkpoint(run_dir)
        save_metadata(
            run_dir / "training_complete.json",
            {
                "status": "completed",
                "best_checkpoint": str(best_checkpoint.relative_to(run_dir)),
                "checkpoint_retention": "best_only",
            },
        )
        log_yolo_results(cfg, run_dir)
    print("Treino concluído:", run_dir)
    print("Melhor checkpoint esperado:", run_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
