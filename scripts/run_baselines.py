#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import subprocess
import sys

from tcc_pipeline.config import load_config, model_run_dir, project_root_from_config, resolve_path


def run(cmd, dry=False):
    print("\n$", " ".join(map(str, cmd)))
    if not dry:
        subprocess.run(list(map(str, cmd)), check=True)


def main():
    ap = argparse.ArgumentParser(description="Orquestra o benchmark baseline completo de forma reiniciável.")
    ap.add_argument("--config", default=str(_ROOT / "configs" / "project.yaml"))
    ap.add_argument("--models", default="yolo,faster,rtdetr")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-prepare-yolo", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    py = sys.executable
    wanted = {x.strip() for x in args.models.split(",") if x.strip()}
    scripts = root / "scripts"
    if not args.skip_download:
        run(
            [
                py,
                scripts / "download_models.py",
                "--out",
                resolve_path(root, cfg["paths"]["pretrained_dir"]),
                "--models",
                args.models,
            ],
            args.dry_run,
        )
    if not args.skip_prepare_yolo and "yolo" in wanted:
        run(
            [
                py,
                scripts / "convert_coco_to_yolo.py",
                "--train-images",
                resolve_path(root, cfg["paths"]["train_images"]),
                "--train-annotations",
                resolve_path(root, cfg["paths"]["train_annotations"]),
                "--val-images",
                resolve_path(root, cfg["paths"]["val_images"]),
                "--val-annotations",
                resolve_path(root, cfg["paths"]["val_annotations"]),
                "--test-images",
                resolve_path(root, cfg["paths"]["test_images"]),
                "--test-annotations",
                resolve_path(root, cfg["paths"]["test_annotations"]),
                "--out",
                resolve_path(root, cfg["paths"]["yolo_dataset_dir"]),
            ],
            args.dry_run,
        )

    metrics_files = []
    test_images = resolve_path(root, cfg["paths"]["test_images"])
    test_gt = resolve_path(root, cfg["paths"]["test_annotations"])
    pred_conf = cfg.get("prediction", {}).get("confidence_threshold", 0.001)
    max_detections = cfg.get("prediction", {}).get("max_detections", 300)
    warmup = cfg.get("prediction", {}).get("benchmark_warmup", 10)
    ev = cfg.get("evaluation", {})
    rel = ev.get("relative_size", {})
    common_eval = [
        "--gt",
        test_gt,
        "--conf",
        ev.get("f1_confidence_threshold", 0.25),
        "--iou",
        ev.get("iou_threshold", 0.5),
        "--small-max",
        rel.get("small_max", 0.01),
        "--medium-max",
        rel.get("medium_max", 0.05),
    ]

    if "yolo" in wanted:
        m = cfg["models"]["yolo"]
        rd = model_run_dir(root, cfg, "yolo", m["name"])
        pred = rd / "predictions.json"
        met = rd / "metrics.json"
        run([py, scripts / "train_yolo.py", "--config", args.config], args.dry_run)
        run(
            [
                py,
                scripts / "predict_yolo.py",
                "--weights",
                rd / "weights" / "best.pt",
                "--images",
                test_images,
                "--annotations",
                test_gt,
                "--mapping",
                resolve_path(root, cfg["paths"]["yolo_dataset_dir"]) / "category_mapping.json",
                "--out",
                pred,
                "--imgsz",
                m["imgsz"],
                "--conf",
                pred_conf,
                "--device",
                m["device"],
                "--max-det",
                max_detections,
                "--warmup",
                warmup,
            ],
            args.dry_run,
        )
        run([py, scripts / "evaluate.py", *common_eval, "--pred", pred, "--out", met], args.dry_run)
        metrics_files.append(met)
    if "faster" in wanted or "faster_rcnn" in wanted:
        m = cfg["models"]["faster_rcnn"]
        rd = model_run_dir(root, cfg, "faster_rcnn", m["name"])
        pred = rd / "predictions.json"
        met = rd / "metrics.json"
        run([py, scripts / "train_faster_rcnn.py", "--config", args.config], args.dry_run)
        run(
            [
                py,
                scripts / "predict_faster_rcnn.py",
                "--checkpoint",
                rd / "best.pth",
                "--images",
                test_images,
                "--annotations",
                test_gt,
                "--out",
                pred,
                "--conf",
                pred_conf,
                "--device",
                m["device"],
                "--warmup",
                warmup,
            ],
            args.dry_run,
        )
        run([py, scripts / "evaluate.py", *common_eval, "--pred", pred, "--out", met], args.dry_run)
        metrics_files.append(met)
    if "rtdetr" in wanted:
        m = cfg["models"]["rtdetr"]
        rd = model_run_dir(root, cfg, "rtdetr", m["name"])
        pred = rd / "predictions.json"
        met = rd / "metrics.json"
        run([py, scripts / "train_rtdetr.py", "--config", args.config], args.dry_run)
        run(
            [
                py,
                scripts / "predict_rtdetr.py",
                "--model",
                rd / "best_model",
                "--mapping",
                rd / "class_mapping.json",
                "--images",
                test_images,
                "--annotations",
                test_gt,
                "--out",
                pred,
                "--conf",
                pred_conf,
                "--warmup",
                warmup,
            ],
            args.dry_run,
        )
        run([py, scripts / "evaluate.py", *common_eval, "--pred", pred, "--out", met], args.dry_run)
        metrics_files.append(met)
    if metrics_files:
        run(
            [
                py,
                scripts / "compare_experiments.py",
                "--metrics",
                *metrics_files,
                "--out",
                resolve_path(root, cfg["paths"]["reports_dir"]) / "comparison",
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
