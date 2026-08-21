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
import os
import shutil
from pathlib import Path


def download_yolo(out_dir: Path):
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / "yolo11n.pt"
    old = Path.cwd()
    try:
        os.chdir(out_dir)
        model = YOLO("yolo11n.pt")
        src = Path(getattr(model, "ckpt_path", expected)).resolve()
    finally:
        os.chdir(old)
    if not expected.exists() and src.exists():
        shutil.copy2(src, expected)
    if not expected.exists():
        raise RuntimeError("YOLO11n foi carregado, mas o checkpoint não foi localizado.")
    return expected


def download_faster_rcnn(out_dir: Path):
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "fasterrcnn_resnet50_fpn_coco.pth"
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    torch.save({"state_dict": model.state_dict(), "weights_name": str(weights)}, dst)
    return dst


def download_rtdetr(out_dir: Path):
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    dst = out_dir / "rtdetr_r18vd"
    dst.mkdir(parents=True, exist_ok=True)
    model_id = "PekingU/rtdetr_r18vd"
    processor = RTDetrImageProcessor.from_pretrained(model_id)
    model = RTDetrForObjectDetection.from_pretrained(model_id)
    processor.save_pretrained(dst)
    model.save_pretrained(dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description="Baixa os checkpoints-base usados no benchmark.")
    ap.add_argument("--out", default="models/pretrained")
    ap.add_argument("--models", default="yolo,faster,rtdetr", help="Lista separada por vírgula")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    wanted = {x.strip().lower() for x in args.models.split(",") if x.strip()}
    manifest_path = out / "download_manifest.json"
    if manifest_path.exists():
        results = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        results = {}
    if "yolo" in wanted:
        results["yolo11n"] = str(download_yolo(out))
    if "faster" in wanted or "faster_rcnn" in wanted:
        results["faster_rcnn_r50_fpn"] = str(download_faster_rcnn(out))
    if "rtdetr" in wanted:
        results["rtdetr_r18vd"] = str(download_rtdetr(out))
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
