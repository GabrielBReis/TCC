#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from tcc_pipeline.benchmarking import save_benchmark, timed_inference
from tcc_pipeline.coco import load_coco_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--metrics-out")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu")
    mapping = payload["mapping"]
    idx0_to_cat = {int(k): int(v) for k, v in mapping["idx0_to_coco_category_id"].items()}
    num_classes = len(mapping["id2label"]) + 1
    imgsz = int(payload.get("config", {}).get("imgsz", 640))
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=imgsz, max_size=imgsz)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    coco = load_coco_json(args.annotations)
    preds = []
    durations = []
    with torch.no_grad():
        for img in coco["images"]:
            image = Image.open(Path(args.images) / img["file_name"]).convert("RGB")
            tensor = F.to_tensor(image).to(device)
            with timed_inference(device, durations):
                out = model([tensor])[0]
            for box, score, label in zip(
                out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()
            ):
                if score < args.conf:
                    continue
                x1, y1, x2, y2 = map(float, box)
                idx0 = int(label) - 1
                if idx0 not in idx0_to_cat:
                    continue
                preds.append(
                    {
                        "image_id": int(img["id"]),
                        "category_id": idx0_to_cat[idx0],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    save_json(preds, args.out)
    save_benchmark(
        args.metrics_out or str(Path(args.out).with_name("inference_metrics.json")),
        durations,
        sum(p.numel() for p in model.parameters()),
        args.warmup,
    )
    print(f"Predições: {len(preds)} -> {args.out}")


if __name__ == "__main__":
    main()
