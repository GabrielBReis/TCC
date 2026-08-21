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
from pathlib import Path

from PIL import Image

from tcc_pipeline.benchmarking import save_benchmark, timed_inference
from tcc_pipeline.coco import load_coco_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--metrics-out")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    coco = load_coco_json(args.annotations)
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))["yolo_id_to_coco_category_id"]
    mapping = {int(k): int(v) for k, v in mapping.items()}
    preds = []
    durations = []
    for img in coco["images"]:
        p = Path(args.images) / img["file_name"]
        image = Image.open(p).convert("RGB")
        with timed_inference(model.device, durations):
            result = model.predict(
                image, imgsz=args.imgsz, conf=args.conf, device=args.device, max_det=args.max_det, verbose=False
            )[0]
        if result.boxes is None:
            continue
        for xyxy, score, cls in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
        ):
            x1, y1, x2, y2 = map(float, xyxy)
            preds.append(
                {
                    "image_id": int(img["id"]),
                    "category_id": mapping[int(cls)],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    save_json(preds, args.out)
    metrics_out = args.metrics_out or str(Path(args.out).with_name("inference_metrics.json"))
    save_benchmark(
        metrics_out, durations, sum(parameter.numel() for parameter in model.model.parameters()), args.warmup
    )
    print(f"Predições: {len(preds)} -> {args.out}")


if __name__ == "__main__":
    main()
