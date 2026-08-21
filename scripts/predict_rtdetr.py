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

import torch
from PIL import Image

from tcc_pipeline.benchmarking import save_benchmark, timed_inference
from tcc_pipeline.coco import load_coco_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--metrics-out")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    processor = RTDetrImageProcessor.from_pretrained(args.model)
    model = RTDetrForObjectDetection.from_pretrained(args.model).to(device).eval()
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))["idx0_to_coco_category_id"]
    mapping = {int(k): int(v) for k, v in mapping.items()}
    coco = load_coco_json(args.annotations)
    preds = []
    durations = []
    with torch.no_grad():
        for img in coco["images"]:
            image = Image.open(Path(args.images) / img["file_name"]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with timed_inference(device, durations):
                outputs = model(**inputs)
                target_sizes = torch.tensor([(image.height, image.width)], device=device)
                r = processor.post_process_object_detection(outputs, threshold=args.conf, target_sizes=target_sizes)[0]
            for box, score, label in zip(
                r["boxes"].cpu().tolist(), r["scores"].cpu().tolist(), r["labels"].cpu().tolist()
            ):
                x1, y1, x2, y2 = map(float, box)
                if int(label) not in mapping:
                    continue
                preds.append(
                    {
                        "image_id": int(img["id"]),
                        "category_id": mapping[int(label)],
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
