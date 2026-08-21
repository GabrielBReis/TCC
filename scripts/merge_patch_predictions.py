#!/usr/bin/env python3
"""Project patch detections to source images and suppress overlap duplicates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.coco import load_coco_json, save_json
from tcc_pipeline.geometry import box_iou_xywh


def non_max_suppression(detections, iou_threshold, max_detections):
    kept = []
    by_class = defaultdict(list)
    for detection in detections:
        by_class[int(detection["category_id"])].append(detection)
    for class_detections in by_class.values():
        pending = sorted(class_detections, key=lambda item: float(item["score"]), reverse=True)
        while pending and len(kept) < max_detections:
            best = pending.pop(0)
            kept.append(best)
            pending = [item for item in pending if box_iou_xywh(best["bbox"], item["bbox"]) < iou_threshold]
    return sorted(kept, key=lambda item: float(item["score"]), reverse=True)[:max_detections]


def merge_predictions(patch_coco, predictions, iou_threshold=0.5, max_detections=300):
    patch_info = {int(image["id"]): image for image in patch_coco["images"]}
    projected = defaultdict(list)
    for prediction in predictions:
        info = patch_info.get(int(prediction["image_id"]))
        if info is None:
            raise ValueError(f"Prediction references unknown patch image_id={prediction['image_id']}")
        x, y, width, height = map(float, prediction["bbox"])
        projected[int(info["source_image_id"])].append(
            {
                "image_id": int(info["source_image_id"]),
                "category_id": int(prediction["category_id"]),
                "bbox": [x + float(info["patch_x"]), y + float(info["patch_y"]), width, height],
                "score": float(prediction["score"]),
            }
        )
    return [
        item
        for image_predictions in projected.values()
        for item in non_max_suppression(image_predictions, iou_threshold, max_detections)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=300)
    args = parser.parse_args()
    patch_coco = load_coco_json(args.patch_annotations)
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    merged = merge_predictions(patch_coco, predictions, args.iou, args.max_detections)
    save_json(merged, args.out)
    print(f"Merged predictions: {len(merged)} -> {args.out}")


if __name__ == "__main__":
    main()
