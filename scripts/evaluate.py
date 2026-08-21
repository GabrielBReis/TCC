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
from collections import defaultdict
from pathlib import Path

import numpy as np

from tcc_pipeline.coco import images_by_id, load_coco_json, save_json
from tcc_pipeline.geometry import box_iou_xywh, relative_area, size_bin


def coco_metrics(gt_path, pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    preds = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if not preds:
        return {
            k: 0.0
            for k in [
                "coco_map5095",
                "coco_map50",
                "coco_map75",
                "coco_ap_small",
                "coco_ap_medium",
                "coco_ap_large",
                "coco_ar100",
            ]
        }
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {
        "coco_map5095": float(s[0]),
        "coco_map50": float(s[1]),
        "coco_map75": float(s[2]),
        "coco_ap_small": float(s[3]),
        "coco_ap_medium": float(s[4]),
        "coco_ap_large": float(s[5]),
        "coco_ar100": float(s[8]),
    }


def _match_class(gt, preds, images, conf, iou, small_max, medium_max, wanted_bin, category_id):
    """Match by GT scale and ignore out-of-range objects, similarly to COCOeval."""
    gt_by = defaultdict(list)
    pred_by = defaultdict(list)
    for ann in gt["annotations"]:
        if int(ann.get("iscrowd", 0)) or int(ann["category_id"]) != category_id:
            continue
        img = images[int(ann["image_id"])]
        ann_bin = size_bin(relative_area(ann["bbox"], img["width"], img["height"]), small_max, medium_max)
        gt_by[int(ann["image_id"])].append((ann, wanted_bin is not None and ann_bin != wanted_bin))
    for p in preds:
        if float(p["score"]) >= conf and int(p["category_id"]) == category_id and int(p["image_id"]) in images:
            pred_by[int(p["image_id"])].append(p)
    records = []
    total_gt = sum(not ignored for items in gt_by.values() for _, ignored in items)
    matched = 0
    for image_id in set(gt_by) | set(pred_by):
        gts = sorted(gt_by.get(image_id, []), key=lambda item: item[1])
        used = [False] * len(gts)
        for p in sorted(pred_by.get(image_id, []), key=lambda x: float(x["score"]), reverse=True):
            candidates = []
            for j, (g, _) in enumerate(gts):
                if used[j]:
                    continue
                value = box_iou_xywh(p["bbox"], g["bbox"])
                if value >= iou:
                    candidates.append((j, value, gts[j][1]))
            # Any valid in-range match takes precedence over ignored GTs.
            eligible = [item for item in candidates if not item[2]] or candidates
            best_j = max(eligible, key=lambda item: item[1])[0] if eligible else -1
            if best_j >= 0:
                used[best_j] = True
                ignored = gts[best_j][1]
                matched += int(not ignored)
                records.append((float(p["score"]), not ignored, False))
            else:
                img = images[image_id]
                pred_bin = size_bin(relative_area(p["bbox"], img["width"], img["height"]), small_max, medium_max)
                ignored = wanted_bin is not None and pred_bin != wanted_bin
                records.append((float(p["score"]), False, not ignored))
    return records, total_gt - matched, total_gt


def prf(gt, preds, images, conf, iou, small_max, medium_max, wanted_bin=None, category_id=None):
    category_ids = [category_id] if category_id is not None else sorted(int(c["id"]) for c in gt["categories"])
    records = []
    fn = 0
    for cid in category_ids:
        class_records, class_fn, _ = _match_class(gt, preds, images, conf, iou, small_max, medium_max, wanted_bin, cid)
        records.extend(class_records)
        fn += class_fn
    tp = sum(int(item[1]) for item in records)
    fp = sum(int(item[2]) for item in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def ap_101(rec, prec):
    if len(rec) == 0:
        return 0.0
    samples = np.linspace(0, 1, 101)
    values = []
    for r in samples:
        mask = rec >= r
        values.append(np.max(prec[mask]) if np.any(mask) else 0.0)
    return float(np.mean(values))


def relative_ap_one_class(gt, preds, images, category_id, wanted_bin, iou_thr, small_max, medium_max):
    records, _, total_gt = _match_class(gt, preds, images, 0.0, iou_thr, small_max, medium_max, wanted_bin, category_id)
    if total_gt == 0:
        return None
    evaluated = sorted((item for item in records if item[1] or item[2]), key=lambda x: x[0], reverse=True)
    if not evaluated:
        return 0.0
    tps = [int(item[1]) for item in evaluated]
    fps = [int(item[2]) for item in evaluated]
    tp_c = np.cumsum(tps)
    fp_c = np.cumsum(fps)
    rec = tp_c / total_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    return ap_101(rec, prec)


def relative_map(gt, preds, images, categories, wanted_bin, thresholds, small_max, medium_max):
    aps = []
    for iou in thresholds:
        per_class = []
        for cid in categories:
            ap = relative_ap_one_class(gt, preds, images, cid, wanted_bin, iou, small_max, medium_max)
            if ap is not None:
                per_class.append(ap)
        if per_class:
            aps.append(float(np.mean(per_class)))
    return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser(description="Avaliação padronizada COCO + P/R/F1 + métricas relativas por escala.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--small-max", type=float, default=0.01)
    ap.add_argument("--medium-max", type=float, default=0.05)
    args = ap.parse_args()
    gt = load_coco_json(args.gt)
    images = images_by_id(gt)
    preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    cats = sorted(int(c["id"]) for c in gt["categories"])
    metrics = coco_metrics(args.gt, args.pred)
    overall = prf(gt, preds, images, args.conf, args.iou, args.small_max, args.medium_max)
    metrics.update({f"overall_{k}": float(v) for k, v in overall.items()})
    for bin_name in ("small", "medium", "large"):
        x = prf(gt, preds, images, args.conf, args.iou, args.small_max, args.medium_max, wanted_bin=bin_name)
        metrics.update({f"relative_{bin_name}_{k}": float(v) for k, v in x.items()})
        metrics[f"relative_{bin_name}_map50"] = relative_map(
            gt, preds, images, cats, bin_name, [0.50], args.small_max, args.medium_max
        )
        metrics[f"relative_{bin_name}_map5095"] = relative_map(
            gt, preds, images, cats, bin_name, np.arange(0.50, 0.96, 0.05), args.small_max, args.medium_max
        )
    per_class = {}
    for c in gt["categories"]:
        x = prf(gt, preds, images, args.conf, args.iou, args.small_max, args.medium_max, category_id=int(c["id"]))
        per_class[c["name"]] = x
    payload = {
        "metrics": metrics,
        "per_class": per_class,
        "definition": {
            "f1_confidence": args.conf,
            "iou_threshold": args.iou,
            "relative_small_max": args.small_max,
            "relative_medium_max": args.medium_max,
            "relative_AP_note": "A escala é definida pelo ground truth; objetos fora da faixa usam semântica de ignore inspirada no COCO. AP usa interpolação em 101 pontos.",
        },
    }
    save_json(payload, args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
