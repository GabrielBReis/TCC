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

import matplotlib.pyplot as plt
import pandas as pd

from tcc_pipeline.coco import images_by_id, load_coco_json
from tcc_pipeline.geometry import relative_area, size_bin


def main():
    ap = argparse.ArgumentParser(description="Analisa classes e escala relativa das bounding boxes.")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="reports/dataset_analysis")
    ap.add_argument("--small-max", type=float, default=0.01)
    ap.add_argument("--medium-max", type=float, default=0.05)
    args = ap.parse_args()

    coco = load_coco_json(args.annotations)
    imgs = images_by_id(coco)
    cats = {int(c["id"]): c["name"] for c in coco["categories"]}
    rows = []
    for ann in coco["annotations"]:
        img = imgs[int(ann["image_id"])]
        rel = relative_area(ann["bbox"], img["width"], img["height"])
        rows.append(
            {
                "annotation_id": int(ann["id"]),
                "image_id": int(ann["image_id"]),
                "class": cats[int(ann["category_id"])],
                "bbox_area_px": float(ann["bbox"][2]) * float(ann["bbox"][3]),
                "relative_area": rel,
                "size_bin": size_bin(rel, args.small_max, args.medium_max),
            }
        )
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "annotations_scale.csv", index=False)

    summary = {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": cats,
        "small_max": args.small_max,
        "medium_max": args.medium_max,
        "size_counts": df["size_bin"].value_counts().to_dict() if not df.empty else {},
        "class_counts": df["class"].value_counts().to_dict() if not df.empty else {},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not df.empty:
        plt.figure(figsize=(8, 5))
        df["relative_area"].clip(upper=df["relative_area"].quantile(0.99)).hist(bins=50)
        plt.xlabel("Área relativa da bounding box")
        plt.ylabel("Quantidade")
        plt.tight_layout()
        plt.savefig(out / "relative_area_histogram.png", dpi=160)
        plt.close()

        pivot = pd.crosstab(df["class"], df["size_bin"])
        pivot.to_csv(out / "class_by_size.csv")
        print(pivot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
