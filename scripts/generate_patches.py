#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import math
import random
from pathlib import Path

from PIL import Image

from tcc_pipeline.coco import annotations_by_image, load_coco_json, save_json


def positions(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    xs = list(range(0, max(length - patch + 1, 1), stride))
    last = length - patch
    if xs[-1] != last:
        xs.append(last)
    return xs


def intersection(box, window):
    x, y, w, h = map(float, box)
    wx, wy, ww, wh = map(float, window)
    x1, y1 = max(x, wx), max(y, wy)
    x2, y2 = min(x + w, wx + ww), min(y + h, wy + wh)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def main():
    ap = argparse.ArgumentParser(description="Gera patches/tiling preservando anotações COCO.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--min-visible", type=float, default=0.30)
    ap.add_argument("--keep-all-negative", action="store_true")
    ap.add_argument(
        "--negative-ratio", type=float, default=0.50, help="Nº máximo de patches negativos / patches positivos"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap deve estar em [0,1)")
    if args.negative_ratio < 0:
        raise ValueError("negative-ratio deve ser >= 0")

    stride = max(1, int(args.patch * (1 - args.overlap)))
    coco = load_coco_json(args.annotations)
    anns_by = annotations_by_image(coco)
    images_dir = Path(args.images)
    out = Path(args.out)
    out_images = out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # Primeiro enumeramos candidatos sem gravar. Isso permite subamostrar negativos
    # de forma reproduzível antes de criar os arquivos.
    positives, negatives = [], []
    for info in coco["images"]:
        src = images_dir / info["file_name"]
        with Image.open(src) as im:
            W, H = im.size
        for y0 in positions(H, args.patch, stride):
            for x0 in positions(W, args.patch, stride):
                ww, wh = min(args.patch, W - x0), min(args.patch, H - y0)
                window = [x0, y0, ww, wh]
                kept = []
                for ann in anns_by.get(int(info["id"]), []):
                    inter = intersection(ann["bbox"], window)
                    if inter is None:
                        continue
                    orig_area = max(1e-9, float(ann["bbox"][2]) * float(ann["bbox"][3]))
                    visible = inter[2] * inter[3] / orig_area
                    if visible < args.min_visible:
                        continue
                    clipped = [inter[0] - x0, inter[1] - y0, inter[2], inter[3]]
                    kept.append((ann, clipped, visible))
                candidate = {"info": info, "src": src, "x": x0, "y": y0, "w": ww, "h": wh, "kept": kept}
                (positives if kept else negatives).append(candidate)

    if args.keep_all_negative:
        selected_neg = negatives
    else:
        rng = random.Random(args.seed)
        rng.shuffle(negatives)
        n_neg = min(len(negatives), math.ceil(len(positives) * args.negative_ratio))
        selected_neg = negatives[:n_neg]

    selected = positives + selected_neg
    # Ordem estável: imagem original, y, x; facilita auditoria/reprodutibilidade.
    selected.sort(key=lambda c: (int(c["info"]["id"]), int(c["y"]), int(c["x"])))

    new_images, new_anns = [], []
    ann_id = 1
    for image_id, c in enumerate(selected, start=1):
        info = c["info"]
        stem = Path(info["file_name"]).stem
        file_name = f"{stem}__src{int(info['id'])}_x{c['x']}_y{c['y']}.jpg"
        with Image.open(c["src"]) as im:
            crop = im.convert("RGB").crop((c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]))
            crop.save(out_images / file_name, quality=95)
        new_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": c["w"],
                "height": c["h"],
                "source_image_id": int(info["id"]),
                "source_file_name": info["file_name"],
                "patch_x": int(c["x"]),
                "patch_y": int(c["y"]),
            }
        )
        for ann, bbox, visible in c["kept"]:
            new_anns.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(ann["category_id"]),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "source_annotation_id": int(ann["id"]),
                    "visible_fraction": float(visible),
                }
            )
            ann_id += 1

    out_coco = {
        "images": new_images,
        "annotations": new_anns,
        "categories": coco["categories"],
        "info": {
            **coco.get("info", {}),
            "patch_size": args.patch,
            "overlap": args.overlap,
            "min_visible": args.min_visible,
            "positive_patches": len(positives),
            "negative_candidates": len(negatives),
            "negative_patches_kept": len(selected_neg),
        },
    }
    save_json(out_coco, out / "annotations.json")
    print(f"Patches positivos: {len(positives)}")
    print(f"Patches negativos mantidos: {len(selected_neg)} / {len(negatives)} candidatos")
    print(f"Total: {len(new_images)} | anotações: {len(new_anns)} | saída: {out}")


if __name__ == "__main__":
    main()
