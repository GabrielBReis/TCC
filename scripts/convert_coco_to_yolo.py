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

import yaml

from tcc_pipeline.coco import annotations_by_image, category_maps, load_coco_json


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def convert_split(images_dir: Path, ann_file: Path, out_dir: Path, split: str, mode: str):
    coco = load_coco_json(ann_file)
    anns_by = annotations_by_image(coco)
    cat_to_idx0, idx0_to_cat, id2label, _ = category_maps(coco)
    for img in coco["images"]:
        image_id = int(img["id"])
        src = images_dir / img["file_name"]
        dst = out_dir / "images" / split / img["file_name"]
        link_or_copy(src, dst, mode)
        label = out_dir / "labels" / split / Path(img["file_name"]).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        W, H = float(img["width"]), float(img["height"])
        for ann in anns_by.get(image_id, []):
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0 or W <= 0 or H <= 0:
                continue
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            lines.append(f"{cat_to_idx0[int(ann['category_id'])]} {cx:.8f} {cy:.8f} {w / W:.8f} {h / H:.8f}")
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return idx0_to_cat, id2label


def main():
    ap = argparse.ArgumentParser(description="Converte splits COCO para o formato YOLO/Ultralytics.")
    for split in ("train", "val", "test"):
        ap.add_argument(f"--{split}-images")
        ap.add_argument(f"--{split}-annotations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping, id2label = None, None
    available = []
    for split in ("train", "val", "test"):
        images = getattr(args, f"{split}_images")
        anns = getattr(args, f"{split}_annotations")
        if images and anns:
            m, labels = convert_split(Path(images).resolve(), Path(anns).resolve(), out, split, args.mode)
            mapping = mapping or m
            id2label = id2label or labels
            available.append(split)
    if mapping is None:
        raise SystemExit("Nenhum split informado.")
    yaml_data = {
        "train": "images/train" if "train" in available else None,
        "val": "images/val" if "val" in available else None,
        "test": "images/test" if "test" in available else None,
        "names": {int(k): v for k, v in id2label.items()},
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "category_mapping.json").write_text(
        json.dumps({"yolo_id_to_coco_category_id": mapping, "id2label": id2label}, indent=2), encoding="utf-8"
    )
    print(out / "dataset.yaml")


if __name__ == "__main__":
    main()
