#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from tcc_pipeline.coco import load_coco_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images)
    coco = load_coco_json(args.annotations)
    image_ids = {int(i["id"]) for i in coco["images"]}
    cat_ids = {int(c["id"]) for c in coco["categories"]}
    errors, warnings = [], []

    seen_files = Counter()
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    content_hashes = {}
    for img in coco["images"]:
        p = images_dir / img["file_name"]
        seen_files[img["file_name"]] += 1
        if not p.exists():
            errors.append(f"Imagem ausente: {p}")
            continue
        if int(img.get("width", 0)) <= 0 or int(img.get("height", 0)) <= 0:
            warnings.append(f"Dimensões ausentes/inválidas no JSON: {img['file_name']}")
        try:
            with Image.open(p) as image:
                image.verify()
            with Image.open(p) as image:
                actual_size = image.size
            declared_size = (int(img.get("width", 0)), int(img.get("height", 0)))
            if actual_size != declared_size:
                errors.append(
                    f"Dimensões divergentes em {img['file_name']}: arquivo={actual_size}, COCO={declared_size}"
                )
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in content_hashes:
                errors.append(f"Conteúdo duplicado: {img['file_name']} e {content_hashes[digest]}")
            else:
                content_hashes[digest] = img["file_name"]
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"Imagem corrompida ou ilegível: {p} ({exc})")

    for name, n in seen_files.items():
        if n > 1:
            errors.append(f"file_name duplicado no COCO: {name} ({n} ocorrências)")

    ann_ids = set()
    for ann in coco["annotations"]:
        aid = int(ann["id"])
        if aid in ann_ids:
            errors.append(f"annotation id duplicado: {aid}")
        ann_ids.add(aid)
        if int(ann["image_id"]) not in image_ids:
            errors.append(f"annotation {aid}: image_id inexistente")
        if int(ann["category_id"]) not in cat_ids:
            errors.append(f"annotation {aid}: category_id inexistente")
        x, y, w, h = map(float, ann["bbox"])
        if w <= 0 or h <= 0:
            errors.append(f"annotation {aid}: bbox com largura/altura <= 0")
        if x < 0 or y < 0:
            warnings.append(f"annotation {aid}: bbox inicia fora da imagem")
        image_info = image_by_id.get(int(ann["image_id"]))
        if image_info and (x + w > float(image_info["width"]) or y + h > float(image_info["height"])):
            warnings.append(f"annotation {aid}: bbox ultrapassa os limites da imagem")

    print(f"Imagens: {len(coco['images'])}")
    print(f"Anotações: {len(coco['annotations'])}")
    print(f"Classes: {len(coco['categories'])}")
    print(f"Erros: {len(errors)} | Avisos: {len(warnings)}")
    for item in errors[:50]:
        print("[ERRO]", item)
    for item in warnings[:50]:
        print("[AVISO]", item)
    if errors or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
