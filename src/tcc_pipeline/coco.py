from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_coco_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"Arquivo COCO sem chave obrigatória '{key}': {path}")
    return data


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def category_maps(coco: dict[str, Any]):
    cats = sorted(coco["categories"], key=lambda x: int(x["id"]))
    category_id_to_contiguous0 = {int(c["id"]): i for i, c in enumerate(cats)}
    contiguous0_to_category_id = {i: int(c["id"]) for i, c in enumerate(cats)}
    id2label = {i: str(c["name"]) for i, c in enumerate(cats)}
    label2id = {v: k for k, v in id2label.items()}
    return category_id_to_contiguous0, contiguous0_to_category_id, id2label, label2id


def annotations_by_image(coco: dict[str, Any]):
    out = defaultdict(list)
    for ann in coco["annotations"]:
        out[int(ann["image_id"])].append(ann)
    return out


def images_by_id(coco: dict[str, Any]):
    return {int(img["id"]): img for img in coco["images"]}
