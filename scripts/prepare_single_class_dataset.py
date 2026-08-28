#!/usr/bin/env python3
"""Filtra e combina datasets COCO mantendo apenas uma classe configurável."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.coco import load_coco_json, save_json
from tcc_pipeline.config import find_project_root, resolve_path

OUTPUT_SPLITS = ("train", "val", "test")
SOURCE_SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}


def normalized_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def discover_split(source_root: Path, output_split: str) -> tuple[Path, Path] | None:
    for source_split in SOURCE_SPLIT_ALIASES[output_split]:
        canonical_annotations = source_root / "annotations" / f"{source_split}.json"
        canonical_images = source_root / source_split / "images"
        if canonical_annotations.is_file() and canonical_images.is_dir():
            return canonical_images, canonical_annotations

        roboflow_root = source_root / source_split
        roboflow_annotations = roboflow_root / "_annotations.coco.json"
        if roboflow_annotations.is_file() and roboflow_root.is_dir():
            return roboflow_root, roboflow_annotations
    return None


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config_path: Path) -> dict:
    project_root = find_project_root(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = raw["dataset"]
    output_root = resolve_path(project_root, settings["output_dir"])
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"A saída deve estar vazia ou não existir: {output_root}")

    aliases = {normalized_class_name(settings["target_class"])}
    aliases.update(normalized_class_name(item) for item in settings.get("class_aliases", []))
    keep_empty = bool(settings.get("keep_images_without_target", False))
    output = {split: {"images": [], "annotations": []} for split in OUTPUT_SPLITS}
    next_image_id = Counter({split: 1 for split in OUTPUT_SPLITS})
    next_annotation_id = Counter({split: 1 for split in OUTPUT_SPLITS})
    report_sources = []
    seen_hashes: dict[str, set[str]] = defaultdict(set)

    for source_index, source in enumerate(settings["sources"], start=1):
        source_name = str(source["name"])
        source_root = resolve_path(project_root, source["root"])
        if not source_root.is_dir():
            raise FileNotFoundError(f"Fonte não encontrada: {source_root}")
        source_report = {"name": source_name, "root": str(source["root"]), "splits": {}}
        safe_source = re.sub(r"[^a-zA-Z0-9_.-]+", "_", source_name).strip("_") or f"source_{source_index}"

        for output_split in OUTPUT_SPLITS:
            discovered = discover_split(source_root, output_split)
            if discovered is None:
                raise FileNotFoundError(f"Split '{output_split}' não encontrado em {source_root}")
            images_dir, annotation_file = discovered
            coco = load_coco_json(annotation_file)
            category_names = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
            target_ids = {cid for cid, name in category_names.items() if normalized_class_name(name) in aliases}
            if not target_ids:
                raise ValueError(
                    f"Classe alvo não encontrada em {annotation_file}. Classes disponíveis: {sorted(category_names.values())}"
                )

            annotations_by_image = defaultdict(list)
            for annotation in coco["annotations"]:
                if int(annotation["category_id"]) in target_ids:
                    annotations_by_image[int(annotation["image_id"])].append(annotation)

            kept_images = kept_annotations = dropped_images = clipped_boxes = 0
            for image in sorted(coco["images"], key=lambda item: int(item["id"])):
                source_annotations = annotations_by_image[int(image["id"])]
                if not source_annotations and not keep_empty:
                    dropped_images += 1
                    continue
                source_image = images_dir / image["file_name"]
                if not source_image.is_file():
                    raise FileNotFoundError(f"Imagem referenciada pelo COCO não encontrada: {source_image}")
                digest = file_digest(source_image)
                seen_hashes[digest].add(output_split)

                image_id = next_image_id[output_split]
                next_image_id[output_split] += 1
                extension = source_image.suffix.lower() or ".jpg"
                output_name = f"{safe_source}_{image_id:07d}{extension}"
                link_or_copy(source_image, output_root / output_split / "images" / output_name)
                width, height = int(image["width"]), int(image["height"])
                output[output_split]["images"].append(
                    {"id": image_id, "file_name": output_name, "width": width, "height": height}
                )
                kept_images += 1

                for annotation in source_annotations:
                    x, y, box_width, box_height = map(float, annotation["bbox"])
                    x1, y1 = max(0.0, x), max(0.0, y)
                    x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
                    if (x1, y1, x2, y2) != (x, y, x + box_width, y + box_height):
                        clipped_boxes += 1
                    if x2 <= x1 or y2 <= y1:
                        continue
                    output[output_split]["annotations"].append(
                        {
                            "id": next_annotation_id[output_split],
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": int(annotation.get("iscrowd", 0)),
                        }
                    )
                    next_annotation_id[output_split] += 1
                    kept_annotations += 1

            source_report["splits"][output_split] = {
                "images_kept": kept_images,
                "images_without_target_removed": dropped_images,
                "annotations_kept": kept_annotations,
                "boxes_clipped": clipped_boxes,
                "matched_source_classes": [category_names[item] for item in sorted(target_ids)],
            }
        report_sources.append(source_report)

    duplicate_hashes = {digest: sorted(splits) for digest, splits in seen_hashes.items() if len(splits) > 1}
    if duplicate_hashes and settings.get("fail_on_duplicate_images_across_splits", True):
        raise ValueError(f"Foram encontradas {len(duplicate_hashes)} imagens idênticas em splits diferentes.")

    summary = {}
    category = {"id": 1, "name": str(settings["target_class"]), "supercategory": "defect"}
    for split in OUTPUT_SPLITS:
        coco = {
            "info": {"description": f"{settings['name']} - classe única {settings['target_class']}"},
            "images": output[split]["images"],
            "annotations": output[split]["annotations"],
            "categories": [category],
        }
        save_json(coco, output_root / "annotations" / f"{split}.json")
        summary[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}

    report = {
        "dataset_name": settings["name"],
        "target_class": settings["target_class"],
        "class_aliases": sorted(aliases),
        "keep_images_without_target": keep_empty,
        "sources": report_sources,
        "output_summary": summary,
        "duplicate_images_across_splits": len(duplicate_hashes),
    }
    save_json(report, output_root / "preparation_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs" / "dataset.yaml", type=Path)
    args = parser.parse_args()
    report = prepare(args.config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
