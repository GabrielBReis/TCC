#!/usr/bin/env python3
"""Clean and create leakage-safe splits from the Innovation Hangar export."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.coco import save_json

SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "val", "test")
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+(?=\.[^.]+$)", re.IGNORECASE)


def source_group(file_name: str) -> str:
    """Recover the source-image name before Roboflow's generated hash."""
    stem = ROBOFLOW_SUFFIX.sub("", Path(file_name).name)
    if stem.lower().endswith("_jpg.jpg"):
        stem = stem[:-8]
    return stem.casefold()


def load_export(root: Path):
    records = []
    categories = None
    source_summary = {}
    for source_split in SOURCE_SPLITS:
        annotation_path = root / source_split / "_annotations.coco.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing Roboflow annotation file: {annotation_path}")
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        current_categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
        if categories is None:
            categories = current_categories
        elif categories != current_categories:
            raise ValueError(f"Category mapping differs in split {source_split}")
        annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        for image in coco["images"]:
            source_path = root / source_split / image["file_name"]
            if not source_path.exists():
                raise FileNotFoundError(f"Missing image: {source_path}")
            records.append(
                {
                    "source_split": source_split,
                    "source_path": source_path,
                    "source_file_name": image["file_name"],
                    "group": source_group(image["file_name"]),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": annotations[int(image["id"])],
                }
            )
        source_summary[source_split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"])}
    return records, categories or {}, source_summary


def assign_groups(records, seed: int, ratios: dict[str, float]):
    groups = defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)

    total_images = len(records)
    total_classes = Counter(
        int(annotation["category_id"]) for record in records for annotation in record["annotations"]
    )
    target_images = {split: total_images * ratios[split] for split in OUTPUT_SPLITS}
    target_classes = {
        split: {category: count * ratios[split] for category, count in total_classes.items()} for split in OUTPUT_SPLITS
    }
    assigned = {split: [] for split in OUTPUT_SPLITS}
    image_counts = Counter()
    class_counts = {split: Counter() for split in OUTPUT_SPLITS}

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(
        key=lambda item: (
            len(item[1]),
            sum(
                1 / max(total_classes[int(annotation["category_id"])], 1)
                for record in item[1]
                for annotation in record["annotations"]
            ),
        ),
        reverse=True,
    )

    for _, group_records in group_items:
        group_classes = Counter(
            int(annotation["category_id"]) for record in group_records for annotation in record["annotations"]
        )

        def score(split, group_records=group_records, group_classes=group_classes):
            image_fill = (image_counts[split] + len(group_records)) / max(target_images[split], 1)
            class_fill = sum(
                (class_counts[split][category] + count) / max(target_classes[split][category], 1)
                for category, count in group_classes.items()
            ) / max(len(group_classes), 1)
            # Allocate to the least-filled split; a small image term also handles
            # groups without annotations while class occupancy drives stratification.
            return 0.35 * image_fill + 0.65 * class_fill

        destination = min(OUTPUT_SPLITS, key=score)
        assigned[destination].extend(group_records)
        image_counts[destination] += len(group_records)
        class_counts[destination].update(group_classes)
    return assigned, len(groups)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize(assigned, categories, output_root: Path):
    used_category_ids = {
        int(annotation["category_id"])
        for records in assigned.values()
        for record in records
        for annotation in record["annotations"]
    }
    clean_categories = [
        {"id": category_id, "name": categories[category_id], "supercategory": "defect"}
        for category_id in sorted(used_category_ids)
    ]
    summary = {}
    clipped_boxes = 0
    for split, records in assigned.items():
        images = []
        annotations = []
        annotation_id = 1
        for image_id, record in enumerate(
            sorted(records, key=lambda item: (item["group"], item["source_file_name"])), start=1
        ):
            extension = record["source_path"].suffix.lower()
            output_name = f"{image_id:06d}{extension}"
            link_or_copy(record["source_path"], output_root / split / "images" / output_name)
            images.append(
                {
                    "id": image_id,
                    "file_name": output_name,
                    "width": record["width"],
                    "height": record["height"],
                    "source_file_name": record["source_file_name"],
                    "source_split": record["source_split"],
                    "source_group": record["group"],
                }
            )
            for source_annotation in record["annotations"]:
                x, y, width, height = map(float, source_annotation["bbox"])
                x1 = min(max(x, 0.0), record["width"])
                y1 = min(max(y, 0.0), record["height"])
                x2 = min(max(x + width, 0.0), record["width"])
                y2 = min(max(y + height, 0.0), record["height"])
                if x < 0 or y < 0 or x + width > record["width"] or y + height > record["height"]:
                    clipped_boxes += 1
                if x2 <= x1 or y2 <= y1:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(source_annotation["category_id"]),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": int(source_annotation.get("iscrowd", 0)),
                        "source_annotation_id": int(source_annotation["id"]),
                    }
                )
                annotation_id += 1
        coco = {
            "info": {"description": "Innovation Hangar v2, leakage-safe TCC split"},
            "images": images,
            "annotations": annotations,
            "categories": clean_categories,
        }
        save_json(coco, output_root / "annotations" / f"{split}.json")
        summary[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "classes": dict(Counter(categories[item["category_id"]] for item in annotations)),
        }
    return summary, clipped_boxes, clean_categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train", type=float, default=0.75)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.out}")

    records, categories, source_summary = load_export(args.source)
    ratios = {"train": args.train, "val": args.val, "test": args.test}
    assigned, source_groups = assign_groups(records, args.seed, ratios)
    output_summary, clipped_boxes, clean_categories = materialize(assigned, categories, args.out)
    report = {
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "ratios": ratios,
        "source_summary": source_summary,
        "source_groups": source_groups,
        "removed_empty_categories": [
            name
            for category_id, name in categories.items()
            if category_id not in {item["id"] for item in clean_categories}
        ],
        "clipped_boxes": clipped_boxes,
        "output_summary": output_summary,
    }
    save_json(report, args.out / "preparation_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
