from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from tcc_pipeline.coco import load_coco_json, save_json
from tcc_pipeline.geometry import relative_area, size_bin

SPLITS = ("train", "val", "test")
SOURCE_FOLDERS = {"train": "train", "val": "valid", "test": "test"}
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+(?=\.[^.]+$)", re.IGNORECASE)
COPY_SUFFIX = re.compile(r"(?:__\d+)+$")


def normalized_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def source_group(file_name: str) -> str:
    """Recover the source name shared by Roboflow-generated variants."""
    name = ROBOFLOW_SUFFIX.sub("", Path(file_name).name)
    stem = Path(name).stem
    if stem.casefold().endswith("_jpg"):
        stem = stem[:-4]
    return COPY_SUFFIX.sub("", stem.casefold())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path, mode: str = "hardlink") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        try:
            os.symlink(source.resolve(), destination)
            return
        except OSError:
            pass
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@dataclass
class ImageRecord:
    source_split: str
    source_path: Path
    source_file_name: str
    group: str
    width: int
    height: int
    annotations: list[dict[str, Any]]
    sha256: str = ""

    @property
    def positive(self) -> bool:
        return bool(self.annotations)

    def size_counts(self, small_max: float, medium_max: float) -> Counter[str]:
        return Counter(
            size_bin(relative_area(annotation["bbox"], self.width, self.height), small_max, medium_max)
            for annotation in self.annotations
        )


@dataclass
class GroupFeatures:
    records: list[ImageRecord]
    values: Counter[str] = field(default_factory=Counter)


def load_records(
    source_root: Path,
    target_class: str,
    aliases: list[str],
    small_max: float,
    medium_max: float,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    accepted = {normalized_class_name(target_class), *(normalized_class_name(item) for item in aliases)}
    records: list[ImageRecord] = []
    source_summary: dict[str, Any] = {}

    for split, folder in SOURCE_FOLDERS.items():
        split_root = source_root / folder
        annotation_file = split_root / "_annotations.coco.json"
        coco = load_coco_json(annotation_file)
        categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
        target_ids = {category_id for category_id, name in categories.items() if normalized_class_name(name) in accepted}
        if not target_ids:
            raise ValueError(f"Classe '{target_class}' não encontrada em {annotation_file}")

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            if int(annotation["category_id"]) in target_ids:
                annotations_by_image[int(annotation["image_id"])].append(annotation)

        split_records = []
        for image in sorted(coco["images"], key=lambda item: int(item["id"])):
            source_path = split_root / str(image["file_name"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            record = ImageRecord(
                source_split=split,
                source_path=source_path,
                source_file_name=str(image["file_name"]),
                group=source_group(str(image["file_name"])),
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=list(annotations_by_image[int(image["id"])]),
            )
            split_records.append(record)
            records.append(record)

        source_summary[split] = summarize_records(split_records, small_max, medium_max)
    return records, source_summary


def portable_path(path: Path, project_root: Path) -> str:
    """Representa caminhos do relatório relativamente à raiz do projeto."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def deduplicate_records(
    records: list[ImageRecord], project_root: Path
) -> tuple[list[ImageRecord], list[dict[str, Any]]]:
    kept: list[ImageRecord] = []
    seen: dict[str, ImageRecord] = {}
    removed = []
    for record in records:
        record.sha256 = sha256_file(record.source_path)
        previous = seen.get(record.sha256)
        if previous is None:
            seen[record.sha256] = record
            kept.append(record)
            continue
        previous_boxes = sorted(tuple(map(float, item["bbox"])) for item in previous.annotations)
        current_boxes = sorted(tuple(map(float, item["bbox"])) for item in record.annotations)
        if previous_boxes != current_boxes and (previous_boxes or current_boxes):
            raise ValueError(
                "Imagens idênticas possuem anotações de crack diferentes: "
                f"{previous.source_path} e {record.source_path}"
            )
        removed.append(
            {
                "sha256": record.sha256,
                "kept": portable_path(previous.source_path, project_root),
                "removed": portable_path(record.source_path, project_root),
                "source_split": record.source_split,
            }
        )
    return kept, removed


def group_features(
    records: list[ImageRecord], small_max: float, medium_max: float
) -> dict[str, GroupFeatures]:
    groups: dict[str, GroupFeatures] = {}
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    for group, items in grouped.items():
        values: Counter[str] = Counter()
        for record in items:
            values["images"] += 1
            values["positive_images" if record.positive else "negative_images"] += 1
            values["boxes"] += len(record.annotations)
            values.update({f"boxes_{key}": value for key, value in record.size_counts(small_max, medium_max).items()})
            values[f"source_{record.source_split}"] += 1
        groups[group] = GroupFeatures(records=items, values=values)
    return groups


def assign_evaluation_groups(
    records: list[ImageRecord],
    val_ratio: float,
    seed: int,
    small_max: float,
    medium_max: float,
) -> dict[str, list[ImageRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("evaluation_val_ratio deve estar entre 0 e 1")
    groups = group_features(records, small_max, medium_max)
    totals: Counter[str] = Counter()
    for item in groups.values():
        totals.update(item.values)

    ratios = {"val": val_ratio, "test": 1.0 - val_ratio}
    targets = {split: {key: value * ratios[split] for key, value in totals.items()} for split in ("val", "test")}
    counts = {"val": Counter(), "test": Counter()}
    assigned = {"val": [], "test": []}
    weights = {
        "images": 1.0,
        "positive_images": 2.0,
        "negative_images": 1.5,
        "boxes": 1.0,
        "boxes_small": 3.0,
        "boxes_medium": 3.0,
        "boxes_large": 3.0,
        "source_val": 0.5,
        "source_test": 0.5,
    }

    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)

    def priority(item: tuple[str, GroupFeatures]) -> tuple[float, int]:
        values = item[1].values
        rarity = sum(
            weights.get(key, 1.0) * value / max(totals[key], 1) for key, value in values.items() if value
        )
        return rarity, len(item[1].records)

    items.sort(key=priority, reverse=True)
    feature_keys = sorted(totals)
    for _, item in items:
        scores = {}
        for destination in ("val", "test"):
            score = 0.0
            for split in ("val", "test"):
                for key in feature_keys:
                    value = counts[split][key]
                    if split == destination:
                        value += item.values[key]
                    target = targets[split][key]
                    score += weights.get(key, 1.0) * ((value - target) / max(target, 1.0)) ** 2
            scores[destination] = score
        destination = min(("val", "test"), key=lambda split: (scores[split], len(assigned[split])))
        assigned[destination].extend(item.records)
        counts[destination].update(item.values)
    return assigned


def summarize_records(records: list[ImageRecord], small_max: float, medium_max: float) -> dict[str, Any]:
    relative_areas = [
        relative_area(annotation["bbox"], record.width, record.height)
        for record in records
        for annotation in record.annotations
    ]
    size_counts = Counter(size_bin(value, small_max, medium_max) for value in relative_areas)
    positives = sum(record.positive for record in records)
    return {
        "images": len(records),
        "positive_images": positives,
        "negative_images": len(records) - positives,
        "annotations": len(relative_areas),
        "source_groups": len({record.group for record in records}),
        "size_counts": dict(size_counts),
        "median_relative_area": float(np.median(relative_areas)) if relative_areas else 0.0,
        "positive_image_fraction": positives / len(records) if records else 0.0,
    }


def materialize_coco(
    assigned: dict[str, list[ImageRecord]],
    output_root: Path,
    target_class: str,
    link_mode: str,
    small_max: float,
    medium_max: float,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        images = []
        annotations = []
        annotation_id = 1
        records = sorted(assigned[split], key=lambda item: (item.group, item.source_file_name, item.sha256))
        for image_id, record in enumerate(records, start=1):
            extension = record.source_path.suffix.lower() or ".jpg"
            output_name = f"aircraft_surface_damage_crack_v2_{image_id:07d}{extension}"
            output_image = output_root / split / "images" / output_name
            link_or_copy(record.source_path, output_image, link_mode)
            images.append(
                {
                    "id": image_id,
                    "file_name": output_name,
                    "width": record.width,
                    "height": record.height,
                    "source_split": record.source_split,
                    "source_file_name": record.source_file_name,
                    "source_group": record.group,
                    "sha256": record.sha256,
                }
            )
            bins = record.size_counts(small_max, medium_max)
            manifest_rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "file_name": output_name,
                    "source_split": record.source_split,
                    "source_file_name": record.source_file_name,
                    "source_group": record.group,
                    "sha256": record.sha256,
                    "positive": int(record.positive),
                    "annotations": len(record.annotations),
                    "small_boxes": bins["small"],
                    "medium_boxes": bins["medium"],
                    "large_boxes": bins["large"],
                }
            )
            for source_annotation in record.annotations:
                x, y, width, height = map(float, source_annotation["bbox"])
                x1, y1 = max(0.0, x), max(0.0, y)
                x2 = min(float(record.width), x + width)
                y2 = min(float(record.height), y + height)
                if x2 <= x1 or y2 <= y1:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": int(source_annotation.get("iscrowd", 0)),
                    }
                )
                annotation_id += 1
        save_json(
            {
                "info": {"description": f"aircraft_surface_damage_crack_v2 - classe única {target_class}"},
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": target_class, "supercategory": "defect"}],
            },
            output_root / "annotations" / f"{split}.json",
        )
    return manifest_rows


def write_manifest(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        columns = fieldnames or (list(rows[0]) if rows else [])
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
        if rows:
            writer.writerows(rows)


def convert_to_yolo(output_root: Path, link_mode: str) -> None:
    yolo_root = output_root / "yolo"
    for split in SPLITS:
        coco = load_coco_json(output_root / "annotations" / f"{split}.json")
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        for image in coco["images"]:
            source = output_root / split / "images" / image["file_name"]
            destination = yolo_root / "images" / split / image["file_name"]
            link_or_copy(source, destination, link_mode)
            width, height = float(image["width"]), float(image["height"])
            lines = []
            for annotation in annotations[int(image["id"])]:
                x, y, box_width, box_height = map(float, annotation["bbox"])
                lines.append(
                    "0 "
                    f"{(x + box_width / 2) / width:.8f} {(y + box_height / 2) / height:.8f} "
                    f"{box_width / width:.8f} {box_height / height:.8f}"
                )
            label = yolo_root / "labels" / split / Path(image["file_name"]).with_suffix(".txt")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    (yolo_root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "crack"}},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    save_json(
        {"yolo_id_to_coco_category_id": {"0": 1}, "id2label": {"0": "crack"}},
        yolo_root / "category_mapping.json",
    )


def perceptual_hash(path: Path) -> int:
    size = 32
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64)
    coordinates = np.arange(size)
    transform = np.cos(np.pi * (2 * coordinates[:, None] + 1) * coordinates[None, :] / (2 * size)).T
    transform[0] *= 1.0 / math.sqrt(2.0)
    transform *= math.sqrt(2.0 / size)
    coefficients = transform @ pixels @ transform.T
    values = coefficients[:8, :8].reshape(-1)[1:]
    median = float(np.median(values))
    result = 0
    for value in values:
        result = (result << 1) | int(value > median)
    return result


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[dict[str, Any]], dict[int, Any]] | None = None

    def add(self, value: int, payload: dict[str, Any]) -> None:
        if self.root is None:
            self.root = (value, [payload], {})
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(payload)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [payload], {})
                return
            node = child

    def query(self, value: int, threshold: int) -> list[tuple[int, dict[str, Any]]]:
        if self.root is None:
            return []
        found = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= threshold:
                found.extend((distance, payload) for payload in node[1])
            low, high = distance - threshold, distance + threshold
            pending.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


def yolo_equivalence(output_root: Path, tolerance: float = 2e-5) -> tuple[int, list[str]]:
    errors = []
    checked = 0
    for split in SPLITS:
        coco = load_coco_json(output_root / "annotations" / f"{split}.json")
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        for image in coco["images"]:
            expected = []
            width, height = float(image["width"]), float(image["height"])
            for annotation in annotations[int(image["id"])]:
                x, y, box_width, box_height = map(float, annotation["bbox"])
                expected.append(
                    (0, (x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height)
                )
            label_path = output_root / "yolo" / "labels" / split / Path(image["file_name"]).with_suffix(".txt")
            actual = []
            if not label_path.is_file():
                errors.append(f"Label ausente: {label_path}")
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Linha YOLO inválida em {label_path}: {line}")
                    continue
                actual.append((int(parts[0]), *(float(value) for value in parts[1:])))
            expected.sort()
            actual.sort()
            checked += len(expected)
            if len(expected) != len(actual):
                errors.append(f"Contagem COCO/YOLO divergente em {split}/{image['file_name']}")
                continue
            for expected_box, actual_box in zip(expected, actual, strict=True):
                if expected_box[0] != actual_box[0] or any(
                    abs(left - right) > tolerance for left, right in zip(expected_box[1:], actual_box[1:], strict=True)
                ):
                    errors.append(f"BBox COCO/YOLO divergente em {split}/{image['file_name']}")
                    break
    return checked, errors


def audit_dataset(
    dataset_root: Path,
    report_root: Path,
    small_max: float,
    medium_max: float,
    phash_threshold: int,
) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    split_summaries: dict[str, Any] = {}
    assignment_rows = []
    annotations_rows = []
    seen_hashes: dict[str, dict[str, Any]] = {}
    groups_by_split: dict[str, set[str]] = {}
    positive_groups_by_split: dict[str, set[str]] = {}
    phash_items = []

    for split in SPLITS:
        coco = load_coco_json(dataset_root / "annotations" / f"{split}.json")
        categories = [(int(item["id"]), str(item["name"])) for item in coco["categories"]]
        if categories != [(1, "crack")]:
            errors.append(f"Categorias inesperadas em {split}: {categories}")
        images = {int(item["id"]): item for item in coco["images"]}
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        groups_by_split[split] = {str(item.get("source_group", item["file_name"])) for item in coco["images"]}
        positive_groups_by_split[split] = set()
        relative_areas = []
        positives = 0
        size_counts: Counter[str] = Counter()
        for image_id, image in images.items():
            image_path = dataset_root / split / "images" / image["file_name"]
            if not image_path.is_file():
                errors.append(f"Imagem ausente: {image_path}")
                continue
            try:
                with Image.open(image_path) as opened:
                    actual_size = opened.size
                if actual_size != (int(image["width"]), int(image["height"])):
                    errors.append(f"Dimensão divergente: {split}/{image['file_name']}")
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"Imagem ilegível: {image_path} ({exc})")
                continue
            digest = sha256_file(image_path)
            previous = seen_hashes.get(digest)
            if previous and previous["split"] != split:
                errors.append(
                    f"Duplicata exata entre splits: {previous['split']}/{previous['file_name']} e {split}/{image['file_name']}"
                )
            else:
                seen_hashes[digest] = {"split": split, "file_name": image["file_name"]}
            item_annotations = annotations.get(image_id, [])
            positives += int(bool(item_annotations))
            if item_annotations:
                positive_groups_by_split[split].add(str(image.get("source_group", image["file_name"])))
            assignment_rows.append(
                {
                    "split": split,
                    "file_name": image["file_name"],
                    "source_split": image.get("source_split", ""),
                    "source_file_name": image.get("source_file_name", ""),
                    "source_group": image.get("source_group", ""),
                    "sha256": digest,
                    "positive": int(bool(item_annotations)),
                    "annotations": len(item_annotations),
                }
            )
            phash_items.append(
                {
                    "split": split,
                    "file_name": image["file_name"],
                    "source_group": str(image.get("source_group", image["file_name"])),
                    "path": image_path,
                }
            )
            for annotation in item_annotations:
                if int(annotation["category_id"]) != 1:
                    errors.append(f"Categoria inválida na anotação {annotation['id']} de {split}")
                x, y, width, height = map(float, annotation["bbox"])
                if width <= 0 or height <= 0 or x < 0 or y < 0:
                    errors.append(f"BBox inválida na anotação {annotation['id']} de {split}")
                    continue
                if x + width > float(image["width"]) + 1e-6 or y + height > float(image["height"]) + 1e-6:
                    errors.append(f"BBox fora da imagem na anotação {annotation['id']} de {split}")
                relative = relative_area(annotation["bbox"], image["width"], image["height"])
                bucket = size_bin(relative, small_max, medium_max)
                relative_areas.append(relative)
                size_counts[bucket] += 1
                annotations_rows.append(
                    {
                        "split": split,
                        "image_id": image_id,
                        "annotation_id": annotation["id"],
                        "relative_area": relative,
                        "size_bin": bucket,
                        "aspect_ratio": max(width / height, height / width),
                    }
                )
        split_summaries[split] = {
            "images": len(images),
            "positive_images": positives,
            "negative_images": len(images) - positives,
            "annotations": len(coco["annotations"]),
            "source_groups": len(groups_by_split[split]),
            "positive_source_groups": len(positive_groups_by_split[split]),
            "images_per_source_group": len(images) / max(len(groups_by_split[split]), 1),
            "positive_image_fraction": positives / len(images) if images else 0.0,
            "median_relative_area": float(np.median(relative_areas)) if relative_areas else 0.0,
            "size_counts": dict(size_counts),
        }

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                errors.append(f"{len(overlap)} source_group(s) presentes em {left} e {right}")

    checked_boxes, yolo_errors = yolo_equivalence(dataset_root)
    errors.extend(yolo_errors)

    tree = BKTree()
    near_duplicates = []
    for item in phash_items:
        value = perceptual_hash(item["path"])
        for distance, previous in tree.query(value, phash_threshold):
            if previous["split"] == item["split"]:
                continue
            near_duplicates.append(
                {
                    "distance": distance,
                    "split_a": previous["split"],
                    "file_a": previous["file_name"],
                    "group_a": previous["source_group"],
                    "split_b": item["split"],
                    "file_b": item["file_name"],
                    "group_b": item["source_group"],
                }
            )
        tree.add(value, item)
    near_duplicates.sort(key=lambda row: (row["distance"], row["split_a"], row["file_a"]))
    if near_duplicates:
        warnings.append(
            f"{len(near_duplicates)} par(es) perceptualmente semelhantes entre splits exigem revisão visual."
        )

    train_summary, val_summary, test_summary = (split_summaries[split] for split in SPLITS)
    positive_gap = abs(val_summary["positive_image_fraction"] - test_summary["positive_image_fraction"])
    if positive_gap > 0.05:
        warnings.append(f"Diferença de imagens positivas entre val/test: {positive_gap:.3f}")
    val_median, test_median = val_summary["median_relative_area"], test_summary["median_relative_area"]
    median_ratio = max(val_median, test_median) / max(min(val_median, test_median), 1e-12)
    if median_ratio > 1.5:
        warnings.append(f"Razão entre medianas de área relativa val/test: {median_ratio:.2f}")

    train_evaluation = {}
    train_boxes = max(train_summary["annotations"], 1)
    train_small_fraction = train_summary["size_counts"].get("small", 0) / train_boxes
    for split in ("val", "test"):
        evaluation_summary = split_summaries[split]
        evaluation_boxes = max(evaluation_summary["annotations"], 1)
        evaluation_small_fraction = evaluation_summary["size_counts"].get("small", 0) / evaluation_boxes
        area_ratio = max(train_summary["median_relative_area"], evaluation_summary["median_relative_area"]) / max(
            min(train_summary["median_relative_area"], evaluation_summary["median_relative_area"]), 1e-12
        )
        small_fraction_gap = abs(train_small_fraction - evaluation_small_fraction)
        positive_fraction_gap = abs(
            train_summary["positive_image_fraction"] - evaluation_summary["positive_image_fraction"]
        )
        train_evaluation[split] = {
            "median_relative_area_ratio": area_ratio,
            "small_box_fraction_gap": small_fraction_gap,
            "positive_image_fraction_gap": positive_fraction_gap,
        }
        if area_ratio > 1.5:
            warnings.append(f"Razão entre medianas de área relativa train/{split}: {area_ratio:.2f}")
        if small_fraction_gap > 0.20:
            warnings.append(
                f"Diferença na fração de caixas pequenas entre train/{split}: {small_fraction_gap:.3f}"
            )
    if train_summary["images_per_source_group"] > 1.5:
        warnings.append(
            "Treino contém em média "
            f"{train_summary['images_per_source_group']:.2f} variantes por source_group; "
            "o total de imagens superestima a diversidade efetiva."
        )

    write_manifest(assignment_rows, report_root / "image_manifest.csv")
    write_manifest(annotations_rows, report_root / "annotation_distribution.csv")
    write_manifest(
        near_duplicates,
        report_root / "near_duplicate_candidates.csv",
        ["distance", "split_a", "file_a", "group_a", "split_b", "file_b", "group_b"],
    )
    summary_rows = []
    for split, summary in split_summaries.items():
        summary_rows.append(
            {
                "split": split,
                **{key: value for key, value in summary.items() if key != "size_counts"},
                **{f"boxes_{key}": summary["size_counts"].get(key, 0) for key in ("small", "medium", "large")},
            }
        )
    write_manifest(summary_rows, report_root / "split_summary.csv")
    create_plots(split_summaries, annotations_rows, report_root)

    fingerprint = hashlib.sha256()
    for split in SPLITS:
        fingerprint.update((dataset_root / "annotations" / f"{split}.json").read_bytes())
    for row in sorted(assignment_rows, key=lambda item: (item["split"], item["file_name"])):
        fingerprint.update(str(row["sha256"]).encode())

    report = {
        "dataset": dataset_root.name,
        "dataset_root": str(dataset_root),
        "status": "failed" if errors else ("warning" if warnings else "passed"),
        "dataset_sha256": fingerprint.hexdigest(),
        "thresholds": {
            "relative_small_max": small_max,
            "relative_medium_max": medium_max,
            "perceptual_hash_hamming": phash_threshold,
        },
        "splits": split_summaries,
        "coco_yolo_boxes_checked": checked_boxes,
        "coco_yolo_equivalence": not yolo_errors,
        "exact_duplicates_across_splits": sum("Duplicata exata" in item for item in errors),
        "source_group_leakage": sum("source_group" in item for item in errors),
        "near_duplicate_candidates": len(near_duplicates),
        "validation_test": {
            "positive_fraction_gap": positive_gap,
            "median_relative_area_ratio": median_ratio,
        },
        "train_evaluation": train_evaluation,
        "errors": errors,
        "warnings": warnings,
    }
    save_json(report, report_root / "audit_report.json")
    (report_root / "audit_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return report


def create_plots(summaries: dict[str, Any], annotations: list[dict[str, Any]], report_root: Path) -> None:
    labels = list(SPLITS)
    positives = [summaries[split]["positive_images"] for split in labels]
    negatives = [summaries[split]["negative_images"] for split in labels]
    positions = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(positions, positives, label="Com crack")
    plt.bar(positions, negatives, bottom=positives, label="Sem crack")
    plt.xticks(positions, labels)
    plt.ylabel("Imagens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_root / "split_composition.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for split in SPLITS:
        values = [row["relative_area"] for row in annotations if row["split"] == split]
        if values:
            plt.hist(values, bins=40, alpha=0.45, label=split, density=True)
    plt.xlim(left=0)
    plt.xlabel("Área relativa da bounding box")
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_root / "relative_area_distribution.png", dpi=160)
    plt.close()


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Auditoria — {report['dataset']}",
        "",
        f"- Status: **{report['status']}**",
        f"- SHA-256 do dataset: `{report['dataset_sha256']}`",
        f"- Equivalência COCO–YOLO: **{'sim' if report['coco_yolo_equivalence'] else 'não'}**",
        f"- Caixas COCO–YOLO verificadas: {report['coco_yolo_boxes_checked']}",
        f"- Duplicatas exatas entre splits: {report['exact_duplicates_across_splits']}",
        f"- Vazamentos de source_group: {report['source_group_leakage']}",
        f"- Candidatos perceptuais entre splits: {report['near_duplicate_candidates']}",
        "",
        "## Resumo dos splits",
        "",
        "| Split | Imagens | Positivas | Negativas | Boxes | Pequenas | Médias | Grandes | Mediana área relativa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = report["splits"][split]
        lines.append(
            f"| {split} | {item['images']} | {item['positive_images']} | {item['negative_images']} | "
            f"{item['annotations']} | {item['size_counts'].get('small', 0)} | "
            f"{item['size_counts'].get('medium', 0)} | {item['size_counts'].get('large', 0)} | "
            f"{item['median_relative_area']:.6f} |"
        )
    lines.extend(["", "## Erros", ""])
    lines.extend([f"- {item}" for item in report["errors"]] or ["- Nenhum."])
    lines.extend(["", "## Avisos", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- Nenhum."])
    lines.extend(
        [
            "",
            "## Arquivos auxiliares",
            "",
            "- `split_summary.csv`",
            "- `image_manifest.csv`",
            "- `annotation_distribution.csv`",
            "- `near_duplicate_candidates.csv`",
            "- `original_vs_v2.csv` e `original_vs_v2.json`",
            "- `split_composition.png`",
            "- `relative_area_distribution.png`",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_dataset(config_path: Path, project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["dataset"]
    source_root = (project_root / config["source_root"]).resolve()
    output_root = (project_root / config["output_dir"]).resolve()
    report_root = (project_root / config["report_dir"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"A saída deve estar vazia ou não existir: {output_root}")
    if report_root.exists() and any(report_root.iterdir()):
        raise FileExistsError(f"A pasta de relatório deve estar vazia ou não existir: {report_root}")

    small_max = float(config.get("relative_size", {}).get("small_max", 0.01))
    medium_max = float(config.get("relative_size", {}).get("medium_max", 0.05))
    records, original_summary = load_records(
        source_root,
        str(config["target_class"]),
        list(config.get("class_aliases", [])),
        small_max,
        medium_max,
    )
    records, duplicates_removed = deduplicate_records(records, project_root)
    train_records = [record for record in records if record.source_split == "train"]
    evaluation_records = [record for record in records if record.source_split in {"val", "test"}]
    train_groups = {record.group for record in train_records}
    evaluation_groups = {record.group for record in evaluation_records}
    overlap = train_groups & evaluation_groups
    evaluation_excluded = [record for record in evaluation_records if record.group in overlap]
    evaluation_records = [record for record in evaluation_records if record.group not in overlap]

    evaluation = assign_evaluation_groups(
        evaluation_records,
        float(config.get("evaluation_val_ratio", 0.5)),
        int(config.get("seed", 42)),
        small_max,
        medium_max,
    )
    assigned = {"train": train_records, **evaluation}
    link_mode = str(config.get("link_mode", "hardlink"))
    manifest_rows = materialize_coco(
        assigned,
        output_root,
        str(config["target_class"]),
        link_mode,
        small_max,
        medium_max,
    )
    write_manifest(manifest_rows, output_root / "split_manifest.csv")
    convert_to_yolo(output_root, link_mode)
    output_summary = {split: summarize_records(assigned[split], small_max, medium_max) for split in SPLITS}
    preparation = {
        "dataset_name": str(config["name"]),
        "target_class": str(config["target_class"]),
        "seed": int(config.get("seed", 42)),
        "strategy": "keep_preaugmented_train_and_rebalance_unaugmented_evaluation_pool",
        "source_root": str(config["source_root"]),
        "output_dir": str(config["output_dir"]),
        "evaluation_val_ratio": float(config.get("evaluation_val_ratio", 0.5)),
        "relative_size": {"small_max": small_max, "medium_max": medium_max},
        "original_summary": original_summary,
        "duplicates_removed": duplicates_removed,
        "evaluation_excluded_due_to_train_group": [
            {
                "source_split": record.source_split,
                "source_file_name": record.source_file_name,
                "source_group": record.group,
                "sha256": record.sha256,
            }
            for record in evaluation_excluded
        ],
        "output_summary": output_summary,
        "train_evaluation_group_overlap_before_filter": len(overlap),
        "train_evaluation_group_overlap": 0,
    }
    save_json(preparation, output_root / "preparation_report.json")
    audit = audit_dataset(
        output_root,
        report_root,
        small_max,
        medium_max,
        int(config.get("perceptual_hash_hamming", 6)),
    )
    save_json(
        {
            "dataset": str(config["name"]),
            "dataset_sha256": audit["dataset_sha256"],
            "audit_status": audit["status"],
            "coco_yolo_equivalence": audit["coco_yolo_equivalence"],
            "seed": int(config.get("seed", 42)),
        },
        output_root / "dataset_fingerprint.json",
    )
    save_json(
        {"original": original_summary, "v2": output_summary},
        report_root / "original_vs_v2.json",
    )
    comparison_rows = []
    for version, summaries in (("original", original_summary), ("v2", output_summary)):
        for split, summary in summaries.items():
            comparison_rows.append(
                {
                    "version": version,
                    "split": split,
                    **{key: value for key, value in summary.items() if key != "size_counts"},
                    **{
                        f"boxes_{key}": summary["size_counts"].get(key, 0)
                        for key in ("small", "medium", "large")
                    },
                }
            )
    write_manifest(comparison_rows, report_root / "original_vs_v2.csv")
    return preparation, audit
