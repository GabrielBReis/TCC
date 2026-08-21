from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

from .coco import annotations_by_image, category_maps, load_coco_json


class CocoDetectionTorchDataset(torch.utils.data.Dataset):
    """Dataset COCO para Faster R-CNN com labels 1..K (0 reservado para background)."""

    def __init__(
        self, images_dir: str | Path, annotation_file: str | Path, train: bool = False, hflip_prob: float = 0.0
    ):
        self.images_dir = Path(images_dir)
        self.coco = load_coco_json(annotation_file)
        self.images = sorted(self.coco["images"], key=lambda x: int(x["id"]))
        self.anns_by_image = annotations_by_image(self.coco)
        self.cat_to_idx0, self.idx0_to_cat, self.id2label, _ = category_maps(self.coco)
        self.train = train
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = int(info["id"])
        image_path = self.images_dir / info["file_name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes, labels, areas, crowds = [], [], [], []
        for ann in self.anns_by_image.get(image_id, []):
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + w), min(float(height), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_to_idx0[int(ann["category_id"])] + 1)
            areas.append((x2 - x1) * (y2 - y1))
            crowds.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }

        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = F.hflip(image)
            if target["boxes"].numel():
                b = target["boxes"].clone()
                b[:, [0, 2]] = width - b[:, [2, 0]]
                target["boxes"] = b

        image = F.to_tensor(image)
        return image, target


class RTDetrCocoDataset(torch.utils.data.Dataset):
    """Dataset COCO adaptado ao RT-DETR/Hugging Face, com category_id contíguo 0..K-1."""

    def __init__(self, images_dir: str | Path, annotation_file: str | Path, processor):
        self.images_dir = Path(images_dir)
        self.coco = load_coco_json(annotation_file)
        self.images = sorted(self.coco["images"], key=lambda x: int(x["id"]))
        self.anns_by_image = annotations_by_image(self.coco)
        self.cat_to_idx0, self.idx0_to_cat, self.id2label, self.label2id = category_maps(self.coco)
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = int(info["id"])
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        anns = []
        for ann in self.anns_by_image.get(image_id, []):
            _, _, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            converted = dict(ann)
            converted["category_id"] = self.cat_to_idx0[int(ann["category_id"])]
            converted["area"] = float(ann.get("area", w * h))
            converted["iscrowd"] = int(ann.get("iscrowd", 0))
            anns.append(converted)
        target = {"image_id": image_id, "annotations": anns}
        enc = self.processor(images=image, annotations=target, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"].squeeze(0), "labels": enc["labels"][0]}


def detection_collate(batch):
    return tuple(zip(*batch))


def rtdetr_collate(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }
