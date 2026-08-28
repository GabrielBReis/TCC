from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms import functional as F

from .coco import annotations_by_image, category_maps, load_coco_json


class DetectionAugmentation:
    """Transformações leves que mantêm caixas alinhadas com a imagem."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        degrees = float(config.get("rotation_degrees", 0.0))
        translate = float(config.get("translate", 0.0))
        scale = float(config.get("scale", 0.0))
        shear = float(config.get("shear_degrees", 0.0))
        perspective = float(config.get("perspective", 0.0))
        horizontal_flip = float(config.get("horizontal_flip", 0.0))
        vertical_flip = float(config.get("vertical_flip", 0.0))
        for name, value in (("translate", translate), ("scale", scale)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"augmentation.{name} deve estar no intervalo [0, 1)")
        for name, value in (("horizontal_flip", horizontal_flip), ("vertical_flip", vertical_flip)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"augmentation.{name} deve estar no intervalo [0, 1]")
        if not 0.0 <= perspective <= 1.0:
            raise ValueError("augmentation.perspective deve estar no intervalo [0, 1]")

        transforms = []
        if degrees or translate or scale or shear:
            transforms.append(
                v2.RandomAffine(
                    degrees=(-degrees, degrees),
                    translate=(translate, translate) if translate else None,
                    scale=(1.0 - scale, 1.0 + scale) if scale else None,
                    shear=(-shear, shear) if shear else None,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=114,
                )
            )
        if perspective:
            transforms.append(
                v2.RandomPerspective(
                    distortion_scale=perspective,
                    p=1.0,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=114,
                )
            )
        if horizontal_flip:
            transforms.append(v2.RandomHorizontalFlip(horizontal_flip))
        if vertical_flip:
            transforms.append(v2.RandomVerticalFlip(vertical_flip))

        brightness = float(config.get("brightness", 0.0))
        contrast = float(config.get("contrast", 0.0))
        saturation = float(config.get("saturation", 0.0))
        hue = float(config.get("hue", 0.0))
        if any((brightness, contrast, saturation, hue)):
            transforms.append(
                v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
            )
        self.transform = v2.Compose(transforms)

    def __call__(self, image: Image.Image, boxes: torch.Tensor):
        width, height = image.size
        bounding_boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width))
        image, bounding_boxes = self.transform(image, bounding_boxes)
        transformed = bounding_boxes.as_subclass(torch.Tensor).to(dtype=torch.float32)
        transformed[:, 0::2].clamp_(0, width)
        transformed[:, 1::2].clamp_(0, height)
        keep = (transformed[:, 2] > transformed[:, 0]) & (transformed[:, 3] > transformed[:, 1])
        return image, transformed[keep], keep


class CocoDetectionTorchDataset(torch.utils.data.Dataset):
    """Dataset COCO para Faster R-CNN com labels 1..K (0 reservado para background)."""

    def __init__(
        self,
        images_dir: str | Path,
        annotation_file: str | Path,
        train: bool = False,
        hflip_prob: float = 0.0,
        augmentation: dict | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.coco = load_coco_json(annotation_file)
        self.images = sorted(self.coco["images"], key=lambda x: int(x["id"]))
        self.anns_by_image = annotations_by_image(self.coco)
        self.cat_to_idx0, self.idx0_to_cat, self.id2label, _ = category_maps(self.coco)
        self.train = train
        self.hflip_prob = hflip_prob
        augmentation = augmentation or {}
        self.augmentation = DetectionAugmentation(augmentation) if train and augmentation.get("enabled", False) else None

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

        if self.augmentation is not None:
            image, transformed_boxes, keep = self.augmentation(image, target["boxes"])
            target["boxes"] = transformed_boxes
            target["labels"] = target["labels"][keep]
            target["iscrowd"] = target["iscrowd"][keep]
            target["area"] = (transformed_boxes[:, 2] - transformed_boxes[:, 0]) * (
                transformed_boxes[:, 3] - transformed_boxes[:, 1]
            )
        elif self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = F.hflip(image)
            if target["boxes"].numel():
                b = target["boxes"].clone()
                b[:, [0, 2]] = width - b[:, [2, 0]]
                target["boxes"] = b

        image = F.to_tensor(image)
        return image, target


class RTDetrCocoDataset(torch.utils.data.Dataset):
    """Dataset COCO adaptado ao RT-DETR/Hugging Face, com category_id contíguo 0..K-1."""

    def __init__(
        self,
        images_dir: str | Path,
        annotation_file: str | Path,
        processor,
        train: bool = False,
        augmentation: dict | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.coco = load_coco_json(annotation_file)
        self.images = sorted(self.coco["images"], key=lambda x: int(x["id"]))
        self.anns_by_image = annotations_by_image(self.coco)
        self.cat_to_idx0, self.idx0_to_cat, self.id2label, self.label2id = category_maps(self.coco)
        self.processor = processor
        augmentation = augmentation or {}
        self.augmentation = DetectionAugmentation(augmentation) if train and augmentation.get("enabled", False) else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = int(info["id"])
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        anns = []
        boxes = []
        for ann in self.anns_by_image.get(image_id, []):
            x, y, w, h = map(float, ann["bbox"])
            if w <= 0 or h <= 0:
                continue
            converted = dict(ann)
            converted["category_id"] = self.cat_to_idx0[int(ann["category_id"])]
            converted["area"] = float(ann.get("area", w * h))
            converted["iscrowd"] = int(ann.get("iscrowd", 0))
            anns.append(converted)
            boxes.append([x, y, x + w, y + h])
        if self.augmentation is not None:
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            image, box_tensor, keep = self.augmentation(image, box_tensor)
            kept_indices = torch.where(keep)[0].tolist()
            transformed_anns = []
            for output_index, source_index in enumerate(kept_indices):
                converted = anns[source_index]
                x1, y1, x2, y2 = box_tensor[output_index].tolist()
                converted["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                converted["area"] = (x2 - x1) * (y2 - y1)
                transformed_anns.append(converted)
            anns = transformed_anns
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
