from pathlib import Path

import torch
import yaml
from PIL import Image

from tcc_pipeline.datasets import DetectionAugmentation


def test_detection_augmentation_keeps_boxes_aligned_after_flips():
    image = Image.new("RGB", (100, 80), "white")
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    augmentation = DetectionAugmentation(
        {
            "horizontal_flip": 1.0,
            "vertical_flip": 1.0,
        }
    )

    transformed_image, transformed_boxes, keep = augmentation(image, boxes)

    assert transformed_image.size == image.size
    assert keep.tolist() == [True]
    torch.testing.assert_close(transformed_boxes, torch.tensor([[70.0, 40.0, 90.0, 60.0]]))


def test_retraining_compares_baseline_with_crack_safe_augmentation():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "project.yaml").read_text(encoding="utf-8"))
    retraining = config["retraining"]

    assert retraining["min_attempts"] == 3
    for model in ("yolo", "faster_rcnn", "rtdetr"):
        attempts = retraining["parameter_sets"][model]
        assert attempts[0]["label"] == "baseline"
        assert attempts[1]["label"] == "crack_safe_augmentation"
        assert attempts[2]["label"] == "expanded_augmentation"
        augmentation = attempts[1]["augmentation"]
        assert augmentation["enabled"] is True
        assert augmentation["rotation_degrees"] == 10.0
        assert augmentation["translate"] == 0.05
        assert augmentation.get("mosaic", 0.0) == 0.0
        assert augmentation.get("mixup", 0.0) == 0.0
        assert augmentation.get("copy_paste", 0.0) == 0.0

        expanded = attempts[2]["augmentation"]
        assert expanded["hue"] > 0.0
        assert expanded["shear_degrees"] > 0.0
        assert expanded["perspective"] > 0.0

    yolo_expanded = retraining["parameter_sets"]["yolo"][2]["augmentation"]
    assert yolo_expanded["mosaic"] > 0.0
    assert yolo_expanded["mixup"] > 0.0
    assert yolo_expanded["copy_paste"] > 0.0
