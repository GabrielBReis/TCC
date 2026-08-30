import json
from pathlib import Path

import yaml
from PIL import Image

from tcc_pipeline.dataset_v2 import ImageRecord, deduplicate_records, prepare_dataset, source_group


def write_source_split(root: Path, folder: str, images: list[tuple[str, list[float] | None]]) -> None:
    coco_images = []
    annotations = []
    folder_offset = {"train": 20, "valid": 90, "test": 160}[folder]
    for index, (name, bbox) in enumerate(images, start=1):
        Image.new("RGB", (100, 100), color=(folder_offset + index, index * 20, 80)).save(root / folder / name)
        coco_images.append({"id": index, "file_name": name, "width": 100, "height": 100})
        if bbox is not None:
            annotations.append({"id": index, "image_id": index, "category_id": 2, "bbox": bbox})
    payload = {
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "dent"}, {"id": 2, "name": "crack"}],
    }
    (root / folder / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_source_group_removes_roboflow_hash():
    assert source_group("IMG_001_jpg.rf.0123456789abcdef0123456789abcdef.jpg") == "img_001"
    assert source_group("IMG_001__01_jpg.rf.0123456789abcdef0123456789abcdef.jpg") == "img_001"


def test_duplicate_report_uses_project_relative_paths(tmp_path):
    source = tmp_path / "data" / "raw"
    source.mkdir(parents=True)
    first = source / "first.jpg"
    second = source / "second.jpg"
    first.write_bytes(b"same image")
    second.write_bytes(b"same image")
    records = [
        ImageRecord("train", first, first.name, "first", 10, 10, []),
        ImageRecord("val", second, second.name, "second", 10, 10, []),
    ]

    _, removed = deduplicate_records(records, tmp_path)

    assert removed[0]["kept"] == "data/raw/first.jpg"
    assert removed[0]["removed"] == "data/raw/second.jpg"


def test_prepare_v2_keeps_train_groups_and_balances_evaluation(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\nversion='0.0.0'\n", encoding="utf-8")
    source = tmp_path / "data" / "raw" / "aircraft_surface_damage"
    for folder in ("train", "valid", "test"):
        (source / folder).mkdir(parents=True)
    write_source_split(
        source,
        "train",
        [
            ("train_a_jpg.rf.00000000000000000000000000000001.jpg", [10, 10, 20, 20]),
            ("train_a_jpg.rf.00000000000000000000000000000002.jpg", [11, 10, 20, 20]),
            ("train_b_jpg.rf.00000000000000000000000000000003.jpg", None),
        ],
    )
    write_source_split(
        source,
        "valid",
        [
            ("val_small_jpg.rf.00000000000000000000000000000004.jpg", [10, 10, 5, 5]),
            ("val_neg.jpg", None),
            ("train_a__01_jpg.rf.00000000000000000000000000000006.jpg", [10, 10, 20, 20]),
        ],
    )
    write_source_split(
        source,
        "test",
        [("test_large_jpg.rf.00000000000000000000000000000005.jpg", [10, 10, 40, 40]), ("test_neg.jpg", None)],
    )
    config = tmp_path / "configs" / "dataset_aircraft_surface_damage_v2.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "aircraft_surface_damage_crack_v2",
                    "target_class": "crack",
                    "source_root": "data/raw/aircraft_surface_damage",
                    "output_dir": "data/processed/datasets/aircraft_surface_damage_crack_v2",
                    "report_dir": "reports/dataset_audit/aircraft_surface_damage_crack_v2",
                    "evaluation_val_ratio": 0.5,
                    "seed": 42,
                    "link_mode": "hardlink",
                    "perceptual_hash_hamming": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    preparation, audit = prepare_dataset(config, tmp_path)

    assert preparation["output_summary"]["train"]["images"] == 3
    assert preparation["train_evaluation_group_overlap_before_filter"] == 1
    assert preparation["train_evaluation_group_overlap"] == 0
    assert len(preparation["evaluation_excluded_due_to_train_group"]) == 1
    assert audit["coco_yolo_equivalence"] is True
    assert audit["errors"] == []
    assert audit["splits"]["val"]["images"] == 2
    assert audit["splits"]["test"]["images"] == 2
    fingerprint = json.loads(
        (
            tmp_path
            / "data/processed/datasets/aircraft_surface_damage_crack_v2/dataset_fingerprint.json"
        ).read_text(encoding="utf-8")
    )
    assert fingerprint["dataset_sha256"] == audit["dataset_sha256"]
    assert fingerprint["audit_status"] == audit["status"]
    near_duplicates = tmp_path / "reports/dataset_audit/aircraft_surface_damage_crack_v2/near_duplicate_candidates.csv"
    assert near_duplicates.read_text(encoding="utf-8").startswith("distance,split_a,file_a")
    train = json.loads(
        (tmp_path / "data/processed/datasets/aircraft_surface_damage_crack_v2/annotations/train.json").read_text(
            encoding="utf-8"
        )
    )
    train_a = [item for item in train["images"] if item["source_group"] == "train_a"]
    assert len(train_a) == 2
