import json

import yaml

from scripts.prepare_single_class_dataset import prepare


def test_prepare_keeps_only_target_class_and_remaps_to_one(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0.0.0'\n", encoding="utf-8")
    source = tmp_path / "source"
    for split in ("train", "val", "test"):
        images = source / split / "images"
        images.mkdir(parents=True)
        (images / "crack.jpg").write_bytes(f"crack-{split}".encode())
        (images / "dent.jpg").write_bytes(f"dent-{split}".encode())
        coco = {
            "images": [
                {"id": 1, "file_name": "crack.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "dent.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 2, 10, 20]},
                {"id": 2, "image_id": 2, "category_id": 8, "bbox": [1, 2, 10, 20]},
            ],
            "categories": [{"id": 5, "name": "Crack"}, {"id": 8, "name": "Dent"}],
        }
        annotations = source / "annotations"
        annotations.mkdir(exist_ok=True)
        (annotations / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    config = tmp_path / "configs" / "dataset.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "crack_only",
                    "target_class": "crack",
                    "class_aliases": ["cracks"],
                    "keep_images_without_target": False,
                    "output_dir": "data/processed/crack_only",
                    "sources": [{"name": "sample", "root": "source", "format": "auto"}],
                }
            }
        ),
        encoding="utf-8",
    )
    report = prepare(config)
    assert report["output_summary"]["train"] == {"images": 1, "annotations": 1}
    output = json.loads((tmp_path / "data/processed/crack_only/annotations/train.json").read_text(encoding="utf-8"))
    assert output["categories"] == [{"id": 1, "name": "crack", "supercategory": "defect"}]
    assert {annotation["category_id"] for annotation in output["annotations"]} == {1}
