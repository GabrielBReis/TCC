from pathlib import Path

import yaml

from scripts.run_yolo_evidence_pipeline import read_source_map5095
from scripts.train_yolo import materialize_runtime_dataset_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_dataset_yaml_uses_absolute_dataset_root(tmp_path):
    yolo_dir = tmp_path / "dataset" / "yolo"
    yolo_dir.mkdir(parents=True)
    source = yolo_dir / "dataset.yaml"
    source.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: crack\n", encoding="utf-8")

    runtime = materialize_runtime_dataset_yaml(source, yolo_dir, tmp_path / "run")
    payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))

    assert Path(payload["path"]).is_absolute()
    assert Path(payload["path"]) == yolo_dir.resolve()
    assert payload["train"] == "images/train"


def test_evidence_pipeline_is_controlled_and_uses_source_checkpoint():
    source = yaml.safe_load((ROOT / "configs" / "yolo_source_reproduction.yaml").read_text(encoding="utf-8"))
    target = yaml.safe_load((ROOT / "configs" / "yolo_domain_adaptation.yaml").read_text(encoding="utf-8"))

    source_attempts = source["retraining"]["parameter_sets"]["yolo"]
    target_attempts = target["retraining"]["parameter_sets"]["yolo"]

    assert len(source_attempts) == 1
    assert source["retraining"]["pipeline_name"] == "yolo_source_reproduction"
    assert source_attempts[0]["optimizer"] == "auto"
    assert source_attempts[0]["augmentation"]["mosaic"] == 0.0
    assert len(target_attempts) == 6
    assert target["retraining"]["pipeline_name"] == "yolo_domain_adaptation"
    assert target["retraining"]["run_all_parameter_sets"] is True
    assert target["models"]["yolo"]["pretrained"].endswith("previous_best_safe_augmentation/weights/best.pt")
    assert [attempt["imgsz"] for attempt in target_attempts] == [640, 640, 640, 800, 960, 800]
    assert max(attempt["batch"] for attempt in target_attempts) <= 16
    assert all(attempt["epochs"] <= 100 for attempt in target_attempts)
    assert target_attempts[-1]["freeze"] == 10


def test_source_reproduction_metric_is_read_from_pipeline_report(tmp_path):
    report = tmp_path / "pipeline_report.json"
    report.write_text(
        '{"selected": {"metrics": {"coco_map5095": 0.2847}}}',
        encoding="utf-8",
    )

    assert read_source_map5095(report) == 0.2847
