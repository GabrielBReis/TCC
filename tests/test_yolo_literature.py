from pathlib import Path

import yaml

from scripts.train_with_retries import save_attempts_comparison, trained_artifact_exists
from scripts.train_yolo import resolve_model_source, retain_best_checkpoint, train_without_ultralytics_mlflow


def test_retain_best_checkpoint_removes_other_yolo_weights(tmp_path: Path):
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    (weights / "epoch100.pt").write_bytes(b"periodic")

    best = retain_best_checkpoint(run_dir)

    assert best == weights / "best.pt"
    assert [path.name for path in weights.iterdir()] == ["best.pt"]


def test_completed_yolo_attempt_requires_marker_and_best_weight(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "best.pt").write_bytes(b"best")
    assert trained_artifact_exists("yolo", tmp_path) is False

    (tmp_path / "training_complete.json").write_text('{"status":"completed"}', encoding="utf-8")
    assert trained_artifact_exists("yolo", tmp_path) is True


def test_resolve_model_source_accepts_local_weights_and_ultralytics_architecture(tmp_path: Path):
    local = tmp_path / "models" / "yolo11n.pt"
    local.parent.mkdir()
    local.write_bytes(b"weights")

    assert resolve_model_source(tmp_path, "models/yolo11n.pt") == str(local.resolve())
    assert resolve_model_source(tmp_path, "yolo11n.yaml") == "yolo11n.yaml"


def test_ultralytics_autolog_is_disabled_only_during_training():
    from ultralytics.utils import SETTINGS

    original = SETTINGS.get("mlflow", False)

    class FakeModel:
        @staticmethod
        def train(**kwargs):
            assert SETTINGS["mlflow"] is False
            return kwargs["epochs"]

    assert train_without_ultralytics_mlflow(FakeModel(), epochs=3) == 3
    assert SETTINGS["mlflow"] == original


def test_yolo_experiment_pipeline_respects_hardware_limits():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "yolo_literature.yaml").read_text(encoding="utf-8"))
    policy = config["retraining"]
    attempts = policy["parameter_sets"]["yolo"]

    assert policy["run_all_parameter_sets"] is True
    assert policy["reuse_completed_attempts"] is True
    assert policy["min_attempts"] == policy["max_attempts"] == len(attempts) == 12
    assert [attempt["label"][:5] for attempt in attempts[:4]] == ["lit01", "lit02", "lit03", "lit04"]
    assert all(attempt["save_period"] == -1 for attempt in attempts)
    assert all(attempt["epochs"] == 100 for attempt in attempts)
    assert all(attempt["imgsz"] == 640 for attempt in attempts)
    assert all(1 <= attempt["batch"] <= 20 for attempt in attempts)
    assert all(attempt["augmentation"]["enabled"] is False for attempt in attempts[:10])
    assert all(attempt["augmentation"]["enabled"] is True for attempt in attempts[10:])
    assert len({attempt["label"] for attempt in attempts}) == len(attempts)
    assert all(attempt["experiment_group"] and attempt["hypothesis"] for attempt in attempts)

    assert attempts[0]["pretrained"] == "yolo11n.yaml"
    assert attempts[0]["optimizer"] == "SGD"
    assert attempts[0]["cos_lr"] is True
    assert attempts[1]["optimizer"] == "AdamW"
    assert attempts[1]["warmup_epochs"] == 3.0
    assert attempts[2]["batch"] == 20
    assert attempts[3]["imgsz"] == 640
    assert attempts[3]["optimizer"] == "Adam"
    assert attempts[4]["pretrained"] == "models/pretrained/yolo11n.pt"
    assert attempts[10]["augmentation"]["mosaic"] == 0.0
    assert attempts[11]["augmentation"]["mosaic"] == 0.5
    assert attempts[11]["augmentation"]["copy_paste"] == 0.0


def test_attempt_comparison_is_saved_as_json_and_csv(tmp_path: Path):
    attempts = [
        {
            "attempt": 1,
            "variant": "baseline",
            "run_name": "run_1",
            "run_dir": "runs/run_1",
            "parameters": {"batch": 16},
            "metrics": {"coco_map5095": 0.25, "overall_f1": 0.4},
            "coco_map5095": 0.25,
        }
    ]

    json_path, csv_path = save_attempts_comparison(attempts, tmp_path)

    assert json_path.is_file()
    assert csv_path.is_file()
    content = csv_path.read_text(encoding="utf-8")
    assert "coco_map5095" in content
    assert "overall_f1" in content
    assert '""batch"": 16' in content
