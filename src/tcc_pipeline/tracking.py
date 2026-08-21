from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


@contextmanager
def tracked_run(
    cfg: dict[str, Any], run_name: str, run_dir: str | Path, params: dict[str, Any]
) -> Iterator[str | None]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tracking = cfg.get("tracking", {})
    mlflow = _mlflow()
    if not tracking.get("enabled", False) or mlflow is None:
        yield None
        return

    mlflow.set_tracking_uri(tracking.get("uri", "sqlite:///mlflow.db"))
    experiment_name = tracking.get("experiment_name", "aircraft_defects_tcc")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_root = tracking.get("artifact_root")
        artifact_uri = Path(artifact_root).resolve().as_uri() if artifact_root else None
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        clean = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool)) or v is None}
        mlflow.log_params(clean)
        run_id = run.info.run_id
        (run_dir / "mlflow_run_id.txt").write_text(run_id, encoding="utf-8")
        yield run_id


def log_artifact_if_enabled(cfg: dict[str, Any], path: str | Path) -> None:
    tracking = cfg.get("tracking", {})
    mlflow = _mlflow()
    path = Path(path)
    if tracking.get("enabled", False) and mlflow is not None and path.exists() and mlflow.active_run() is not None:
        mlflow.log_artifact(str(path))


def log_metrics_to_existing_run(
    cfg: dict[str, Any], run_dir: str | Path, metrics: dict[str, float], artifacts: list[str | Path] | None = None
) -> None:
    tracking = cfg.get("tracking", {})
    mlflow = _mlflow()
    run_dir = Path(run_dir)
    id_file = run_dir / "mlflow_run_id.txt"
    if not tracking.get("enabled", False) or mlflow is None or not id_file.exists():
        return
    mlflow.set_tracking_uri(tracking.get("uri", "sqlite:///mlflow.db"))
    run_id = id_file.read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        for item in artifacts or []:
            p = Path(item)
            if p.exists():
                mlflow.log_artifact(str(p))


def save_metadata(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
