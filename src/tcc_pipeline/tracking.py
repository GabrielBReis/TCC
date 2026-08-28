from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import is_uri


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def _flatten_params(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = value
    return flattened


@contextmanager
def tracked_run(
    cfg: dict[str, Any],
    run_name: str,
    run_dir: str | Path,
    params: dict[str, Any],
    model_key: str | None = None,
) -> Iterator[str | None]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        yield None
        return
    mlflow = _mlflow()
    if mlflow is None:
        yield None
        return

    mlflow.set_tracking_uri(tracking.get("uri", "sqlite:///mlflow.db"))
    if tracking.get("log_system_metrics", True) and hasattr(mlflow, "enable_system_metrics_logging"):
        mlflow.enable_system_metrics_logging()
    dataset_name = str(cfg.get("dataset", {}).get("name", "dataset"))
    model_name = model_key or str(params.get("model", "model"))
    experiment_name = str(tracking.get("experiment_pattern", "{prefix}/{dataset}/{model}")).format(
        prefix=tracking.get("experiment_prefix", cfg.get("project", {}).get("name", "tcc")),
        dataset=dataset_name,
        model=model_name,
    )
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_root = tracking.get("artifact_root")
        experiment_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", experiment_name).strip("_")
        if artifact_root and is_uri(artifact_root):
            artifact_uri = f"{str(artifact_root).rstrip('/')}/experiments/{experiment_slug}"
        else:
            artifact_uri = (
                (Path(artifact_root) / "experiments" / experiment_slug).resolve().as_uri() if artifact_root else None
            )
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten_params(params))
        mlflow.set_tags({"dataset": dataset_name, "model_family": model_name, "local_run_dir": str(run_dir)})
        run_id = run.info.run_id
        (run_dir / "mlflow_run_id.txt").write_text(run_id, encoding="utf-8")
        yield run_id


def log_artifact_if_enabled(cfg: dict[str, Any], path: str | Path) -> None:
    tracking = cfg.get("tracking", {})
    path = Path(path)
    if not tracking.get("enabled", False) or not path.exists():
        return
    mlflow = _mlflow()
    if mlflow is not None and mlflow.active_run() is not None:
        try:
            mlflow.log_artifact(str(path))
        except Exception as exc:  # noqa: BLE001 - tracking não invalida treino concluído
            warnings.warn(f"Falha ao registrar artefato no MLflow: {exc}", stacklevel=2)


def log_metrics_if_enabled(cfg: dict[str, Any], metrics: dict[str, Any], step: int | None = None) -> None:
    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        return
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    clean = {}
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metric_name = re.sub(r"[^a-zA-Z0-9_./ -]", "_", str(key).strip())
            clean[metric_name] = number
    if clean:
        try:
            mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # noqa: BLE001 - tracking não invalida treino concluído
            warnings.warn(f"Falha ao registrar métricas no MLflow: {exc}", stacklevel=2)


def log_table_if_enabled(cfg: dict[str, Any], rows: list[dict[str, Any]], artifact_file: str) -> None:
    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False) or not rows:
        return
    mlflow = _mlflow()
    if mlflow is not None and mlflow.active_run() is not None:
        import pandas as pd

        try:
            mlflow.log_table(data=pd.DataFrame(rows), artifact_file=artifact_file)
        except Exception as exc:  # noqa: BLE001 - tracking não invalida treino concluído
            warnings.warn(f"Falha ao registrar tabela no MLflow: {exc}", stacklevel=2)


def log_directory_if_enabled(cfg: dict[str, Any], path: str | Path, artifact_path: str) -> None:
    tracking = cfg.get("tracking", {})
    path = Path(path)
    if not tracking.get("enabled", False) or not path.is_dir():
        return
    mlflow = _mlflow()
    if mlflow is not None and mlflow.active_run() is not None:
        try:
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        except Exception as exc:  # noqa: BLE001 - tracking não invalida treino concluído
            warnings.warn(f"Falha ao registrar diretório no MLflow: {exc}", stacklevel=2)


def log_metrics_to_existing_run(
    cfg: dict[str, Any], run_dir: str | Path, metrics: dict[str, float], artifacts: list[str | Path] | None = None
) -> None:
    tracking = cfg.get("tracking", {})
    run_dir = Path(run_dir)
    id_file = run_dir / "mlflow_run_id.txt"
    if not tracking.get("enabled", False) or not id_file.exists():
        return
    mlflow = _mlflow()
    if mlflow is None:
        return
    mlflow.set_tracking_uri(tracking.get("uri", "sqlite:///mlflow.db"))
    run_id = id_file.read_text(encoding="utf-8").strip()
    with mlflow.start_run(run_id=run_id):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        for item in artifacts or []:
            p = Path(item)
            if p.exists():
                try:
                    mlflow.log_artifact(str(p))
                except Exception as exc:  # noqa: BLE001 - avaliação local permanece válida
                    warnings.warn(f"Falha ao registrar artefato de avaliação: {exc}", stacklevel=2)


def save_metadata(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
