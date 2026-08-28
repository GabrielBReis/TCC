from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: str | Path) -> Path:
    """Encontra a raiz do repositório sem depender do sistema operacional ou CWD."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Raiz do projeto não encontrada a partir de: {start}")


def is_uri(value: str | Path) -> bool:
    text = str(value)
    return "://" in text or text.startswith(("mlflow-artifacts:", "file:"))


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Configuração inválida: {path}")
    root = find_project_root(path)
    tracking = cfg.setdefault("tracking", {})
    tracking["uri"] = os.getenv("TCC_MLFLOW_TRACKING_URI", str(tracking.get("uri", "sqlite:///mlflow.db")))
    tracking["experiment_prefix"] = os.getenv(
        "TCC_MLFLOW_EXPERIMENT_PREFIX",
        str(tracking.get("experiment_prefix", cfg.get("project", {}).get("name", "tcc"))),
    )
    tracking["artifact_root"] = os.getenv("TCC_MLFLOW_ARTIFACT_ROOT", str(tracking.get("artifact_root", "mlartifacts")))

    uri = str(tracking["uri"])
    if uri.startswith("sqlite:///"):
        database = Path(uri.removeprefix("sqlite:///"))
        if not database.is_absolute():
            database = (root / database).resolve()
        tracking["uri"] = f"sqlite:///{database.as_posix()}"
    artifact_root = str(tracking["artifact_root"])
    if artifact_root and not is_uri(artifact_root):
        artifact_path = Path(artifact_root)
        tracking["artifact_root"] = str(
            artifact_path.resolve() if artifact_path.is_absolute() else (root / artifact_path).resolve()
        )
    return cfg


def project_root_from_config(config_path: str | Path) -> Path:
    return find_project_root(config_path)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def model_run_dir(root: Path, cfg: dict[str, Any], model_key: str, run_name: str) -> Path:
    dataset_name = str(cfg.get("dataset", {}).get("name", "dataset"))
    return resolve_path(root, cfg["paths"]["runs_dir"]) / dataset_name / model_key / run_name
