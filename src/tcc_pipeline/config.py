from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Configuração inválida: {path}")
    root = path.resolve().parent.parent
    tracking = cfg.get("tracking", {})
    uri = str(tracking.get("uri", ""))
    if uri.startswith("sqlite:///"):
        database = Path(uri.removeprefix("sqlite:///"))
        if not database.is_absolute():
            database = (root / database).resolve()
        tracking["uri"] = f"sqlite:///{database.as_posix()}"
    artifact_root = tracking.get("artifact_root")
    if artifact_root:
        artifact_path = Path(artifact_root)
        tracking["artifact_root"] = str(
            artifact_path.resolve() if artifact_path.is_absolute() else (root / artifact_path).resolve()
        )
    return cfg


def project_root_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    # configs/project.yaml -> raiz do projeto
    return config_path.parent.parent


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()
