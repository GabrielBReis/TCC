#!/usr/bin/env python3
from __future__ import annotations

import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))
if str(_ROOT / "scripts") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "scripts"))

import argparse

import mlflow
from train_yolo import log_yolo_results

from tcc_pipeline.config import load_config, project_root_from_config, resolve_path


def main():
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Sincroniza métricas e artefatos de um treino YOLO existente.")
    parser.add_argument("--config", default=str(_ROOT / "configs" / "project.yaml"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    run_dir = resolve_path(root, args.run_dir)
    id_file = run_dir / "mlflow_run_id.txt"
    if not id_file.exists():
        raise FileNotFoundError(f"ID da execução não encontrado: {id_file}")
    mlflow.set_tracking_uri(cfg["tracking"]["uri"])
    with mlflow.start_run(run_id=id_file.read_text(encoding="utf-8").strip()):
        log_yolo_results(cfg, run_dir)
    print("Sincronização concluída:", run_dir)


if __name__ == "__main__":
    main()
