#!/usr/bin/env python3
"""Start the local MLflow server configured by the project YAML."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.config import load_config, project_root_from_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "project.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--allowed-hosts", help="Hosts separados por vírgula aceitos pelo servidor")
    parser.add_argument("--cors-allowed-origins", help="Origens HTTP separadas por vírgula aceitas pela interface")
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root_from_config(args.config)
    server = config.get("mlflow_server", {})
    backend_uri = server.get("backend_store_uri", "sqlite:///mlflow.db")
    if not backend_uri:
        raise ValueError("mlflow_server.backend_store_uri não foi definido")
    if str(backend_uri).startswith("sqlite:///"):
        database = resolve_path(root, str(backend_uri).removeprefix("sqlite:///"))
        backend_uri = f"sqlite:///{database.as_posix()}"
    artifacts_destination = resolve_path(root, server.get("artifacts_destination", "mlartifacts"))
    artifacts_destination.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        str(backend_uri),
        "--artifacts-destination",
        str(artifacts_destination),
        "--serve-artifacts",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.allowed_hosts:
        command.extend(["--allowed-hosts", args.allowed_hosts])
    if args.cors_allowed_origins:
        command.extend(["--cors-allowed-origins", args.cors_allowed_origins])
    print("MLflow UI:", f"http://{args.host}:{args.port}")
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
