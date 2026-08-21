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

from tcc_pipeline.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    tracking = config.get("tracking", {})
    backend_uri = tracking.get("uri")
    if not backend_uri:
        raise ValueError("tracking.uri não foi definido na configuração")
    artifact_root = Path(tracking.get("artifact_root", ROOT / "mlartifacts")).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        str(backend_uri),
        "--default-artifact-root",
        artifact_root.as_uri(),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("MLflow UI:", f"http://{args.host}:{args.port}")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
