#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import json
from pathlib import Path

from tcc_pipeline.config import load_config
from tcc_pipeline.tracking import log_metrics_to_existing_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    log_metrics_to_existing_run(
        cfg, args.run_dir, {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, [args.metrics]
    )
    print("Métricas enviadas ao MLflow quando o tracking está habilitado.")


if __name__ == "__main__":
    main()
