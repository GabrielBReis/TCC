#!/usr/bin/env python3
"""Cria o dataset crack v2 e gera uma auditoria reproduzível dos splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.config import find_project_root
from tcc_pipeline.dataset_v2 import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dataset_aircraft_surface_damage_v2.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    preparation, audit = prepare_dataset(config_path, find_project_root(config_path))
    print(
        json.dumps(
            {
                "dataset": preparation["dataset_name"],
                "output_summary": preparation["output_summary"],
                "audit_status": audit["status"],
                "dataset_sha256": audit["dataset_sha256"],
                "errors": audit["errors"],
                "warnings": audit["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
