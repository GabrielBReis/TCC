#!/usr/bin/env python3
"""Cria o dataset crack v2 e gera uma auditoria reproduzível dos splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.config import find_project_root
from tcc_pipeline.dataset_v2 import audit_dataset, prepare_dataset
from tcc_pipeline.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dataset_aircraft_surface_damage_v2.yaml",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Regera somente a auditoria do dataset processado, sem recriar imagens ou anotações.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    project_root = find_project_root(config_path)
    if args.audit_only:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["dataset"]
        output_root = resolve_path(project_root, config["output_dir"])
        report_root = resolve_path(project_root, config["report_dir"])
        relative_size = config.get("relative_size", {})
        audit = audit_dataset(
            output_root,
            report_root,
            float(relative_size.get("small_max", 0.01)),
            float(relative_size.get("medium_max", 0.05)),
            int(config.get("perceptual_hash_hamming", 6)),
        )
        preparation = {
            "dataset_name": str(config["name"]),
            "output_summary": audit["splits"],
        }
    else:
        preparation, audit = prepare_dataset(config_path, project_root)
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
