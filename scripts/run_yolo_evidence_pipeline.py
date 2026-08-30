#!/usr/bin/env python3
"""Reproduz o melhor YOLO anterior e executa a adaptação ao dataset v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PIPELINE = ROOT / "scripts" / "train_with_retries.py"
DEFAULT_SOURCE_CONFIG = ROOT / "configs" / "yolo_source_reproduction.yaml"
DEFAULT_TARGET_CONFIG = ROOT / "configs" / "yolo_domain_adaptation.yaml"
SOURCE_CHECKPOINT = (
    ROOT
    / "runs"
    / "aircraft_crack"
    / "yolo"
    / "yolo11n_source_reproduction__attempt_01_previous_best_safe_augmentation"
    / "weights"
    / "best.pt"
)
SOURCE_REPORT = ROOT / "runs" / "aircraft_crack" / "pipelines" / "yolo_source_reproduction" / "pipeline_report.json"


def run_stage(config: Path, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(TRAIN_PIPELINE),
        "--model",
        "yolo",
        "--config",
        str(config),
    ]
    if dry_run:
        command.append("--dry-run")
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_source_map5095(report_path: Path = SOURCE_REPORT) -> float:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return float(payload["selected"]["metrics"]["coco_map5095"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    parser.add_argument(
        "--minimum-source-map5095",
        type=float,
        default=0.20,
        help="Interrompe antes da adaptação se a reprodução ficar abaixo deste mAP50-95.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_stage(args.source_config.resolve(), args.dry_run)
    if not args.dry_run and not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint da etapa de origem não encontrado: {SOURCE_CHECKPOINT}")
    if not args.dry_run:
        source_score = read_source_map5095()
        print(f"Reprodução concluída: mAP50-95={source_score:.4f}", flush=True)
        if source_score < args.minimum_source_map5095:
            raise RuntimeError(
                "A reprodução do resultado anterior ficou abaixo do mínimo "
                f"({source_score:.4f} < {args.minimum_source_map5095:.4f}). "
                "A adaptação foi cancelada para evitar conclusões inválidas."
            )
    run_stage(args.target_config.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
