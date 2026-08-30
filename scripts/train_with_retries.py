#!/usr/bin/env python3
"""Treina, avalia na validação e repete com parâmetros definidos no YAML."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = ("yolo", "faster_rcnn", "rtdetr")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tcc_pipeline.coco import save_json
from tcc_pipeline.config import load_config, model_run_dir, project_root_from_config, resolve_path


def save_attempts_comparison(attempts: list[dict], pipeline_dir: Path) -> tuple[Path, Path]:
    """Salva um resumo incremental e portátil de todas as validações."""
    json_path = pipeline_dir / "attempts_comparison.json"
    csv_path = pipeline_dir / "attempts_comparison.csv"
    save_json(attempts, json_path)

    metric_names = sorted({name for attempt in attempts for name in attempt.get("metrics", {})})
    fields = ["attempt", "variant", "run_name", "run_dir", *metric_names, "parameters"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attempt in attempts:
            row = {key: attempt.get(key, "") for key in fields}
            row.update(attempt.get("metrics", {}))
            row["parameters"] = json.dumps(attempt.get("parameters", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    return json_path, csv_path


def execute(command: list[object], dry_run: bool) -> None:
    print("\n$", " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run([str(item) for item in command], check=True)


def trained_artifact_exists(model_key: str, run_dir: Path) -> bool:
    if model_key == "yolo":
        return (run_dir / "weights" / "best.pt").is_file() and (run_dir / "training_complete.json").is_file()
    if model_key == "faster_rcnn":
        return (run_dir / "best.pth").is_file()
    return (run_dir / "best_model").is_dir()


def evaluation_arguments(cfg: dict, annotations: Path) -> list[object]:
    evaluation = cfg.get("evaluation", {})
    relative = evaluation.get("relative_size", {})
    return [
        "--gt",
        annotations,
        "--conf",
        evaluation.get("f1_confidence_threshold", 0.25),
        "--iou",
        evaluation.get("iou_threshold", 0.5),
        "--small-max",
        relative.get("small_max", 0.01),
        "--medium-max",
        relative.get("medium_max", 0.05),
    ]


def predict_and_evaluate(
    model_key: str,
    cfg: dict,
    root: Path,
    run_dir: Path,
    split: str,
    scripts: Path,
    py: str,
    dry_run: bool,
    tracking_config: Path,
) -> Path:
    images = resolve_path(root, cfg["paths"][f"{split}_images"])
    annotations = resolve_path(root, cfg["paths"][f"{split}_annotations"])
    predictions = run_dir / f"predictions_{split}.json"
    metrics = run_dir / f"metrics_{split}.json"
    prediction = cfg.get("prediction", {})
    confidence = prediction.get("confidence_threshold", 0.001)
    warmup = prediction.get("benchmark_warmup", 10)
    model = cfg["models"][model_key]

    if model_key == "yolo":
        command = [
            py,
            scripts / "predict_yolo.py",
            "--weights",
            run_dir / "weights" / "best.pt",
            "--images",
            images,
            "--annotations",
            annotations,
            "--mapping",
            resolve_path(root, cfg["paths"]["yolo_dataset_dir"]) / "category_mapping.json",
            "--out",
            predictions,
            "--imgsz",
            model["imgsz"],
            "--conf",
            confidence,
            "--device",
            model["device"],
            "--max-det",
            prediction.get("max_detections", 300),
            "--warmup",
            warmup,
        ]
    elif model_key == "faster_rcnn":
        command = [
            py,
            scripts / "predict_faster_rcnn.py",
            "--checkpoint",
            run_dir / "best.pth",
            "--images",
            images,
            "--annotations",
            annotations,
            "--out",
            predictions,
            "--conf",
            confidence,
            "--device",
            model["device"],
            "--warmup",
            warmup,
        ]
    else:
        command = [
            py,
            scripts / "predict_rtdetr.py",
            "--model",
            run_dir / "best_model",
            "--mapping",
            run_dir / "class_mapping.json",
            "--images",
            images,
            "--annotations",
            annotations,
            "--out",
            predictions,
            "--conf",
            confidence,
            "--warmup",
            warmup,
        ]
    execute(command, dry_run)
    execute(
        [py, scripts / "evaluate.py", *evaluation_arguments(cfg, annotations), "--pred", predictions, "--out", metrics],
        dry_run,
    )
    if not dry_run:
        execute(
            [
                py,
                scripts / "log_evaluation_to_mlflow.py",
                "--config",
                tracking_config,
                "--run-dir",
                run_dir,
                "--metrics",
                metrics,
                "--prefix",
                split,
            ],
            False,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "project.yaml"))
    parser.add_argument("--model", required=True, choices=(*MODEL_KEYS, "all"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.model == "all":
        for model_key in MODEL_KEYS:
            command = [sys.executable, Path(__file__).resolve(), "--config", args.config, "--model", model_key]
            if args.dry_run:
                command.append("--dry-run")
            execute(command, False)
        return
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    scripts = root / "scripts"
    py = sys.executable
    policy = cfg.get("retraining", {})
    if not policy.get("enabled", False):
        raise ValueError("retraining.enabled precisa estar habilitado")
    parameter_sets = policy.get("parameter_sets", {}).get(args.model, [])
    maximum = min(int(policy.get("max_attempts", len(parameter_sets))), len(parameter_sets))
    minimum = min(int(policy.get("min_attempts", 1)), maximum)
    if maximum < 1:
        raise ValueError(f"Nenhum conjunto de parâmetros configurado para {args.model}")
    if minimum < 1:
        raise ValueError("retraining.min_attempts deve ser pelo menos 1")

    pipeline_dir = resolve_path(root, cfg["paths"]["runs_dir"]) / cfg["dataset"]["name"] / "pipelines" / args.model
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    base_name = str(cfg["models"][args.model]["name"])
    metric_name = str(policy.get("metric", "coco_map5095"))
    threshold = float(policy.get("threshold", 0.30))
    mode = str(policy.get("mode", "max"))
    run_all = bool(policy.get("run_all_parameter_sets", False))
    reuse_completed = bool(policy.get("reuse_completed_attempts", True))
    attempts = []

    for attempt_number, parameter_set in enumerate(parameter_sets[:maximum], start=1):
        overrides = copy.deepcopy(parameter_set)
        variant = str(overrides.pop("label", f"variant_{attempt_number:02d}"))
        attempt_cfg = copy.deepcopy(cfg)
        attempt_cfg["models"][args.model].update(overrides)
        run_name = f"{base_name}__attempt_{attempt_number:02d}_{variant}"
        attempt_cfg["models"][args.model]["name"] = run_name
        config_file = pipeline_dir / f"attempt_{attempt_number:02d}.yaml"
        config_file.write_text(yaml.safe_dump(attempt_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        run_dir = model_run_dir(root, attempt_cfg, args.model, run_name)
        selection_split = str(policy.get("selection_split", "val"))
        metrics_file = run_dir / f"metrics_{selection_split}.json"
        training_completed = trained_artifact_exists(args.model, run_dir)
        evaluation_completed = metrics_file.is_file()
        if training_completed and evaluation_completed and reuse_completed and not args.dry_run:
            print(f"\n[reuse] Treino e avaliação já concluídos: {run_name}", flush=True)
        elif training_completed and reuse_completed and not args.dry_run:
            print(f"\n[reuse] Treino concluído; executando avaliação: {run_name}", flush=True)
            metrics_file = predict_and_evaluate(
                args.model,
                attempt_cfg,
                root,
                run_dir,
                selection_split,
                scripts,
                py,
                False,
                config_file,
            )
        else:
            execute([py, scripts / f"train_{args.model}.py", "--config", config_file], args.dry_run)
            metrics_file = predict_and_evaluate(
                args.model,
                attempt_cfg,
                root,
                run_dir,
                selection_split,
                scripts,
                py,
                args.dry_run,
                config_file,
            )
        if args.dry_run:
            continue
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        numeric_metrics = {
            key: float(value)
            for key, value in payload["metrics"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        value = numeric_metrics[metric_name]
        attempts.append(
            {
                "attempt": attempt_number,
                "variant": variant,
                "run_name": run_name,
                "run_dir": str(run_dir.relative_to(root)),
                "parameters": overrides,
                "metrics": numeric_metrics,
                metric_name: value,
            }
        )
        save_attempts_comparison(attempts, pipeline_dir)
        threshold_reached = any(
            item[metric_name] >= threshold if mode == "max" else item[metric_name] <= threshold for item in attempts
        )
        if not run_all and threshold_reached and attempt_number >= minimum:
            break

    if args.dry_run:
        return
    selector = max if mode == "max" else min
    selected = selector(attempts, key=lambda item: item[metric_name])
    selected_dir = resolve_path(root, selected["run_dir"])
    selected_config = pipeline_dir / f"attempt_{int(selected['attempt']):02d}.yaml"
    selected_cfg = load_config(selected_config)
    test_metrics = selected_dir / "metrics_test.json"
    if not (reuse_completed and test_metrics.is_file()):
        test_metrics = predict_and_evaluate(
            args.model, selected_cfg, root, selected_dir, "test", scripts, py, False, selected_config
        )
    comparison_json, comparison_csv = save_attempts_comparison(attempts, pipeline_dir)
    report = {
        "model": args.model,
        "selection_metric": metric_name,
        "selection_threshold": threshold,
        "minimum_attempts": minimum,
        "maximum_attempts": maximum,
        "run_all_parameter_sets": run_all,
        "reuse_completed_attempts": reuse_completed,
        "attempts": attempts,
        "selected": selected,
        "test_metrics": str(test_metrics.relative_to(root)),
        "comparison_json": str(comparison_json.relative_to(root)),
        "comparison_csv": str(comparison_csv.relative_to(root)),
    }
    save_json(report, pipeline_dir / "pipeline_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
