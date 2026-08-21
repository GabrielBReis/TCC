#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse
import inspect

from tcc_pipeline.config import load_config, project_root_from_config, resolve_path
from tcc_pipeline.datasets import RTDetrCocoDataset, rtdetr_collate
from tcc_pipeline.tracking import save_metadata, tracked_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = project_root_from_config(args.config)
    m = cfg["models"]["rtdetr"].copy()
    if args.name:
        m["name"] = args.name

    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, Trainer, TrainingArguments

    pretrained = resolve_path(root, m["pretrained"])
    processor = RTDetrImageProcessor.from_pretrained(
        pretrained, size={"height": int(m["imgsz"]), "width": int(m["imgsz"])}
    )
    train_ds = RTDetrCocoDataset(
        resolve_path(root, cfg["paths"]["train_images"]),
        resolve_path(root, cfg["paths"]["train_annotations"]),
        processor,
    )
    val_ds = RTDetrCocoDataset(
        resolve_path(root, cfg["paths"]["val_images"]), resolve_path(root, cfg["paths"]["val_annotations"]), processor
    )
    if train_ds.id2label != val_ds.id2label:
        raise ValueError("Mapeamento de classes diferente entre treino e validação.")
    model = RTDetrForObjectDetection.from_pretrained(
        pretrained,
        num_labels=len(train_ds.id2label),
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )
    run_dir = resolve_path(root, cfg["paths"]["runs_dir"]) / "rtdetr" / m["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"idx0_to_coco_category_id": train_ds.idx0_to_cat, "id2label": train_ds.id2label}
    save_metadata(run_dir / "class_mapping.json", mapping)
    params = {"model": "PekingU/rtdetr_r18vd", **m, "num_classes": len(train_ds.id2label)}
    save_metadata(run_dir / "run_config.json", params)

    kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "num_train_epochs": float(m["epochs"]),
        "per_device_train_batch_size": int(m["train_batch"]),
        "per_device_eval_batch_size": int(m["eval_batch"]),
        "dataloader_num_workers": int(m["workers"]),
        "learning_rate": float(m["learning_rate"]),
        "weight_decay": float(m["weight_decay"]),
        "warmup_ratio": float(m.get("warmup_ratio", 0.0)),
        "fp16": bool(m.get("fp16", False)),
        "bf16": bool(m.get("bf16", False)),
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": int(cfg["project"].get("seed", 42)),
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=rtdetr_collate
    )
    with tracked_run(cfg, m["name"], run_dir, params):
        trainer.train()
        best_dir = run_dir / "best_model"
        trainer.save_model(str(best_dir))
        processor.save_pretrained(str(best_dir))
    print("Melhor modelo salvo em:", run_dir / "best_model")


if __name__ == "__main__":
    main()
