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

import matplotlib.pyplot as plt
import pandas as pd


def flatten_metrics(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data.get("metrics", data)
    row = {"experiment": path.parent.name, "metrics_file": str(path)}
    row.update({k: v for k, v in m.items() if isinstance(v, (int, float))})
    inference_path = path.with_name("inference_metrics.json")
    if inference_path.exists():
        inference = json.loads(inference_path.read_text(encoding="utf-8"))
        row.update({f"inference_{k}": v for k, v in inference.items() if isinstance(v, (int, float))})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True, help="Arquivos metrics.json")
    ap.add_argument("--out", default="reports/comparison")
    args = ap.parse_args()
    rows = [flatten_metrics(Path(p)) for p in args.metrics]
    df = pd.DataFrame(rows)
    sort_col = "coco_map5095" if "coco_map5095" in df.columns else df.select_dtypes("number").columns[0]
    df = df.sort_values(sort_col, ascending=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    cols = [
        c
        for c in [
            "coco_map5095",
            "coco_map50",
            "overall_f1",
            "relative_small_f1",
            "relative_small_map50",
            "inference_latency_ms_mean",
            "inference_fps",
            "inference_parameters",
        ]
        if c in df.columns
    ]
    if cols:
        for c in cols:
            plt.figure(figsize=(9, 5))
            plt.bar(df["experiment"], df[c])
            plt.ylabel(c)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(out / f"{c}.png", dpi=160)
            plt.close()
    print(df[["experiment"] + cols].to_string(index=False))
    print("Saída:", out)


if __name__ == "__main__":
    main()
