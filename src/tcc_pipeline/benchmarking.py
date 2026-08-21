from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .coco import save_json


def synchronize(device) -> None:
    if getattr(device, "type", str(device).split(":")[0]) == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed_inference(device, durations: list[float]):
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    durations.append(time.perf_counter() - start)


def save_benchmark(path: str | Path, durations: list[float], parameter_count: int, warmup: int = 0) -> None:
    measured = durations[min(warmup, len(durations)) :]
    total = sum(measured)
    save_json(
        {
            "images": len(measured),
            "warmup_images": min(warmup, len(durations)),
            "latency_ms_mean": 1000 * total / len(measured) if measured else None,
            "fps": len(measured) / total if total else None,
            "parameters": int(parameter_count),
            "scope": "model inference and framework post-processing; image loading excluded",
        },
        path,
    )
