#!/usr/bin/env python3
from __future__ import annotations

# Permite executar diretamente do repositório, mesmo antes de `pip install -e .`.
import sys as _sys
from pathlib import Path as _BootstrapPath

_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import importlib
import platform
import sys

PACKAGES = ["torch", "torchvision", "ultralytics", "transformers", "pycocotools", "mlflow", "yaml", "pandas", "PIL"]

print("Python:", sys.version.replace("\n", " "))
print("SO:", platform.platform())
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(mod, '__version__', 'instalado')}")
    except (ImportError, OSError, RuntimeError) as e:
        print(f"[ERRO] {name}: {e}")

try:
    import torch

    print("CUDA disponível:", torch.cuda.is_available())
    print("CUDA runtime (torch):", torch.version.cuda)
    print("GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except (ImportError, OSError, RuntimeError) as error:
    print(f"[ERRO] Não foi possível consultar CUDA: {error}")
