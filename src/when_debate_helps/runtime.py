from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from .io import write_json


def parameter_counts(model: torch.nn.Module, perturbed_parameters: int | None = None) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "perturbed": perturbed_parameters or 0,
    }


def save_manifest(path: str | Path, config: dict[str, Any], counts: dict[str, int] | None = None) -> None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write_json(
        path,
        {
            "config": config,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "git_commit": commit,
            },
            "parameter_counts": counts,
        },
    )
