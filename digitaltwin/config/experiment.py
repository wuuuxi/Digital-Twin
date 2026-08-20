"""Small experiment-config helpers independent of ``Subject``."""

from __future__ import annotations

import copy
import json
import os
from typing import Any


def load_experiment_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_experiment_config(
    config: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    write: bool = False,
) -> dict[str, Any]:
    """Return a copied config and write it only when ``write=True``."""

    result = copy.deepcopy(config)
    if write:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
    return result


__all__ = ["load_experiment_config", "save_experiment_config"]
