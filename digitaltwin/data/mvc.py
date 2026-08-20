"""Reusable multi-file MVC calculation and configuration helpers."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np

from .emg_processor import EMGProcessor


def compute_mvc_from_file_groups(
    file_groups: list[dict[str, Any]],
    subject,
    *,
    motion_flag: str = "all",
    remove_leading_zeros: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute per-muscle MVC across EMG file groups.

    Each group may use a different ``emg_folder``.  The maximum MVC across
    groups is returned together with per-file diagnostics for plotting.
    """

    musc_max = None
    per_file: dict[str, Any] = {}
    file_names: list[str] = []

    for group in file_groups:
        label = group["label"]
        emg_files = list(group.get("emg_files") or [])
        if not emg_files:
            continue
        emg_folder = group.get("emg_folder")
        if verbose:
            print(f"\n[{label}] EMG folder: {emg_folder}")
            print(f"[{label}] files ({len(emg_files)}): {emg_files}")

        result = EMGProcessor.compute_mvc_from_files(
            emg_files=emg_files,
            emg_folder=emg_folder,
            folder=subject.folder,
            fs=subject.emg_fs,
            musc_label=subject.musc_label,
            motion_flag=motion_flag,
            remove_leading_zeros=remove_leading_zeros,
        )
        values = np.asarray(result["musc_mvc"], dtype=float)
        musc_max = values if musc_max is None else np.maximum(musc_max, values)

        for file_name, data in result.get("per_file", {}).items():
            display_name = f"{label}/{file_name}"
            per_file[display_name] = data
            file_names.append(display_name)

    if musc_max is None:
        musc_max = np.zeros(len(subject.musc_label), dtype=float)

    return {
        "musc_mvc": [round(float(value), 4) for value in musc_max],
        "per_file": per_file,
        "file_names": file_names,
    }


def create_mvc_config(
    subject,
    musc_mvc,
    output_path: str | os.PathLike[str] | None = None,
    *,
    write: bool = False,
) -> tuple[dict[str, Any], str | None]:
    """Return a config containing MVC values and optionally write it.

    ``write`` defaults to ``False``.  When no path is supplied, the default
    output is the source config stem plus ``_mvc``.  The source file is never
    overwritten by this helper.
    """

    config = copy.deepcopy(subject.config)
    config.setdefault("emg_settings", {})["musc_mvc"] = [
        round(float(value), 4) for value in musc_mvc
    ]
    if output_path is None:
        base, ext = os.path.splitext(subject.config_path)
        output_path = f"{base}_mvc{ext}"
    output_path = os.fspath(output_path)
    if write:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
    return config, output_path if write else None


__all__ = ["compute_mvc_from_file_groups", "create_mvc_config"]
