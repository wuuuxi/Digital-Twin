"""Preparation of standard pipeline data for activation fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from digitaltwin.config_manager import numeric_load_value


def estimate_load_from_df(df: pd.DataFrame, g: float = 9.81) -> pd.Series:
    """Estimate load in kg from bilateral force and acceleration columns."""

    required = ["force_l", "force_r", "acc_l", "acc_r"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"estimate_load_from_df: missing columns {missing}")
    force_total = df["force_l"] + df["force_r"]
    denominator = (df["acc_l"] + df["acc_r"]) / 2.0 + g
    denominator = denominator.where(denominator.abs() > 1e-3, other=np.nan)
    return force_total / denominator


def collect_segments(results, movement_types=None, log=None) -> pd.DataFrame | None:
    """Collect fixed-load segments from a result mapping.

    The function accepts both the new ``PipelineResults``/``TrialResult``
    objects and the legacy dictionary shape during migration.
    """

    if not results:
        if log:
            log("Please run the pipeline before collecting segments.")
        return None

    if hasattr(results, "collect_segments"):
        combined = results.collect_segments(movement_types=movement_types)
        if combined is not None and log:
            log(f"Collected {len(combined)} segment rows.")
        return combined

    frames = []
    for load_weight, result in results.items():
        segments = result.get("segments", result.get("cutted_data"))
        if segments is None:
            continue
        if isinstance(segments, list):
            if not segments:
                continue
            segments = pd.concat(segments, ignore_index=True)
        if len(segments) == 0:
            continue
        frame = segments.copy()
        if "load" not in frame.columns:
            frame["load"] = result.get(
                "load_value", numeric_load_value(load_weight))
        if "load_weight" not in frame.columns:
            frame["load_weight"] = load_weight
        frames.append(frame)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if movement_types is not None and "movement_type" in combined.columns:
        wanted = set(str(item) for item in movement_types)
        keep = combined["movement_type"].astype(str).isin(wanted)
        keep |= combined["movement_type"].astype(str).isin(
            {"isometric", "isokinetic"})
        combined = combined.loc[keep].reset_index(drop=True)
    return combined if len(combined) else None
