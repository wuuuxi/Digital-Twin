"""Structured results for the standard data pipelines.

The project historically passed dictionaries containing keys such as
``aligned_data`` and ``cutted_data``.  These dataclasses provide named fields
for new callers while retaining a small, read-only mapping compatibility
surface for library code that is migrated incrementally.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class TrialMetadata:
    """Metadata attached to one fixed-load trial."""

    load_weight: str | None = None
    load_value: float | None = None
    load_mode: str | None = None
    source_files: dict[str, str] = field(default_factory=dict)
    robot_samples: int | None = None
    emg_samples: int | None = None
    processing_time: datetime | str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a metadata field, including values in ``extra``."""

        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None and not hasattr(self, key) and key not in self.extra:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow legacy metadata mutation during the migration window."""

        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value


@dataclass
class TrialResult:
    """Data produced while processing one fixed-load trial.

    ``segments`` is the new name for the historical ``cutted_data`` field.
    It remains a DataFrame (or, for legacy processors, a list of DataFrames)
    so this structural change does not alter analysis algorithms.
    """

    metadata: TrialMetadata = field(default_factory=TrialMetadata)
    robot_data: pd.DataFrame | None = None
    emg_data: dict[str, Any] | None = None
    xsens_data: dict[str, Any] | None = None
    aligned_data: pd.DataFrame | None = None
    segments: pd.DataFrame | list[pd.DataFrame] | None = None
    average_data: dict[str, Any] = field(default_factory=dict)

    @property
    def load_weight(self) -> str | None:
        return self.metadata.load_weight

    @property
    def load_value(self) -> float | None:
        return self.metadata.load_value

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibility accessor for code being migrated incrementally."""

        if key == "cutted_data":
            return self.segments
        if key == "metadata":
            return self.metadata
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None and key not in {
            "robot_data", "emg_data", "xsens_data", "aligned_data",
            "cutted_data", "segments", "average_data",
        }:
            raise KeyError(key)
        return value

    def __contains__(self, key: object) -> bool:
        return key in {
            "metadata", "load_weight", "load_value", "robot_data",
            "emg_data", "xsens_data", "aligned_data", "cutted_data",
            "segments", "average_data",
        }


@dataclass
class PipelineResults(Mapping[str, TrialResult]):
    """Mapping of canonical load keys to :class:`TrialResult` objects."""

    trials: dict[str, TrialResult] = field(default_factory=dict)

    def __getitem__(self, load_key: str) -> TrialResult:
        if load_key in self.trials:
            return self.trials[load_key]
        # Configurations historically mixed numeric keys and strings.  Keep
        # lookup forgiving while preserving the original key ordering.
        wanted = str(load_key)
        for key, value in self.trials.items():
            if str(key) == wanted:
                return value
        raise KeyError(load_key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.trials)

    def __len__(self) -> int:
        return len(self.trials)

    def get(self, load_key: str, default: Any = None) -> TrialResult | Any:
        try:
            return self[load_key]
        except KeyError:
            return default

    def collect_segments(self, movement_types=None) -> pd.DataFrame | None:
        """Combine trial segments and optionally filter movement types.

        The implementation delegates to the activation data collector when
        available so the result schema is shared by standard and activation
        workflows without making this model depend on a pipeline.
        """

        frames: list[pd.DataFrame] = []
        for load_key, result in self.trials.items():
            segments = result.segments
            if segments is None:
                continue
            if isinstance(segments, list):
                if not segments:
                    continue
                segments = pd.concat(segments, ignore_index=True)
            if len(segments) == 0:
                continue
            frame = segments.copy()
            if "load" not in frame.columns and result.load_value is not None:
                frame["load"] = result.load_value
            if "load_weight" not in frame.columns:
                frame["load_weight"] = load_key
            if "load_value" not in frame.columns and result.load_value is not None:
                frame["load_value"] = result.load_value
            frames.append(frame)

        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        if movement_types is not None and "movement_type" in combined.columns:
            wanted = set(str(item) for item in movement_types)
            # Keep force-derived isometric/isokinetic groups available for
            # activation workflows, matching the historical collector.
            keep = combined["movement_type"].astype(str).isin(wanted)
            keep |= combined["movement_type"].astype(str).isin(
                {"isometric", "isokinetic"})
            combined = combined.loc[keep].reset_index(drop=True)
        return combined if len(combined) else None
