"""Feature injection API moved from the historical analysis namespace."""

from digitaltwin.analysis.feature_injector import (
    compute_mdf_for_results,
    compute_segmented_mdf_for_results,
    inject_emg_features,
    inject_xsens_features,
)

__all__ = [
    "inject_emg_features", "inject_xsens_features",
    "compute_mdf_for_results", "compute_segmented_mdf_for_results",
]
