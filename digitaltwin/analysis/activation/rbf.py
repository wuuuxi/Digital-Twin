"""RBF activation-surface API under the activation domain."""

from digitaltwin.analysis.heatmap.rbf_fitting import (
    compute_rmse_by_load,
    compute_rmse_percentage,
    fit_activation_map,
    fit_activation_map_3d,
    load_rbf_params,
    predict_at,
    rbf_fit,
    rbf_function,
    rbf_predict,
    save_rbf_params,
)

# Clearer domain name for new callers; the serialized format is intentionally
# unchanged in this migration phase.
load_rbf_model = load_rbf_params

__all__ = [
    "rbf_function", "rbf_fit", "rbf_predict", "predict_at",
    "fit_activation_map", "fit_activation_map_3d",
    "save_rbf_params", "load_rbf_params", "load_rbf_model",
    "compute_rmse_percentage", "compute_rmse_by_load",
]
