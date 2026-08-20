"""EMG RMS example for fixed-load and variable-load trials.

The reusable plotting implementation lives in
``digitaltwin.visualization.emg_feature_plot``.  This example only selects a
configuration, runs the two pipeline branches, and passes structured
``PipelineResults`` to the public plotting API.
"""

import matplotlib.pyplot as plt

from digitaltwin import MultiLoadPipeline, Subject
from digitaltwin.visualization.emg_feature_plot import (
    plot_feature_bar_combined,
    plot_feature_vs_position_combined,
    plot_feature_vs_time_combined,
    plot_pos_vel_emg_feature_grid_combined,
)


FEATURE = "rms"
LABEL = "RMS (mV)"


def main():
    subject = Subject("../config/20250409_squat_NCMP001.json")
    pipeline = MultiLoadPipeline(subject)
    pipeline.debug = True

    fixed_results = pipeline.run(include_xsens=False, write=True)
    vload_results = pipeline.run_vload()
    muscles = subject.musc_label[:6]

    plot_feature_vs_time_combined(
        fixed_results, vload_results, muscles,
        feature=FEATURE, feature_label=LABEL)
    plot_feature_vs_position_combined(
        fixed_results, vload_results, muscles,
        feature=FEATURE, feature_label=LABEL)
    plot_pos_vel_emg_feature_grid_combined(
        fixed_results, vload_results, ["VL", "RF"],
        subject=subject, feature=FEATURE, feature_label=LABEL)
    plot_feature_bar_combined(
        fixed_results, vload_results, muscles,
        feature=FEATURE, feature_label=LABEL)

    plt.show()


if __name__ == "__main__":
    main()
