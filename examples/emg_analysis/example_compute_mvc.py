"""Compute MVC from configured EMG groups and write a derived config.

The reusable aggregation and config-writing code lives in
``digitaltwin.data.mvc``.  This example keeps only dataset selection,
diagnostic plotting, and the explicit ``write=True`` side effect.
"""

import matplotlib.pyplot as plt

from digitaltwin import Subject
from digitaltwin.data.mvc import (
    compute_mvc_from_file_groups,
    create_mvc_config,
)
from digitaltwin.visualization.mvc import (
    plot_artifact_pct_bar,
    plot_emg_signals_grid,
    plot_frequency_spectrum_grid,
    plot_mvc_candidates_bar,
    plot_psd_grid,
)


CONFIG_FILE = "../config/20260513_squat_FTS09_xsens.json"


def main():
    subject = Subject(CONFIG_FILE)

    modeling_emg_files = []
    for file_info in subject.modeling_data.values():
        emg_file = file_info.get("emg_file")
        if emg_file and emg_file not in modeling_emg_files:
            modeling_emg_files.append(emg_file)

    file_groups = [
        {
            "label": "mvc_file",
            "emg_folder": subject.emg_emg_folder,
            "emg_files": list(subject.mvc_files),
        },
        {
            "label": "modeling_file",
            "emg_folder": subject.modeling_emg_folder,
            "emg_files": modeling_emg_files,
        },
    ]

    result = compute_mvc_from_file_groups(
        file_groups,
        subject,
        motion_flag=subject.motion_flag,
        remove_leading_zeros=subject.remove_leading_zeros,
    )
    musc_mvc = result["musc_mvc"]
    print(f"MVC result: {musc_mvc[:6]}...")

    # Preserve the historical example behaviour explicitly: write a new
    # *_mvc.json, never overwrite the source config.
    _, output_path = create_mvc_config(subject, musc_mvc, write=True)
    print(f"MVC config saved to: {output_path}")

    muscles = subject.musc_label[:12]
    file_names = result["file_names"]
    if not file_names or not muscles:
        print("No data to plot")
        return

    per_file = result["per_file"]
    plot_emg_signals_grid(per_file, file_names, muscles, subject.emg_fs)
    plot_frequency_spectrum_grid(per_file, file_names, muscles, subject.emg_fs)
    plot_psd_grid(per_file, file_names, muscles, subject.emg_fs)
    plot_artifact_pct_bar(per_file, file_names, muscles)
    plot_mvc_candidates_bar(
        per_file, file_names, muscles, subject.musc_label, musc_mvc)
    plt.show()


if __name__ == "__main__":
    main()
