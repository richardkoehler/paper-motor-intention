"""Plot timelocked features."""
from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import mne_bids
import pte

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

PLOT_PATH = constants.PLOTS / "raw_emg.svg"


def task_plot_raw_emg(SUBJECT: str = "sub-EL014") -> None:
    file_finder = pte.filetools.get_filefinder(datatype="bids")
    file_finder.find_files(
        directory=constants.RAWDATA_ORIG,
        extensions=".vhdr",
        keywords=SUBJECT,
        medication="Off",
    )
    print(file_finder)
    raw = mne_bids.read_raw_bids(file_finder.files[0])
    # raw.plot(scalings="auto", block=True)
    raw.pick(["EMG_L_BR_TM"]).load_data().crop(tmin=56.1, tmax=64).resample(
        1000
    ).filter(
        l_freq=15, picks="all", h_freq=None  # 500,
    )
    events, _ = mne.events_from_annotations(raw, event_id={"EMG_onset": 1})
    epochs = mne.Epochs(raw, events, tmin=-3.0, tmax=2.0)
    data = epochs.get_data()
    data = data.squeeze()
    motor_intention.plotting_settings.activate()
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.plot(epochs.times, data, color="black", linewidth=0.1)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([-3, 0, 2])
    ax.set_xlabel("Time [s]")
    motor_intention.plotting_settings.save_fig(fig, PLOT_PATH)


if __name__ == "__main__":
    task_plot_raw_emg()
    plt.show(block=True)
