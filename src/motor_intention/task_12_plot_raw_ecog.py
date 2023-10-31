"""Plot timelocked features."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import mne
import mne_bids
import pte
from pytask import Product

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

PLOT_PATH = constants.PLOTS / "raw_ecog.svg"


def task_plot_raw_ecog(
    subject: str = "sub-EL002",
    plot_path: Annotated[Path, Product] = PLOT_PATH,
) -> None:
    file_finder = pte.filetools.BIDSFinder()
    file_finder.find_files(
        directory=constants.RAWDATA_ORIG,
        extensions=".vhdr",
        keywords=[subject],
        medication="Off",
    )
    print(file_finder)
    raw = mne_bids.read_raw_bids(file_finder.files[0])
    raw.pick(["ECOG_L_06_SMC_AT"])  # .pick("ecog")  # .
    # raw.plot(scalings="auto", block=True, highpass=0.5)  # , lowpass=90)
    raw.load_data().crop(tmin=108, tmax=115).resample(1000).filter(
        l_freq=0.5, picks="all", h_freq=None  # 500,
    )
    events, _ = mne.events_from_annotations(raw, event_id={"EMG_onset": 1})
    epochs = mne.Epochs(raw, events, tmin=-3.0, tmax=2.0)
    data = epochs.get_data(units={"ecog": "uV", "dbs": "uV"})
    data = data.squeeze()
    motor_intention.plotting_settings.activate()
    fig, ax = plt.subplots(1, 1, figsize=(2.3, 0.5))
    ax.plot(epochs.times, data, color="black", linewidth=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([-3, 0, 2])
    ax.set_xticklabels([])
    motor_intention.plotting_settings.save_fig(fig, plot_path)


if __name__ == "__main__":
    task_plot_raw_ecog(subject="sub-EL002")
    # plt.show(block=True)
