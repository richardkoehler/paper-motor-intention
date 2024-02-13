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

PLOT_PATH = constants.PLOTS / "raw_lfp.svg"
SUBJECT = "sub-EL011"


def task_plot_raw_lfp(
    subject: str = SUBJECT,
    plot_path: Annotated[Path, Product] = PLOT_PATH,
) -> None:
    plt.rcParams["axes.xmargin"] = 0
    print(f"{plt.rcParams['axes.xmargin'] = }")
    file_finder = pte.filetools.get_filefinder(datatype="bids")
    file_finder.find_files(
        directory=constants.RAWDATA_ORIG,
        extensions=".vhdr",
        keywords=[subject],
        medication="On",
    )
    print(file_finder)
    raw = mne_bids.read_raw_bids(file_finder.files[0]).load_data()
    raw.pick("dbs")
    new_ch = "LFP_R_050607-08"
    raw = mne.set_bipolar_reference(
        raw, ["LFP_R_(05+06+07)_STN_MT"], ["LFP_R_08_STN_MT"], [new_ch]
    )
    raw.load_data().resample(1000).filter(l_freq=4, picks="all", h_freq=90).crop(
        tmin=548, tmax=556
    )
    events, _ = mne.events_from_annotations(raw, event_id={"EMG_onset": 1})
    epochs = mne.Epochs(raw, events, tmin=-3.0, tmax=2.0, baseline=None)
    # epochs.plot(picks=new_ch, n_epochs=1, block=True)
    data = epochs.get_data(units={"ecog": "uV", "dbs": "uV"})
    data = data.squeeze()
    motor_intention.plotting_settings.activate()
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 0.5))
    ax.plot(epochs.times, data, color="black", linewidth=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([-3, 0, 2])
    ax.set_xticklabels([])
    motor_intention.plotting_settings.save_fig(fig, plot_path)


if __name__ == "__main__":
    task_plot_raw_lfp()
    plt.show(block=True)
