"""Perform and save time frequency analysis of given files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pte

import motor_intention.project_constants as constants


def main() -> None:
    """Main function of this script."""
    OUT_DIR = constants.DERIVATIVES / "time_frequency"
    OUT_DIR.mkdir(exist_ok=True)

    NM_CHANNELS_PATH = constants.DATA / "nm_channels" / "bip_stim_on"  # "bip_stim_off"

    PATH_BAD_EPOCHS = constants.DATA / "bad_epochs"
    if not PATH_BAD_EPOCHS.is_dir():
        raise ValueError(f"Directory not found: {PATH_BAD_EPOCHS}")

    KEYWORDS = None
    STIMULATION = None  # "On"  # "Off"
    MEDICATION = None  # "Off"
    EXCLUDE = None

    OUT_DIR = OUT_DIR / f"Med{MEDICATION}Stim{STIMULATION}"
    OUT_DIR.mkdir(exist_ok=True)

    # parameters for analysis
    N_JOBS = -1
    RESAMPLE_FREQ = 500
    FREQS = np.arange(3, 201, 1).round(1)

    HIGH_PASS = 2
    TMIN = -3.5
    TMAX = 2.5
    PICKS = (
        "dbs",
        "ecog",
    )
    AVERAGE_EPOCHS = False
    NOTCH_FILTER = None

    # Initialize filefinder instance
    file_finder = pte.filetools.get_filefinder(
        datatype="bids", hemispheres=constants.ECOG_HEMISPHERES
    )
    file_finder.find_files(
        directory=constants.RAWDATA,
        extensions=[".vhdr"],
        keywords=KEYWORDS,
        hemisphere="contralateral",
        medication=MEDICATION,
        stimulation=STIMULATION,
        exclude=EXCLUDE,
    )
    print(file_finder)

    for bids_path in file_finder.files:
        # raw = mne_bids.read_raw_bids(bids_path)
        # fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
        # # fig.show()
        # raw.plot_psd(
        #     picks="ecog", tmin=2.0, tmax=200.0, fmax=20, ax=axs[0], show=False
        # )
        # raw.load_data().filter(l_freq=3.0, h_freq=None)
        # raw.plot_psd(
        #     picks="ecog", tmin=2.0, tmax=200.0, fmax=20, ax=axs[1], show=False
        # )
        # plt.show(block=True)
        power = pte.time_frequency.power_from_bids(
            bids_path=bids_path,
            nm_channels_dir=NM_CHANNELS_PATH,
            events_trial_onset=["EMG_onset", "interpolated_EMG_onset"],
            events_trial_end=["EMG_end", "interpolated_EMG_end"],
            min_distance_trials=3.0,
            bad_epochs_dir=PATH_BAD_EPOCHS,
            out_dir=OUT_DIR,
            kwargs_preprocess={
                "resample_freq": RESAMPLE_FREQ,
                "high_pass": HIGH_PASS,
                "average_ref_types": None,
                "notch_filter": NOTCH_FILTER,
            },
            kwargs_epochs={"tmin": TMIN, "tmax": TMAX, "picks": PICKS},
            kwargs_power={
                "n_jobs": N_JOBS,
                "freqs": FREQS,
                "average": AVERAGE_EPOCHS,
            },
        )
        if power is not None:
            power.crop(tmin=-3.0, tmax=2.0, include_tmax=True)
            fname = Path(OUT_DIR, str(bids_path.fpath.stem) + "_tfr.h5")
            power.save(fname=fname, verbose=True, overwrite=True)


if __name__ == "__main__":
    main()
