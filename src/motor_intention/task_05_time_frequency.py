"""Perform and save time frequency analysis of given files."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pte
from pytask import Product

import motor_intention.project_constants as constants

STIM = ("Off", "On")
OUT_DIRS = {stim: constants.DERIVATIVES / "time_frequency" for stim in STIM}


def task_compute_tfr(
    out_dirs: dict[Literal["Off", "On"], Annotated[Path, Product]] = OUT_DIRS
) -> None:
    """Main function of this script."""
    for stimulation, out_dir in out_dirs.items():
        PIPELINE = f"stim_{stimulation.lower()}"
        NM_CHANNELS_PATH = constants.DATA / "nm_channels" / f"bip_{PIPELINE}"
        if stimulation == "Off":
            KEYWORDS = None
            MEDICATION = None
        else:
            KEYWORDS = constants.STIM_PAIRED_SUBS
            MEDICATION = "Off"

        PATH_BAD_EPOCHS = constants.DATA / "bad_epochs"
        if not PATH_BAD_EPOCHS.is_dir():
            msg = f"Directory not found: {PATH_BAD_EPOCHS}"
            raise ValueError(msg)

        OUT_DIR = out_dir / f"Med{MEDICATION}Stim{stimulation}"
        OUT_DIR.mkdir(exist_ok=True, parents=True)

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
        file_finder = pte.filetools.BIDSFinder(hemispheres=constants.ECOG_HEMISPHERES)
        file_finder.find_files(
            directory=constants.RAWDATA,
            extensions=[".vhdr"],
            keywords=KEYWORDS,
            hemisphere="contralateral",
            medication=MEDICATION,
            stimulation=stimulation,
            exclude=None,
        )
        print(file_finder)

        for bids_path in file_finder.files:
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
    task_compute_tfr()
