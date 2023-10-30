"""Calculate and save number of trials for all recordings."""
from __future__ import annotations

import csv
import pathlib
from typing import Annotated

import mne
import mne_bids
import pandas as pd
import pte
from pytask import Product

import motor_intention.project_constants as constants

OUT_DIR = constants.RESULTS / "descriptive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FNAME_TRIALS = OUT_DIR / "trial_numbers.csv"
FNAME_STATS = OUT_DIR / "trial_numbers_stats.csv"


def _get_events(raw: mne.io.Raw, event_ids: dict) -> list:
    event_list = []
    for key, value in event_ids.items():
        event_list.append(
            mne.events_from_annotations(raw=raw, event_id={key: value})[0][  # type: ignore  # noqa: PGH003
                ..., 0
            ]
        )
    return event_list


def task_write_trial_numbers(
    in_path: pathlib.Path = constants.RAWDATA_ORIG,
    outpath_trials: Annotated[pathlib.Path, Product] = FNAME_TRIALS,
    outpath_stats: Annotated[pathlib.Path, Product] = FNAME_STATS,
) -> None:
    """Main function of this script."""

    file_finder = pte.filetools.BIDSFinder(hemispheres=constants.ECOG_HEMISPHERES)
    file_finder.find_files(
        directory=in_path,
        extensions=".vhdr",
        hemisphere="contralateral",
    )
    print(file_finder)

    trials_single = []
    for file in file_finder.files:
        raw: mne.io.Raw = mne_bids.read_raw_bids(file, verbose=False)
        basename = file.update(suffix=None, extension=None).basename
        sub, med, stim = pte.filetools.sub_med_stim_from_fname(basename)
        try:
            emg_onset, emg_end = _get_events(
                raw=raw,
                event_ids={
                    "EMG_onset": 1,
                    "EMG_end": -1,
                },
            )
        except ValueError as e:
            print(e, "Retrying...")
            emg_onset, emg_end = _get_events(
                raw=raw,
                event_ids={
                    "interpolated_EMG_onset": 1,
                    "interpolated_EMG_end": -1,
                },
            )
        assert emg_end.shape == emg_onset.shape
        trials_single.append([sub, med, stim, emg_onset.shape[0]])

    trials = pd.DataFrame(
        trials_single,
        columns=[
            "Subject",
            "Medication",
            "Stimulation",
            "Number of trials",
        ],
    )
    trials.to_csv(outpath_trials, index=False, na_rep="n/a")
    FNAME_STATS.unlink(missing_ok=True)
    with outpath_stats.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["description", "mean", "std"])
        writer.writerow(
            [
                "Number of trials",
                trials["Number of trials"].mean(),
                trials["Number of trials"].std(),
            ]
        )


if __name__ == "__main__":
    task_write_trial_numbers()
