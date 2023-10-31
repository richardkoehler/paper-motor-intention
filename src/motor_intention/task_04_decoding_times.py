"""Calculate and save earliest decoding times."""
from __future__ import annotations

from collections.abc import Sequence
import pathlib
import time
from typing import Annotated, Literal

import numpy as np
import pte
import pte_decode
from joblib import Parallel, delayed
from pytask import Product

import motor_intention.project_constants as constants


CHANNELS = ("ecog", "dbs")


INPATHS_STIM_OFF = {
    (ch, "stim_off"): constants.DERIVATIVES / "decode" / "stim_off" / ch
    for ch in CHANNELS
}
OUTPATHS_STIM_OFF = {
    (ch, "stim_off"): constants.RESULTS
    / "decode"
    / "stim_off"
    / ch
    / "decodingtimes.csv"
    for ch in CHANNELS
}
INPATHS_STIM_ON = {
    (ch, "stim_on"): constants.DERIVATIVES / "decode" / "stim_on" / ch
    for ch in CHANNELS
}
OUTPATHS_STIM_ON = {
    (ch, "stim_on"): constants.RESULTS
    / "decode"
    / "stim_on"
    / ch
    / "decodingtimes.csv"
    for ch in CHANNELS
}
INPATHS_SINGLE_STIM_OFF = {
    (ch, "stim_off_single_chs"): constants.DERIVATIVES
    / "decode"
    / "stim_off_single_chs"
    / ch
    for ch in ("ecog",)
}
OUTPATHS_SINGLE_STIM_OFF = {
    (ch, "stim_off_single_chs"): constants.RESULTS
    / "decode"
    / "stim_off_single_chs"
    / ch
    / "decodingtimes.csv"
    for ch in ("ecog",)
}
INPATHS_SINGLE_STIM_ON = {
    (ch, "stim_on_single_chs"): constants.DERIVATIVES
    / "decode"
    / "stim_off_single_chs"
    / ch
    for ch in ("ecog",)
}
OUTPATHS_SINGLE_STIM_ON = {
    (ch, "stim_on_single_chs"): constants.RESULTS
    / "decode"
    / "stim_on_single_chs"
    / ch
    / "decodingtimes.csv"
    for ch in ("ecog",)
}


def task_decoding_times_stimoff(
    in_paths: dict[
        tuple[Literal["ecog", "dbs"], Literal["stim_on", "stim_off"]],
        Sequence[pathlib.Path],
    ] = INPATHS_STIM_OFF,
    out_paths: Sequence[Annotated[pathlib.Path, Product]] = OUTPATHS_STIM_OFF,
) -> None:
    calculate_decoding_times(
        stimulation="Off", in_paths=in_paths, out_paths=out_paths
    )


def task_decoding_times_stimon(
    in_paths: dict[
        tuple[Literal["ecog", "dbs"], Literal["stim_on", "stim_off"]],
        Sequence[pathlib.Path],
    ] = None,
    out_paths: dict[str, Annotated[pathlib.Path, Product]] | None = None,
) -> None:
    calculate_decoding_times(stimulation="On")


def task_decoding_times_single_ch(
    in_paths: dict[
        tuple[Literal["ecog", "dbs"], Literal["stim_on", "stim_off"]],
        Sequence[pathlib.Path],
    ] = None,
    out_paths: dict[str, Literal["ecog", "dbs"][pathlib.Path, Product]]
    | None = None,
) -> None:
    calculate_decoding_times(stimulation="Off", channels_used="single")


def task_decoding_times_single_ch_stimon(
    in_paths: dict[
        tuple[Literal["ecog", "dbs"], Literal["stim_on", "stim_off"]],
        Sequence[pathlib.Path],
    ] = None,
    out_paths: dict[str, Annotated[pathlib.Path, Product]] | None = None,
) -> None:
    calculate_decoding_times(stimulation="On", channels_used="single")


def calculate_decoding_times(
    stimulation: Literal["Off", "On"],
    channels_used: Literal["all", "single"],
    inpaths: dict[
        tuple[Literal["ecog", "dbs"], Literal["stim_on", "stim_off"]],
        Sequence[pathlib.Path],
    ],
    outpaths: Sequence[pathlib.Path],
) -> None:
    """Main function of this script"""
    N_JOBS = -1

    RESAMPLE_TRIALS = 50

    ALPHA = 0.05
    N_PERM = 500
    CORRECTION_METHOD = "cluster_pvals"

    N_ITERATIONS = 500

    BASELINE = (-3.0, -2.0)
    THRESHOLD = 0

    TIME_LIMS = (-3.0, 2.0)  # seconds

    def _single_timepoint(
        default_value: int | float, **kwargs
    ) -> tuple[int | float, int]:
        timepoint, trials_used = pte_decode.get_earliest_timepoint(**kwargs)
        if timepoint is None:
            return default_value, trials_used
        return timepoint, trials_used

    PIPELINE = f"stim_{stimulation.lower()}"

    if channels_used == "single":
        PIPELINE = f"{PIPELINE}_single_chs"

    stimulation = "On" if stimulation == "On" else "Off"

    medication = None if stimulation == "Off" else "Off"

    channel_types = ("dbs", "ecog") if channels_used == "all" else ("ecog",)

    file_finder = pte.filetools.DefaultFinder()
    start = time.time()
    for () in channel_types:
        INPUT_PATH = constants.DERIVATIVES / "decode" / PIPELINE / channel
        OUTPUT_PATH = constants.RESULTS / "decode" / PIPELINE / channel
        OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

        file_finder.find_files(
            directory=INPUT_PATH,
            extensions=["PredTimelocked.json"],
            medication=medication,
        )
        print(file_finder)
        print("Files found:", len(file_finder.files))

        data = pte_decode.load_predictions(
            files=file_finder.files,
            baseline=BASELINE,
            baseline_mode="zscore",
            baseline_trialwise=False,
            average_predictions=False,
        )
        times = np.array(data.loc[:, "times"].iloc[0])
        TIME_SLICE = (TIME_LIMS[0] <= times) & (times <= TIME_LIMS[1])
        TIMES_USED = times[TIME_SLICE]

        timepoints = []
        trials_used = []

        for sample in data["Predictions"].to_numpy():
            samples_used = sample[..., TIME_SLICE]

            kwargs = {
                "default_value": TIMES_USED[-1],
                "data": samples_used,
                "times": TIMES_USED,
                "threshold": THRESHOLD,
                "n_perm": N_PERM,
                "alpha": ALPHA,
                "correction_method": CORRECTION_METHOD,
                "min_cluster_size": 2,
                "resample_trials": RESAMPLE_TRIALS,
                "verbose": False,
            }
            if N_JOBS == 1 or N_ITERATIONS == 1:
                timepoints_singlesub = []
                for _ in range(N_ITERATIONS):
                    tp, trials = _single_timepoint(**kwargs)
                    timepoints_singlesub.append(tp)
                timepoint_avg = np.mean(timepoints_singlesub)
            else:
                timepoint_avg, trials = np.array(
                    Parallel(n_jobs=N_JOBS, verbose=1)(
                        delayed(_single_timepoint)(**kwargs)
                        for _ in range(N_ITERATIONS)
                    )
                ).mean(axis=0)
            print(f"{timepoint_avg = :.2f}")
            timepoints.append(timepoint_avg)
            trials_used.append(int(trials))  # type: ignore  # noqa: PGH003

        data["Earliest Timepoint"] = timepoints
        data["trials_used"] = trials_used
        data = data.drop(columns=["Predictions", "times", "trial_ids"])

        data.to_csv(
            OUTPUT_PATH / "decodingtimes.csv",
            na_rep="n/a",
            index=False,
        )
    print(f"Time elapsed: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    task_decoding_times_stimoff()
    task_decoding_times_stimon()
    task_decoding_times_single_ch()
    task_decoding_times_single_ch_stimon()
