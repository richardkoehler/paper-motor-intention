"""Perform and save time frequency analysis of given files."""
from __future__ import annotations

import json
from typing import Literal

import mne
import mne_bids
import numpy as np
import pandas as pd
import pte
import pytask
from matplotlib import figure
from matplotlib import pyplot as plt

import motor_intention.project_constants as constants


def compute_rp_ecog(
    stimulation: Literal["Off", "On"], show_plots: bool = False
) -> None:
    """Main function of this script."""
    PIPELINE = f"stim_{stimulation.lower()}"
    if stimulation == "Off":
        KEYWORDS = None
        MEDICATION = None
    else:
        KEYWORDS = constants.STIM_PAIRED_SUBS
        MEDICATION = "Off"

    OUT_DIR = constants.DERIVATIVES / "readiness_potential" / PIPELINE / "ecog"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_SINGLE_SUBS = OUT_DIR / "single_subs"
    OUT_DIR_SINGLE_SUBS.mkdir(parents=True, exist_ok=True)

    PLOT_DIR = constants.PLOTS / "readiness_potential" / PIPELINE
    PLOT_DIR_SINGLE_SUBS = PLOT_DIR / "single_subs" / "ecog"
    PLOT_DIR_SINGLE_SUBS.mkdir(parents=True, exist_ok=True)

    NM_CHANNELS_DIR = constants.DATA / "nm_channels" / f"unip_{PIPELINE}"

    BAD_EPOCHS_DIR = constants.DATA / "bad_epochs"

    # parameters for analysis
    RESAMPLE_FREQ = 100
    HIGH_PASS = 0.1
    LOW_PASS = 40
    NOTCH_FILTER = None

    TMIN = -3.5
    TMAX = 2.5
    BASELINE = (-3, -2)
    SFREQ_SINGLE = 10

    CORTICAL_REGION = "Motor"

    coords = (
        pd.read_csv(
            constants.DATA / "elec_ecog_unip.csv",
            dtype={"Subject": str, "name": str},
        )
        .query(f"region == '{CORTICAL_REGION}'")  #  and used == 1")
        .set_index("Subject")
    )

    # Initialize filefinder instance
    file_finder = pte.filetools.get_filefinder(
        datatype="bids", hemispheres=constants.ECOG_HEMISPHERES
    )
    file_finder.find_files(
        directory=constants.RAWDATA_ORIG,
        extensions=[".vhdr"],
        hemisphere="contralateral",
        keywords=KEYWORDS,
        medication=MEDICATION,
    )
    print(file_finder)

    results = []
    times = None
    for bids_path in file_finder.files:
        basename = bids_path.basename.removesuffix("_ieeg.vhdr")
        file_channels = NM_CHANNELS_DIR / f"{basename}_ieeg_nm_channels.csv"
        if not file_channels.is_file():
            continue
        print(f"\nFILE: {basename}")
        sub, med, stim = pte.filetools.sub_med_stim_from_fname(bids_path)

        raw = mne_bids.read_raw_bids(
            bids_path, verbose=False, extra_params={"preload": True}
        )
        if "ButtonPress" in basename and "LFP_L_01D_STN_PI" not in raw.ch_names:
            print("\nREREFERENCING:", basename)
            raw.set_eeg_reference(["LFP_L_01_STN_PI"], ch_type="ecog")
        raw = pte.preprocessing.preprocess(
            raw=raw,
            nm_channels_dir=NM_CHANNELS_DIR,
            filename=bids_path,
            average_ref_types=None,
            resample_freq=RESAMPLE_FREQ,
            low_pass=LOW_PASS,
            high_pass=HIGH_PASS,
            notch_filter=NOTCH_FILTER,
            pick_used_channels=True,
        )
        epochs = pte.time_frequency.epochs_from_raw(
            raw=raw,
            tmin=TMIN,
            tmax=TMAX,
            baseline=BASELINE,
            events_trial_onset=["EMG_onset", "interpolated_EMG_onset"],
            events_trial_end=["EMG_end", "interpolated_EMG_end"],
            min_distance_trials=3.0,
            picks="ecog",
        )
        del raw

        bad_epochs_df = pte.filetools.get_bad_epochs(
            filename=bids_path,
            bad_epochs_dir=BAD_EPOCHS_DIR,
        )
        if bad_epochs_df is not None:
            bad_epochs = bad_epochs_df.event_id.to_numpy()
            bad_indices = np.array(
                [
                    idx
                    for idx, event in enumerate(epochs.selection)
                    if event in bad_epochs
                ]
            )
            epochs = epochs.drop(indices=bad_indices)

        reject_criteria = {"ecog": 1e-3}  # 1 mV
        epochs.load_data().drop_bad(reject=reject_criteria)  # type: ignore
        evoked_all: mne.Evoked = (
            epochs.copy().crop(tmin=-3, tmax=2).average(by_event_type=False)
        )
        if times is None:
            times = evoked_all.times

        # Motor cortex channels
        PICKS: list[str] = (
            coords.loc[sub, ["name"]]  # sub.removeprefix("EL"), ["name"]]
            .to_numpy()
            .squeeze()
            .tolist()
        )
        evoked_motorcortex = evoked_all.copy().pick(PICKS)
        evoked_motorcortex.save(
            OUT_DIR_SINGLE_SUBS / f"{basename}_proc-motorcortex-ave.fif.gz",  # type: ignore
            overwrite=True,
        )
        epochs_motorcortex = epochs.copy().pick(PICKS)
        if SFREQ_SINGLE != RESAMPLE_FREQ:
            epochs_motorcortex.resample(SFREQ_SINGLE)
        data_motorcortex = (
            epochs_motorcortex.crop(tmin=-3, tmax=2)
            .get_data(units="µV")
            .mean(axis=1)
            .squeeze()
        )
        res_single = {
            "times": epochs_motorcortex.times.round(1).tolist(),
            "predictions": data_motorcortex.tolist(),
            "trial_ids": epochs_motorcortex.selection.tolist(),
        }
        OUT_DIR_SINGLE = OUT_DIR / "data_motorcortex" / basename
        OUT_DIR_SINGLE.mkdir(exist_ok=True, parents=True)
        with open(
            OUT_DIR_SINGLE / f"{basename}_RPTimelocked.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(res_single, file)
        results.append(
            (
                sub,
                med,
                stim,
                "MotorCortex",
                *evoked_motorcortex.get_data(units="µV").mean(axis=0),
            )
        )
        fig: figure.Figure = evoked_motorcortex.plot(show=False)
        fig.suptitle(basename.replace("_", " "))
        fig.tight_layout()
        fig.savefig(PLOT_DIR_SINGLE_SUBS / f"{basename}_proc-motorcortex.png")
        if show_plots:
            plt.show(block=True)
        else:
            plt.close(fig)

    final = pd.DataFrame(
        results,
        columns=["Subject", "Medication", "Stimulation", "Channels", *times],
    )
    final.to_csv(str(OUT_DIR / "readiness_potential.csv"), index=False)


@pytask.mark.depends_on(constants.RAWDATA)
@pytask.mark.produces(
    constants.DERIVATIVES / "readiness_potential" / "stim_off" / "ecog"
)
def task_compute_rp_ecog_stimoff() -> None:
    """Main function of this script."""
    compute_rp_ecog(stimulation="Off")


@pytask.mark.depends_on(constants.RAWDATA)
@pytask.mark.produces(
    constants.DERIVATIVES / "readiness_potential" / "stim_on" / "ecog"
)
def task_compute_rp_ecog_stimon() -> None:
    """Main function of this script."""
    compute_rp_ecog(stimulation="On")


if __name__ == "__main__":
    # compute_rp_ecog(stimulation="Off", show_plots=False)
    compute_rp_ecog(stimulation="On", show_plots=False)
