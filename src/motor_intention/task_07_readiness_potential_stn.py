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
from matplotlib import pyplot as plt

import motor_intention.project_constants as constants


def compute_rp_stn(stimulation: Literal["Off", "On"], show_plots: bool = False) -> None:
    """Main function of this script."""
    PIPELINE = f"stim_{stimulation.lower()}"
    if stimulation == "Off":
        KEYWORDS = None
        MEDICATION = None
        STIMULATION = None
    else:
        KEYWORDS = None  # constants.STIM_PAIRED_SUBS
        MEDICATION = "Off"
        STIMULATION = None

    OUT_DIR = constants.DERIVATIVES / "readiness_potential" / PIPELINE / "dbs"
    # OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_SINGLE_SUBS = OUT_DIR / "single_subs"
    OUT_DIR_SINGLE_SUBS.mkdir(parents=True, exist_ok=True)

    PLOT_DIR = constants.PLOTS / "readiness_potential" / PIPELINE
    PLOT_DIR_SINGLE_SUBS = PLOT_DIR / "single_subs" / "dbs"
    PLOT_DIR_SINGLE_SUBS.mkdir(parents=True, exist_ok=True)

    NM_CHANNELS_DIR = (
        constants.DATA / "nm_channels" / f"unip_{PIPELINE}"  # f"bip_{PIPELINE}"
    )  #

    BAD_EPOCHS_DIR = constants.DATA / "bad_epochs"

    # parameters for analysis
    RESAMPLE_FREQ = 100
    HIGH_PASS = 0.1
    LOW_PASS = 40
    NOTCH_FILTER = None

    TMIN = -3.5
    TMAX = 2.5
    BASELINE = (-3, -2)

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
        stimulation=STIMULATION,
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
        raw = mne_bids.read_raw_bids(
            bids_path, verbose=False, extra_params={"preload": True}
        )
        sub, med, stim = pte.filetools.sub_med_stim_from_fname(bids_path)
        side = "L" if constants.ECOG_HEMISPHERES[sub] == "R" else "R"
        with open(bids_path.copy().update(extension=".json"), encoding="utf-8") as file:
            sidecar = json.load(file)
        ref_orig = sidecar["iEEGReference"]

        if not ref_orig.startswith(f"LFP_{side}_01") and sub != "EL002":
            ref_kw = f"LFP_{side}_01"
            ref_ch = [
                ch for ch in raw.ch_names if ch.startswith(ref_kw) and "STN" in ch
            ]
            assert len(ref_ch) == 1
            raw.set_eeg_reference(ref_ch, ch_type="dbs")

        raw = pte.preprocessing.preprocess(
            raw=raw,
            nm_channels_dir=NM_CHANNELS_DIR,
            filename=bids_path,
            average_ref_types=None,
            ref_nm_channels=False,
            resample_freq=RESAMPLE_FREQ,
            low_pass=LOW_PASS,
            high_pass=HIGH_PASS,
            notch_filter=NOTCH_FILTER,
            pick_used_channels=True,
        )
        raw.pick(["dbs"])
        if stimulation == "On":
            if sub == "EL008":
                raw.drop_channels(
                    # ['LFP_L_(01+02+03)-(14+15)_STN_BS', 'LFP_L_(04+05+06)-(11+12)_STN_BS']
                    ["LFP_L_(01+02+03)_STN_BS", "LFP_L_(04+05+06)_STN_BS"]
                )
            elif sub == "EL005":
                # raw.drop_channels(['LFP_R_(02+03+04)-08_STN_MT', 'LFP_R_01-(02+03+04)_STN_MT']) #
                raw.drop_channels(["LFP_R_(02+03+04)_STN_MT"])
        epochs = pte.time_frequency.epochs_from_raw(
            raw=raw,
            tmin=TMIN,
            tmax=TMAX,
            baseline=BASELINE,
            events_trial_onset=["EMG_onset", "interpolated_EMG_onset"],
            events_trial_end=["EMG_end", "interpolated_EMG_end"],
            min_distance_trials=3.0,
            picks="dbs",
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
        else:
            raise ValueError("No bad epochs file found.")

        reject_criteria = {"dbs": 1e-3}  # 1 mV
        epochs.load_data().crop(tmin=-3.0, tmax=2.0).drop_bad(reject=reject_criteria)  # type: ignore
        evoked_all: mne.Evoked = (
            epochs.copy()
            # .crop(tmin=-3.0, tmax=0.0)
            .average(by_event_type=False)
        )
        evoked_all.save(
            OUT_DIR_SINGLE_SUBS / f"{basename}_proc-dbsall-ave.fif.gz",  # type: ignore
            overwrite=True,
        )
        if times is None:
            times = evoked_all.times
        data_all = evoked_all.get_data(units="µV").mean(axis=0)
        if evoked_all.get_data(tmin=-0.2, tmax=0.2).mean() > 0:
            data_all *= -1

        results.append((sub, med, stim, "All", *data_all))
        fig = evoked_all.plot(show=False)
        fig.suptitle(basename.replace("_", " "))
        fig.tight_layout()
        fig.savefig(PLOT_DIR_SINGLE_SUBS / f"{basename}_proc-dbsall.png")
        if show_plots:
            plt.show(block=True)

    final = pd.DataFrame(
        results,
        columns=["Subject", "Medication", "Stimulation", "Channels", *times],
    )
    final.to_csv(str(OUT_DIR / "readiness_potential.csv"), index=False)


@pytask.mark.depends_on(constants.RAWDATA)
@pytask.mark.produces(
    constants.DERIVATIVES / "readiness_potential" / "stim_off" / "dbs"
)
def task_compute_rp_stn_stimoff() -> None:
    """Main function of this script."""
    compute_rp_stn(stimulation="Off")


@pytask.mark.depends_on(constants.RAWDATA)
@pytask.mark.produces(constants.DERIVATIVES / "readiness_potential" / "stim_on" / "dbs")
def task_compute_rp_stn_stimon() -> None:
    """Main function of this script."""
    compute_rp_stn(stimulation="On")


if __name__ == "__main__":
    compute_rp_stn(stimulation="Off", show_plots=False)
    # compute_rp_stn(stimulation="On", show_plots=False)
