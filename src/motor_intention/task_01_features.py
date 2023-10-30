"""Calculate features using py_neuromodulation."""
from __future__ import annotations

import pathlib
import time
from pathlib import Path
from typing import Annotated, Literal

import mne_bids
import numpy as np
import pte.filetools
import pte_neuromodulation as nm
from joblib import Parallel, delayed
from pytask import Product

import motor_intention.project_constants as constants

OUT_PATHS = {
    stim: constants.DERIVATIVES / "features" / f"stim_{stim.lower()}"
    for stim in ["Off", "On"]
}


def compute_features(
    in_path: pathlib.Path,
    stimulation: Literal["Off", "On"],
    out_path: pathlib.Path,
) -> None:
    PIPELINE = f"stim_{stimulation.lower()}"
    OUT_DIR = out_path
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    NM_CHANNELS_PATH = constants.DATA / "nm_channels" / f"bip_{PIPELINE}"

    N_JOBS = -1

    file_finder = pte.filetools.BIDSFinder(hemispheres=constants.ECOG_HEMISPHERES)
    file_finder.find_files(
        directory=in_path,
        hemisphere="contralateral",
        medication=None,
        stimulation=None,
        exclude=None,
    )
    print(file_finder)
    files = file_finder.files

    kwargs = {
        "root_nm_channels": NM_CHANNELS_PATH,
        "path_settings": str(constants.DATA / "nm_settings.json"),
        "path_out": str(OUT_DIR),
        "stimulation": stimulation,
    }

    start = time.perf_counter()

    if N_JOBS != 1:
        Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(run)(fname=file, **kwargs) for file in files
        )
    else:
        for file in files:
            run(fname=file, **kwargs)

    print(f"Time elapsed: {((time.perf_counter()-start)/60):.0f} minutes")


def run(
    fname: mne_bids.BIDSPath,
    root_nm_channels: Path,
    path_settings: str,
    path_out: str,
    stimulation: Literal["Off", "On"],
) -> None:
    """Calculate features for single file."""
    path_nm_channels = (
        root_nm_channels
        / (fname.copy().update(extension=None).basename + "_nm_channels.csv")
    ).resolve()
    if not path_nm_channels.is_file():
        print(f"No nm_channel found. Skipping file: {fname.fpath}")
        return
    print(f"Reading file: {fname.fpath}")
    path_nm_channels = str(path_nm_channels)
    raw = mne_bids.read_raw_bids(fname, extra_params={"verbose": 0})
    coord_list, coord_names = nm.io.get_coord_list(raw)
    if stimulation == "On":
        raw.load_data()
        freqs = np.arange(130, 512, 130)
        notch_widths = (freqs * 0.2).clip(26)
        raw.notch_filter(
            freqs=freqs,
            picks=["ecog", "dbs"],
            notch_widths=notch_widths,
            verbose=True,
        )
    settings = nm.io.read_settings(path_settings)
    nm_channels = nm.io.load_nm_channels(path_nm_channels)

    stream = nm.Stream(
        sfreq=raw.info["sfreq"],
        nm_channels=nm_channels,
        settings=settings,
        line_noise=int(raw.info["line_freq"]),
        coord_list=coord_list,
        coord_names=coord_names,
        verbose=False,
    )
    stream.run(
        data=raw.get_data(),
        out_path_root=path_out,
        folder_name=fname.copy().update(extension=None).basename,
    )


def task_compute_features_stimoff(
    in_path: pathlib.Path = constants.RAWDATA,
    stimulation: Literal["Off", "On"] = "Off",
    outpath: Annotated[pathlib.Path, Product] = OUT_PATHS["Off"],
) -> None:
    """Main function of this script."""
    compute_features(in_path, stimulation, outpath)


def task_compute_features_stimon(
    in_path: pathlib.Path = constants.RAWDATA,
    stimulation: Literal["Off", "On"] = "On",
    outpath: Annotated[pathlib.Path, Product] = OUT_PATHS["On"],
) -> None:
    """Main function of this script."""
    compute_features(in_path, stimulation, outpath)


if __name__ == "__main__":
    task_compute_features_stimoff()
    task_compute_features_stimon()
