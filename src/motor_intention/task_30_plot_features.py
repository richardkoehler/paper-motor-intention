"""Plot timelocked features."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pte
from pytask import Product

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

FEATURE_PATH = constants.DERIVATIVES / "decode" / "stim_off" / "ecog"
PLOT_PATH = constants.PLOTS / "features_timelocked.svg"


def task_plot_features(
    in_path: Path = FEATURE_PATH,
    out_path: Annotated[Path, Product] = PLOT_PATH,
    subject: str = "sub-EL014",
) -> None:
    "Plot features for single subject."
    motor_intention.plotting_settings.activate()

    coords_raw = pd.read_csv(constants.DATA / "elec_ecog_bip.csv").dropna(
        axis="columns", how="all"
    )
    coords = (
        coords_raw.query(f"used == 1 and region == 'Motor' and Subject == '{subject}'")
        .rename(columns={"name": "Channel"})
        .set_index(["Subject"])
    )
    assert len(coords) == 1
    file_finder = pte.filetools.get_filefinder(datatype="any")
    file_finder.find_files(
        directory=in_path,
        extensions="FeaturesTimelocked.json.gz",
        keywords=[f"sub-{subject}"],
        medication="Off",
    )
    print(file_finder)
    assert len(file_finder.files) == 1
    with gzip.open(file_finder.files[0], "rb") as file:
        feat = json.load(file)
    f_bands = []
    features = []
    for feature, data in feat["features"].items():
        if not feature.startswith(coords.loc[subject, "Channel"]):
            continue
        f_band = feature.split("_fft_")[1]
        f_band = "HFA" if f_band == "high frequency activity" else f_band.capitalize()
        f_bands.append(f_band)
        features.append(data)
    f_bands = f_bands[::-1]
    features = np.array(features).mean(axis=1)[::-1, :]  # -10
    max_val = np.abs(features).max()

    fig, axs = plt.subplots(3, 1, figsize=(1.6, 3.1), height_ratios=[8, 1.5, 0.3])
    ax = axs[0]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    im = ax.imshow(
        features,
        cmap="viridis",
        aspect="auto",
        vmin=-max_val,
        vmax=max_val,
    )
    ax.axvline(x=30, color="white", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_xticks([0, 30, 50])
    ax.set_xticklabels([-3, 0, 2])
    ax.set_yticks(ticks=np.arange(len(f_bands)), labels=f_bands, rotation=45)
    ax.xaxis.set_label_coords(0.65, -0.14)
    ax.yaxis.set_tick_params(length=0)
    axs[1].set_axis_off()
    ticks = [-max_val, 0, max_val]
    cbar = fig.colorbar(
        im,
        cax=axs[2],
        label="Power [AU]",
        ticks=ticks,
        orientation="horizontal",
    )
    cbar.outline.set_visible(False)
    cbar.ax.set_xticklabels([round(val, 1) for val in ticks])
    motor_intention.plotting_settings.save_fig(fig, out_path)
    plt.show(block=True)


if __name__ == "__main__":
    task_plot_features(subject="EL014")
