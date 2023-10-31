"""Perform and save time frequency analysis of given files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pte
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

BASENAME = "time_frequency_plot"
IN_ROOT = constants.DERIVATIVES / "time_frequency"
PLOT_ROOT = constants.PLOTS
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

BASELINE = (-3, -2)
FMIN: int = 3
FMAX: int = 200
Y_LIMS = (FMIN, FMAX)
TMIN: int | float = -3.0
TMAX: int | float = 2.0

VALS_CBAR = {"ecog": 3.0, "dbs": 1.0}


def task_plot_time_frequency(in_path: Path = IN_ROOT, show_plots: bool = False) -> None:
    """Main function of this script."""
    motor_intention.plotting_settings.activate()

    extent = (TMIN, TMAX, FMIN, FMAX)
    fig_height = 1.4
    coords = (
        pd.read_csv(constants.DATA / "elec_ecog_bip.csv")
        .dropna(axis="columns", how="all")
        .query("used == 1 and region == 'Motor'")
        .rename(columns={"name": "Channel"})
        .set_index("Subject")
    )
    file_finder = pte.filetools.DefaultFinder()
    file_finder.find_files(
        directory=in_path, exclude="sub-EL002", extensions=["tfr.h5"]
    )
    print(file_finder)
    powers = {}
    for file in file_finder.files[:]:
        sub, _, _ = pte.filetools.sub_med_stim_from_fname(file)
        power = pte.time_frequency.load_power(files=[file])[0]
        power = (
            power.average()
            .apply_baseline(baseline=BASELINE, mode="zscore")
            .crop(tmin=TMIN, tmax=TMAX, fmin=FMIN, fmax=FMAX)
        )
        powers[sub] = power

    for channel in ["ecog", "dbs"]:  #
        ch_str = "motorcortex" if channel == "ecog" else "dbs"
        power_norm = []
        for sub, power in powers.items():
            ch_pick = coords.loc[sub, "Channel"] if channel == "ecog" else "dbs"
            power_pick = power.copy().pick(ch_pick)
            power_norm.append(power_pick.data.mean(axis=0))
        power_norm_all = np.stack(power_norm)
        power_av = power_norm_all.mean(axis=0)

        borderval_cbar = VALS_CBAR[channel]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.3, fig_height))
        img = ax.imshow(
            power_av,
            extent=extent,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            vmin=borderval_cbar * -1,
            vmax=borderval_cbar,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim([TMIN, TMAX])
        ax.set_xticks([TMIN, 0, TMAX])
        ax.set_xticklabels([])
        ax.set_ylim([Y_LIMS[0], Y_LIMS[1]])
        ax.set_yticks([Y_LIMS[0], 100, Y_LIMS[1]])
        if channel == "ecog":
            ax.set_ylabel("Frequency [Hz]")
        else:
            ax.set_yticklabels([])

        motor_intention.plotting_settings.save_fig(
            fig, PLOT_ROOT / (f"{BASENAME}_{ch_str}.svg")
        )
        plt.show(block=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(0.05, fig_height))
        cbar = fig.colorbar(
            img,
            cax=ax,
            label="Power [AU]",
        )
        cbar.outline.set_visible(False)
        cbar.ax.get_yaxis().set_ticks([-1 * borderval_cbar, 0, borderval_cbar])
        motor_intention.plotting_settings.save_fig(
            fig, PLOT_ROOT / (f"{BASENAME}_cbar_{ch_str}.svg")
        )

    if show_plots:
        plt.show(block=True)
    else:
        plt.close("all")


if __name__ == "__main__":
    task_plot_time_frequency(show_plots=True)
