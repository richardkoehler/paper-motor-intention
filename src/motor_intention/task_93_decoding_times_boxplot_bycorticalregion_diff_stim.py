"""Plot earliest prediction timepoints."""
from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pte_decode
import scipy.stats
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants
import motor_intention.stats_helpers

COND = "Stimulation"
COND_ABB = "Stim" if COND == "Stimulation" else "Med"
STIMULATION = "on" if COND == "Stimulation" else "off"
DECODE = "decode"
PLOT_PATH = constants.PLOTS / DECODE
PLOT_PATH.mkdir(parents=True, exist_ok=True)

CHANNEL = "ecog"
IN_PATH = (
    constants.RESULTS
    / DECODE
    / f"stim_{STIMULATION}_single_chs"
    / CHANNEL
    / "decodingtimes.csv"
)

BASENAME = "decodingtimes_boxplot"


class Cond(Enum):
    PARIETAL = "Parietal"
    SENSORY = "Sensory"
    MOTOR = "Motor"


def task_plot_decoding_times_bycorticalregion(in_path: Path = IN_PATH) -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.cortical_region()

    x = "Cortical Region"
    y = r"Time $Δ_{ON-OFF}$ [s]"  # "Time [s]"
    coords = (
        pd.read_csv(constants.DATA / "elec_ecog_bip.csv")
        .dropna(axis="columns", how="all")
        .query("used == 1")
        .rename(columns={"name": "Channel", "region": x})
        .set_index(["Subject", "Channel"])
    )
    times = pd.read_csv(in_path, index_col=["Subject", "Channel"])

    data_list = []
    for state in ("OFF", "ON"):
        times_pick = times.query(f"{COND} == '{state}'")
        data_list.append(
            pd.concat([times_pick, coords], axis="columns").dropna().reset_index()
        )
    data_raw = (
        pd.concat(data_list)
        .query(f"`{x}` in ['Parietal', 'Sensory', 'Motor']")
        .rename(columns={"Earliest Timepoint": y})
        .sort_values(by=["Subject", "Channel"])
        .set_index(["Subject"])
        # .set_index(["Subject", "Channel", x])
    )
    data_raw[y] = data_raw[y].clip(upper=0.0)

    data_paired = data_raw.query(f"Subject in {constants.STIM_PAIRED_SUBS}")
    keep = []
    for index, row in data_paired.iterrows():
        if constants.STIM_PAIRED[f"sub-{index}"] == row["Medication"]:
            keep.append(True)
        else:
            keep.append(False)
    data_paired = data_paired[keep]

    data_paired = (
        data_paired.reset_index().set_index(["Subject", "Channel", x]).sort_index()
    )
    off = data_paired.query(f"{COND} == 'OFF'").loc[:, y]
    on = data_paired.query(f"{COND} == 'ON'").loc[:, y]
    data = on - off
    data = data.reset_index().set_index(["Subject", "Channel"]).sort_index()

    basename = f"{BASENAME}_bycorticalregion_{COND_ABB.lower()}offvson_diff"
    fname_plot = PLOT_PATH / f"{basename}.svg"
    fname_stats = PLOT_PATH / f"{basename}_stats.csv"
    figsize = (1.7, 1.3)
    fig = pte_decode.boxplot_all_conds(
        data=data,
        outpath=None,
        x=x,
        y=y,
        order=[
            Cond.PARIETAL.value,
            Cond.SENSORY.value,
            Cond.MOTOR.value,
        ],
        title=None,
        figsize=figsize,
        show=False,
    )
    ax = fig.axes[0]
    print(ax.get_xlim())
    xlims = (-1.2, 1.0)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_xticks([xlims[0], 0, xlims[1]])
    ax.set_yticklabels(
        ax.get_yticklabels(),
        weight="bold",
    )
    motor_intention.plotting_settings.save_fig(fig, fname_plot)
    plt.show(block=True)
    fname_stats.unlink(missing_ok=True)

    with fname_stats.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["description", "mean", "std", "statistic", "P"])
        statistic = np.mean
        for cond in Cond:
            description = cond.value
            print(f"{description = }")
            data_cond = data.query(f"`{x}` == '{cond.value}'")[y].to_numpy()
            test = scipy.stats.permutation_test(
                (data_cond - 0.0,),
                statistic,
                vectorized=True,
                n_resamples=int(1e6),
                permutation_type="samples",
            )
            print(f"statistic = {test.statistic}, P = {test.pvalue}")
            writer.writerow(
                [
                    description,
                    statistic(data_cond),
                    np.std(data_cond),
                    test.statistic,
                    test.pvalue,
                ]
            )


if __name__ == "__main__":
    task_plot_decoding_times_bycorticalregion()
    plt.show(block=True)
