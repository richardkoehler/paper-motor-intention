"""Plot decoding times."""
from __future__ import annotations

import csv
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
import pte_decode
import pytask
import scipy.stats
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants
import motor_intention.stats_helpers

DECODE = "decode"
PLOT_PATH = constants.PLOTS / "supplements" / DECODE
PLOT_PATH.mkdir(parents=True, exist_ok=True)
BASENAME = "decodingtimes_boxplot_ecogvslfp"


class Cond(Enum):
    ECOG = "ECOG"
    STN_LFP = "STN-LFP"


def plot_decoding_times_ecogvslfp(
    stimulation: Literal["Off", "On"],
) -> None:
    """Main function of this script"""
    stim = stimulation.upper()
    PIPELINE = f"stim_{stimulation.lower()}"
    channel_types = ("dbs", "ecog")

    med_conds = ("OFF", "ON") if stimulation == "Off" else ("OFF",)
    channel_types = ("dbs", "ecog")
    x = "Channels"
    y = "Time [s]"

    motor_intention.plotting_settings.activate()
    colormap = {
        ("OFF", "OFF"): motor_intention.plotting_settings.ecogvsstn_medoff,
        ("ON", "OFF"): motor_intention.plotting_settings.ecogvsstn_medon,
        ("OFF", "ON"): motor_intention.plotting_settings.ecogvsstn_stimon,
    }

    data_list = []
    for channel in channel_types:
        INPUT_ROOT = constants.RESULTS / DECODE / PIPELINE / channel

        data_raw = (
            pd.read_csv(INPUT_ROOT / "decodingtimes.csv")
            .rename(
                columns={
                    "Earliest Timepoint": y,
                    "Channel": x,
                }
            )
            .set_index("Subject")
        )

        data_raw.loc[:, "Time (s)"] = data_raw.loc[:, "Time (s)"].clip(upper=0.0)
        data_raw["Channels"] = "STN-LFP" if channel == "dbs" else "ECOG"

        data_list.append(data_raw)

    data_all = pd.concat(data_list, join="outer")

    bottom_lims = []
    top_lims = []
    figs = []
    for med in med_conds:
        colormap[(med, stim)]()
        basename = f"{BASENAME}_stim{stimulation.lower()}_med{med.lower()}"
        outpath = PLOT_PATH / (basename + ".svg")
        FNAME_STATS = PLOT_PATH / (basename + "_stats.csv")

        data = data_all.query(f"Medication == '{med}'").sort_values("Subject")

        fig = pte_decode.boxplot_results(
            data=data,
            outpath=outpath,
            x=x,
            y=y,
            hue=None,
            order=["ECOG", "STN-LFP"],
            hue_order=None,
            stat_test=None,
            add_lines="Subject",
            title=None,
            figsize=(0.8, 1.4),
            show=False,
        )
        bottom, top = fig.axes[0].get_ylim()
        bottom_lims.append(bottom)
        top_lims.append(top)
        figs.append((fig, outpath))

        FNAME_STATS.unlink(missing_ok=True)
        with FNAME_STATS.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["description", "mean", "std", "statistic", "P"])

            for cond_a, cond_b in ((Cond.ECOG, Cond.STN_LFP),):
                description = f"{cond_a.value} vs {cond_b.value}"
                print(f"{description = }")
                data_a = (
                    data.query(f"{x} == '{cond_a.value}'")
                    .sort_values("Subject")[y]
                    .to_numpy()
                )
                data_b = (
                    data.query(f"{x} == '{cond_b.value}'")
                    .sort_values("Subject")[y]
                    .to_numpy()
                )
                test = scipy.stats.permutation_test(
                    (data_a - data_b,),
                    statistic := np.mean,
                    vectorized=True,
                    n_resamples=int(1e6),
                    permutation_type="samples",
                )
                print(f"statistic = {test.statistic}, P = {test.pvalue}")
                writer.writerow(
                    [
                        description,
                        statistic(data_a - data_b),
                        np.std(data_a - data_b),
                        test.statistic,
                        test.pvalue,
                    ]
                )

    bottom_lim = min(bottom_lims)
    top_lim = max(top_lims)
    print("xlims:", bottom_lim, top_lim)

    for fig, outpath in figs:
        ax = fig.axes[0]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        # ax.axhline(0.5, color="black", linestyle="--", alpha=0.5)
        y_lims = (-2.0, 0.0)
        ax.set_ylim(y_lims[0], y_lims[1])
        ax.set_yticks([y_lims[0], -1.0, y_lims[1]])
        ax.set_xticklabels(ax.get_xticklabels(), weight="bold", rotation=30)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xlabel("")
        motor_intention.plotting_settings.save_fig(fig, outpath)


@pytask.mark.depends_on(
    (
        constants.RESULTS / DECODE / "stim_off" / "ecog" / "decodingtimes.csv",
        constants.RESULTS / DECODE / "stim_off" / "dbs" / "decodingtimes.csv",
    )
)
@pytask.mark.produces(
    (
        PLOT_PATH / f"{BASENAME}_stimoff_medoff.svg",
        PLOT_PATH / f"{BASENAME}_stimoff_medon.svg",
        PLOT_PATH / f"{BASENAME}_stimoff_medoff_stats.csv",
        PLOT_PATH / f"{BASENAME}_stimoff_medon_stats.csv",
    )
)
def task_plot_decodingtimes_stim_off() -> None:
    plot_decoding_times_ecogvslfp(stimulation="Off")


@pytask.mark.depends_on(
    (
        constants.RESULTS / DECODE / "stim_on" / "ecog" / "decodingtimes.csv",
        constants.RESULTS / DECODE / "stim_on" / "dbs" / "decodingtimes.csv",
    )
)
@pytask.mark.produces(
    (
        PLOT_PATH / f"{BASENAME}_stimon_medoff.svg",
        PLOT_PATH / f"{BASENAME}_stimon_medon.svg",
        PLOT_PATH / f"{BASENAME}_stimon_medoff_stats.csv",
        PLOT_PATH / f"{BASENAME}_stimon_medon_stats.csv",
    )
)
def task_plot_decodingtimes_stim_on() -> None:
    plot_decoding_times_ecogvslfp(stimulation="On")


if __name__ == "__main__":
    task_plot_decodingtimes_stim_off()
    plt.show()
    task_plot_decodingtimes_stim_on()
    plt.show()
