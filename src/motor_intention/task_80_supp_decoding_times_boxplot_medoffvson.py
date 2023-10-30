"""Plot decoding times."""
from __future__ import annotations

import csv
from enum import Enum

import numpy as np
import pandas as pd
import pte_decode
import pytask
import scipy.stats

import motor_intention.plotting_settings
import motor_intention.project_constants as constants
import motor_intention.stats_helpers

DECODE = "decode"
PLOT_PATH = constants.PLOTS / "supplements" / DECODE
PLOT_PATH.mkdir(parents=True, exist_ok=True)

CHANNEL = "ecog"
TIMES = constants.RESULTS / "decode" / "stim_off" / CHANNEL / "decodingtimes.csv"
BASENAME = f"decodingtimes_boxplot_{CHANNEL}_medoffvson"
SUBJECT_PICKS = ("paired",)  #  "all")
FNAMES_PLOT = [PLOT_PATH / f"{BASENAME}_{picks}.svg" for picks in SUBJECT_PICKS]
FNAMES_STATS = [PLOT_PATH / f"{BASENAME}_{picks}_stats.csv" for picks in SUBJECT_PICKS]


class Cond(Enum):
    OFF = "OFF"
    ON = "ON"


@pytask.mark.depends_on(TIMES)
@pytask.mark.produces(*FNAMES_PLOT, *FNAMES_STATS)
def task_plot_decoding_times_medoffvson() -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.medoffvson()

    MED_PAIRED = [sub.strip("sub-") for sub in constants.MED_PAIRED]

    x = "Medication"
    y = "Time [s]"
    data_raw = (
        pd.read_csv(
            TIMES,
            dtype={"Subject": str},
        )
        .rename(
            columns={
                "Earliest Timepoint": y,
                "Channel": "Channels",
            }
        )
        .set_index("Subject")
    )
    data_raw = data_raw.loc[:, y].clip(upper=0.0)

    bottom_lims = []
    top_lims = []
    figs = []
    for picks, outpath, fname_stats in zip(
        SUBJECT_PICKS, FNAMES_PLOT, FNAMES_STATS, strict=True
    ):
        if picks == "paired":
            data = data_raw.query(f"Subject in {MED_PAIRED}")
            add_lines = "Subject"
        else:
            data = data_raw
            add_lines = None
        fig = pte_decode.boxplot_results(
            data=data,
            x=x,
            y=y,
            order=[Cond.OFF.value, Cond.ON.value],
            stat_test=None,
            add_lines=add_lines,
            figsize=(0.8, 1.4),
            show=False,
        )
        bottom, top = fig.axes[0].get_ylim()
        bottom_lims.append(bottom)
        top_lims.append(top)
        figs.append((fig, outpath))

        fname_stats.unlink(missing_ok=True)
        with fname_stats.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["description", "mean", "std", "statistic", "P"])
            cond_a, cond_b = Cond.OFF.value, Cond.ON.value
            description = f"{cond_a} vs {cond_b}"
            print(f"{description = }")
            data_a = (
                data.query(f"{x} == '{cond_a}'")
                .sort_values("Subject")
                .loc[:, y]
                .to_numpy()
            )
            data_b = (
                data.query(f"{x} == '{cond_b}'")
                .sort_values("Subject")
                .loc[:, y]
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
        # ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        y_lims = (-1.9, -0.4)
        ax.set_ylim(y_lims[0], y_lims[1])
        ax.set_yticks([y_lims[0], -1.0, y_lims[1]])
        ax.set_xticklabels(ax.get_xticklabels(), weight="bold", rotation=30)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xlabel("")
        motor_intention.plotting_settings.save_fig(fig, outpath)


if __name__ == "__main__":
    task_plot_decoding_times_medoffvson()
    # plt.show(block=True)
