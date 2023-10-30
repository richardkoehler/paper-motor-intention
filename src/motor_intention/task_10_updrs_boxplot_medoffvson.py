"""Plot UPDRS Med. OFF vs ON"""
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

PLOT_PATH = constants.PLOTS / "updrs"
PLOT_PATH.mkdir(parents=True, exist_ok=True)

PART_INFO = constants.DATA / "participant_info.csv"
BASENAME = "UPDRS_boxplot_medoffvson"
FNAME_PLOT = PLOT_PATH / (BASENAME + ".svg")
FNAME_STATS = PLOT_PATH / (BASENAME + "_stats.csv")


class Cond(Enum):
    OFF_THERAPY = "OFF Therapy"
    ON_LEVODOPA = "ON Levodopa"
    ON_STN_DBS = "ON STN-DBS"


@pytask.mark.depends_on(PART_INFO)
@pytask.mark.produces(FNAME_PLOT)
@pytask.mark.produces(FNAME_STATS)
def task_plot_updrs_medoffvson() -> None:
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.medoffvson()
    x = "Medication"
    y = "UPDRS-III"
    order = [Cond.OFF_THERAPY, Cond.ON_LEVODOPA]
    data_raw = pd.read_csv(PART_INFO).rename(
        columns={
            "ID": "Subject",
            f"{y} recordingMedOFF": Cond.OFF_THERAPY.value,
            f"{y} recordingMedON": Cond.ON_LEVODOPA.value,
        }
    )
    data = (
        pd.melt(
            data_raw,
            value_vars=[cond.value for cond in order],
            id_vars=["Subject"],
            var_name="Medication",
            value_name=y,
        )
        .dropna(axis="index", how="any")
        .query(f"Subject in {[sub.strip('sub-') for sub in constants.MED_PAIRED]}")
        .sort_values("Subject")
    )
    print(data.head())
    outpath = PLOT_PATH / (BASENAME + ".svg")
    figsize = (1.0, 1.3)
    fig = pte_decode.boxplot_updrs(
        data=data,
        outpath=None,
        x=x,
        y=y,
        add_lines="Subject",
        order=[cond.value for cond in order],
        title=None,
        figsize=figsize,
        show=False,
    )
    ax = fig.axes[0]
    print(ax.get_ylim())
    ylims = (5, 50)
    ax.set_ylim(ylims[0], ylims[-1])
    ax.set_yticks([ylims[0], ylims[-1]])
    ax.set_xticklabels(ax.get_xticklabels(), weight="bold")
    motor_intention.plotting_settings.save_fig(fig, outpath)

    FNAME_STATS.unlink(missing_ok=True)
    with open(FNAME_STATS, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["description", "mean", "std", "statistic", "P"])

        statistic = np.mean
        for cond in order:
            description = cond.value
            print(f"{description = }")
            data_cond = data.query(f"{x} == '{cond.value}'")[y].to_numpy()
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

        for cond_a, cond_b in ((Cond.OFF_THERAPY, Cond.ON_LEVODOPA),):
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


if __name__ == "__main__":
    task_plot_updrs_medoffvson()
    # plt.show(block=True)
