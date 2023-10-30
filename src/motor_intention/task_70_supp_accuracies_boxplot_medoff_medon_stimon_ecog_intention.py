"""Plot decoding performance."""
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
PLOT_PATH = constants.PLOTS / DECODE
PLOT_PATH.mkdir(parents=True, exist_ok=True)

CHANNEL = "ecog"
ACCURACIES = (
    constants.RESULTS / DECODE / "stim_off" / CHANNEL / "accuracies.csv",
    constants.RESULTS / DECODE / "stim_on" / CHANNEL / "accuracies.csv",
)
BASENAME = f"accuracies_boxplot_{CHANNEL}_medoff_medon_stimon"
FNAME_PLOT = PLOT_PATH / (BASENAME + ".svg")
FNAME_STATS = PLOT_PATH / (BASENAME + "_stats.csv")


class Cond(Enum):
    OFF_THERAPY = "OFF Therapy"
    ON_LEVODOPA = "ON Levodopa"
    ON_STN_DBS = "ON STN-DBS"


@pytask.mark.depends_on(ACCURACIES)
@pytask.mark.produces(FNAME_PLOT)
@pytask.mark.produces(FNAME_STATS)
def task_plot_accuracies_medoffvson() -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.medoff_medon_stimon()

    x = "Condition"
    y = "Balanced Accuracy"
    data_list = []
    for stimulation in ("Off", "On"):
        PIPELINE = f"stim_{stimulation.lower()}"
        fpath = constants.RESULTS / DECODE / PIPELINE / CHANNEL / "accuracies.csv"
        acc = pd.read_csv(fpath).rename(
            columns={
                "Channel": "Channels",
                "balanced_accuracy": "Balanced Accuracy",
            }
        )
        if stimulation == "On":
            acc = acc.query("Stimulation == 'ON' and Medication == 'OFF'")
        data_list.append(acc)
    data = pd.concat(data_list, ignore_index=True)
    for i, row in data.iterrows():
        med, stim = row["Medication"], row["Stimulation"]
        if med == "OFF" and stim == "ON":
            data.loc[i, "Condition"] = Cond.ON_STN_DBS.value
        elif med == "ON" and stim == "OFF":
            data.loc[i, "Condition"] = Cond.ON_LEVODOPA.value
        elif med == "OFF" and stim == "OFF":
            data.loc[i, "Condition"] = Cond.OFF_THERAPY.value
        else:
            raise ValueError(
                "Unknown combination of medication and stimulation. Got:"
                f"{med = }, {stim = }"
            )

    outpath = PLOT_PATH / (BASENAME + ".svg")
    figsize = (1.7, 1.3)
    fig = pte_decode.boxplot_all_conds(
        data=data,
        outpath=None,
        x=x,
        y=y,
        order=[
            Cond.OFF_THERAPY.value,
            Cond.ON_LEVODOPA.value,
            Cond.ON_STN_DBS.value,
        ],
        title=None,
        figsize=figsize,
        show=False,
    )
    ax = fig.axes[0]
    # print(ax.get_xlim())
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlim(0.5, 0.95)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.95])
    ax.set_yticklabels(ax.get_yticklabels(), weight="bold")
    motor_intention.plotting_settings.save_fig(fig, outpath)

    FNAME_STATS.unlink(missing_ok=True)
    with open(FNAME_STATS, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["description", "mean", "std", "statistic", "P"])
        statistic = np.mean
        for cond in Cond:
            description = cond.value
            print(f"{description = }")
            data_cond = data.query(f"{x} == '{cond.value}'")[y].to_numpy()
            test = scipy.stats.permutation_test(
                (data_cond - 0.5,),
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

        def statistic(x, y, axis=0):
            return np.mean(a=x, axis=axis) - np.mean(a=y, axis=axis)

        for cond_a, cond_b in (
            (Cond.OFF_THERAPY, Cond.ON_LEVODOPA),
            (Cond.OFF_THERAPY, Cond.ON_STN_DBS),
        ):
            description = f"{cond_a.value} vs {cond_b.value}"
            print(f"{description = }")
            data_a = data.query(f"{x} == '{cond_a.value}'")[y].to_numpy()
            data_b = data.query(f"{x} == '{cond_b.value}'")[y].to_numpy()
            test = scipy.stats.permutation_test(
                (data_a, data_b),
                statistic,
                vectorized=True,
                n_resamples=int(1e6),
                permutation_type="independent",
            )
            print(f"statistic = {test.statistic}, P = {test.pvalue}")
            writer.writerow(
                [
                    description,
                    statistic(data_a, data_b),
                    "n/a",
                    test.statistic,
                    test.pvalue,
                ]
            )


if __name__ == "__main__":
    task_plot_accuracies_medoffvson()
    # plt.show(block=True)
