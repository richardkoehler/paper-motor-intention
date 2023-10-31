"""Plot performance Medication OFF vs ON."""
from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import pte_decode
import scipy.stats

import motor_intention.plotting_settings
import motor_intention.project_constants as constants
import motor_intention.stats_helpers

DECODE = "decode"
PLOT_PATH = constants.PLOTS / "supplements" / DECODE
PLOT_PATH.mkdir(parents=True, exist_ok=True)

CHANNEL = "ecog"
STIM = "On"
IN_PATH = (
    constants.RESULTS / DECODE / f"stim_{STIM.lower()}" / CHANNEL / "accuracies.csv"
)
BASENAME = f"accuracies_boxplot_{CHANNEL}_stimoffvson"
SUBJECT_PICKS = ("paired",)
FNAMES_PLOT = (PLOT_PATH / f"{BASENAME}_{pick}.svg" for pick in SUBJECT_PICKS)
FNAMES_STATS = (PLOT_PATH / f"{BASENAME}_{pick}_stats.csv" for pick in SUBJECT_PICKS)


class Cond(Enum):
    OFF = "OFF"
    ON = "ON"


def task_plot_accuracies_stimoffvson(in_path: Path = IN_PATH) -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.stimoffvson()

    x = "Stimulation"
    y = "Balanced Accuracy"
    data_raw = (
        pd.read_csv(in_path)
        .rename(columns={"Channel": "Channels", "balanced_accuracy": y})
        .query("Medication == 'OFF'")
        .set_index("Subject")
    )
    bottom_lims = []
    top_lims = []
    figs = []
    for picks, fname_plot, fname_stats in zip(
        SUBJECT_PICKS, FNAMES_PLOT, FNAMES_STATS, strict=True
    ):
        if picks == "paired":
            data = data_raw.query(f"Subject in {constants.STIM_PAIRED_SUBS}")
            keep = []
            for index, row in data.iterrows():
                if constants.STIM_PAIRED[f"sub-{index}"] == row["Medication"]:
                    keep.append(True)
                else:
                    keep.append(False)
            data = data[keep]
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
        figs.append((fig, fname_plot))

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

    for fig, fname_plot in figs:
        ax = fig.axes[0]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.axhline(0.5, color="black", linestyle="--", alpha=0.5)
        y_lims = (0.5, 0.95)
        ax.set_ylim(y_lims[0], y_lims[1])
        ax.set_yticks([y_lims[0], 0.6, 0.7, 0.8, y_lims[1]])
        ax.set_xticklabels(ax.get_xticklabels(), weight="bold", rotation=30)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xlabel("")
        motor_intention.plotting_settings.save_fig(fig, fname_plot)


if __name__ == "__main__":
    task_plot_accuracies_stimoffvson()
