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

STIMULATION = "off"
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
SUBJECT_PICKS = ("paired", "all")
MEDICATION = ("OFF", "ON")

BASENAME = "decodingtimes_boxplot"


class Cond(Enum):
    PARIETAL = "Parietal"
    SENSORY = "Sensory"
    MOTOR = "Motor"


def task_plot_decoding_times_bycorticalregion(in_path: Path = IN_PATH) -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()

    x = "Cortical Region"
    y = "Time [s]"
    coords = (
        pd.read_csv(constants.DATA / "elec_ecog_bip.csv")
        .dropna(axis="columns", how="all")
        .rename(columns={"name": "Channel", "region": x})
        .set_index(["Subject", "Channel"])
    )
    times = pd.read_csv(in_path, index_col=["Subject", "Channel"])

    data_list = []
    prefrontal_subs = set()
    for med in ("OFF", "ON"):
        times_pick = times.query(f"Medication == '{med}'")
        data_list.append(
            pd.concat([times_pick, coords], axis="columns").reset_index().dropna()
        )
        total = data_list[-1]
        print(f"Total channels {med} Med.: {len(total)}")
        query = f"`{x}` == 'Prefrontal'"
        prefrontal = total.query(query)
        print(f"Prefrontal channels {med} Med.: {len(prefrontal)}")
        prefrontal_subs = prefrontal_subs.union(set(prefrontal["Subject"].unique()))
        print(f"Prefrontal Subjects Current Total: {len(prefrontal_subs)}")
    data_raw = (
        pd.concat(data_list)
        .query("used == 1")
        .rename(columns={"Earliest Timepoint": y})
        .sort_values(by=["Subject", "Channel"])
        .set_index(["Subject", "Channel"])
    )
    data_raw[y] = data_raw[y].clip(upper=0.0)

    motor_intention.plotting_settings.cortical_region()
    for med in MEDICATION:
        basename = f"{BASENAME}_bycorticalregion_med{med.lower()}"
        fname_plot = PLOT_PATH / f"{basename}.svg"
        fname_stats = PLOT_PATH / f"{basename}_stats.csv"
        data = data_raw.query(
            f"`{x}` in ['Parietal', 'Sensory', 'Motor'] and Medication == '{med}'"
        )
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
        xlims = (-2, 0)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xticks([xlims[0], -1, xlims[1]])
        ax.set_yticklabels(
            ax.get_yticklabels(),
            weight="bold",
        )
        motor_intention.plotting_settings.save_fig(fig, fname_plot)
        fname_stats.unlink(missing_ok=True)

        with fname_stats.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["description", "mean", "std", "statistic", "P"])
            for cond in Cond:
                description = cond.value
                print(f"{description = }")
                data_cond = data.query(f"`{x}` == '{cond.value}'").loc[:, y].to_numpy()
                test = scipy.stats.permutation_test(
                    (data_cond - 0.0,),
                    statistic := np.mean,
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
                (Cond.PARIETAL, Cond.SENSORY),
                (Cond.PARIETAL, Cond.MOTOR),
                (Cond.SENSORY, Cond.MOTOR),
            ):
                description = f"{cond_a.value} vs {cond_b.value}"
                print(f"{description = }")
                data_a = data.query(f"`{x}` == '{cond_a.value}'")[y].to_numpy()
                data_b = data.query(f"`{x}` == '{cond_b.value}'")[y].to_numpy()
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
    task_plot_decoding_times_bycorticalregion()
    plt.show(block=True)
