"""Plot earliest prediction timepoints."""
from __future__ import annotations

import csv
from enum import Enum

import numpy as np
import pandas as pd
import pte_decode
import pytask
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
TIMES = (
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


@pytask.mark.depends_on(TIMES)
@pytask.mark.produces(
    [PLOT_PATH / (BASENAME + f"_med{med}.svg") for med in MEDICATION]
    + [PLOT_PATH / (BASENAME + f"_med{med}_stats.csv") for med in MEDICATION]
)
def task_plot_decoding_times_bycorticalregion() -> None:
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
    times = pd.read_csv(TIMES, index_col=["Subject", "Channel"])

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
    # MED_PAIRED = [sub.strip("sub-") for sub in constants.MED_PAIRED]
    # data_paired = data_raw.query(f"Subject in {MED_PAIRED}")

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

    # motor_intention.plotting_settings.medoffvson()
    # region_picks = ["Parietal", "Sensory", "Motor"]
    # # data_pick = data_paired.query(f"region in {region_picks}")
    # # ymax = data_pick["Time [s]"].max() + 0.1
    # # ymin = data_pick["Time [s]"].min() - 0.1

    # bottom_lims = []
    # top_lims = []
    # figs = []
    # for region in region_picks:
    #     basename = (
    #         f"decodingtimes_boxplot_{region.lower()}_medoffvson_paired.svg"
    #     )
    #     outpath = constants.PLOTS / basename
    #     fig = pte_decode.boxplot_results(
    #         data=data_paired.query(f"`{x}` == '{region}'"),
    #         outpath=None,
    #         x="Medication",
    #         y="Time [s]",
    #         hue=None,
    #         order=["OFF", "ON"],
    #         hue_order=None,
    #         stat_test=motor_intention.stats_helpers.permutation_onesample(),
    #         alpha=0.05,
    #         add_lines="Subject",
    #         figsize="auto",
    #         show=False,
    #     )
    #     # fig.axes[0].set_ylim(bottom=ymin, top=ymax)
    #     fig.tight_layout()

    #     bottom, top = fig.axes[0].get_ylim()
    #     bottom_lims.append(bottom)
    #     top_lims.append(top)
    #     figs.append((fig, outpath))

    # bottom_lim = min(bottom_lims)
    # top_lim = max(top_lims)

    # for fig, outpath in figs:
    #     fig.axes[0].set_ylim(bottom=bottom_lim, top=top_lim)
    #     fig.savefig(str(outpath))
    #     # fig.show()
    # # plt.show(block=True)

    # bottom_lims = []
    # top_lims = []
    # figs = []
    # for region in region_picks:
    #     basename = f"decodingtimes_boxplot_{region.lower()}_medoffvson_all.svg"
    #     outpath = constants.PLOTS / basename
    #     fig = pte_decode.boxplot_results(
    #         data=data_raw.query(f"region == '{region}'"),
    #         outpath=None,
    #         x="Medication",
    #         y="Time [s]",
    #         hue=None,
    #         order=["OFF", "ON"],
    #         hue_order=None,
    #         stat_test=motor_intention.stats_helpers.permutation_twosample(),
    #         alpha=0.05,
    #         add_lines=None,
    #         # title=region,
    #         figsize="auto",
    #         show=False,
    #     )
    #     # fig.axes[0].set_ylim(bottom=ymin, top=ymax)
    #     fig.tight_layout()

    #     bottom, top = fig.axes[0].get_ylim()
    #     bottom_lims.append(bottom)
    #     top_lims.append(top)
    #     figs.append((fig, outpath))

    # bottom_lim = min(bottom_lims)
    # top_lim = max(top_lims)

    # for fig, outpath in figs:
    #     fig.axes[0].set_ylim(bottom=bottom_lim, top=top_lim)
    #     fig.savefig(str(outpath))
    #     # fig.show()


if __name__ == "__main__":
    task_plot_decoding_times_bycorticalregion()
    plt.show(block=True)
