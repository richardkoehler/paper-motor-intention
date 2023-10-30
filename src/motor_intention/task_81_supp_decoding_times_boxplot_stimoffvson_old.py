"""Plot decoding times."""
from __future__ import annotations

import pandas as pd
import pte_decode
import pytask
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants
import motor_intention.stats_helpers

PLOT_PATH = constants.PLOTS
PLOT_PATH.mkdir(parents=True, exist_ok=True)

TIMES = constants.RESULTS / "decode" / "stim_on" / "ecog" / "decodingtimes.csv"
BASENAME = "decodingtimes_boxplot_ecog_stimoffvson"
SUBJECT_PICKS = ("paired", "all")


@pytask.mark.depends_on(TIMES)
@pytask.mark.produces(
    PLOT_PATH / (BASENAME + f"_{picks}.svg") for picks in SUBJECT_PICKS
)
def task_plot_decoding_times_stimoffvson() -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.stimoffvson()

    data = (
        pd.read_csv(
            TIMES,
            dtype={"Subject": str},
        )
        .rename(
            columns={
                "Earliest Timepoint": "Time (s)",
                "Channel": "Channels",
            }
        )
        .query("Medication == 'OFF'")
        .set_index("Subject")
    )
    data.loc[:, "Time (s)"] = data.loc[:, "Time (s)"].clip(upper=0.0)

    bottom_lims = []
    top_lims = []
    figs = []
    for picks in SUBJECT_PICKS:
        if picks == "paired":
            data_pick = data.query(f"Subject in {constants.STIM_PAIRED_SUBS}")
            keep = []
            for index, row in data_pick.iterrows():
                if constants.STIM_PAIRED[f"sub-{index}"] == row["Medication"]:
                    keep.append(True)
                else:
                    keep.append(False)
            data_pick = data_pick[keep]
            add_lines = "Subject"
            stat_test = motor_intention.stats_helpers.permutation_onesample()
        else:
            data_pick = data
            add_lines = None
            stat_test = motor_intention.stats_helpers.permutation_twosample()
        outpath = PLOT_PATH / (BASENAME + f"_{picks}.svg")
        fig = pte_decode.boxplot_results(
            data=data_pick,
            outpath=None,
            x="Stimulation",
            y="Time (s)",
            hue=None,
            order=["OFF", "ON"],
            hue_order=None,
            stat_test=stat_test,
            alpha=0.05,
            add_lines=add_lines,
            title=None,
            figsize="auto",
            show=False,
        )
        bottom, top = fig.axes[0].get_ylim()
        bottom_lims.append(bottom)
        top_lims.append(top)
        figs.append((fig, outpath))

    bottom_lim = min(bottom_lims)
    top_lim = max(top_lims)

    for fig, outpath in figs:
        fig.axes[0].set_ylim(bottom=bottom_lim, top=top_lim)
        fig.savefig(str(outpath))


if __name__ == "__main__":
    task_plot_decoding_times_stimoffvson()
    plt.show(block=True)
