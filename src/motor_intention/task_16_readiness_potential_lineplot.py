"""Perform and save time frequency analysis of given files."""
from __future__ import annotations

from typing import Literal

import pandas as pd
import pte_decode
import pytask
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

PLOT_DIR = constants.PLOTS / "readiness_potential"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

BASENAME = "rp_lineplot"


def rp_lineplot(
    ch_type: Literal["ecog", "dbs"],
    show_plots: bool = False,  # stimulation: Literal["Off", "On"],
) -> None:
    """Main function of this script."""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.medoff_medon_stimon()

    ch_str = "motorcortex" if ch_type == "ecog" else "stn"
    outpath = PLOT_DIR / (f"{BASENAME}_{ch_str}.svg")
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(2.3, 3.4))
    i = 0
    legend = True
    for stimulation in ("Off", "On"):
        PIPELINE = f"stim_{stimulation.lower()}"
        IN_DIR = constants.DERIVATIVES / "readiness_potential" / PIPELINE / ch_type

        X_LABEL = "Time [s]"
        if ch_type == "ecog":
            Y_LIMS = (-68, 10)
        elif ch_type == "dbs":
            Y_LIMS = (-26, 4)
        Y_LABEL = "Voltage [µV]"
        THRESHOLD = 0.0
        CORRECTION_METHOD = "cluster_pvals"
        ALPHA = 0.05
        N_PERM = 10000

        rp = pd.read_csv(
            str(IN_DIR / "readiness_potential.csv"),
            dtype={
                "Subject": str,
                "Medication": str,
                "Stimulation": str,
                "Channels": str,
            },
        ).replace({"MotorCortex": "Motor Cortex"})

        if stimulation == "Off":
            COND = "Medication"
            COND_ABB = "Med."
        else:
            COND = "Stimulation"
            COND_ABB = "Stim."
            rp = rp.query("Medication == 'OFF'")
        rp = rp.set_index(["Subject", "Medication", "Stimulation", "Channels"])
        times = rp.columns.to_numpy(dtype=float)
        rps = {
            "OFF": rp.query(f"{COND} == 'OFF'").to_numpy().T,
            "ON": rp.query(f"{COND} == 'ON'").to_numpy().T,
        }
        conds = ("OFF", "ON") if COND == "Medication" else ("ON",)
        for cond in conds:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
            x_label = X_LABEL if i == 2 else None
            pte_decode.lineplot_single(
                data=rps[cond],
                times=times,
                ax=axs[i],
                # figsize=(6.4, 3.2),
                outpath=None,
                x_label=x_label,
                y_label=Y_LABEL,
                y_lims=None,  # Y_LIMS,
                title=None,  # f"{COND_ABB} {cond}",
                threshold=THRESHOLD,
                color=color,
                alpha=ALPHA,
                n_perm=N_PERM,
                correction_method=CORRECTION_METHOD,
                two_tailed=False,
                one_tailed_test="smaller",
                add_vline=0.0,
                print_n=True,
                legend=legend,
                show=False,
            )
            axs[i].set_xlim([-3, 2])
            axs[i].set_xticks([-3, 0, 2])
            axs[i].set_ylim([Y_LIMS[0], Y_LIMS[1]])
            axs[i].set_yticks([Y_LIMS[0], 0, Y_LIMS[1]])
            axs[i].spines["left"].set_position(("outward", 3))
            axs[i].spines["bottom"].set_position(("outward", 3))
            legend = False
            i += 1
    for ax in axs:
        bottom_lim, top_lim = ax.get_ylim()
        print(f"{bottom_lim = }, {top_lim = }")
    motor_intention.plotting_settings.save_fig(fig, outpath)
    if show_plots:
        plt.show()
    else:
        plt.close()


@pytask.mark.depends_on(
    (
        constants.DERIVATIVES / "readiness_potential" / "stim_off" / "ecog",
        constants.DERIVATIVES / "readiness_potential" / "stim_on" / "ecog",
    )
)
@pytask.mark.produces(PLOT_DIR / (f"{BASENAME}_motorcortex.svg"))
def task_rp_lineplot_ecog() -> None:
    """Run main function."""
    rp_lineplot(ch_type="ecog", show_plots=False)


@pytask.mark.depends_on(
    (
        constants.DERIVATIVES / "readiness_potential" / "stim_off" / "dbs",
        constants.DERIVATIVES / "readiness_potential" / "stim_on" / "dbs",
    )
)
@pytask.mark.produces(PLOT_DIR / (f"{BASENAME}_stn.svg"))
def task_rp_lineplot_dbs() -> None:
    """Run main function."""
    rp_lineplot(ch_type="dbs", show_plots=False)


if __name__ == "__main__":
    rp_lineplot(ch_type="ecog", show_plots=True)
    rp_lineplot(ch_type="dbs", show_plots=True)
