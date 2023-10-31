"""Plot continuous predictions for all subjects"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import matplotlib as mpl
import numpy as np
import pte
import pte_decode
from matplotlib import pyplot as plt
from pytask import Product

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

DECODE = "decode"
PLOT_PATH = constants.PLOTS / DECODE
PLOT_PATH.mkdir(exist_ok=True, parents=True)

BASENAME = "prediction_lineplot_ecogvslfp"

CH_TYPES = ("ecog", "dbs")
STIM = ("Off", "On")
IN_PATHS = {
    (ch_type, stim): constants.DERIVATIVES / DECODE / f"stim_{stim.lower()}" / ch_type
    for ch_type in CH_TYPES
    for stim in STIM
}


def task_prediction_lineplot_ecogvslfp(
    in_paths: dict[
        tuple[Literal["ecog", "dbs"], Literal["Off", "On"]], Path
    ] = IN_PATHS,
    plot_path: Annotated[Path, Product] = PLOT_PATH / (BASENAME + ".svg"),
    cluster_path: Annotated[Path, Product] = PLOT_PATH / f"{BASENAME}_clusters.json",
) -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()
    motor_intention.plotting_settings.medoff_medon_stimon()

    N_PERM = 10000
    BASELINE = (-3.0, -2.0)
    CORRECTION_METHOD = "cluster_pvals"
    ALPHA = 0.05
    Y_LIMS = (-0.6, 4.5)

    file_finder = pte.filetools.DefaultFinder()

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(2.3, 3.4), sharey=True)
    i = 0
    legend = True
    clusters = {}
    for stimulation in STIM:
        conds_med = ("OFF", "ON") if stimulation == "Off" else ("OFF",)
        for med in conds_med:
            data_map = {}
            for ch_type in CH_TYPES:
                in_path = in_paths[(ch_type, stimulation)]

                file_finder.find_files(
                    directory=in_path,
                    keywords=None,
                    exclude=None,
                    extensions=["PredTimelocked.json"],
                    medication=med,
                    stimulation=stimulation,
                )
                print(file_finder)
                print("Files found:", len(file_finder.files))

                data = pte_decode.load_predictions(
                    files=file_finder.files,
                    baseline=BASELINE,
                    baseline_mode="zscore",
                    baseline_trialwise=False,
                    average_predictions=True,
                )
                data_map[ch_type] = data

            with Path(file_finder.files[0]).open("w", encoding="utf-8") as f:
                pred_data = json.load(f)
            times = np.array(pred_data["times"])

            ecog = data_map["ecog"].sort_values(by=["Subject"])
            lfp = data_map["dbs"].sort_values(by=["Subject"])
            ecog_data = np.stack(ecog.loc[:, "Predictions"].to_list()).squeeze().T
            lfp_data = np.stack(lfp.loc[:, "Predictions"].to_list()).squeeze().T
            assert ecog_data.shape == lfp_data.shape
            print("Subjects used:", ecog.shape[0])

            colors = (
                mpl.rcParams["axes.prop_cycle"].by_key()["color"][i],
                motor_intention.plotting_settings.Color.STN.value,
            )
            x_label = "Time [s]" if i == 2 else None
            _, cluster_times = pte_decode.lineplot_compare(
                x_1=ecog_data,
                x_2=lfp_data,
                times=times,
                ax=axs[i],
                y_lims=None,
                data_labels=["ECOG", "STN-LFP"],
                x_label=x_label,
                y_label="Distance from\nHyperplane [Z]",
                alpha=ALPHA,
                n_perm=N_PERM,
                correction_method=CORRECTION_METHOD,
                two_tailed=True,
                paired_x1x2=True,
                outpath=None,
                legend=legend,
                add_vline=0.0,
                print_n=True,
                colors=colors,
                show=False,
            )
            clusters[f"Med. {med}, Stim. {stimulation.upper()}"] = cluster_times
            axs[i].set_title("")
            axs[i].set_xlim([-3, 2])
            axs[i].set_xticks([-3, 0, 2])
            axs[i].set_ylim([Y_LIMS[0], Y_LIMS[1]])
            axs[i].set_yticks([Y_LIMS[0], 0, Y_LIMS[1]])
            axs[i].set_ybound(lower=Y_LIMS[0], upper=Y_LIMS[1])
            axs[i].spines["left"].set_position(("outward", 3))
            axs[i].spines["bottom"].set_position(("outward", 3))
            legend = False
            i += 1
    with cluster_path.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=4)
    motor_intention.plotting_settings.save_fig(fig, plot_path)


if __name__ == "__main__":
    task_prediction_lineplot_ecogvslfp()
    plt.show(block=True)
