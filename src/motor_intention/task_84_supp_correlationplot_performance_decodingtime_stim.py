"""Plot correlation between performance and decoding time for stimulation on and off."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pte_stats
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt

import motor_intention.plotting_settings
import motor_intention.project_constants as constants

ITEM_X = "Time [s]"
ITEM_Y = "Balanced Accuracy"


def task_plot_correlation_perf_time(
    in_path: Path = constants.RESULTS
    / "decode"
    / "stim_on"
    / "ecog"
    / "accuracies.csv",
) -> None:
    """Main function of this script"""
    motor_intention.plotting_settings.activate()

    PLOT_PATH = constants.PLOTS / "supplements" / "decode"
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    STIM_PAIRED_SUBS = [sub.strip("sub-") for sub in constants.STIM_PAIRED]

    diff_str = r"$Δ_{ON-OFF}$"
    x = f"{ITEM_X} {diff_str}"
    y = f"{ITEM_Y} {diff_str}"
    x_str = x.replace(":", "").replace(".", "")
    y_str = y.replace(":", "").replace(".", "")
    basename = f"correlation_{x_str}_{y_str}"

    data = (
        pd.read_csv(in_path)
        .rename(
            columns={
                "channel": "Channels",
                "balanced_accuracy": ITEM_Y,
            }
        )
        .query(f"Subject in {STIM_PAIRED_SUBS} and Medication == 'OFF'")
        .set_index("Subject")
    )
    keep = []
    for index, row in data.iterrows():
        if constants.STIM_PAIRED[f"sub-{index}"] == row["Medication"]:
            keep.append(True)
        else:
            keep.append(False)
    performance = (
        data[keep]
        .loc[:, ["Stimulation", ITEM_Y]]
        .reset_index()
        .set_index(["Subject", "Stimulation"])
    )
    data = (
        pd.read_csv(in_path / "decodingtimes.csv")
        .rename(
            columns={
                "Earliest Timepoint": ITEM_X,
                "Channel": "Channels",
            }
        )
        .query(f"Subject in {STIM_PAIRED_SUBS} and Medication == 'OFF'")
        .set_index("Subject")
    )
    keep = []
    for index, row in data.iterrows():
        if constants.STIM_PAIRED[f"sub-{index}"] == row["Medication"]:
            keep.append(True)
        else:
            keep.append(False)
    data[ITEM_X] = data[ITEM_X].clip(upper=0.0)
    time = (
        data[keep]
        .loc[:, ["Stimulation", ITEM_X]]
        .reset_index()
        .set_index(["Subject", "Stimulation"])
    )

    data = pd.concat((time, performance), axis="columns")

    data = (
        data.loc[:, [ITEM_X, ITEM_Y]]
        .reset_index()
        .sort_values(by="Subject")
        .set_index("Subject")
    )
    data_on = data.query("Stimulation == 'ON'").drop(columns="Stimulation")
    data_off = data.query("Stimulation == 'OFF'").drop(columns="Stimulation")
    data_diff = (data_on - data_off).rename(
        columns={col: f"{col} {diff_str}" for col in data.columns}
    )
    data_xy = data_diff[[x, y]].dropna()
    rho, p = pte_stats.spearmans_rho_permutation(data_xy[x], data_xy[y], n_perm=10000)
    x_lin = data_diff[x].to_numpy()
    y_lin = data_diff[y].to_numpy()
    res_lin = scipy.stats.linregress(x_lin, y_lin)
    p_lin = res_lin.pvalue
    r_lin = res_lin.rvalue
    print(
        f"correlation_{x}_{y}:"
        f"Rho={f'{rho:.2f}'},  P={f'{p:.4f}'};"
        f" r = {r_lin:.2f}, P={p_lin:.4f}"
    )
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
    ax = sns.regplot(
        x=x,
        y=y,
        data=data_xy,
        color="black",
        scatter_kws={"s": 6},
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title = (
        f"\u03C1 = {rho:.2f}"
        f", P = {round(p, 2):.2f}"
        f"; r = {r_lin:.2f}"
        f", P = {round(p_lin, 2):.2f}"
    )
    ax.set_ylabel("\n".join(y.split(": ")))
    ax.set_xlim([-1, 1])
    ax.set_xticks([-1, 0, 1])
    Y_LIMS = [-0.16, 0.16]
    ax.set_ylim(Y_LIMS)
    ax.set_yticks([Y_LIMS[0], 0, Y_LIMS[1]])
    ax.spines["left"].set_position(("outward", 3))
    ax.spines["bottom"].set_position(("outward", 3))
    motor_intention.plotting_settings.save_fig(fig, PLOT_PATH / f"{basename}.svg")
    FNAME_STATS = PLOT_PATH / f"{basename}_stats.txt"
    # FNAME_STATS.unlink(missing_ok=True)
    FNAME_STATS.write_text(title, encoding="utf-8")


if __name__ == "__main__":
    task_plot_correlation_perf_time()
    plt.show(block=True)
