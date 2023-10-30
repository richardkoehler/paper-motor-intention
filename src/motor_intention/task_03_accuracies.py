"""Average classification accuracies."""
from __future__ import annotations

import pathlib
from typing import Annotated

import pte
import pte_decode
from pytask import Product

import motor_intention.project_constants as constants


def write_accuracies(
    in_paths: dict[str, pathlib.Path],
    out_paths: dict[str, pathlib.Path],
) -> None:
    """Main function of this script"""
    channel_types = ("ecog", "dbs")

    MEDICATION = None
    STIMULATION = None
    PICKS = None

    for channel in channel_types:
        file_finder = pte.filetools.get_filefinder(datatype="any")
        file_finder.find_files(
            directory=in_paths[channel],
            keywords=PICKS,
            extensions=["Scores.csv"],
            medication=MEDICATION,
            stimulation=STIMULATION,
            exclude=None,
        )
        print(file_finder)
        print("Number of files found:", len(file_finder.files))

        data = pte_decode.load_scores(files=file_finder.files, average_runs=False)
        outpath_results = out_paths[channel]
        outpath_results.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(outpath_results, index=False)


def task_write_accuracies_stimoff(
    in_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.DERIVATIVES / "decode" / "stim_off" / ch for ch in ("ecog", "dbs")
    },
    out_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.RESULTS / "decode" / "stim_off" / ch / "accuracies.csv"
        for ch in ("ecog", "dbs")
    },
) -> None:
    write_accuracies(in_paths=in_paths, out_paths=out_paths)


def task_write_accuracies_stimon(
    in_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.DERIVATIVES / "decode" / "stim_on" / ch for ch in ("ecog", "dbs")
    },
    out_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.RESULTS / "decode" / "stim_on" / ch / "accuracies.csv"
        for ch in ("ecog", "dbs")
    },
) -> None:
    write_accuracies(in_paths=in_paths, out_paths=out_paths)


if __name__ == "__main__":
    task_write_accuracies_stimon()
    task_write_accuracies_stimoff()
