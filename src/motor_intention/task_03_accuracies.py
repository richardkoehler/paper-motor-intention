"""Average classification accuracies."""
from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated

import pte
import pte_decode
from pytask import Product

import motor_intention.project_constants as constants

CHANNELS = ("ecog", "dbs")
INPATHS_STIM_OFF = tuple(
    constants.DERIVATIVES / "decode" / "stim_off" / ch for ch in CHANNELS
)
INPATHS_STIM_ON = tuple(
    constants.DERIVATIVES / "decode" / "stim_on" / ch for ch in CHANNELS
)
OUTPATHS_STIM_OFF = tuple(
    constants.RESULTS / "decode" / "stim_off" / ch / "accuracies.csv" for ch in CHANNELS
)
OUTPATHS_STIM_ON = tuple(
    constants.RESULTS / "decode" / "stim_on" / ch / "accuracies.csv" for ch in CHANNELS
)


def task_write_accuracies_stimoff(
    in_paths: Sequence[pathlib.Path] = INPATHS_STIM_OFF,
    out_paths: Sequence[Annotated[pathlib.Path, Product]] = OUTPATHS_STIM_OFF,
) -> None:
    write_accuracies(in_paths=in_paths, out_paths=out_paths)


def task_write_accuracies_stimon(
    in_paths: Sequence[pathlib.Path] = INPATHS_STIM_ON,
    out_paths: Sequence[Annotated[pathlib.Path, Product]] = OUTPATHS_STIM_ON,
) -> None:
    write_accuracies(in_paths=in_paths, out_paths=out_paths)


def write_accuracies(
    in_paths: Sequence[pathlib.Path],
    out_paths: Sequence[pathlib.Path],
) -> None:
    """Main function of this script"""
    MEDICATION = None
    STIMULATION = None
    PICKS = None

    for in_path, out_path in zip(in_paths, out_paths, strict=True):
        file_finder = pte.filetools.get_filefinder(datatype="any")
        file_finder.find_files(
            directory=in_path,
            keywords=PICKS,
            extensions=["Scores.csv"],
            medication=MEDICATION,
            stimulation=STIMULATION,
            exclude=None,
        )
        print(file_finder)
        print("Number of files found:", len(file_finder.files))

        data = pte_decode.load_scores(files=file_finder.files, average_runs=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out_path, index=False)


if __name__ == "__main__":
    task_write_accuracies_stimon()
    task_write_accuracies_stimoff()
