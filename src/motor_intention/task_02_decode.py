"""Run decoding pipeline. """
from __future__ import annotations

import pathlib
import time
from typing import Annotated, Literal

import pte
import pte_decode
from pytask import Product

import motor_intention.project_constants as constants


def decode(
    channels_used: Literal["all", "single"],
    in_path: pathlib.Path,
    out_paths: dict[str, pathlib.Path],
) -> None:
    if channels_used == "all":
        channel_types = ("ecog", "dbs")
    else:
        channel_types = ("ecog",)

    n_jobs = -1

    classifier_parameters = [
        {
            "classifier": "lda",
            "balancing_method": "balance_weights",
            "optimize": False,
        },
    ]
    scoring = "balanced_accuracy"
    prediction_mode = "decision_function"
    hemispheres_used = "contralat"
    channels_used = channels_used
    n_splits_outer = "max"
    n_splits_inner = 10
    feature_keywords = [
        "fft_theta",
        "fft_alpha",
        "fft_low beta",
        "fft_high beta",
        "fft_low gamma",
        "fft_high gamma",
        "fft_high frequency activity",
    ]
    calculate_feature_importance = True  # Must be True, False or an Integer
    # How many previous samples are used at each time point. Set to [1] to only
    # use the current time point.
    timepoint_features = range(1, 2)  # range(1, 2) is equal to [1]
    feature_normalization_mode = None  # "by_latest_sample"
    # Classification targets
    targets = [(-0.1, "trial_onset")]

    label_channels = [
        "SQUARED_EMG",
        "SQUARED_INTERPOLATED_EMG",
    ]
    targets_for_plotting = [
        "rms_100",
        "SQUARED_EMG",
        "SQUARED_INTERPOLATED_EMG",
        "analog",
        "SQUARED_ROTATION",
    ]

    PATH_BAD_EPOCHS = constants.DATA / "bad_epochs"
    if not PATH_BAD_EPOCHS.is_dir():
        raise ValueError(f"Directory not found: {PATH_BAD_EPOCHS}")

    start = time.perf_counter()

    # Initialize filefinder instance
    file_finder = pte.filetools.get_filefinder(datatype="any")
    file_finder.find_files(
        directory=in_path,
        extensions="FEATURES.csv",
        stimulation=None,
        exclude=None,
    )
    print(file_finder)
    feature_files = file_finder.files[-1::-1]

    for CLASSIFIER in classifier_parameters:
        classifier, balancing, optimize = CLASSIFIER.values()
        for target_begin, target_end in targets:
            for types_used in channel_types:
                for use_times in timepoint_features:
                    feature_normalization_mode = (
                        None if use_times == 1 else feature_normalization_mode
                    )
                    print(
                        "\n",
                        classifier,
                        balancing,
                        optimize,
                        target_begin,
                        target_end,
                        types_used,
                        use_times,
                    )
                    out_path = out_paths[types_used]
                    out_path.mkdir(exist_ok=True)
                    pte_decode.run_pipeline_multiproc(
                        pipeline_steps=[
                            "engineer",
                            "select",
                            "decode",
                        ],
                        feature_root=in_path,
                        filepaths_features=feature_files,  # type: ignore
                        n_jobs=n_jobs,
                        classifier=classifier,
                        label_channels=label_channels,
                        target_begin=target_begin,
                        target_end=target_end,
                        optimize=optimize,
                        balancing=balancing,
                        out_root=out_path,
                        channels_used=channels_used,
                        types_used=types_used,
                        hemispheres_used=hemispheres_used,
                        feature_keywords=feature_keywords,
                        n_splits_outer=n_splits_outer,
                        scoring=scoring,
                        feature_importance=calculate_feature_importance,
                        plotting_target_channels=targets_for_plotting,
                        prediction_mode=prediction_mode,
                        use_times=use_times,
                        normalization_mode=feature_normalization_mode,
                        bad_epochs_path=PATH_BAD_EPOCHS,
                        rest_begin=-3.0,
                        rest_end=-2.0,
                        dist_end=1.0,
                        verbose=False,
                        n_splits_inner=n_splits_inner,
                        side="auto",
                    )

    print(f"Time elapsed: {(time.perf_counter()-start)/60:.2f} minutes")


def task_decode_stimoff(
    in_path: pathlib.Path = constants.DERIVATIVES / "features" / "stim_off",
    out_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.DERIVATIVES / "decode" / "stim_off" / ch for ch in ("dbs", "ecog")
    },
) -> None:
    decode(
        channels_used="all",
        in_path=in_path,
        out_paths=out_paths,
    )


def task_decode_stimon(
    in_path: pathlib.Path = constants.DERIVATIVES / "features" / "stim_on",
    out_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.DERIVATIVES / "decode" / "stim_on" / ch for ch in ("dbs", "ecog")
    },
) -> None:
    decode(
        channels_used="all",
        in_path=in_path,
        out_paths=out_paths,
    )


def task_decode_single_ch_stimoff(
    in_path: pathlib.Path = constants.DERIVATIVES / "features" / "stim_off",
    out_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.DERIVATIVES / "decode" / "stim_off_single_chs" / ch
        for ch in ("ecog",)
    },
) -> None:
    decode(
        stimulation="Off",
        channels_used="single",
        in_path=in_path,
        out_paths=out_paths,
    )


def task_decode_single_ch_stimon(
    in_path: pathlib.Path = constants.DERIVATIVES / "features" / "stim_on",
    out_paths: dict[str, Annotated[pathlib.Path, Product]] = {
        ch: constants.DERIVATIVES / "decode" / "stim_on_single_chs" / ch
        for ch in ("ecog",)
    },
) -> None:
    decode(
        stimulation="On",
        channels_used="single",
        in_path=in_path,
        out_paths=out_paths,
    )


if __name__ == "__main__":
    task_decode_stimoff()
    task_decode_stimon()
    task_decode_single_ch_stimoff()
    task_decode_single_ch_stimon()
