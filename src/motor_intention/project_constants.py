"""Project constants for the motor intention decoding project."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)  # noqa: NPY002
random.seed(RANDOM_SEED)


SRC = Path(__file__).parent.resolve()
DATA = SRC.joinpath("..", "..", "data").resolve()
RAWDATA = DATA / "00_rawdata_mot_onset_pred_downsample"
RAWDATA_ORIG = DATA / "00_rawdata_mot_onset_pred"
DERIVATIVES = DATA / "01_derivatives"
RESULTS = DATA / "02_results"
PLOTS = DATA / "03_plots"
for folder in [DATA, DERIVATIVES, RESULTS, PLOTS]:
    folder.mkdir(exist_ok=True)

MED_PAIRED = [
    "sub-EL003",
    "sub-EL004",
    "sub-EL005",
    "sub-EL006",
    "sub-EL007",
    "sub-EL008",
    "sub-EL009",
    "sub-EL010",
    "sub-EL011",
    "sub-EL012",
    "sub-EL013",
    "sub-EL014",
    "sub-EL015",
    "sub-EL016",
    "sub-EL017",
]
STIM_PAIRED = {
    "sub-EL002": "OFF",
    "sub-EL005": "OFF",
    "sub-EL006": "OFF",
    "sub-EL007": "OFF",
    "sub-EL008": "OFF",
    "sub-EL009": "OFF",
    "sub-EL010": "ON",
    "sub-EL012": "OFF",
    "sub-EL014": "ON",
    "sub-EL016": "ON",
    "sub-EL017": "OFF",
}
STIM_PAIRED_SUBS = [sub.strip("sub-") for sub in STIM_PAIRED]
ECOG_HEMISPHERES = {
    "EL002": "L",
    "EL003": "L",
    "EL004": "L",
    "EL005": "R",
    "EL006": "R",
    "EL007": "R",
    "EL008": "L",
    "EL009": "L",
    "EL010": "R",
    "EL011": "R",
    "EL012": "R",
    "EL013": "R",
    "EL014": "R",
    "EL015": "R",
    "EL016": "R",
    "EL017": "R",
    "FOG006": "R",
    "FOG008": "R",
    "FOG010": "R",
    "FOG012": "R",
    "FOG014": "R",
    "FOG016": "R",
    "FOG021": "R",
    "FOG022": "R",
    "FOGC001": "R",
}
FREQ_BANDS = {
    "theta": [4, 7],
    "alpha": [8, 12],
    "low_beta": [13, 20],
    "high_beta": [21, 35],
    "all_beta": [13, 35],
    "low_gamma": [60, 90],
    "high_gamma": [91, 200],
    "all_gamma": [60, 200],
    "high_frequency_activity": [201, 400],
}
SUBJECTS = [
    "sub-EL002",
    "sub-EL003",
    "sub-EL004",
    "sub-EL005",
    "sub-EL006",
    "sub-EL007",
    "sub-EL008",
    "sub-EL009",
    "sub-EL010",
    "sub-EL011",
    "sub-EL012",
    "sub-EL013",
    "sub-EL014",
    "sub-EL015",
    "sub-EL016",
    "sub-EL017",
    "FOG006",
    "FOG008",
    "FOG010",
    "FOG012",
    "FOG014",
    "FOG016",
    "FOG021",
    "FOG022",
    "FOGC001",
]
