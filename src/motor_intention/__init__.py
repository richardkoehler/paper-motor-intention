"""
Copyright (c) 2023 Richard M. Köhler. All rights reserved.

motor-intention: Code used for paper investigating motor intention in Parkinson's disease patients.
"""


from __future__ import annotations

from motor_intention.plotting_settings import set_mne_backends
from motor_intention.project_constants import set_random_seed

from ._version import version as __version__

set_random_seed()
set_mne_backends()

__all__ = ["__version__"]
