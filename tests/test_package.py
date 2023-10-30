from __future__ import annotations

import importlib.metadata

import motor_intention as m


def test_version():
    assert importlib.metadata.version("motor_intention") == m.__version__
