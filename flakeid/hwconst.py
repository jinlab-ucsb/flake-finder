"""Constants of the microscope that the analysis and the scan program share: the 100x objective against the 10x one.

The defaults are the BX51's (measured there).  Another microscope gives its own through the environment, which the
scan program sets from its settings (ratio_100x, offset_100x_um) before anything else is imported, and which the
analysis process it starts inherits:

    FLAKEID_RATIO_100X     100x pixels per 10x pixel (same camera)
    FLAKEID_OFFSET_100X    "dx,dy": the stage move (um) that puts under the 100x objective what was centred at 10x
"""
import os


def _float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _pair(name, default):
    try:
        a, b = os.environ[name].split(",")
        return (float(a), float(b))
    except (KeyError, ValueError):
        return default


RATIO_100X = _float("FLAKEID_RATIO_100X", 9.54)
OFFSET_100X = _pair("FLAKEID_OFFSET_100X", (24.0, -18.3))
