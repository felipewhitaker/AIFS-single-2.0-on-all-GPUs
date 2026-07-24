"""
Run an AIFS weather forecast on any hardware (CPU or GPU).

Public API
----------
Device helpers:
    :func:`get_device`, :func:`device_label`, :func:`configure_ssl`

Initial-conditions:
    :func:`load_ics`, :func:`list_cached`

Forecasting:
    :func:`run_forecast`, :func:`run_forecast_streaming`,
    :func:`save_forecast`, :func:`load_forecast`

Plotting:
    :func:`plot_field`, :data:`PLOTTABLE`

Notes
-----
Importing this package installs a ``flash_attn`` compatibility shim (see
:mod:`aifs.compat`). This is required so that ``anemoi-inference`` can be
loaded on hardware without real flash-attn support (non-Ampere GPUs, Apple
MPS, CPU). Numerical output may differ from a reference CUDA + flash-attn
run; do not use these forecasts for operational or safety-critical purposes.
"""

# Install the flash_attn shim *before* any anemoi import triggered by
# submodules below.
from aifs import compat  # noqa: F401

from aifs.device import configure_ssl, device_label, get_device
from aifs.forecast import (
    CHECKPOINTS,
    DEFAULT_CHECKPOINT,
    load_forecast,
    run_forecast,
    run_forecast_streaming,
    save_forecast,
)
from aifs.initial_conditions import list_cached, load_ics
from aifs.plot import PLOTTABLE, plot_field

__all__ = [
    # device
    "configure_ssl",
    "device_label",
    "get_device",
    # initial conditions
    "list_cached",
    "load_ics",
    # forecast
    "CHECKPOINTS",
    "DEFAULT_CHECKPOINT",
    "load_forecast",
    "run_forecast",
    "run_forecast_streaming",
    "save_forecast",
    # plotting
    "PLOTTABLE",
    "plot_field",
]
