import os
import datetime
from typing import Generator

import numpy as np
import pandas as pd
import xarray as xr

# Apply flash-attn shim before any anemoi import
import aifs.compat  # noqa: F401
from aifs.device import get_device, device_label

DEFAULT_CHECKPOINT = "aifs-single-2.0"

CHECKPOINTS = {
    DEFAULT_CHECKPOINT: {"huggingface": f"ecmwf/{DEFAULT_CHECKPOINT}"},
    "aifs-ens-2.0": {"huggingface": "ecmwf/aifs-ens-2.0"},
    "aifs-single-1.1": {"huggingface": "ecmwf/aifs-single-1.1"},
    "aifs-ens-1.0": {"huggingface": "ecmwf/aifs-ens-1.0"},
}


def run_forecast(
    fields: dict,
    date: datetime.datetime,
    lead_time: int = 24,
    num_chunks: int = 16,
    checkpoint: str = DEFAULT_CHECKPOINT,
    verbose: bool = True,
):
    """
    Run an AIFS forecast and return all output states.

    Parameters
    ----------
    fields:
        Initial-condition field dict as returned by ``load_ics()``.
        Shape of each array: ``(2, N320_nodes)``.
    date:
        Forecast initialisation datetime.
    lead_time:
        Forecast horizon in hours. Must be a multiple of 6.
    num_chunks:
        Number of chunks for the attention computation.
        Increase if you run out of memory; decrease for speed.
        16 is a safe default for 16 GB RAM / VRAM.
    checkpoint:
        Key into ``CHECKPOINTS`` dict, or a raw ``{"huggingface": "..."}``
        dict you can pass directly.
    verbose:
        Print step-by-step progress.

    Returns
    -------
    list of state dicts, one per 6-hour output step.
    Each state dict has at minimum:
        ``state["date"]``   — output datetime
        ``state["fields"]`` — ``{variable: np.ndarray}``
        ``state["latitudes"]``
        ``state["longitude"]``
    """
    if lead_time % 6 != 0:
        raise ValueError(f"lead_time must be a multiple of 6, got {lead_time}")

    from anemoi.inference.runners.simple import SimpleRunner

    device = get_device()

    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)

    if verbose:
        print(
            f"🖥️   Device  : {device_label()} \n\n Checkpoint: {checkpoint} \n\n Lead time : {lead_time} h  ({lead_time // 6} steps)"
        )

    ckpt = CHECKPOINTS.get(checkpoint, checkpoint)

    if verbose:
        print("🤖  Loading model …")

    runner = SimpleRunner(ckpt)

    if verbose:
        print("🌍  Running inference …")

    states: list[dict] = []
    input_state = {"fields": fields, "date": date}
    for state in runner.run(input_states=input_state, lead_time=lead_time):
        states.append(
            {
                "date": state["date"],
                "fields": {k: v.copy() for k, v in state["fields"].items()},
                "latitudes": state["latitudes"],
                "longitudes": state["longitudes"],
            }
        )
        if verbose:
            print(f"    ✓  {state['date']}")

    if verbose:
        print(f"✅  Done — {len(states)} steps produced.")

    return states


def run_forecast_streaming(
    fields: dict,
    date: datetime.datetime,
    lead_time: int = 24,
    num_chunks: int = 16,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Generator[dict, None, None]:
    """
    Generator variant of :func:`run_forecast`.

    Yields each state dict as soon as it is computed, which is useful for
    Gradio apps or notebooks that want to display results incrementally.

    Example
    -------
        for state in run_forecast_streaming(fields, date, lead_time=48):
            plot_field(state)
    """
    if lead_time % 6 != 0:
        raise ValueError(f"lead_time must be a multiple of 6, got {lead_time}")

    from anemoi.inference.runners.simple import SimpleRunner

    device = get_device()
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)

    ckpt = CHECKPOINTS.get(checkpoint, checkpoint)
    runner = SimpleRunner(ckpt)

    input_state = {"fields": fields, "date": date}
    for state in runner.run(input_states=input_state, lead_time=lead_time):
        yield {
            "date": state["date"],
            "fields": {k: v.copy() for k, v in state["fields"].items()},
        }


def save_forecast(states: list[dict], path: str = "../forecasts/output.nc"):
    """
    Save the results of the forecasts as .nc file.

    :param states: list of state dicts, one per 6-hour output step.
                   Each state dict has at minimum:
                        ``state["date"]``   — output datetime
                        ``state["fields"]`` — ``{variable: np.ndarray}``
                        ``state["latitudes"]``
                        ``state["longitudes"]``
    :param path:   path to save the states.
    """
    dates = [s["date"] for s in states]
    lat = states[0]["latitudes"]
    lon = states[0]["longitudes"]

    var_names = states[0]["fields"].keys()
    data_vars = {}

    for var in var_names:
        arr = np.stack([s["fields"][var] for s in states], axis=0)
        data_vars[var] = (["date", "values"], arr)

    ds = xr.Dataset(
        data_vars,
        coords={
            "date": dates,
            "latitudes": ("values", lat),
            "longitudes": ("values", lon),
        },
    )
    ds.to_netcdf(path)


def load_forecast(path: str):
    """
    Load the results of the forecasts, saved as either a .nc or .npz file.

    :param path:   path of the .nc or .npz file to load.
    :return:       list of state dicts, one per 6-hour output step.
                   Each state dict has at minimum:
                        ``state["date"]``   — output datetime
                        ``state["fields"]`` — ``{variable: np.ndarray}``
                        ``state["latitudes"]``
                        ``state["longitudes"]``
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".nc":
        return _load_forecast_nc(path)
    elif ext == ".npz":
        return _load_forecast_npz(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext!r} (expected .nc or .npz)")


def _load_forecast_nc(path: str):
    ds = xr.open_dataset(path)
    lat = ds["latitudes"].values
    lon = ds["longitudes"].values

    states = []
    for i, t in enumerate(ds["date"].values):
        date = pd.Timestamp(t).to_pydatetime()

        fields = {var: ds[var].isel(date=i).values for var in ds.keys()}

        states.append(
            {
                "date": date,
                "fields": fields,
                "latitudes": lat,
                "longitudes": lon,
            }
        )

    ds.close()
    return states


def _load_forecast_npz(path: str):
    data = np.load(path, allow_pickle=True)

    lat = data["latitudes"]
    lon = data["longitudes"]
    dates_raw = data["date"]

    if dates_raw.ndim == 0:
        dates_raw = dates_raw.item()
    dates_raw = np.atleast_1d(dates_raw)

    # Only field arrays are stored with the "field_" prefix in run_forecast.py.
    # Everything else (latitudes, longitudes, date) is metadata.
    var_keys = [k for k in data.files if k.startswith("field_")]

    states = []
    for i, t in enumerate(dates_raw):
        date = pd.Timestamp(str(t)).to_pydatetime()
        states.append(
            {
                "date": date,
                "fields": {k.replace("field_", "", 1): data[k] for k in var_keys},
                "latitudes": lat,
                "longitudes": lon,
            }
        )

    return states
