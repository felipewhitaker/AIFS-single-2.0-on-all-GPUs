import datetime
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Model configuration ───────────────────────────────────────────────────────

ModelVariant = Literal["single", "ens"]


@dataclass
class ModelConfig:
    """Describes what data a given AIFS variant needs."""

    variant: ModelVariant

    param_sfc: list[str] = field(default_factory=list)
    param_sfc_const: list[str] = field(default_factory=list)

    param_wave: list[str] = field(default_factory=list)
    param_soil: list[str] = field(default_factory=list)
    param_pl: list[str] = field(default_factory=list)

    levels: list[int] = field(default_factory=list)
    soil_levels: list[int] = field(default_factory=list)

    q_levels_drop: list[int] = field(default_factory=list)


G = 9.80665

_PARAM_WAVE = [
    "wmb",
    "h1012",
    "h1214",
    "h1417",
    "h1721",
    "h2125",
    "h2530",
    "mwd",
    "cdww",
    "mwp",
    "swh",
]
_PARAM_SOIL = ["vsw", "sot"]
_SOIL_LEVELS = [1, 2]
_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]

CONFIGS: dict[ModelVariant, ModelConfig] = {
    "single": ModelConfig(
        variant="single",
        param_sfc=[
            "10u",
            "10v",
            "2d",
            "2t",
            "msl",
            "skt",
            "sp",
            "tcw",
            "lsm",
            "z",
            "slor",
            "sdor",
            "sd",
        ],
        param_sfc_const=[],
        param_wave=_PARAM_WAVE,
        param_soil=_PARAM_SOIL,
        param_pl=["gh", "t", "u", "v", "q"],
        levels=_LEVELS,
        soil_levels=_SOIL_LEVELS,
        q_levels_drop=[10, 50],
    ),
    "ens": ModelConfig(
        variant="ens",
        param_sfc=[
            "10u",
            "10v",
            "2d",
            "2t",
            "msl",
            "skt",
            "sp",
            "tcw",
            "sd",
        ],
        param_sfc_const=["lsm", "z", "slor", "sdor"],
        param_wave=_PARAM_WAVE,
        param_soil=_PARAM_SOIL,
        param_pl=["gh", "t", "u", "v", "w", "q"],
        levels=_LEVELS,
        soil_levels=_SOIL_LEVELS,
        q_levels_drop=[10],
    ),
}

SOURCE = "ecmwf"

# ── Cache helpers ─────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path("ic_cache")


def _cache_path(
    date: datetime.datetime, variant: ModelVariant, cache_dir: Path
) -> Path:
    """Build the cache path from the variable and date"""
    return cache_dir / f"ic_{variant}_{date.strftime('%Y%m%dT%H%M%S')}.npz"


def _save(
    date: datetime.datetime, variant: ModelVariant, fields: dict, cache_dir: Path
) -> Path:
    """Save the data in cache. The path is constructed from the variable and date."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(date, variant, cache_dir)
    np.savez_compressed(str(path), **fields)
    return path


def _try_load(date: datetime.datetime, variant: ModelVariant, cache_dir: Path):
    """Try to load the data from ECMWF. Return ``(fields_dict, path)`` if cached, else ``(None, None)``."""
    path = _cache_path(date, variant, cache_dir)
    if path.exists():
        return dict(np.load(str(path))), path
    return None, None


def list_cached(cache_dir: Path = DEFAULT_CACHE_DIR) -> list[Path]:
    """Return all cached .npz files, newest first."""
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("ic_*.npz"), reverse=True)


# ── Download helpers ──────────────────────────────────────────────────────────


def _fetch_fields(ekd, ekr, date, param, levelist=None, number=None, **kwargs) -> dict:
    """
    Download ``param`` for two time-steps (t-6h, t) and return a dict
    ``{variable_name: np.ndarray shape (2, N320_nodes)}``.

    Parameters
    ----------
    number:
        Ensemble member index (1–50). ``None`` for deterministic / control
        runs or constant fields that have no perturbation.
    """
    levelist = levelist or []
    raw: dict[str, list] = defaultdict(list)

    for t in [date - datetime.timedelta(hours=6), date]:
        fetch_kwargs = dict(kwargs)
        if number is not None:
            fetch_kwargs.setdefault("stream", "enfo")
            fetch_kwargs["number"] = [number]

        dataset = ekd.from_source(
            "ecmwf-open-data",
            date=t,
            param=param,
            levelist=levelist,
            source=SOURCE,
            **fetch_kwargs,
        )
        for f in dataset:
            assert f.to_numpy().shape == (721, 1440), (
                f"Unexpected grid shape for {f.metadata('param')}: "
                f"{f.to_numpy().shape}"
            )
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})

            name = (
                f"{f.metadata('param')}_{f.metadata('levelist')}"
                if levelist
                else f.metadata("param")
            )
            raw[name].append(values)

    return {k: np.stack(v) for k, v in raw.items()}


def _build_fields(
    ekd, ekr, date: datetime.datetime, cfg: ModelConfig, number=None
) -> dict:
    """Download and transform all required fields for ``date``."""
    fields: dict = {}

    # ── Surface ───────────────────────────────────────────────────────────────
    print("  ⬇  Surface fields …")
    fields.update(
        _fetch_fields(ekd, ekr, date, cfg.param_sfc, number=number, levtype="sfc")
    )

    if cfg.param_sfc_const:
        print("  ⬇  Constant surface fields …")
        fields.update(_fetch_fields(ekd, ekr, date, cfg.param_sfc_const, levtype="sfc"))

    # ── Wave ──────────────────────────────────────────────────────────────────
    print("  ⬇  Wave fields …")
    wave_stream = "waef" if (cfg.variant == "ens" and number is not None) else "wave"
    fields.update(
        _fetch_fields(ekd, ekr, date, cfg.param_wave, number=number, stream=wave_stream)
    )

    # ── Soil ──────────────────────────────────────────────────────────────────
    print("  ⬇  Soil fields …")
    soil = _fetch_fields(
        ekd, ekr, date, cfg.param_soil, levelist=cfg.soil_levels, number=number
    )

    # ── Pressure levels ───────────────────────────────────────────────────────
    print("  ⬇  Pressure-level fields …")
    fields.update(
        _fetch_fields(ekd, ekr, date, cfg.param_pl, levelist=cfg.levels, number=number)
    )

    # ── Transformations ───────────────────────────────────────────────────────

    mwd = fields.pop("mwd")
    mwd_rad = np.deg2rad(mwd)
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    # Rename soil fields to ECMWF short-names expected by AIFS
    _soil_rename = {
        "sot_1": "stl1",
        "sot_2": "stl2",
        "vsw_1": "swvl1",
        "vsw_2": "swvl2",
    }
    for src, dst in _soil_rename.items():
        fields[dst] = soil[src]

    # Drop q levels not used by this model variant
    for lev in cfg.q_levels_drop:
        fields.pop(f"q_{lev}", None)

    # Land-sea mask: set ocean points to NaN for snow depth and soil moisture
    try:
        lsm = ekd.from_source(
            "file", f"{Path(__file__).resolve().parent}/data/lsm.grib"
        )[0].to_numpy(flatten=True)
        ocean_mask = np.equal(lsm, 0)
        for var in ("sd", "swvl1", "swvl2"):
            if var in fields:
                fields[var][:, ocean_mask] = np.nan
    except Exception:
        pass

    for level in cfg.levels:
        gh = fields.pop(f"gh_{level}", None)
        if gh is not None:
            fields[f"z_{level}"] = gh * G

    return fields


# ── Public API ────────────────────────────────────────────────────────────────


def load_ics(
    variant: ModelVariant = "single",
    number: int | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> tuple[dict, datetime.datetime]:
    """
    Return ``(fields, date)`` for the latest available ECMWF Open Data run.

    Parameters
    ----------
    variant:
        ``"single"`` for AIFS Single 2.0, ``"ens"`` for AIFS ENS 2.0.
    number:
        Ensemble member (1–50). ``None`` → control / deterministic run.
        Ignored when ``variant="single"``.
    cache_dir:
        Directory where .npz caches are stored.
    force:
        Re-download even when a local cache exists.

    Returns
    -------
    fields:
        ``{variable_name: np.ndarray shape (2, N320_nodes)}``.
        The first axis indexes the two input time-steps: ``[t-6h, t]``.
    date:
        The forecast initialisation date/time (the *later* of the two
        time-steps).
    """
    import earthkit.data as ekd
    import earthkit.regrid as ekr
    from ecmwf.opendata import Client as OpendataClient

    ekd.config.set({"cache-policy": "user"})
    cache_dir = Path(cache_dir)

    if variant not in CONFIGS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from: {list(CONFIGS)}")

    cfg = CONFIGS[variant]

    # ENS-only: normalise number
    member = number if variant == "ens" else None

    date: datetime.datetime = OpendataClient(SOURCE).latest()
    print(f"📅  Latest ECMWF run : {date}")
    print(
        f"🤖  Model variant    : {variant}" + (f"  (member {member})" if member else "")
    )

    if not force:
        cached, path = _try_load(date, variant, cache_dir)
        if cached is not None:
            sz_mb = path.stat().st_size / 1e6
            print(f"✅  Loaded from cache  ({sz_mb:.0f} MB)  →  {path}")
            return cached, date

    print("⬇️   Downloading initial conditions …")
    fields = _build_fields(ekd, ekr, date, cfg, number=member)

    path = _save(date, variant, fields, cache_dir)
    sz_mb = path.stat().st_size / 1e6
    print(f"💾  Saved to {path}  ({sz_mb:.0f} MB)")

    return fields, date
