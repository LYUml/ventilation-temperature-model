"""Reusable RDF + weather-file interface for indoor-temperature simulation.

Canonical weather CSV columns are::

    timestamp,outdoor_temperature_c,diffuse_solar_w_m2,direct_solar_w_m2

For compatibility with this project's source data, ``year/mon/day/hour``,
``TEM``, ``diffuse`` and ``direct`` are normalized to the canonical schema.
The thermal calculation itself remains in :mod:`model.mz5r1c`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model.mz5r1c import RdfMz5r1cModel


REQUIRED_WEATHER_COLUMNS = (
    "timestamp",
    "outdoor_temperature_c",
    "diffuse_solar_w_m2",
    "direct_solar_w_m2",
)


def _load_weather(weatherPath: str | Path) -> pd.DataFrame:
    """Read, normalize, and validate one hourly weather CSV."""
    path = Path(weatherPath)
    if not path.is_file():
        raise FileNotFoundError(f"Weather CSV does not exist: {path}")
    weather = pd.read_csv(path)

    if set(REQUIRED_WEATHER_COLUMNS).issubset(weather.columns):
        normalized = weather.loc[:, REQUIRED_WEATHER_COLUMNS].copy()
    elif {"year", "mon", "day", "hour", "TEM", "diffuse", "direct"}.issubset(
        weather.columns
    ):
        normalized = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    {
                        "year": weather["year"],
                        "month": weather["mon"],
                        "day": weather["day"],
                        "hour": weather["hour"],
                    },
                    errors="raise",
                ),
                "outdoor_temperature_c": weather["TEM"],
                "diffuse_solar_w_m2": weather["diffuse"],
                "direct_solar_w_m2": weather["direct"],
            }
        )
    else:
        raise ValueError(
            "Weather CSV must contain canonical columns "
            f"{list(REQUIRED_WEATHER_COLUMNS)}, or project columns "
            "year/mon/day/hour/TEM/diffuse/direct"
        )

    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], errors="raise"
    )
    numeric_columns = list(REQUIRED_WEATHER_COLUMNS[1:])
    try:
        normalized[numeric_columns] = normalized[numeric_columns].apply(
            pd.to_numeric, errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Weather temperature and radiation must be numeric") from error
    if not len(normalized):
        raise ValueError("Weather CSV cannot be empty")
    if normalized["timestamp"].duplicated().any():
        raise ValueError("Weather timestamps must be unique")
    if not normalized["timestamp"].is_monotonic_increasing:
        raise ValueError("Weather timestamps must be increasing")
    gaps = normalized["timestamp"].diff().dropna()
    if len(gaps) and not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError("Weather must be a continuous hourly series")
    values = normalized[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Weather contains missing or non-finite values")
    if (normalized[["diffuse_solar_w_m2", "direct_solar_w_m2"]] < 0).any().any():
        raise ValueError("Solar radiation cannot be negative")
    return normalized


def simulateIndoorTemperature(
    rdfPath: str | Path,
    weatherPath: str | Path,
    targetSpace: str | None = None,
) -> dict[str, list[float]]:
    """Simulate hourly temperature for all usable RDF spaces by default.

    Pass ``targetSpace`` to restrict the result to one space. An explicitly
    selected target is treated as a free-running baseline space: its people,
    lighting, equipment gains, and ideal HVAC are disabled. With the default
    ``None``, every usable RDF space is returned under its RDF conditions.
    The bundled ``data/weather_54399.csv`` and canonical schema are accepted.
    """
    if targetSpace is not None and (
        not isinstance(targetSpace, str) or not targetSpace.strip()
    ):
        raise ValueError("targetSpace must be a non-empty RDF space name")
    weather = _load_weather(weatherPath).rename(
        columns={
            "timestamp": "Timestamp",
            "outdoor_temperature_c": "Ta",
            "diffuse_solar_w_m2": "diffuse",
            "direct_solar_w_m2": "direct",
        }
    )
    model = RdfMz5r1cModel(
        outdoor_column="Ta",
        parameters={"allow_non_project_assumptions": True},
        baseline_spaces=[targetSpace] if targetSpace is not None else None,
    )
    targets = [targetSpace] if targetSpace is not None else None
    return model.simulate(rdfPath, weather, targets)


__all__ = [
    "REQUIRED_WEATHER_COLUMNS",
    "simulateIndoorTemperature",
]
