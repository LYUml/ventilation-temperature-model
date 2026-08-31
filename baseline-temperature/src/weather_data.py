from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_COLUMNS = ["PRS", "RHU", "PRE_1h", "WIN_S_Avg_2mi", "wind_east", "wind_north", "diffuse", "direct"]


def merge_hourly_weather(frame: pd.DataFrame, timestamp_column: str, weather_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Join station 543990 data to building observations with strict hourly QA."""
    weather = pd.read_csv(weather_csv)
    required = {"year", "mon", "day", "hour", "TEM", "PRS", "RHU", "PRE_1h", "WIN_S_Avg_2mi", "WIN_D_Avg_2mi", "diffuse", "direct"}
    missing = sorted(required.difference(weather.columns))
    if missing:
        raise ValueError(f"Weather file is missing columns: {missing}")
    weather["_timestamp"] = pd.to_datetime(
        dict(year=weather.year, month=weather.mon, day=weather.day, hour=weather.hour), errors="raise"
    )
    if weather["_timestamp"].duplicated().any():
        raise ValueError("Weather file contains duplicate hourly timestamps")
    direction = np.deg2rad(weather["WIN_D_Avg_2mi"].to_numpy(float))
    speed = weather["WIN_S_Avg_2mi"].to_numpy(float)
    weather["wind_east"] = speed * np.sin(direction)
    weather["wind_north"] = speed * np.cos(direction)
    source = weather[["_timestamp", "TEM", *WEATHER_COLUMNS]].copy()
    merged = frame.merge(source, how="left", left_on=timestamp_column, right_on="_timestamp", validate="one_to_one")
    if merged[WEATHER_COLUMNS].isna().any().any():
        absent = merged.loc[merged[WEATHER_COLUMNS].isna().any(axis=1), timestamp_column]
        raise ValueError(f"Weather data do not cover {len(absent)} building timestamps")
    temperature_rmse = float(np.sqrt(np.mean((merged["TEM"].to_numpy(float) - merged["OUTDOOR"].to_numpy(float)) ** 2)))
    qa = {
        "source": str(weather_csv),
        "matched_rows": int(len(merged)),
        "station_vs_building_outdoor_rmse_c": temperature_rmse,
        "features": WEATHER_COLUMNS,
    }
    return merged.drop(columns=["_timestamp"]), qa


def weather_windows(weather: np.ndarray, keys: list[tuple[int, int]], horizon: int = 24) -> np.ndarray:
    """Create compact future forcing trajectories; omit noisy pressure/rain/direction channels."""
    speed_index = WEATHER_COLUMNS.index("WIN_S_Avg_2mi")
    humidity_index = WEATHER_COLUMNS.index("RHU")
    diffuse_index = WEATHER_COLUMNS.index("diffuse")
    direct_index = WEATHER_COLUMNS.index("direct")
    rows = []
    for origin, _ in keys:
        block = weather[origin : origin + horizon]
        total_solar = block[:, diffuse_index] + block[:, direct_index]
        rows.append(np.r_[total_solar, block[:, speed_index], block[:, humidity_index]])
    return np.asarray(rows, dtype=np.float32)
