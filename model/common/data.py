from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or ROOT / "config.json").read_text(encoding="utf-8"))


def validate_hourly(frame: pd.DataFrame, timestamp: str, columns: list[str]) -> dict[str, Any]:
    if frame[timestamp].duplicated().any():
        raise ValueError("Duplicate timestamps")
    if frame[columns].isna().any().any():
        raise ValueError("Missing required values")
    gaps = frame[timestamp].diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError("Input is not a continuous hourly series")
    return {"rows": len(frame), "start": frame[timestamp].iloc[0].isoformat(), "end": frame[timestamp].iloc[-1].isoformat()}


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    temperature = pd.read_csv(ROOT / config["temperature_csv"])
    weather = pd.read_csv(ROOT / config["weather_csv"])
    timestamp = config["timestamp_column"]
    temperature[timestamp] = pd.to_datetime(temperature[timestamp], errors="raise")
    weather["_timestamp"] = pd.to_datetime(dict(year=weather.year, month=weather.mon, day=weather.day, hour=weather.hour), errors="raise")
    if weather["_timestamp"].duplicated().any():
        raise ValueError("Weather data contain duplicate timestamps")
    merged = temperature.merge(weather, how="left", left_on=timestamp, right_on="_timestamp", validate="one_to_one").drop(columns="_timestamp")
    required = [config["outdoor_column"], *config["floor_columns"], "TEM", "PRS", "RHU", "PRE_1h", "WIN_S_Avg_2mi", "WIN_D_Avg_2mi", "diffuse", "direct"]
    qa = validate_hourly(merged, timestamp, required)
    temperature_rmse = float(np.sqrt(np.mean((merged["TEM"] - merged[config["outdoor_column"]]) ** 2)))
    if temperature_rmse > 1e-9:
        raise ValueError("Station TEM does not match the supplied OUTDOOR series")
    qa["weather_temperature_rmse_c"] = temperature_rmse
    return merged, qa


def split_indices(length: int) -> tuple[int, int]:
    return int(length * 0.60), int(length * 0.80)


def origins(start: int, end: int, horizon: int = 24) -> list[int]:
    return list(range(start, end - horizon + 1))


def windows(values: np.ndarray, starts: list[int], horizon: int = 24) -> np.ndarray:
    return np.asarray([values[start:start + horizon] for start in starts])


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    return {"rmse_c": float(np.sqrt(np.mean(residual ** 2))), "mae_c": float(np.mean(np.abs(residual))), "mean_error_c": float(np.mean(residual))}


def ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=float); result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * result[index - 1] + (1.0 - alpha) * values[index]
    return result


def fit_kernel(outdoor: np.ndarray, indoor: np.ndarray, train_end: int) -> tuple[np.ndarray, float]:
    inner = int(train_end * 0.70); best = (float("inf"), 0.5)
    for alpha in np.linspace(0.5, 0.995, 100):
        state = ewma(outdoor, float(alpha)); errors = []
        for floor in range(indoor.shape[1]):
            coefficient = np.linalg.lstsq(np.c_[np.ones(inner), state[:inner]], indoor[:inner, floor], rcond=None)[0]
            prediction = coefficient[0] + coefficient[1] * state[inner:train_end]
            errors.append(error_metrics(indoor[inner:train_end, floor], prediction)["rmse_c"])
        if np.mean(errors) < best[0]: best = (float(np.mean(errors)), float(alpha))
    state = ewma(outdoor, best[1]); prediction = []
    for floor in range(indoor.shape[1]):
        coefficient = np.linalg.lstsq(np.c_[np.ones(train_end), state[:train_end]], indoor[:train_end, floor], rcond=None)[0]
        prediction.append(coefficient[0] + coefficient[1] * state)
    return np.column_stack(prediction), best[1]
