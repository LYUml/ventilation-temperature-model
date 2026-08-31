from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .benchmark_open_innovation import STATION_LATITUDE, STATION_LONGITUDE, _facade_irradiance
from .rdf_building import extract_building
from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data
from .weather_data import merge_hourly_weather


HORIZON = 24
AIR_LAGS = 12
AIR_STATE_SIZE = 3 * AIR_LAGS
MASS_START = AIR_STATE_SIZE
SHARED_INDEX = MASS_START + 3
OUTDOOR_STATE_INDEX = SHARED_INDEX + 1


def _states(indoor: np.ndarray, outdoor: np.ndarray, alpha_mass: float, alpha_shared: float) -> np.ndarray:
    air_history = []
    for time in range(len(indoor)):
        blocks = [indoor[max(0, time - lag)] for lag in range(AIR_LAGS)]
        air_history.append(np.concatenate(blocks))
    mass = np.column_stack([causal_ewma(indoor[:, floor], alpha_mass) for floor in range(3)])
    shared = causal_ewma(np.mean(indoor, axis=1), alpha_shared)[:, None]
    outdoor_mass = causal_ewma(outdoor, alpha_shared)[:, None]
    return np.column_stack([np.asarray(air_history), mass, shared, outdoor_mass])


def _rdf_context(building: dict[str, Any], frame: pd.DataFrame, floors: list[str]) -> tuple[dict[str, Any], np.ndarray]:
    floor_index = {name: index for index, name in enumerate(floors)}
    context: dict[str, Any] = {"floors": []}
    room_boundaries = []
    for floor in floors:
        corridor = building["corridors"][floor]
        ua_by_space: dict[str, float] = {}
        vertical_by_floor: dict[int, float] = {}
        for edge in corridor["adjacent_spaces"]:
            neighbour = edge["space"]
            ua = float(edge["ua_w_k"])
            if neighbour in floor_index:
                vertical_by_floor[floor_index[neighbour]] = vertical_by_floor.get(floor_index[neighbour], 0.0) + ua
            elif neighbour in frame.columns:
                ua_by_space[neighbour] = ua_by_space.get(neighbour, 0.0) + ua
        if not ua_by_space:
            raise ValueError(f"RDF has no measured adjacent temperature boundary for {floor}")
        total_room_ua = float(sum(ua_by_space.values()))
        boundary = sum(frame[column].to_numpy(float) * ua for column, ua in ua_by_space.items()) / total_room_ua
        room_boundaries.append(boundary)
        context["floors"].append({
            "name": floor,
            "volume_m3": float(corridor["volume_m3"]),
            "exterior_ua_w_k": float(corridor["exterior_ua_w_k"]),
            "window_area_m2": float(corridor["operable_window_area_m2"]),
            "measured_adjacent_spaces": ua_by_space,
            "measured_adjacent_ua_w_k": total_room_ua,
            "vertical_corridor_ua_w_k": vertical_by_floor,
        })
    return context, np.column_stack(room_boundaries)


def _inputs(frame: pd.DataFrame, timestamp: str, outdoor_name: str, floors: list[str], building: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    context, rooms = _rdf_context(building, frame, floors)
    facade = _facade_irradiance(frame, timestamp) / 500.0
    hour = frame[timestamp].dt.hour.to_numpy(float)
    angle = 2.0 * np.pi * hour / 24.0
    forcing = np.column_stack([
        frame[outdoor_name].to_numpy(float), rooms, facade,
        frame["WIN_S_Avg_2mi"].to_numpy(float), frame["RHU"].to_numpy(float) / 100.0,
        np.sin(angle), np.cos(angle), np.ones(len(frame)),
    ])
    return forcing, context


def _features(state: np.ndarray, forcing: np.ndarray, row: int, rdf_context: dict[str, Any]) -> np.ndarray:
    if row < 3:
        floor = row
        rdf_floor = rdf_context["floors"][floor]
        connected_floors = [floor, *sorted(rdf_floor["vertical_corridor_ua_w_k"])]
        history_indices = [lag * 3 + connected for lag in range(AIR_LAGS) for connected in connected_floors]
        current = state[floor]
        exterior_drive = rdf_floor["exterior_ua_w_k"] / rdf_floor["volume_m3"] * (forcing[0] - current)
        room_drive = rdf_floor["measured_adjacent_ua_w_k"] / rdf_floor["volume_m3"] * (forcing[1 + floor] - current)
        vertical_drive = sum(ua / rdf_floor["volume_m3"] * (state[neighbour] - current) for neighbour, ua in rdf_floor["vertical_corridor_ua_w_k"].items())
        solar_drive = rdf_floor["window_area_m2"] / rdf_floor["volume_m3"] * forcing[4:8]
        return np.r_[state[history_indices], state[MASS_START + floor], state[SHARED_INDEX], state[OUTDOOR_STATE_INDEX], exterior_drive, room_drive, vertical_drive, solar_drive, forcing[8:]]
    if row < 6:
        floor = row - 3
        return np.r_[state[MASS_START + floor], state[floor], state[SHARED_INDEX], forcing[0], 1.0]
    if row == 6:
        return np.r_[state[SHARED_INDEX], np.mean(state[:3]), np.mean(state[MASS_START:MASS_START + 3]), state[OUTDOOR_STATE_INDEX], forcing[0], 1.0]
    return np.r_[state[OUTDOOR_STATE_INDEX], forcing[0], 1.0]


def _fit_models(states: np.ndarray, forcing: np.ndarray, train_end: int, alpha: float, rdf_context: dict[str, Any]) -> list[Any]:
    models = []
    target_indices = [0, 1, 2, MASS_START, MASS_START + 1, MASS_START + 2, SHARED_INDEX, OUTDOOR_STATE_INDEX]
    for row, target_index in enumerate(target_indices):
        x = np.asarray([_features(states[t], forcing[t + 1], row, rdf_context) for t in range(train_end - 1)])
        y = states[1:train_end, target_index]
        models.append(make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y))
    return models


def _transition(models: list[Any], state: np.ndarray, forcing: np.ndarray, rdf_context: dict[str, Any]) -> np.ndarray:
    predicted = np.asarray([models[row].predict(_features(state, forcing, row, rdf_context)[None, :])[0] for row in range(8)])
    new_air = predicted[:3]
    shifted_air = np.r_[new_air, state[:AIR_STATE_SIZE - 3]]
    return np.r_[shifted_air, predicted[3:6], predicted[6], predicted[7]]


def _forecast(models: list[Any], initial: np.ndarray, forcing: np.ndarray, origin: int, rdf_context: dict[str, Any]) -> np.ndarray:
    state = initial.copy(); result = []
    for time in range(origin, origin + HORIZON):
        state = _transition(models, state, forcing[time], rdf_context)
        result.append(state[:3].copy())
    return np.asarray(result)


def _window(values: np.ndarray, origins: list[int]) -> np.ndarray:
    return np.asarray([values[origin:origin + HORIZON] for origin in origins])


def _spectral_diagnostic(models: list[Any], reference_state: np.ndarray, reference_input: np.ndarray, rdf_context: dict[str, Any]) -> float:
    base = _transition(models, reference_state, reference_input, rdf_context)
    jacobian = np.zeros((len(reference_state), len(reference_state)))
    epsilon = 1e-4
    for column in range(len(reference_state)):
        shifted = reference_state.copy(); shifted[column] += epsilon
        value = _transition(models, shifted, reference_input, rdf_context)
        jacobian[:, column] = (value - base) / epsilon
    return float(np.max(np.abs(np.linalg.eigvals(jacobian))))


def run_validation(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    timestamp, outdoor_name, floors = config["timestamp_column"], config["outdoor_column"], list(config["floor_columns"])
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    qa = validate_data(frame, timestamp, [outdoor_name, *floors])
    frame, weather_qa = merge_hourly_weather(frame, timestamp, (PROJECT_ROOT / config["weather_csv"]).resolve())
    indoor = frame[floors].to_numpy(float); outdoor = frame[outdoor_name].to_numpy(float)
    rdf_path = (PROJECT_ROOT / config["rdf_file"]).resolve()
    building = extract_building(rdf_path)
    forcing, rdf_context = _inputs(frame, timestamp, outdoor_name, floors, building)
    train_end, validation_end = int(len(frame) * 0.6), int(len(frame) * 0.8)
    validation_origins = list(range(train_end, validation_end - HORIZON + 1))
    test_origins = list(range(validation_end, len(frame) - HORIZON + 1))
    validation_actual = _window(indoor, validation_origins)

    candidates = []
    for alpha_mass in (0.8, 0.9, 0.98):
        for alpha_shared in (0.95, 0.99, 0.995):
            states = _states(indoor, outdoor, alpha_mass, alpha_shared)
            for ridge_alpha in (0.1, 10.0, 1000.0):
                models = _fit_models(states, forcing, train_end, ridge_alpha, rdf_context)
                prediction = np.asarray([_forecast(models, states[origin - 1], forcing, origin, rdf_context) for origin in validation_origins])
                score = math.sqrt(mean_squared_error(validation_actual.reshape(-1), prediction.reshape(-1)))
                radius = _spectral_diagnostic(models, states[train_end - 1], forcing[train_end], rdf_context)
                penalty = max(0.0, radius - 1.01) * 10.0
                candidates.append((score + penalty, score, radius, alpha_mass, alpha_shared, ridge_alpha, models, states))
    _, validation_rmse, spectral_radius, alpha_mass, alpha_shared, ridge_alpha, models, states = min(candidates, key=lambda item: item[0])
    predicted = np.asarray([_forecast(models, states[origin - 1], forcing, origin, rdf_context) for origin in test_origins])
    actual = _window(indoor, test_origins)

    inner = int(train_end * 0.7); best_kernel_alpha, best_error = 0.5, float("inf")
    for alpha in np.linspace(0.5, 0.995, 100):
        kernel_state = causal_ewma(outdoor, float(alpha))
        error = np.mean([metrics(indoor[inner:train_end, j], predict_linear(kernel_state[inner:train_end], fit_linear(kernel_state[:inner], indoor[:inner, j])))["rmse_c"] for j in range(3)])
        if error < best_error: best_kernel_alpha, best_error = float(alpha), float(error)
    kernel_state = causal_ewma(outdoor, best_kernel_alpha)
    kernel_series = np.column_stack([predict_linear(kernel_state, fit_linear(kernel_state[:train_end], indoor[:train_end, j])) for j in range(3)])
    kernel_prediction = _window(kernel_series, test_origins)

    floor_results = {}
    kernel_results = {}
    for floor, label in enumerate(floors):
        floor_results[label] = metrics(actual[:, :, floor].reshape(-1), predicted[:, :, floor].reshape(-1))
        kernel_results[label] = metrics(actual[:, :, floor].reshape(-1), kernel_prediction[:, :, floor].reshape(-1))
    model_mean = float(np.mean([value["rmse_c"] for value in floor_results.values()]))
    kernel_mean = float(np.mean([value["rmse_c"] for value in kernel_results.values()]))
    result = {
        "method": "RDF-constrained Multi-scale Spatial Thermal State Model (RDF-MSTS)",
        "equation": "x[t+1] = A_topology x[t] + B_weather u[t+1]; T_corridor[t] = C x[t]",
        "state_definition": [*[f"corridor_air_lag_{lag + 1}_{floor}" for lag in range(AIR_LAGS) for floor in ("2f", "3f", "4f")], "floor_mass_2f", "floor_mass_3f", "floor_mass_4f", "shared_building_mass", "outdoor_slow_state"],
        "data_quality": qa, "weather_data_quality": weather_qa,
        "rdf": {"source": building["source_rdf"], "space_count": building["space_count"], "interface_count": building["interface_count"], "floor_constraints": rdf_context["floors"]},
        "split": {"train": [0, train_end], "validation": [train_end, validation_end], "test": [validation_end, len(frame)]},
        "test_windows_per_floor": len(test_origins), "validation_rmse_c": validation_rmse,
        "selected": {"floor_mass_alpha": alpha_mass, "shared_mass_alpha": alpha_shared, "ridge_alpha": ridge_alpha, "transition_spectral_radius": spectral_radius, "station_latitude": STATION_LATITUDE, "station_longitude": STATION_LONGITUDE},
        "model_mean_floor_rmse_c": model_mean, "kernel_mean_floor_rmse_c": kernel_mean,
        "improvement_vs_kernel_pct": 100.0 * (kernel_mean - model_mean) / kernel_mean,
        "floors": floor_results, "kernel_floors": kernel_results,
        "unification": "One RDF-masked companion-form latent thermal transition system produces all three temperatures. Measured room boundaries, vertical floor links, volume, envelope UA and window area are derived from the RDF.",
        "caveats": ["Filtered mass states are operational latent-state definitions, not directly measured wall temperatures.", "Only 421 hourly observations are available.", "Window initial states use measurements available immediately before the forecast origin."],
    }
    output = PROJECT_ROOT / "outputs" / "rdf_msts_validation"; output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    result = run_validation(parser.parse_args().config)
    print(f"RDF-MSTS={result['model_mean_floor_rmse_c']:.3f} C kernel={result['kernel_mean_floor_rmse_c']:.3f} C improvement={result['improvement_vs_kernel_pct']:.1f}%")
    print(json.dumps(result["selected"], ensure_ascii=False, indent=2))
    for floor, value in result["floors"].items(): print(f"{floor}: {value['rmse_c']:.3f} C")


if __name__ == "__main__":
    main()
