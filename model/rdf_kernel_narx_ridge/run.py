from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common.data import ROOT, error_metrics, fit_kernel, load_config, load_inputs, origins, split_indices, windows
from ..common.rdf import extract_building


HORIZON = 24


def rdf_boundaries(building: dict[str, Any], frame: pd.DataFrame, floors: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    floor_index = {name: index for index, name in enumerate(floors)}
    series, audit = [], []
    for floor in floors:
        corridor = building["corridors"][floor]
        measured: dict[str, float] = {}
        connected: dict[int, float] = {}
        for edge in corridor["adjacent_spaces"]:
            name, ua = edge["space"], float(edge["ua_w_k"])
            if name in floor_index:
                connected[floor_index[name]] = connected.get(floor_index[name], 0.0) + ua
            elif name in frame.columns:
                measured[name] = measured.get(name, 0.0) + ua
        if not measured:
            raise ValueError(f"No measured RDF-adjacent boundary for {floor}")
        total = sum(measured.values())
        series.append(sum(frame[name].to_numpy(float) * ua for name, ua in measured.items()) / total)
        audit.append({"floor": floor, "measured_spaces": measured, "connected_floor_indices": connected,
                      "volume_m3": corridor["volume_m3"], "exterior_ua_w_k": corridor["exterior_ua_w_k"],
                      "window_area_m2": corridor["operable_window_area_m2"]})
    return np.column_stack(series), audit


def exogenous(frame: pd.DataFrame, timestamp: str, outdoor: str, boundaries: np.ndarray) -> np.ndarray:
    hour = frame[timestamp].dt.hour.to_numpy(float)
    angle = 2 * np.pi * hour / 24
    return np.column_stack([
        frame[outdoor].to_numpy(float), boundaries,
        frame["diffuse"].to_numpy(float), frame["direct"].to_numpy(float),
        frame["WIN_S_Avg_2mi"].to_numpy(float), frame["RHU"].to_numpy(float),
        frame["PRS"].to_numpy(float), frame["PRE_1h"].to_numpy(float),
        np.sin(angle), np.cos(angle),
    ])


def feature(history: list[np.ndarray], forcing: np.ndarray, floor: int, topology: list[dict[str, Any]], lag: int) -> np.ndarray:
    connected = [floor, *sorted(topology[floor]["connected_floor_indices"])]
    thermal_history = [history[-offset][index] for offset in range(lag, 0, -1) for index in connected]
    local_boundary = forcing[1 + floor]
    weather = np.r_[forcing[0], forcing[4:]]
    return np.r_[thermal_history, local_boundary, weather]


def fit_models(indoor: np.ndarray, forcing: np.ndarray, topology: list[dict[str, Any]], end: int, lag: int, alpha: float):
    models = []
    for floor in range(3):
        x = np.asarray([feature(list(indoor[:time]), forcing[time], floor, topology, lag) for time in range(lag, end)])
        models.append(make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, indoor[lag:end, floor]))
    return models


def forecast(models, indoor: np.ndarray, forcing: np.ndarray, topology: list[dict[str, Any]], origin: int, lag: int) -> np.ndarray:
    history = [row.copy() for row in indoor[origin - lag:origin]]
    result = []
    for time in range(origin, origin + HORIZON):
        value = np.asarray([models[floor].predict(feature(history, forcing[time], floor, topology, lag)[None])[0] for floor in range(3)])
        history.append(value); result.append(value)
    return np.asarray(result)


def spectral_radius(models, topology: list[dict[str, Any]], lag: int) -> float:
    size = 3 * lag
    a = np.zeros((size, size))
    if lag > 1:
        a[3:, :-3] = np.eye(size - 3)
    for floor, model in enumerate(models):
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        coefficients = ridge.coef_ / scaler.scale_
        connected = [floor, *sorted(topology[floor]["connected_floor_indices"])]
        cursor = 0
        for offset in range(lag):
            state_block = lag - 1 - offset
            for neighbour in connected:
                a[floor, state_block * 3 + neighbour] = coefficients[cursor]
                cursor += 1
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def run(config_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    frame, qa = load_inputs(config)
    timestamp, outdoor_name, floors = config["timestamp_column"], config["outdoor_column"], list(config["floor_columns"])
    building = extract_building((ROOT / config["rdf_file"]).resolve())
    boundaries, topology = rdf_boundaries(building, frame, floors)
    forcing = exogenous(frame, timestamp, outdoor_name, boundaries)
    indoor = frame[floors].to_numpy(float); outdoor = frame[outdoor_name].to_numpy(float)
    train_end, val_end = split_indices(len(frame))
    val_origins = origins(train_end, val_end); test_origins = origins(val_end, len(frame))
    val_actual = windows(indoor, val_origins)
    candidates = []
    for lag in (1, 2, 3, 6, 12, 24):
        for alpha in (.1, 1., 10., 100., 1000.):
            models = fit_models(indoor, forcing, topology, train_end, lag, alpha)
            radius = spectral_radius(models, topology, lag)
            pred = np.asarray([forecast(models, indoor, forcing, topology, origin, lag) for origin in val_origins])
            score = math.sqrt(mean_squared_error(val_actual.reshape(-1), pred.reshape(-1)))
            candidates.append((score + 10 * max(0., radius - 1.01), score, radius, lag, alpha, models))
    _, validation_rmse, radius, lag, alpha, models = min(candidates, key=lambda item: item[0])
    actual = windows(indoor, test_origins)
    prediction = np.asarray([forecast(models, indoor, forcing, topology, origin, lag) for origin in test_origins])

    kernel_series, kernel_alpha = fit_kernel(outdoor, indoor, train_end)
    kernel_prediction = windows(kernel_series, test_origins)
    residual = indoor - kernel_series
    residual_val_actual = windows(residual, val_origins)
    residual_candidates = []
    for residual_lag in (1, 2, 3, 6, 12, 24):
        for residual_alpha in (.1, 1., 10., 100., 1000.):
            residual_models = fit_models(residual, forcing, topology, train_end, residual_lag, residual_alpha)
            residual_radius = spectral_radius(residual_models, topology, residual_lag)
            residual_pred = np.asarray([forecast(residual_models, residual, forcing, topology, origin, residual_lag) for origin in val_origins])
            residual_score = math.sqrt(mean_squared_error(residual_val_actual.reshape(-1), residual_pred.reshape(-1)))
            residual_candidates.append((residual_score + 10 * max(0., residual_radius - 1.01), residual_score, residual_radius, residual_lag, residual_alpha, residual_models))
    _, residual_validation_rmse, residual_radius, residual_lag, residual_alpha, residual_models = min(residual_candidates, key=lambda item: item[0])
    residual_test = np.asarray([forecast(residual_models, residual, forcing, topology, origin, residual_lag) for origin in test_origins])
    slow_state_prediction = kernel_prediction + residual_test
    floor_results = {label: error_metrics(actual[:, :, j].reshape(-1), prediction[:, :, j].reshape(-1)) for j, label in enumerate(floors)}
    kernel_results = {label: error_metrics(actual[:, :, j].reshape(-1), kernel_prediction[:, :, j].reshape(-1)) for j, label in enumerate(floors)}
    model_mean = float(np.mean([value["rmse_c"] for value in floor_results.values()])); kernel_mean = float(np.mean([value["rmse_c"] for value in kernel_results.values()]))
    slow_floor_results = {label: error_metrics(actual[:, :, j].reshape(-1), slow_state_prediction[:, :, j].reshape(-1)) for j, label in enumerate(floors)}
    slow_mean = float(np.mean([value["rmse_c"] for value in slow_floor_results.values()]))
    result = {"method": "RDF-constrained graph companion thermal state model", "data_quality": qa,
            "rdf": {"source": building["source_rdf"], "topology": topology}, "selected": {"lag": lag, "ridge_alpha": alpha, "spectral_radius": radius, "validation_rmse_c": validation_rmse},
            "test_windows_per_floor": len(test_origins), "model_mean_floor_rmse_c": model_mean, "kernel_mean_floor_rmse_c": kernel_mean,
            "improvement_vs_kernel_pct": 100 * (kernel_mean - model_mean) / kernel_mean, "floors": floor_results, "kernel_floors": kernel_results,
            "slow_state_model": {"method": "RDF-Kernel NARX-Ridge", "selected": {"kernel_alpha": kernel_alpha, "lag": residual_lag, "ridge_alpha": residual_alpha, "spectral_radius": residual_radius, "validation_residual_rmse_c": residual_validation_rmse}, "mean_floor_rmse_c": slow_mean, "improvement_vs_kernel_pct": 100 * (kernel_mean - slow_mean) / kernel_mean, "floors": slow_floor_results}}
    output = ROOT / "results" / "rdf_kernel_narx_ridge"; output.mkdir(parents=True, exist_ok=True); (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path); result = run(parser.parse_args().config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
