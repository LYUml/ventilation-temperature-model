from __future__ import annotations

import itertools
import json
import math

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.common.data import ROOT, error_metrics, fit_kernel, load_config, load_inputs, origins, split_indices, windows
from model.common.rdf import extract_building
from model.rdf_kernel_narx_ridge.run import rdf_boundaries


HORIZON = 24
LAGS = (1, 2, 3, 6, 12, 24)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _rollout_1r1c(parameters, initial, outdoor_window):
    gain, response = parameters
    current = float(initial); result = []
    for outside in outdoor_window:
        current += gain + response * (outside - current)
        result.append(current)
    return np.asarray(result)


def _rollout_2r2c(parameters, initial, outdoor_window):
    ke, km, ka, gain = parameters
    air = mass = float(initial); result = []
    for outside in outdoor_window:
        new_air = air + gain + ke * (outside - air) + km * (mass - air)
        mass += ka * (air - mass); air = new_air; result.append(air)
    return np.asarray(result)


def _fit_rc(indoor, outdoor, train_end, kind):
    if kind == "1r1c_one_step":
        delta = indoor[1:train_end] - indoor[:train_end - 1]
        gap = outdoor[:train_end - 1] - indoor[:train_end - 1]
        gain, response = np.linalg.lstsq(np.c_[np.ones(len(delta)), gap], delta, rcond=None)[0]
        return np.asarray([gain, np.clip(response, 1e-5, 0.3)])
    train_origins = origins(24, train_end)
    if kind == "1r1c_rollout":
        initial = _fit_rc(indoor, outdoor, train_end, "1r1c_one_step")
        residual = lambda p: np.concatenate([_rollout_1r1c(p, indoor[o - 1], outdoor[o - 1:o + HORIZON - 1]) - indoor[o:o + HORIZON] for o in train_origins])
        return least_squares(residual, initial, bounds=([-0.2, 1e-5], [0.2, 0.3]), max_nfev=500).x
    initial = np.asarray([0.01, 0.03, 0.005, 0.02])
    residual = lambda p: np.concatenate([_rollout_2r2c(p, indoor[o - 1], outdoor[o - 1:o + HORIZON - 1]) - indoor[o:o + HORIZON] for o in train_origins])
    return least_squares(residual, initial, bounds=([1e-5, 1e-5, 1e-5, -0.2], [0.3, 0.5, 0.3, 0.2]), max_nfev=800).x


def _rc_windows(outdoor, indoor, keys, parameters, kind):
    rollout = _rollout_2r2c if kind == "2r2c_rollout" else _rollout_1r1c
    return np.asarray([rollout(parameters[f], indoor[o - 1, f], outdoor[o - 1:o + HORIZON - 1]) for o, f in keys])


def _one_step_residuals(outdoor, indoor, parameters, kind):
    result = np.zeros_like(indoor)
    rollout = _rollout_2r2c if kind == "2r2c_rollout" else _rollout_1r1c
    for floor in range(indoor.shape[1]):
        for time in range(1, len(indoor)):
            result[time, floor] = indoor[time, floor] - rollout(parameters[floor], indoor[time - 1, floor], outdoor[time - 1:time])[0]
    return result


def _make_features(outdoor, indoor, hours, boundaries, topology, rc_residual, keys, mode):
    rows = []
    for origin, floor in keys:
        code = np.zeros(indoor.shape[1]); code[floor] = 1.0
        angle = 2 * np.pi * hours[origin] / 24
        row = list(np.r_[outdoor[origin:origin + HORIZON], np.sin(angle), np.cos(angle), code])
        if mode in ("residual_lags", "residual_lags_rdf"):
            row.extend(rc_residual[origin - np.asarray(LAGS), floor])
        if mode in ("rdf_lags", "residual_lags_rdf"):
            connected = sorted(topology[floor]["connected_floor_indices"])
            neighbour = indoor[:, connected].mean(axis=1) if connected else indoor[:, floor]
            row.extend(boundaries[origin - np.asarray(LAGS), floor])
            row.extend(neighbour[origin - np.asarray(LAGS)])
        rows.append(row)
    return np.asarray(rows)


def _fit_residual_models(train_x, train_y, val_x, val_y, train_keys, val_keys, strategy):
    if strategy == "shared":
        candidates = []
        for alpha in ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)
            score = float(np.sqrt(np.mean((model.predict(val_x) - val_y) ** 2)))
            candidates.append((score, alpha, [model]))
        return min(candidates, key=lambda item: item[0])
    models, alphas, predictions = [], [], np.empty_like(val_y)
    for floor in range(3):
        train_mask = np.asarray([key[1] == floor for key in train_keys])
        val_mask = np.asarray([key[1] == floor for key in val_keys])
        candidates = []
        for alpha in ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x[train_mask], train_y[train_mask])
            score = float(np.sqrt(np.mean((model.predict(val_x[val_mask]) - val_y[val_mask]) ** 2)))
            candidates.append((score, alpha, model))
        _, alpha, model = min(candidates, key=lambda item: item[0])
        models.append(model); alphas.append(alpha); predictions[val_mask] = model.predict(val_x[val_mask])
    return float(np.sqrt(np.mean((predictions - val_y) ** 2))), alphas, models


def _predict(models, x, keys, strategy):
    if strategy == "shared":
        return models[0].predict(x)
    result = np.empty((len(x), HORIZON))
    for floor, model in enumerate(models):
        mask = np.asarray([key[1] == floor for key in keys]); result[mask] = model.predict(x[mask])
    return result


def run() -> dict:
    config = load_config(); frame, qa = load_inputs(config)
    floors = list(config["floor_columns"])
    outdoor = frame[config["outdoor_column"]].to_numpy(float)
    indoor = frame[floors].to_numpy(float)
    hours = frame[config["timestamp_column"]].dt.hour.to_numpy()
    building = extract_building(ROOT / config["rdf_file"])
    boundaries, topology = rdf_boundaries(building, frame, floors)
    train_end, val_end = split_indices(len(frame))
    split_origins = (origins(24, train_end), origins(train_end, val_end), origins(val_end, len(frame)))
    key_sets = [[(origin, floor) for origin in origin_set for floor in range(3)] for origin_set in split_origins]
    actual_sets = [np.asarray([indoor[o:o + HORIZON, f] for o, f in keys]) for keys in key_sets]
    experiments = []
    for rc_kind, strategy, feature_mode in itertools.product(
        ("1r1c_one_step", "1r1c_rollout", "2r2c_rollout"),
        ("shared", "per_floor"),
        ("base", "residual_lags", "rdf_lags", "residual_lags_rdf"),
    ):
        parameters = [_fit_rc(indoor[:, floor], outdoor, train_end, rc_kind) for floor in range(3)]
        rc_sets = [_rc_windows(outdoor, indoor, keys, parameters, rc_kind) for keys in key_sets]
        residual_series = _one_step_residuals(outdoor, indoor, parameters, rc_kind)
        x_sets = [
            np.c_[_make_features(outdoor, indoor, hours, boundaries, topology, residual_series, keys, feature_mode), rc]
            for keys, rc in zip(key_sets, rc_sets)
        ]
        residual_targets = [actual - rc for actual, rc in zip(actual_sets, rc_sets)]
        validation_rmse, alpha, models = _fit_residual_models(x_sets[0], residual_targets[0], x_sets[1], residual_targets[1], key_sets[0], key_sets[1], strategy)
        test_prediction = rc_sets[2] + _predict(models, x_sets[2], key_sets[2], strategy)
        floor_metrics = {}
        for floor, label in enumerate(floors):
            mask = np.asarray([key[1] == floor for key in key_sets[2]])
            floor_metrics[label] = error_metrics(actual_sets[2][mask], test_prediction[mask])
        experiments.append({
            "name": f"{rc_kind}__{strategy}__{feature_mode}",
            "rc_kind": rc_kind, "residual_strategy": strategy, "features": feature_mode,
            "selected_alpha": alpha, "validation_rmse_c": validation_rmse,
            "test_mean_floor_rmse_c": float(np.mean([value["rmse_c"] for value in floor_metrics.values()])),
            "floors": floor_metrics,
            "rc_parameters": {label: parameters[index].tolist() for index, label in enumerate(floors)},
        })
    selected = min(experiments, key=lambda item: item["validation_rmse_c"])
    oracle = min(experiments, key=lambda item: item["test_mean_floor_rmse_c"])
    kernel_series, kernel_alpha = fit_kernel(outdoor, indoor, train_end)
    kernel_actual = windows(indoor, split_origins[2]); kernel_pred = windows(kernel_series, split_origins[2])
    kernel_floors = {label: error_metrics(kernel_actual[:, :, i], kernel_pred[:, :, i]) for i, label in enumerate(floors)}
    result = {
        "method": "RC NARX-Ridge combinatorial ablation",
        "data_quality": qa,
        "split": [train_end, val_end, len(frame)],
        "selection_rule": "Select exactly one configuration by aggregate validation residual RMSE; test metrics are reported only after selection.",
        "kernel": {"alpha": kernel_alpha, "mean_floor_rmse_c": float(np.mean([v["rmse_c"] for v in kernel_floors.values()])), "floors": kernel_floors},
        "selected_by_validation": selected,
        "test_oracle_for_audit_only": oracle,
        "experiments": sorted(experiments, key=lambda item: item["validation_rmse_c"]),
        "caveats": [
            "Only 421 hourly observations from one building are available.",
            "Sliding 24 h windows overlap and are not independent samples.",
            "Future outdoor temperature is treated as an available weather forecast.",
            "The test oracle is diagnostic only and must not be presented as the selected model.",
        ],
    }
    destination = ROOT / "results" / "rc_narx_ridge_optimization"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
