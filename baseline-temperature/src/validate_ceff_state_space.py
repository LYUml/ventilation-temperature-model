from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data


DT_SECONDS = 3600.0
AREAL_CEFF_J_M2K = 165_000.0
FLOOR_AREA_M2 = 53.58
TOTAL_CEFF_J_K = AREAL_CEFF_J_M2K * FLOOR_AREA_M2


def disturbance_features(hours, outdoor, indoor):
    angle = 2 * np.pi * hours / 24.0
    d_out = np.r_[0.0, np.diff(outdoor)]
    d_in_previous = np.r_[0.0, 0.0, np.diff(indoor)[:-1]]
    return np.c_[np.ones(len(outdoor)), np.sin(angle), np.cos(angle), d_out, d_in_previous]


def fit_parameters(outdoor, indoor, thermal_state, hours, ridge):
    q_features = disturbance_features(hours, outdoor, indoor)[:-1]
    design = np.c_[outdoor[:-1] - indoor[:-1], thermal_state[:-1] - indoor[:-1], q_features]
    target_w = TOTAL_CEFF_J_K / DT_SECONDS * np.diff(indoor)
    penalty = np.zeros((q_features.shape[1], design.shape[1]))
    penalty[:, 2:] = math.sqrt(ridge) * np.eye(q_features.shape[1])
    augmented_x = np.r_[design, penalty]
    augmented_y = np.r_[target_w, np.zeros(q_features.shape[1])]
    lower = np.r_[[0.0, 0.0], np.full(q_features.shape[1], -np.inf)]
    upper = np.r_[[500.0, 500.0], np.full(q_features.shape[1], np.inf)]
    return lsq_linear(augmented_x, augmented_y, bounds=(lower, upper), max_iter=2000).x


def rollout(parameters, outdoor, thermal_state, hours, initial, previous_outdoor, previous_indoor):
    hout, hmass, *qcoef = parameters
    qcoef = np.asarray(qcoef)
    current = float(initial)
    previous_change = float(initial - previous_indoor)
    result = []
    for index, value in enumerate(outdoor):
        angle = 2 * math.pi * float(hours[index]) / 24.0
        d_out = float(value - (previous_outdoor if index == 0 else outdoor[index - 1]))
        phi = np.asarray([1.0, math.sin(angle), math.cos(angle), d_out, previous_change])
        heat_w = hout * (value - current) + hmass * (thermal_state[index] - current) + float(phi @ qcoef)
        updated = current + DT_SECONDS / TOTAL_CEFF_J_K * heat_w
        previous_change = updated - current
        current = updated
        result.append(current)
    return np.asarray(result)


def run_validation(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    ts, outdoor_name, floors = config["timestamp_column"], config["outdoor_column"], list(config["floor_columns"])
    frame[ts] = pd.to_datetime(frame[ts], errors="raise")
    qa = validate_data(frame, ts, [outdoor_name, *floors])
    outdoor, indoor, hours = frame[outdoor_name].to_numpy(float), frame[floors].to_numpy(float), frame[ts].dt.hour.to_numpy()
    train_end, val_end = int(len(frame) * .6), int(len(frame) * .8)
    val_origins = list(range(train_end, val_end - 24 + 1)); test_origins = list(range(val_end, len(frame) - 24 + 1))

    floor_results = {}; all_predictions = []
    for floor_index, floor in enumerate(floors):
        y = indoor[:, floor_index]; candidates = []
        for alpha in np.linspace(.55, .98, 18):
            state = causal_ewma(outdoor, float(alpha))
            for ridge in (0.0, .1, 1.0, 10.0, 100.0, 1000.0):
                parameters = fit_parameters(outdoor[:train_end], y[:train_end], state[:train_end], hours[:train_end], ridge)
                actual_values, predicted_values = [], []
                for origin in val_origins:
                    prediction = rollout(parameters, outdoor[origin:origin + 24], state[origin:origin + 24], hours[origin:origin + 24], y[origin - 1], outdoor[origin - 1], y[origin - 2])
                    actual_values.extend(y[origin:origin + 24]); predicted_values.extend(prediction)
                score = metrics(np.asarray(actual_values), np.asarray(predicted_values))["rmse_c"]
                candidates.append((score, float(alpha), float(ridge), parameters))
        _, alpha, ridge, parameters = min(candidates, key=lambda item: item[0])
        state = causal_ewma(outdoor, alpha)
        actual_values, predicted_values = [], []
        for origin in test_origins:
            prediction = rollout(parameters, outdoor[origin:origin + 24], state[origin:origin + 24], hours[origin:origin + 24], y[origin - 1], outdoor[origin - 1], y[origin - 2])
            actual_values.extend(y[origin:origin + 24]); predicted_values.extend(prediction)
            for step in range(24):
                all_predictions.append({"floor": floor, "origin": frame[ts].iloc[origin].isoformat(), "horizon_h": step + 1, "measured_c": y[origin + step], "predicted_c": prediction[step]})
        floor_results[floor] = {"metrics": metrics(np.asarray(actual_values), np.asarray(predicted_values)), "selected_alpha": alpha, "selected_ridge": ridge, "h_out_w_k": float(parameters[0]), "h_mass_w_k": float(parameters[1]), "disturbance_coefficients": parameters[2:].tolist(), "time_constant_out_h": TOTAL_CEFF_J_K / max(parameters[0], 1e-12) / 3600.0}

    # Kernel comparison on the identical locked test interval.
    inner = int(train_end * .7); best_alpha, best_error = None, float("inf")
    for alpha in np.linspace(.5, .995, 100):
        state = causal_ewma(outdoor, float(alpha)); error = np.mean([metrics(indoor[inner:train_end, j], predict_linear(state[inner:train_end], fit_linear(state[:inner], indoor[:inner, j])))["rmse_c"] for j in range(3)])
        if error < best_error: best_alpha, best_error = float(alpha), float(error)
    state = causal_ewma(outdoor, best_alpha); kernel_floor = {}
    for j, floor in enumerate(floors):
        coef = fit_linear(state[:train_end], indoor[:train_end, j]); actual_values, predicted_values = [], []
        for origin in test_origins:
            actual_values.extend(indoor[origin:origin + 24, j]); predicted_values.extend(predict_linear(state[origin:origin + 24], coef))
        kernel_floor[floor] = metrics(np.asarray(actual_values), np.asarray(predicted_values))

    model_mean = float(np.mean([v["metrics"]["rmse_c"] for v in floor_results.values()])); kernel_mean = float(np.mean([v["rmse_c"] for v in kernel_floor.values()]))
    result = {"method": "Fixed-Ceff physics-regularized NARX state-space model", "data_quality": qa, "ceff": {"areal_j_m2k": AREAL_CEFF_J_M2K, "floor_area_m2": FLOOR_AREA_M2, "total_j_k": TOTAL_CEFF_J_K}, "split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, len(frame)]}, "test_windows_per_floor": len(test_origins), "model_mean_floor_rmse_c": model_mean, "kernel_mean_floor_rmse_c": kernel_mean, "improvement_vs_kernel_pct": 100 * (kernel_mean - model_mean) / kernel_mean, "floors": floor_results, "kernel_floors": kernel_floor, "interpretation": "Ceff is fixed from the supplied areal heat capacity; H_out, H_mass and regularized equivalent heat-disturbance coefficients are calibrated."}
    output = PROJECT_ROOT / "outputs" / "ceff_state_space_validation"; output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(all_predictions).to_csv(output / "predictions.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    result = run_validation(parser.parse_args().config)
    print(f"Ceff total={result['ceff']['total_j_k']:.0f} J/K")
    print(f"model={result['model_mean_floor_rmse_c']:.3f} C kernel={result['kernel_mean_floor_rmse_c']:.3f} C improvement={result['improvement_vs_kernel_pct']:.1f}%")
    for floor, value in result["floors"].items(): print(f"{floor}: RMSE={value['metrics']['rmse_c']:.3f} Hout={value['h_out_w_k']:.1f} Hmass={value['h_mass_w_k']:.1f}")


if __name__ == "__main__": main()
