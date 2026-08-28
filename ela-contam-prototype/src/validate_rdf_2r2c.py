from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .rdf_building import extract_building
from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data


DT_SECONDS = 3600.0
RHO_AIR_KG_M3 = 1.2
CP_AIR_J_KGK = 1005.0
NOMINAL_CEFF_J_M2K = 165_000.0


def _physical_parameters(raw: np.ndarray, corridor: dict[str, float]) -> dict[str, float]:
    area = float(corridor["floor_area_m2"])
    volume = float(corridor["volume_m3"])
    wall_ua = float(corridor["opaque_exterior_ua_w_k"])
    split = float(raw[1])
    return {
        "ceff_multiplier": float(raw[0]),
        "ceff_j_k": NOMINAL_CEFF_J_M2K * area * float(raw[0]),
        "wall_resistance_outdoor_fraction": split,
        "g_outdoor_mass_w_k": wall_ua / split,
        "g_mass_air_w_k": wall_ua / (1.0 - split),
        "air_capacity_multiplier": float(raw[2]),
        "air_capacity_j_k": RHO_AIR_KG_M3 * CP_AIR_J_KGK * volume * float(raw[2]),
        "window_ua_w_k": float(corridor["window_ua_w_k"]),
        "infiltration_ua_w_k": float(raw[3]),
        "q0_w": float(raw[4]),
        "qsin_w": float(raw[5]),
        "qcos_w": float(raw[6]),
        "initial_mass_offset_c": float(raw[7]),
    }


def _heat_gain(p: dict[str, float], hour: float) -> float:
    angle = 2.0 * math.pi * hour / 24.0
    return p["q0_w"] + p["qsin_w"] * math.sin(angle) + p["qcos_w"] * math.cos(angle)


def _step(ti: float, tm: float, tout: float, hour: float, p: dict[str, float]) -> tuple[float, float]:
    g_direct = p["window_ua_w_k"] + p["infiltration_ua_w_k"]
    d_ti = (g_direct * (tout - ti) + p["g_mass_air_w_k"] * (tm - ti) + _heat_gain(p, hour)) / p["air_capacity_j_k"]
    d_tm = (p["g_outdoor_mass_w_k"] * (tout - tm) + p["g_mass_air_w_k"] * (ti - tm)) / p["ceff_j_k"]
    return ti + DT_SECONDS * d_ti, tm + DT_SECONDS * d_tm


def _one_step_predictions(raw: np.ndarray, corridor: dict[str, float], outdoor: np.ndarray, indoor: np.ndarray, hours: np.ndarray) -> np.ndarray:
    p = _physical_parameters(raw, corridor)
    tm = float(indoor[0] + p["initial_mass_offset_c"])
    predictions = []
    for index in range(len(indoor) - 1):
        predicted, tm = _step(float(indoor[index]), tm, float(outdoor[index]), float(hours[index]), p)
        predictions.append(predicted)
    return np.asarray(predictions)


def _mass_state_at(raw: np.ndarray, corridor: dict[str, float], outdoor: np.ndarray, indoor: np.ndarray, hours: np.ndarray, origin: int) -> float:
    p = _physical_parameters(raw, corridor)
    tm = float(indoor[0] + p["initial_mass_offset_c"])
    for index in range(origin - 1):
        _, tm = _step(float(indoor[index]), tm, float(outdoor[index]), float(hours[index]), p)
    return tm


def _forecast(raw: np.ndarray, corridor: dict[str, float], outdoor: np.ndarray, indoor: np.ndarray, hours: np.ndarray, origin: int, horizon: int) -> np.ndarray:
    p = _physical_parameters(raw, corridor)
    ti = float(indoor[origin - 1])
    tm = _mass_state_at(raw, corridor, outdoor, indoor, hours, origin)
    result = []
    for index in range(origin, origin + horizon):
        ti, tm = _step(ti, tm, float(outdoor[index]), float(hours[index]), p)
        result.append(ti)
    return np.asarray(result)


def _fit(corridor: dict[str, float], outdoor: np.ndarray, indoor: np.ndarray, hours: np.ndarray) -> np.ndarray:
    upper_infiltration = max(10.0, float(corridor["exterior_ua_w_k"]))
    lower = np.asarray([110 / 165, 0.15, 1.0, 0.0, -500.0, -500.0, -500.0, -3.0])
    upper = np.asarray([260 / 165, 0.85, 20.0, upper_infiltration, 500.0, 500.0, 500.0, 3.0])
    starts = [
        np.asarray([1.0, split, air_mult, 0.1 * upper_infiltration, 0.0, 0.0, 0.0, 0.0])
        for split in (0.3, 0.5, 0.7)
        for air_mult in (2.0, 8.0)
    ]

    def residual(raw: np.ndarray) -> np.ndarray:
        prediction = _one_step_predictions(raw, corridor, outdoor, indoor, hours)
        data_residual = prediction - indoor[1:]
        # Weak priors prevent unmeasured infiltration and heat gains from replacing the RDF envelope.
        prior = np.asarray([raw[3] / upper_infiltration, raw[4] / 500.0, raw[5] / 500.0, raw[6] / 500.0]) * 0.05
        return np.r_[data_residual, prior]

    fits = [least_squares(residual, start, bounds=(lower, upper), max_nfev=2500) for start in starts]
    return min(fits, key=lambda fit: float(np.sum(fit.fun**2))).x


def _boundary_diagnostics(raw: np.ndarray) -> list[str]:
    names = ["ceff_multiplier", "wall_resistance_outdoor_fraction", "air_capacity_multiplier", "infiltration_ua_w_k"]
    lower = np.asarray([110 / 165, 0.15, 1.0, 0.0])
    upper = np.asarray([260 / 165, 0.85, 20.0, np.inf])
    messages = []
    for index, name in enumerate(names):
        if abs(raw[index] - lower[index]) <= 1e-4 * max(1.0, abs(lower[index])):
            messages.append(f"{name} reached lower bound")
        if np.isfinite(upper[index]) and abs(raw[index] - upper[index]) <= 1e-4 * max(1.0, abs(upper[index])):
            messages.append(f"{name} reached upper bound")
    return messages


def run_validation(config_path: Path, rdf_path: Path | None = None) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    timestamp, outdoor_name = config["timestamp_column"], config["outdoor_column"]
    floors = list(config["floor_columns"])
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    qa = validate_data(frame, timestamp, [outdoor_name, *floors])
    outdoor = frame[outdoor_name].to_numpy(float)
    indoor = frame[floors].to_numpy(float)
    hours = frame[timestamp].dt.hour.to_numpy(float)
    building = extract_building(rdf_path or PROJECT_ROOT.parent / "data" / "NBuilding.rdf")
    train_end, validation_end = int(len(frame) * 0.6), int(len(frame) * 0.8)
    test_origins = list(range(validation_end, len(frame) - 24 + 1))
    floor_results: dict[str, Any] = {}
    prediction_rows = []

    for floor_index, floor in enumerate(floors):
        y = indoor[:, floor_index]
        raw = _fit(building["corridors"][floor], outdoor[:train_end], y[:train_end], hours[:train_end])
        actual, predicted = [], []
        for origin in test_origins:
            forecast = _forecast(raw, building["corridors"][floor], outdoor, y, hours, origin, 24)
            actual.extend(y[origin:origin + 24]); predicted.extend(forecast)
            for step, value in enumerate(forecast):
                prediction_rows.append({"floor": floor, "origin": frame[timestamp].iloc[origin].isoformat(), "horizon_h": step + 1, "measured_c": y[origin + step], "predicted_c": value})
        physical = _physical_parameters(raw, building["corridors"][floor])
        floor_results[floor] = {"metrics": metrics(np.asarray(actual), np.asarray(predicted)), "parameters": physical, "boundary_diagnostics": _boundary_diagnostics(raw), "rdf": building["corridors"][floor]}

    inner = int(train_end * 0.7); best_alpha, best_error = 0.0, float("inf")
    for alpha in np.linspace(0.5, 0.995, 100):
        state = causal_ewma(outdoor, float(alpha))
        error = np.mean([metrics(indoor[inner:train_end, j], predict_linear(state[inner:train_end], fit_linear(state[:inner], indoor[:inner, j])))["rmse_c"] for j in range(len(floors))])
        if error < best_error: best_alpha, best_error = float(alpha), float(error)
    state = causal_ewma(outdoor, best_alpha); kernel = {}
    for index, floor in enumerate(floors):
        coefficient = fit_linear(state[:train_end], indoor[:train_end, index]); actual, predicted = [], []
        for origin in test_origins:
            actual.extend(indoor[origin:origin + 24, index]); predicted.extend(predict_linear(state[origin:origin + 24], coefficient))
        kernel[floor] = metrics(np.asarray(actual), np.asarray(predicted))

    model_mean = float(np.mean([item["metrics"]["rmse_c"] for item in floor_results.values()]))
    kernel_mean = float(np.mean([item["rmse_c"] for item in kernel.values()]))
    result = {"method": "RDF-constrained 2R2C", "data_quality": qa, "split": {"train": [0, train_end], "validation": [train_end, validation_end], "test": [validation_end, len(frame)]}, "test_windows_per_floor": len(test_origins), "model_mean_floor_rmse_c": model_mean, "kernel_mean_floor_rmse_c": kernel_mean, "improvement_vs_kernel_pct": 100.0 * (kernel_mean - model_mean) / kernel_mean, "floors": floor_results, "kernel_floors": kernel}
    output = PROJECT_ROOT / "outputs" / "rdf_2r2c_validation"; output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(output / "predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--rdf", type=Path)
    args = parser.parse_args()
    result = run_validation(args.config, args.rdf)
    print(f"2R2C={result['model_mean_floor_rmse_c']:.3f} C kernel={result['kernel_mean_floor_rmse_c']:.3f} C improvement={result['improvement_vs_kernel_pct']:.1f}%")
    for floor, item in result["floors"].items(): print(f"{floor}: RMSE={item['metrics']['rmse_c']:.3f} Ceff multiplier={item['parameters']['ceff_multiplier']:.3f}")


if __name__ == "__main__":
    main()
