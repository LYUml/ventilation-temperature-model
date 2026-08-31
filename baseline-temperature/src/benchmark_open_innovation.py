from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pvlib
from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data
from .weather_data import merge_hourly_weather


ROOM_COLUMNS = {
    "2FCORRIDOR": ["2F215"],
    "3FCORRIDOR": ["3F308", "3F309", "3F310"],
    "4FCORRIDOR": ["4F408", "4F409", "4F410", "4F411", "5F510"],
}
HORIZON = 24
STATION_LATITUDE = 39.9869
STATION_LONGITUDE = 116.2908


def _origins(start: int, end: int) -> list[int]:
    return list(range(start, end - HORIZON + 1))


def _kernel(outdoor: np.ndarray, indoor: np.ndarray, train_end: int) -> tuple[np.ndarray, float]:
    inner = int(train_end * 0.7)
    best_alpha, best_error = 0.5, float("inf")
    for alpha in np.linspace(0.5, 0.995, 100):
        state = causal_ewma(outdoor, float(alpha))
        error = np.mean([
            metrics(indoor[inner:train_end, j], predict_linear(state[inner:train_end], fit_linear(state[:inner], indoor[:inner, j])))["rmse_c"]
            for j in range(indoor.shape[1])
        ])
        if error < best_error:
            best_alpha, best_error = float(alpha), float(error)
    state = causal_ewma(outdoor, best_alpha)
    prediction = np.column_stack([predict_linear(state, fit_linear(state[:train_end], indoor[:train_end, j])) for j in range(indoor.shape[1])])
    return prediction, best_alpha


def _window_cube(values: np.ndarray, origins: list[int]) -> np.ndarray:
    return np.asarray([values[o:o + HORIZON] for o in origins])


def _varx_rows(target: np.ndarray, inputs: np.ndarray, end: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for t in range(lag, end):
        x.append(np.r_[target[t - lag:t].reshape(-1), inputs[t]])
        y.append(target[t])
    return np.asarray(x), np.asarray(y)


def _varx_forecast(model: Ridge, scaler: StandardScaler, target: np.ndarray, inputs: np.ndarray, origin: int, lag: int) -> np.ndarray:
    history = [row.copy() for row in target[origin - lag:origin]]
    result = []
    for t in range(origin, origin + HORIZON):
        x = np.r_[np.asarray(history[-lag:]).reshape(-1), inputs[t]][None, :]
        value = model.predict(scaler.transform(x))[0]
        history.append(value); result.append(value)
    return np.asarray(result)


def _fit_varx(train_target: np.ndarray, inputs: np.ndarray, train_end: int, val_origins: list[int]) -> tuple[int, float, Ridge, StandardScaler, np.ndarray]:
    candidates = []
    for lag in (1, 2, 3, 6, 12, 24):
        x, y = _varx_rows(train_target, inputs, train_end, lag)
        scaler = StandardScaler().fit(x)
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
            pred = np.asarray([_varx_forecast(model, scaler, train_target, inputs, o, lag) for o in val_origins])
            actual = _window_cube(train_target, val_origins)
            candidates.append((math.sqrt(mean_squared_error(actual.reshape(-1), pred.reshape(-1))), lag, alpha, model, scaler, pred))
    _, lag, alpha, model, scaler, pred = min(candidates, key=lambda item: item[0])
    return lag, alpha, model, scaler, pred


def _n4sid_forecasts(train_frame: pd.DataFrame, full_inputs: np.ndarray, full_outputs: np.ndarray, origins: list[int], rank: int) -> np.ndarray:
    input_names = [column for column in train_frame.columns if column.startswith("u")]
    output_names = [column for column in train_frame.columns if column.startswith("y")]
    identifier = NFourSID(train_frame, output_columns=output_names, input_columns=input_names, num_block_rows=8)
    identifier.subspace_identification()
    state_space, covariance = identifier.system_identification(rank=rank)
    result = []
    for origin in origins:
        kalman = Kalman(state_space, covariance)
        for t in range(origin):
            kalman.step(full_outputs[t, :, None], full_inputs[t, :, None])
        window = []
        for t in range(origin, origin + HORIZON):
            filtered, _ = kalman.step(None, full_inputs[t, :, None])
            window.append(filtered[:, 0])
        result.append(window)
    return np.asarray(result)


def _simplex_weights(actual: np.ndarray, components: list[np.ndarray], upper_bounds: list[float] | None = None) -> np.ndarray:
    matrix = np.stack([item.reshape(-1) for item in components], axis=1)
    target = actual.reshape(-1)
    objective = lambda w: float(np.mean((matrix @ w - target) ** 2) + 0.002 * np.sum((w - 1 / len(w)) ** 2))
    upper_bounds = upper_bounds or [1.0] * len(components)
    fit = minimize(objective, np.full(len(components), 1 / len(components)), bounds=[(0.0, bound) for bound in upper_bounds], constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    return fit.x


def _facade_irradiance(frame: pd.DataFrame, timestamp: str) -> np.ndarray:
    times = pd.DatetimeIndex(frame[timestamp]).tz_localize("Asia/Shanghai")
    position = pvlib.solarposition.get_solarposition(times, STATION_LATITUDE, STATION_LONGITUDE)
    zenith = position["apparent_zenith"].to_numpy(float)
    cos_zenith = np.maximum(np.cos(np.deg2rad(zenith)), 0.065)
    dhi = frame["diffuse"].to_numpy(float)
    beam_horizontal = frame["direct"].to_numpy(float)
    dni = np.where(zenith < 90.0, beam_horizontal / cos_zenith, 0.0)
    ghi = dhi + beam_horizontal
    facades = []
    for azimuth in (0.0, 90.0, 180.0, 270.0):
        poa = pvlib.irradiance.get_total_irradiance(
            90.0, azimuth, zenith, position["azimuth"].to_numpy(float), dni, ghi, dhi,
            model="isotropic",
        )["poa_global"]
        facades.append(np.nan_to_num(np.asarray(poa, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    return np.column_stack(facades)


def run_benchmark(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    ts, outdoor_name, floors = config["timestamp_column"], config["outdoor_column"], list(config["floor_columns"])
    frame[ts] = pd.to_datetime(frame[ts], errors="raise")
    qa = validate_data(frame, ts, [outdoor_name, *floors])
    frame, weather_qa = merge_hourly_weather(frame, ts, (PROJECT_ROOT / config["weather_csv"]).resolve())
    indoor = frame[floors].to_numpy(float); outdoor = frame[outdoor_name].to_numpy(float)
    rooms = np.column_stack([frame[ROOM_COLUMNS[floor]].mean(axis=1).to_numpy(float) for floor in floors])
    hour = frame[ts].dt.hour.to_numpy(float); angle = 2 * np.pi * hour / 24
    solar = (frame["direct"] + frame["diffuse"]).to_numpy(float)
    facade_solar = _facade_irradiance(frame, ts)
    inputs = np.column_stack([outdoor, rooms, solar / 500.0, facade_solar / 500.0, frame["WIN_S_Avg_2mi"].to_numpy(float), np.sin(angle), np.cos(angle)])
    train_end, val_end = int(len(frame) * 0.6), int(len(frame) * 0.8)
    val_origins, test_origins = _origins(train_end, val_end), _origins(val_end, len(frame))
    actual_val, actual_test = _window_cube(indoor, val_origins), _window_cube(indoor, test_origins)

    kernel, kernel_alpha = _kernel(outdoor, indoor, train_end)
    kernel_val, kernel_test = _window_cube(kernel, val_origins), _window_cube(kernel, test_origins)

    # Open baseline 1: regularized multi-output ARX/ARMAX approximation.
    arx_lag, arx_alpha, arx_model, arx_scaler, arx_val = _fit_varx(indoor, inputs, train_end, val_origins)
    arx_test = np.asarray([_varx_forecast(arx_model, arx_scaler, indoor, inputs, o, arx_lag) for o in test_origins])

    # Open baseline 2: N4SID MIMO latent state model, with rank selected only on validation.
    input_scaler, output_scaler = StandardScaler().fit(inputs[:train_end]), StandardScaler().fit(indoor[:train_end])
    scaled_inputs, scaled_outputs = input_scaler.transform(inputs), output_scaler.transform(indoor)
    train_df = pd.DataFrame(np.c_[scaled_inputs[:train_end], scaled_outputs[:train_end]], columns=[*[f"u{i}" for i in range(inputs.shape[1])], *[f"y{i}" for i in range(3)]])
    n4_candidates = []
    for rank in (1, 2, 3, 4, 5, 6):
        pred = output_scaler.inverse_transform(_n4sid_forecasts(train_df, scaled_inputs, scaled_outputs, val_origins, rank).reshape(-1, 3)).reshape(len(val_origins), HORIZON, 3)
        n4_candidates.append((math.sqrt(mean_squared_error(actual_val.reshape(-1), pred.reshape(-1))), rank, pred))
    _, n4_rank, n4_val = min(n4_candidates, key=lambda item: item[0])
    n4_test = output_scaler.inverse_transform(_n4sid_forecasts(train_df, scaled_inputs, scaled_outputs, test_origins, n4_rank).reshape(-1, 3)).reshape(len(test_origins), HORIZON, 3)

    # Original: kernel slow manifold + shared multi-zone residual dynamics + physical exogenous forcing.
    residual = indoor - kernel
    res_lag, res_alpha, res_model, res_scaler, residual_val = _fit_varx(residual, inputs, train_end, val_origins)
    residual_test = np.asarray([_varx_forecast(res_model, res_scaler, residual, inputs, o, res_lag) for o in test_origins])
    innovation_val, innovation_test = kernel_val + residual_val, kernel_test + residual_test

    # Organic fusion: floor-specific validation simplex. Kernel acts as safety anchor.
    fused_test = np.empty_like(actual_test); weights = {}
    for floor, label in enumerate(floors):
        # The unconstrained subspace model is deliberately capped: with 252 training hours,
        # it is useful as a latent-state correction but not reliable enough to dominate a floor.
        w = _simplex_weights(actual_val[:, :, floor], [kernel_val[:, :, floor], arx_val[:, :, floor], innovation_val[:, :, floor], n4_val[:, :, floor]], [1.0, 1.0, 1.0, 0.30])
        fused_test[:, :, floor] = w[0] * kernel_test[:, :, floor] + w[1] * arx_test[:, :, floor] + w[2] * innovation_test[:, :, floor] + w[3] * n4_test[:, :, floor]
        weights[label] = {"kernel": float(w[0]), "arx_boundary": float(w[1]), "thermal_innovation": float(w[2]), "n4sid": float(w[3])}

    predictions = {"kernel": kernel_test, "arx_weather_rooms": arx_test, "mimo_n4sid": n4_test, "thermal_innovation": innovation_test, "organic_fusion": fused_test}
    results = {}
    for name, prediction in predictions.items():
        floor_metrics = {label: metrics(actual_test[:, :, j].reshape(-1), prediction[:, :, j].reshape(-1)) for j, label in enumerate(floors)}
        results[name] = {"mean_floor_rmse_c": float(np.mean([value["rmse_c"] for value in floor_metrics.values()])), "floors": floor_metrics}
    ranking = sorted(results, key=lambda name: results[name]["mean_floor_rmse_c"])
    result = {
        "method": "Open-source baselines and original Kernel Latent Thermal Innovation fusion",
        "data_quality": qa, "weather_data_quality": weather_qa,
        "split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, len(frame)]},
        "test_windows_per_floor": len(test_origins), "ranking": ranking, "models": results,
        "selected": {"kernel_alpha": kernel_alpha, "arx_lag": arx_lag, "arx_ridge_alpha": arx_alpha, "n4sid_rank": n4_rank, "innovation_lag": res_lag, "innovation_ridge_alpha": res_alpha, "fusion_weights": weights, "pvlib_facade_azimuths_deg": [0, 90, 180, 270]},
        "originality": "Kernel estimates the slow outdoor-driven manifold; boundary ARX represents observable room/weather forcing; a shared multi-floor VARX evolves deviations; N4SID supplies an independently identified latent thermal state; validation-only simplex weights anchor uncertain floors to Kernel.",
        "caveats": ["Only 421 hourly observations are available.", "The 24-hour windows overlap.", "All structure and weights are selected on validation; the final test is not used for tuning."],
    }
    output = PROJECT_ROOT / "outputs" / "open_innovation_benchmark"; output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"rank": i + 1, "model": name, "mean_floor_rmse_c": results[name]["mean_floor_rmse_c"]} for i, name in enumerate(ranking)]).to_csv(output / "ranking.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    result = run_benchmark(parser.parse_args().config)
    for i, name in enumerate(result["ranking"], 1):
        print(f"{i}. {name}: {result['models'][name]['mean_floor_rmse_c']:.3f} C")
    print(json.dumps(result["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
