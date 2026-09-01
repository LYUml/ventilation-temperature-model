from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.common.data import ROOT, error_metrics, load_config, load_inputs, origins, split_indices


HORIZON = 24


def _features(outdoor: np.ndarray, indoor: np.ndarray, hours: np.ndarray, starts: list[int]):
    x, y, keys = [], [], []
    floors = indoor.shape[1]
    for origin in starts:
        angle = 2.0 * math.pi * float(hours[origin]) / 24.0
        for floor in range(floors):
            floor_code = np.zeros(floors); floor_code[floor] = 1.0
            x.append(np.r_[outdoor[origin:origin + HORIZON], math.sin(angle), math.cos(angle), floor_code])
            y.append(indoor[origin:origin + HORIZON, floor])
            keys.append((origin, floor))
    return np.asarray(x), np.asarray(y), keys


def _fit_rc(indoor: np.ndarray, outdoor: np.ndarray, train_end: int) -> tuple[float, float]:
    delta = indoor[1:train_end] - indoor[:train_end - 1]
    gap = outdoor[:train_end - 1] - indoor[:train_end - 1]
    gain, response = np.linalg.lstsq(np.c_[np.ones(len(delta)), gap], delta, rcond=None)[0]
    if not 0.0 < response < 1.0:
        raise ValueError(f"Fitted RC response {response} is outside (0, 1)")
    return float(gain), float(response)


def _rc_windows(outdoor: np.ndarray, indoor: np.ndarray, keys, coefficients) -> np.ndarray:
    result = []
    for origin, floor in keys:
        gain, response = coefficients[floor]
        current = float(indoor[origin - 1, floor])
        trajectory = []
        for step in range(HORIZON):
            current += gain + response * (outdoor[origin + step - 1] - current)
            trajectory.append(current)
        result.append(trajectory)
    return np.asarray(result)


def run(output_dir: Path | None = None) -> dict:
    config = load_config(); frame, qa = load_inputs(config)
    floors = config["floor_columns"]
    outdoor = frame[config["outdoor_column"]].to_numpy(float)
    indoor = frame[floors].to_numpy(float)
    hours = frame[config["timestamp_column"]].dt.hour.to_numpy()
    train_end, val_end = split_indices(len(frame))
    starts = (origins(1, train_end), origins(train_end, val_end), origins(val_end, len(frame)))
    train_x, train_y, train_keys = _features(outdoor, indoor, hours, starts[0])
    val_x, val_y, val_keys = _features(outdoor, indoor, hours, starts[1])
    test_x, test_y, test_keys = _features(outdoor, indoor, hours, starts[2])

    rc = [_fit_rc(indoor[:, floor], outdoor, train_end) for floor in range(len(floors))]
    rc_train = _rc_windows(outdoor, indoor, train_keys, rc)
    rc_val = _rc_windows(outdoor, indoor, val_keys, rc)
    rc_test = _rc_windows(outdoor, indoor, test_keys, rc)
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(
            np.c_[train_x, rc_train], train_y - rc_train
        )
        prediction = rc_val + model.predict(np.c_[val_x, rc_val])
        candidates.append((float(np.sqrt(np.mean((prediction - val_y) ** 2))), alpha, model))
    validation_rmse, alpha, model = min(candidates, key=lambda item: item[0])
    prediction = rc_test + model.predict(np.c_[test_x, rc_test])

    floor_metrics = {}
    for index, floor in enumerate(floors):
        mask = np.asarray([key[1] == index for key in test_keys])
        floor_metrics[floor] = error_metrics(test_y[mask], prediction[mask])
    result = {
        "method": "RC NARX-Ridge direct multi-step calibration",
        "data_quality": qa,
        "split": [train_end, val_end, len(frame)],
        "test_windows_per_floor": len(starts[2]),
        "selected_ridge_alpha": alpha,
        "validation_rmse_c": validation_rmse,
        "mean_floor_rmse_c": float(np.mean([item["rmse_c"] for item in floor_metrics.values()])),
        "floors": floor_metrics,
        "fitted_rc": {floor: {"gain_c_per_h": rc[i][0], "response_per_h": rc[i][1]} for i, floor in enumerate(floors)},
        "caveat": "The RC coefficients are fitted effective parameters; the residual model is a direct 24 h NARX-style calibrator and reads the corridor temperature immediately before each forecast window.",
    }
    destination = output_dir or ROOT / "results" / "rc_narx_ridge"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
