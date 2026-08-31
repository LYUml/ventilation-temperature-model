from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temperature_model import (
    PROJECT_ROOT,
    causal_ewma,
    fit_linear,
    metrics,
    predict_linear,
    validate_data,
)
from .temperature_rc_model import fit_first_order_rc


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def extended_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = metrics(actual, predicted)
    result["r2"] = r2_score(actual, predicted)
    return result


def select_kernel_alpha(
    outdoor: np.ndarray,
    actual: np.ndarray,
    inner_split: int,
    train_end: int,
    candidates: np.ndarray,
) -> float:
    best_alpha = None
    best_rmse = float("inf")
    for alpha in candidates:
        baseline = causal_ewma(outdoor, float(alpha))
        floor_errors = []
        for floor_index in range(actual.shape[1]):
            coefficients = fit_linear(
                baseline[:inner_split], actual[:inner_split, floor_index]
            )
            prediction = predict_linear(
                baseline[inner_split:train_end], coefficients
            )
            floor_errors.append(
                metrics(actual[inner_split:train_end, floor_index], prediction)[
                    "rmse_c"
                ]
            )
        score = float(np.mean(floor_errors))
        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("no kernel smoothing parameter selected")
    return best_alpha


def run_24h_validation(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    input_path = (PROJECT_ROOT / config["input_csv"]).resolve()
    output_dir = PROJECT_ROOT / config.get("output_root", "outputs") / "baseline_24h_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = config["timestamp_column"]
    outdoor_name = config["outdoor_column"]
    floors = list(config["floor_columns"])
    frame = pd.read_csv(input_path)
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    qa = validate_data(frame, timestamp, [outdoor_name, *floors])

    outdoor = frame[outdoor_name].to_numpy(dtype=float)
    actual = frame[floors].to_numpy(dtype=float)
    train_end = int(len(frame) * float(config["train_fraction"]))
    inner_split = int(train_end * float(config.get("inner_train_fraction", 0.7)))
    horizon = 24

    candidates = np.linspace(
        float(config.get("alpha_min", 0.5)),
        float(config.get("alpha_max", 0.995)),
        int(config.get("alpha_steps", 100)),
    )
    alpha = select_kernel_alpha(outdoor, actual, inner_split, train_end, candidates)
    kernel_state = causal_ewma(outdoor, alpha)

    origins = list(range(train_end, len(frame) - horizon + 1))
    if not origins:
        raise ValueError("test period is shorter than 24 hours")

    result: dict[str, Any] = {
        "question": "24-hour corridor baseline forecast: kernel moving average versus first-order RC",
        "data_quality": qa,
        "train_rows": train_end,
        "test_rows": len(frame) - train_end,
        "horizon_hours": horizon,
        "rolling_origins": len(origins),
        "selected_kernel_alpha": alpha,
        "initialization": (
            "Each RC forecast reads the measured corridor temperature immediately before "
            "the 24-hour window once; it does not read corridor measurements inside the window."
        ),
        "floors": {},
    }

    rows: list[dict[str, Any]] = []
    for floor_index, floor in enumerate(floors):
        indoor = actual[:, floor_index]
        kernel_coefficients = fit_linear(
            kernel_state[:train_end], indoor[:train_end]
        )
        heat_gain, response = fit_first_order_rc(indoor, outdoor, train_end)

        measured_all = []
        kernel_all = []
        rc_all = []
        persistence_all = []
        for origin in origins:
            current = float(indoor[origin - 1])
            initial = current
            for step in range(horizon):
                index = origin + step
                kernel_prediction = float(
                    predict_linear(
                        np.asarray([kernel_state[index]]), kernel_coefficients
                    )[0]
                )
                current = (
                    current
                    + heat_gain
                    + response * (float(outdoor[index - 1]) - current)
                )
                measured = float(indoor[index])
                measured_all.append(measured)
                kernel_all.append(kernel_prediction)
                rc_all.append(current)
                persistence_all.append(initial)
                rows.append(
                    {
                        "floor": floor,
                        "origin": frame[timestamp].iloc[origin].isoformat(),
                        "horizon_h": step + 1,
                        "timestamp": frame[timestamp].iloc[index].isoformat(),
                        "measured_c": measured,
                        "kernel_c": kernel_prediction,
                        "rc_c": current,
                        "persistence_c": initial,
                    }
                )

        measured_array = np.asarray(measured_all)
        kernel_array = np.asarray(kernel_all)
        rc_array = np.asarray(rc_all)
        persistence_array = np.asarray(persistence_all)
        horizon_24 = np.arange(horizon - 1, len(measured_array), horizon)
        kernel_metrics = extended_metrics(measured_array, kernel_array)
        rc_metrics = extended_metrics(measured_array, rc_array)
        result["floors"][floor] = {
            "kernel_coefficients": {
                "intercept_c": kernel_coefficients[0],
                "slope": kernel_coefficients[1],
            },
            "rc_coefficients": {
                "heat_gain_c_per_h": heat_gain,
                "response_per_h": response,
            },
            "all_horizons": {
                "kernel": kernel_metrics,
                "rc": rc_metrics,
                "persistence": extended_metrics(measured_array, persistence_array),
                "rc_rmse_improvement_vs_kernel_pct": float(
                    100.0
                    * (kernel_metrics["rmse_c"] - rc_metrics["rmse_c"])
                    / kernel_metrics["rmse_c"]
                ),
            },
            "at_24_hours": {
                "kernel": extended_metrics(
                    measured_array[horizon_24], kernel_array[horizon_24]
                ),
                "rc": extended_metrics(measured_array[horizon_24], rc_array[horizon_24]),
                "persistence": extended_metrics(
                    measured_array[horizon_24], persistence_array[horizon_24]
                ),
            },
        }

    prediction_path = output_dir / "rolling_24h_predictions.csv"
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    metrics_path = output_dir / "rolling_24h_metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 24-hour baseline forecasts")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_24h_validation(args.config)
    print(f"rolling origins: {result['rolling_origins']}")
    print(f"kernel alpha: {result['selected_kernel_alpha']:.3f}")
    for floor, values in result["floors"].items():
        kernel = values["all_horizons"]["kernel"]["rmse_c"]
        rc = values["all_horizons"]["rc"]["rmse_c"]
        change = values["all_horizons"]["rc_rmse_improvement_vs_kernel_pct"]
        print(f"{floor}: kernel={kernel:.3f}, RC={rc:.3f} °C, improvement={change:.1f}%")


if __name__ == "__main__":
    main()
