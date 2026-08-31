from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .temperature_model import PROJECT_ROOT, validate_data


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "rmse_c": float(math.sqrt(float(np.mean(error**2)))),
        "mae_c": float(np.mean(np.abs(error))),
        "mean_error_c": float(np.mean(error)),
    }


def fit_first_order_rc(
    indoor: np.ndarray, outdoor: np.ndarray, train_end: int
) -> tuple[float, float]:
    """Fit dT = heat_gain + response*(Tout-Tin) on training transitions."""
    delta = indoor[1:train_end] - indoor[: train_end - 1]
    temperature_gap = outdoor[: train_end - 1] - indoor[: train_end - 1]
    design = np.column_stack([np.ones(len(delta)), temperature_gap])
    heat_gain_c_per_h, response_per_h = np.linalg.lstsq(design, delta, rcond=None)[0]
    response_per_h = float(response_per_h)
    if not 0.0 < response_per_h < 1.0:
        raise ValueError(
            f"fitted thermal response {response_per_h} is outside physical range (0, 1)"
        )
    return float(heat_gain_c_per_h), response_per_h


def fit_rc_with_neighbors(
    indoor: np.ndarray, outdoor: np.ndarray, neighbors: np.ndarray, train_end: int
) -> tuple[float, float, float]:
    """Fit a physical RC model with measured adjacent rooms as boundary conditions."""
    delta = indoor[1:train_end] - indoor[: train_end - 1]
    gaps = np.column_stack(
        [
            outdoor[: train_end - 1] - indoor[: train_end - 1],
            neighbors[: train_end - 1] - indoor[: train_end - 1],
        ]
    )
    design = np.column_stack([np.ones(len(delta)), gaps])
    heat_gain, outdoor_response, neighbor_response = np.linalg.lstsq(design, delta, rcond=None)[0]
    # Heat-transfer responses must not reverse direction. Refit the intercept after projection.
    outdoor_response = float(max(0.0, outdoor_response))
    neighbor_response = float(max(0.0, neighbor_response))
    if outdoor_response + neighbor_response >= 1.0:
        scale = 0.999 / (outdoor_response + neighbor_response)
        outdoor_response *= scale
        neighbor_response *= scale
    heat_gain = float(np.mean(delta - gaps @ np.array([outdoor_response, neighbor_response])))
    return heat_gain, outdoor_response, neighbor_response


def neighbor_predictions(
    indoor: np.ndarray,
    outdoor: np.ndarray,
    neighbors: np.ndarray,
    start: int,
    coefficients: tuple[float, float, float],
    rollout: bool,
) -> np.ndarray:
    heat_gain, outdoor_response, neighbor_response = coefficients
    predictions = []
    current = float(indoor[start - 1])
    for index in range(start, len(indoor)):
        previous = current if rollout else float(indoor[index - 1])
        current = (
            previous
            + heat_gain
            + outdoor_response * (outdoor[index - 1] - previous)
            + neighbor_response * (neighbors[index - 1] - previous)
        )
        predictions.append(current)
    return np.asarray(predictions, dtype=float)


def one_step_predictions(
    indoor: np.ndarray,
    outdoor: np.ndarray,
    start: int,
    heat_gain: float,
    response: float,
) -> np.ndarray:
    previous_indoor = indoor[start - 1 : -1]
    previous_outdoor = outdoor[start - 1 : -1]
    return previous_indoor + heat_gain + response * (previous_outdoor - previous_indoor)


def rollout_predictions(
    initial_indoor: float,
    outdoor: np.ndarray,
    start: int,
    heat_gain: float,
    response: float,
) -> np.ndarray:
    predictions = []
    current = float(initial_indoor)
    for index in range(start, len(outdoor)):
        current = current + heat_gain + response * (outdoor[index - 1] - current)
        predictions.append(current)
    return np.asarray(predictions, dtype=float)


def run_rc_model(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    input_path = (PROJECT_ROOT / config["input_csv"]).resolve()
    output_dir = PROJECT_ROOT / config.get("output_root", "outputs") / config["case_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = config["timestamp_column"]
    outdoor_name = config["outdoor_column"]
    floors = list(config["floor_columns"])
    neighbor_columns = config.get("neighbor_columns", {})
    frame = pd.read_csv(input_path)
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    all_neighbors = [column for columns in neighbor_columns.values() for column in columns]
    qa = validate_data(frame, timestamp, [outdoor_name, *floors, *all_neighbors])
    split = int(len(frame) * float(config["train_fraction"]))
    outdoor = frame[outdoor_name].to_numpy(dtype=float)

    floor_results: dict[str, Any] = {}
    saved_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for floor in floors:
        indoor = frame[floor].to_numpy(dtype=float)
        heat_gain, response = fit_first_order_rc(indoor, outdoor, split)
        actual_test = indoor[split:]
        persistence = indoor[split - 1 : -1]
        one_step = one_step_predictions(indoor, outdoor, split, heat_gain, response)
        rollout = rollout_predictions(indoor[split - 1], outdoor, split, heat_gain, response)
        saved_predictions[floor] = one_step, rollout
        floor_results[floor] = {
            "heat_gain_c_per_h": heat_gain,
            "thermal_response_per_h": response,
            "time_constant_h_approx": 1.0 / response,
            "equilibrium_offset_from_outdoor_c": heat_gain / response,
            "persistence_test_metrics": error_metrics(actual_test, persistence),
            "one_step_test_metrics": error_metrics(actual_test, one_step),
            "rollout_test_metrics": error_metrics(actual_test, rollout),
        }
        columns = list(neighbor_columns.get(floor, []))
        if columns:
            neighbor = frame[columns].mean(axis=1).to_numpy(dtype=float)
            coefficients = fit_rc_with_neighbors(indoor, outdoor, neighbor, split)
            neighbor_one = neighbor_predictions(indoor, outdoor, neighbor, split, coefficients, False)
            neighbor_rollout = neighbor_predictions(indoor, outdoor, neighbor, split, coefficients, True)
            saved_predictions[floor] = one_step, rollout, neighbor_one, neighbor_rollout
            floor_results[floor]["neighbor_model"] = {
                "columns": columns,
                "aggregation": "arithmetic mean of adjacent measured rooms",
                "heat_gain_c_per_h": coefficients[0],
                "outdoor_response_per_h": coefficients[1],
                "neighbor_response_per_h": coefficients[2],
                "one_step_test_metrics": error_metrics(actual_test, neighbor_one),
                "rollout_test_metrics": error_metrics(actual_test, neighbor_rollout),
            }

    prediction_path = output_dir / "temperature_rc_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [timestamp, outdoor_name]
        for floor in floors:
            fields += [f"{floor}_actual_c", f"{floor}_one_step_c", f"{floor}_rollout_c"]
            if floor in neighbor_columns:
                fields += [f"{floor}_neighbor_one_step_c", f"{floor}_neighbor_rollout_c"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, index in enumerate(range(split, len(frame))):
            row: dict[str, Any] = {
                timestamp: frame[timestamp].iloc[index].isoformat(),
                outdoor_name: outdoor[index],
            }
            for floor in floors:
                predictions = saved_predictions[floor]
                one_step, rollout = predictions[:2]
                row[f"{floor}_actual_c"] = frame[floor].iloc[index]
                row[f"{floor}_one_step_c"] = one_step[offset]
                row[f"{floor}_rollout_c"] = rollout[offset]
                if len(predictions) == 4:
                    row[f"{floor}_neighbor_one_step_c"] = predictions[2][offset]
                    row[f"{floor}_neighbor_rollout_c"] = predictions[3][offset]
            writer.writerow(row)

    figure, axes = plt.subplots(len(floors), 1, figsize=(12, 9), sharex=True)
    test_times = frame[timestamp].iloc[split:]
    for axis, floor in zip(axes, floors):
        predictions = saved_predictions[floor]
        one_step, rollout = predictions[:2]
        axis.plot(test_times, frame[floor].iloc[split:], label="Measured", linewidth=1.5)
        axis.plot(test_times, one_step, label="One-step RC", linewidth=1.1)
        axis.plot(test_times, rollout, label="Continuous RC rollout", linewidth=1.1)
        if len(predictions) == 4:
            axis.plot(test_times, predictions[2], label="RC + adjacent rooms", linewidth=1.1)
        rmse = floor_results[floor]["one_step_test_metrics"]["rmse_c"]
        axis.set_title(f"{floor}: one-step RC RMSE={rmse:.3f} °C")
        axis.set_ylabel("Temperature (°C)")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Time")
    figure.suptitle("First-order thermal resistance-capacitance model: test period")
    figure.tight_layout()
    figure.savefig(output_dir / "temperature_rc_comparison.png", dpi=160)
    plt.close(figure)

    result = {
        "method": "first-order thermal resistance-capacitance model",
        "equations": {
            "baseline": "T_next = T_now + heat_gain + response * (T_outdoor - T_now)",
            "neighbor": "T_next = T_now + heat_gain + a*(T_outdoor-T_now) + d*(T_neighbor_mean-T_now)",
        },
        "input_csv": str(input_path),
        "data_quality": qa,
        "train_rows": split,
        "test_rows": len(frame) - split,
        "floors": floor_results,
        "interpretation": {
            "one_step": "Uses the measured indoor temperature from the previous hour.",
            "rollout": "Uses only the final training indoor temperature and then predicts continuously.",
            "warning": "Temperature alone cannot uniquely identify infiltration ELA and envelope heat transfer.",
            "neighbor_boundary": "Adjacent-room temperatures are observed boundary conditions; they improve corridor prediction but do not identify air leakage.",
        },
    }
    (output_dir / "temperature_rc_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a first-order floor temperature RC model")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_rc_model(args.config)
    for floor, values in result["floors"].items():
        base = values["persistence_test_metrics"]["rmse_c"]
        one = values["one_step_test_metrics"]["rmse_c"]
        rollout = values["rollout_test_metrics"]["rmse_c"]
        message = f"{floor}: persistence={base:.3f}, one-step RC={one:.3f}, rollout RC={rollout:.3f} °C"
        if "neighbor_model" in values:
            neighbor = values["neighbor_model"]
            message += (
                f", neighbor one-step={neighbor['one_step_test_metrics']['rmse_c']:.3f}"
                f", neighbor rollout={neighbor['rollout_test_metrics']['rmse_c']:.3f} °C"
            )
        print(message)


if __name__ == "__main__":
    main()
