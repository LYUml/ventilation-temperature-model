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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    """Past-only exponentially weighted moving average."""
    result = np.empty(len(values), dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * result[index - 1] + (1.0 - alpha) * values[index]
    return result


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def predict_linear(x: np.ndarray, coefficients: tuple[float, float]) -> np.ndarray:
    return coefficients[0] + coefficients[1] * x


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "rmse_c": float(math.sqrt(float(np.mean(error**2)))),
        "mae_c": float(np.mean(np.abs(error))),
        "mean_error_c": float(np.mean(error)),
    }


def validate_data(
    frame: pd.DataFrame,
    timestamp_column: str,
    required_columns: list[str],
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("temperature CSV is empty")
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")
    if frame[required_columns].isna().any().any():
        raise ValueError("temperature CSV contains missing required values")
    if frame[timestamp_column].duplicated().any():
        raise ValueError("temperature CSV contains duplicate timestamps")
    if not frame[timestamp_column].is_monotonic_increasing:
        raise ValueError("timestamps are not increasing")
    intervals = frame[timestamp_column].diff().dropna().dt.total_seconds() / 3600.0
    non_hourly = int((np.abs(intervals.to_numpy() - 1.0) > 1.0e-9).sum())
    if non_hourly:
        raise ValueError(f"temperature CSV has {non_hourly} non-hourly gaps")
    return {
        "row_count": int(len(frame)),
        "start": frame[timestamp_column].iloc[0].isoformat(),
        "end": frame[timestamp_column].iloc[-1].isoformat(),
        "duplicate_timestamps": 0,
        "missing_required_values": 0,
        "non_hourly_gaps": 0,
    }


def run_temperature_model(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = (PROJECT_ROOT / config["input_csv"]).resolve()
    output_dir = PROJECT_ROOT / config.get("output_root", "outputs") / config["case_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = config["timestamp_column"]
    outdoor_column = config["outdoor_column"]
    floors = list(config["floor_columns"])
    frame = pd.read_csv(input_path)
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    qa = validate_data(frame, timestamp, [outdoor_column, *floors])

    count = len(frame)
    split = int(count * float(config["train_fraction"]))
    inner_split = int(split * float(config["inner_train_fraction"]))
    if inner_split < 2 or split >= count:
        raise ValueError("invalid train split")
    outdoor = frame[outdoor_column].to_numpy(dtype=float)
    actual = frame[floors].to_numpy(dtype=float)

    candidates = np.linspace(
        float(config["alpha_min"]),
        float(config["alpha_max"]),
        int(config["alpha_steps"]),
    )
    best_alpha = None
    best_validation_rmse = float("inf")
    for alpha in candidates:
        baseline = causal_ewma(outdoor, float(alpha))
        floor_rmses = []
        for floor_index in range(len(floors)):
            coefficients = fit_linear(
                baseline[:inner_split], actual[:inner_split, floor_index]
            )
            prediction = predict_linear(
                baseline[inner_split:split], coefficients
            )
            floor_rmses.append(
                metrics(actual[inner_split:split, floor_index], prediction)["rmse_c"]
            )
        candidate_score = float(np.mean(floor_rmses))
        if candidate_score < best_validation_rmse:
            best_validation_rmse = candidate_score
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("no smoothing parameter selected")

    baseline = causal_ewma(outdoor, best_alpha)
    predictions = np.empty_like(actual)
    floor_results: dict[str, Any] = {}
    for floor_index, floor in enumerate(floors):
        coefficients = fit_linear(baseline[:split], actual[:split, floor_index])
        predictions[:, floor_index] = predict_linear(baseline, coefficients)
        floor_results[floor] = {
            "intercept_c": coefficients[0],
            "baseline_slope": coefficients[1],
            "test_metrics": metrics(
                actual[split:, floor_index], predictions[split:, floor_index]
            ),
        }

    prediction_path = output_dir / "temperature_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [timestamp, outdoor_column, "common_baseline_c", "split", *[
            item for floor in floors for item in (floor + "_actual_c", floor + "_predicted_c")
        ]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            row: dict[str, Any] = {
                timestamp: frame[timestamp].iloc[index].isoformat(),
                outdoor_column: outdoor[index],
                "common_baseline_c": baseline[index],
                "split": "train" if index < split else "test",
            }
            for floor_index, floor in enumerate(floors):
                row[floor + "_actual_c"] = actual[index, floor_index]
                row[floor + "_predicted_c"] = predictions[index, floor_index]
            writer.writerow(row)

    figure, axes = plt.subplots(len(floors), 1, figsize=(12, 9), sharex=True)
    times = frame[timestamp]
    for floor_index, floor in enumerate(floors):
        axis = axes[floor_index]
        axis.plot(times, actual[:, floor_index], label="Measured", linewidth=1.5)
        axis.plot(times, predictions[:, floor_index], label="Predicted", linewidth=1.2)
        axis.axvline(times.iloc[split], color="black", linestyle="--", linewidth=1)
        axis.set_ylabel("Temperature (°C)")
        axis.set_title(
            f"{floor}: test RMSE={floor_results[floor]['test_metrics']['rmse_c']:.3f} °C"
        )
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Time")
    figure.suptitle("Causal outdoor-temperature baseline with floor corrections")
    figure.tight_layout()
    figure.savefig(output_dir / "temperature_comparison.png", dpi=160)
    plt.close(figure)

    result = {
        "method": "causal exponentially weighted outdoor baseline plus floor-specific linear correction",
        "input_csv": str(input_path),
        "data_quality": qa,
        "train": {
            "rows": split,
            "start": frame[timestamp].iloc[0].isoformat(),
            "end": frame[timestamp].iloc[split - 1].isoformat(),
        },
        "test": {
            "rows": count - split,
            "start": frame[timestamp].iloc[split].isoformat(),
            "end": frame[timestamp].iloc[-1].isoformat(),
        },
        "selected_alpha": best_alpha,
        "inner_validation_mean_rmse_c": best_validation_rmse,
        "floors": floor_results,
        "limitations": [
            "The dataset covers a short spring period and is not an annual model.",
            "Solar radiation, wind, door state and neighbouring-room heat transfer are unavailable.",
            "Floor corrections are empirical associations, not proof of physical causation.",
        ],
    }
    (output_dir / "temperature_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and validate corridor baseline temperatures")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_temperature_model(args.config)
    print(f"selected smoothing alpha: {result['selected_alpha']:.3f}")
    for floor, values in result["floors"].items():
        test = values["test_metrics"]
        print(f"{floor}: RMSE={test['rmse_c']:.3f} °C, MAE={test['mae_c']:.3f} °C")


if __name__ == "__main__":
    main()
