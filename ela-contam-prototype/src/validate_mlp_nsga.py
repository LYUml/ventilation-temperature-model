from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data


@dataclass(frozen=True)
class Genome:
    width: int
    depth: int
    learning_rate: float
    weight_decay: float


WIDTHS = (4, 8, 16, 32, 64)
DEPTHS = (1, 2, 3)
LEARNING_RATES = (0.0005, 0.001, 0.003, 0.01)
WEIGHT_DECAYS = (0.0, 0.0001, 0.001, 0.01)


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, genome: Genome) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for _ in range(genome.depth):
            layers.extend((nn.Linear(previous, genome.width), nn.Tanh()))
            previous = genome.width
        layers.append(nn.Linear(previous, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def make_windows(
    outdoor: np.ndarray,
    indoor: np.ndarray,
    hours: np.ndarray,
    start: int,
    end: int,
    horizon: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    features, targets, keys = [], [], []
    for origin in range(start, end - horizon + 1):
        hour_angle = 2.0 * math.pi * float(hours[origin]) / 24.0
        weather = outdoor[origin : origin + horizon]
        for floor_index in range(indoor.shape[1]):
            floor = np.zeros(indoor.shape[1], dtype=float)
            floor[floor_index] = 1.0
            features.append(np.r_[weather, math.sin(hour_angle), math.cos(hour_angle), floor])
            targets.append(indoor[origin : origin + horizon, floor_index])
            keys.append((origin, floor_index))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), keys


def standardize(
    train_x: np.ndarray, train_y: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    x_mean, x_std = train_x.mean(0), train_x.std(0)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0
    result = [(train_x - x_mean) / x_std, (train_y - y_mean) / y_std]
    result.extend((item - x_mean) / x_std for item in others)
    return (*result, y_mean, y_std)


def train_candidate(
    genome: Genome,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed: int,
    epochs: int = 300,
) -> tuple[float, int, dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    model = MLP(train_x.shape[1], train_y.shape[1], genome)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay
    )
    loss_function = nn.MSELoss()
    tx, ty = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx = torch.from_numpy(val_x)
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(tx), ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_function(model(vx), torch.from_numpy(val_y)))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    assert best_state is not None
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return math.sqrt(best_loss), parameter_count, best_state


def dominates(a: tuple[float, int], b: tuple[float, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def nondominated(population: list[Genome], scores: dict[Genome, tuple[float, int]]) -> list[Genome]:
    return [
        item
        for item in population
        if not any(dominates(scores[other], scores[item]) for other in population if other != item)
    ]


def random_genome(rng: random.Random) -> Genome:
    return Genome(rng.choice(WIDTHS), rng.choice(DEPTHS), rng.choice(LEARNING_RATES), rng.choice(WEIGHT_DECAYS))


def mutate(genome: Genome, rng: random.Random) -> Genome:
    values = list(asdict(genome).values())
    index = rng.randrange(4)
    values[index] = rng.choice((WIDTHS, DEPTHS, LEARNING_RATES, WEIGHT_DECAYS)[index])
    return Genome(*values)


def run_experiment(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    timestamp, outdoor_name = config["timestamp_column"], config["outdoor_column"]
    floors = list(config["floor_columns"])
    frame[timestamp] = pd.to_datetime(frame[timestamp], errors="raise")
    qa = validate_data(frame, timestamp, [outdoor_name, *floors])
    outdoor = frame[outdoor_name].to_numpy(float)
    indoor = frame[floors].to_numpy(float)
    hours = frame[timestamp].dt.hour.to_numpy()

    train_end, val_end = int(len(frame) * 0.60), int(len(frame) * 0.80)
    train_x, train_y, _ = make_windows(outdoor, indoor, hours, 0, train_end)
    val_x, val_y, _ = make_windows(outdoor, indoor, hours, train_end, val_end)
    test_x, test_y, test_keys = make_windows(outdoor, indoor, hours, val_end, len(frame))
    train_x, train_y, val_x, test_x, y_mean, y_std = standardize(train_x, train_y, val_x, test_x)
    val_y_scaled = (val_y - y_mean) / y_std

    rng = random.Random(20260828)
    population_size, generations = 12, 4
    population = list({random_genome(rng) for _ in range(population_size * 3)})[:population_size]
    scores: dict[Genome, tuple[float, int]] = {}
    states: dict[Genome, dict[str, torch.Tensor]] = {}
    for generation in range(generations):
        for genome in population:
            if genome not in scores:
                rmse_scaled, count, state = train_candidate(
                    genome, train_x, train_y, val_x, val_y_scaled, seed=1000 + generation
                )
                scores[genome] = (rmse_scaled, count)
                states[genome] = state
        front = nondominated(population, scores)
        ranked = sorted(population, key=lambda g: (0 if g in front else 1, scores[g][0], scores[g][1]))
        parents = ranked[: max(4, population_size // 2)]
        children = set(parents)
        while len(children) < population_size:
            children.add(mutate(rng.choice(parents), rng))
        population = list(children)

    for genome in population:
        if genome not in scores:
            rmse_scaled, count, state = train_candidate(genome, train_x, train_y, val_x, val_y_scaled, seed=2000)
            scores[genome], states[genome] = (rmse_scaled, count), state
    all_candidates = list(scores)
    pareto = nondominated(all_candidates, scores)
    best = min(pareto, key=lambda g: scores[g][0])

    model = MLP(test_x.shape[1], test_y.shape[1], best)
    model.load_state_dict(states[best])
    model.eval()
    with torch.no_grad():
        mlp_prediction = model(torch.from_numpy(test_x)).numpy() * y_std + y_mean

    # Kernel benchmark is tuned only on the first 60% and evaluated on the same test windows.
    candidates = np.linspace(0.5, 0.995, 100)
    inner = int(train_end * 0.7)
    best_alpha, best_score = None, float("inf")
    for alpha in candidates:
        state = causal_ewma(outdoor, float(alpha))
        errors = []
        for j in range(len(floors)):
            coefficients = fit_linear(state[:inner], indoor[:inner, j])
            prediction = predict_linear(state[inner:train_end], coefficients)
            errors.append(metrics(indoor[inner:train_end, j], prediction)["rmse_c"])
        if np.mean(errors) < best_score:
            best_alpha, best_score = float(alpha), float(np.mean(errors))
    assert best_alpha is not None
    kernel_state = causal_ewma(outdoor, best_alpha)
    kernel_coefficients = [fit_linear(kernel_state[:train_end], indoor[:train_end, j]) for j in range(len(floors))]
    kernel_prediction = np.empty_like(test_y)
    for row, (origin, floor_index) in enumerate(test_keys):
        kernel_prediction[row] = predict_linear(
            kernel_state[origin : origin + 24], kernel_coefficients[floor_index]
        )

    floor_results = {}
    for floor_index, floor in enumerate(floors):
        mask = np.asarray([key[1] == floor_index for key in test_keys])
        actual_values = test_y[mask].reshape(-1)
        mlp_values = mlp_prediction[mask].reshape(-1)
        kernel_values = kernel_prediction[mask].reshape(-1)
        floor_results[floor] = {
            "kernel": metrics(actual_values, kernel_values),
            "mlp_nsga": metrics(actual_values, mlp_values),
            "mlp_improvement_vs_kernel_pct": float(
                100.0 * (metrics(actual_values, kernel_values)["rmse_c"] - metrics(actual_values, mlp_values)["rmse_c"])
                / metrics(actual_values, kernel_values)["rmse_c"]
            ),
        }

    result = {
        "method": "24-hour direct multi-output MLP with compact NSGA-II hyperparameter search",
        "data_quality": qa,
        "split": {"train_end": train_end, "validation_end": val_end, "test_rows": len(frame) - val_end},
        "test_windows_per_floor": int(len(test_keys) / len(floors)),
        "input": "24 outdoor temperatures + start-hour sine/cosine + floor one-hot",
        "selected_genome": asdict(best),
        "selected_parameter_count": scores[best][1],
        "pareto_front": [{**asdict(g), "validation_scaled_rmse": scores[g][0], "parameter_count": scores[g][1]} for g in sorted(pareto, key=lambda item: scores[item][0])],
        "kernel_alpha": best_alpha,
        "floors": floor_results,
        "limitations": [
            "Only one building and 421 hourly records are available.",
            "Sliding 24-hour windows overlap and are not independent samples.",
            "RDF geometry cannot be learned as a transferable effect from a single building.",
        ],
    }
    output_dir = PROJECT_ROOT / "outputs" / "mlp_nsga_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"origin": frame[timestamp].iloc[origin].isoformat(), "floor": floors[floor_index], "horizon_h": step + 1,
         "measured_c": float(test_y[row, step]), "kernel_c": float(kernel_prediction[row, step]), "mlp_nsga_c": float(mlp_prediction[row, step])}
        for row, (origin, floor_index) in enumerate(test_keys) for step in range(24)
    ]).to_csv(output_dir / "predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    result = run_experiment(parser.parse_args().config)
    print("selected", result["selected_genome"], "parameters", result["selected_parameter_count"])
    for floor, values in result["floors"].items():
        print(f"{floor}: kernel={values['kernel']['rmse_c']:.3f}, MLP={values['mlp_nsga']['rmse_c']:.3f} C, improvement={values['mlp_improvement_vs_kernel_pct']:.1f}%")


if __name__ == "__main__":
    main()
