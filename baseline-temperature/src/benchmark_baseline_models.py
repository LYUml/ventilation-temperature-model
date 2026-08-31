from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data
from .validate_mlp_nsga import make_windows
from .weather_data import WEATHER_COLUMNS, merge_hourly_weather, weather_windows


def choose(candidates, x, y, vx, vy):
    scored = []
    for name, model in candidates:
        model.fit(x, y)
        scored.append((math.sqrt(mean_squared_error(vy, model.predict(vx))), name, model))
    return min(scored, key=lambda item: item[0])[1:]


def fit_2r2c(outdoor: np.ndarray, indoor: np.ndarray) -> np.ndarray:
    def rollout(p):
        ke, km, ka, gain = p
        air = mass = float(indoor[0]); result = []
        for index in range(1, len(indoor)):
            new_air = air + gain + ke * (outdoor[index - 1] - air) + km * (mass - air)
            mass += ka * (air - mass); air = new_air; result.append(air)
        return np.asarray(result)
    return least_squares(lambda p: rollout(p) - indoor[1:], [0.01, 0.03, 0.005, 0.02],
                         bounds=([0, 0, 0, -0.2], [0.3, 0.5, 0.3, 0.2]), max_nfev=3000).x


def predict_2r2c(p, initial, weather):
    ke, km, ka, gain = p
    air = mass = float(initial); result = []
    for value in weather:
        new_air = air + gain + ke * (value - air) + km * (mass - air)
        mass += ka * (air - mass); air = new_air; result.append(air)
    return np.asarray(result)


def run_benchmark(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    ts, outdoor_name, floors = config["timestamp_column"], config["outdoor_column"], list(config["floor_columns"])
    frame[ts] = pd.to_datetime(frame[ts], errors="raise")
    weather_qa = None
    weather = None
    if config.get("weather_csv"):
        frame, weather_qa = merge_hourly_weather(frame, ts, (PROJECT_ROOT / config["weather_csv"]).resolve())
        weather = frame[WEATHER_COLUMNS].to_numpy(float)
    qa = validate_data(frame, ts, [outdoor_name, *floors])
    outdoor, indoor, hours = frame[outdoor_name].to_numpy(float), frame[floors].to_numpy(float), frame[ts].dt.hour.to_numpy()
    train_end, val_end = int(len(frame) * .60), int(len(frame) * .80)
    train_x, train_y, train_keys = make_windows(outdoor, indoor, hours, 0, train_end)
    val_x, val_y, val_keys = make_windows(outdoor, indoor, hours, train_end, val_end)
    test_x, test_y, test_keys = make_windows(outdoor, indoor, hours, val_end, len(frame))

    inner = int(train_end * .7); best_alpha, best_error = None, float("inf")
    for alpha in np.linspace(.5, .995, 100):
        state = causal_ewma(outdoor, float(alpha))
        error = np.mean([metrics(indoor[inner:train_end, j], predict_linear(state[inner:train_end], fit_linear(state[:inner], indoor[:inner, j])))["rmse_c"] for j in range(3)])
        if error < best_error: best_alpha, best_error = float(alpha), float(error)
    state = causal_ewma(outdoor, best_alpha)
    kernel_coef = [fit_linear(state[:train_end], indoor[:train_end, j]) for j in range(3)]
    kernel_for = lambda keys: np.asarray([predict_linear(state[o:o + 24], kernel_coef[f]) for o, f in keys])
    ktrain, kval, ktest = kernel_for(train_keys), kernel_for(val_keys), kernel_for(test_keys)

    models = {}
    _, models["ridge_distributed_lag"] = choose([(str(a), make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (.01, .1, 1, 10, 100)], train_x, train_y, val_x, val_y)
    _, models["direct_mlp"] = choose([(str(h), make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, alpha=.01, early_stopping=True, max_iter=1200, random_state=42))) for h in ((8,), (16,), (32,), (16, 8))], train_x, train_y, val_x, val_y)
    _, residual = choose([(str(h), make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, alpha=.03, early_stopping=True, max_iter=1200, random_state=43))) for h in ((4,), (8,), (16,), (8, 4))], train_x, train_y - ktrain, val_x, val_y - kval)
    models["kernel_plus_mlp_residual"] = residual
    models["extra_trees"] = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.8, random_state=44, n_jobs=-1).fit(train_x, train_y)
    models["gradient_boosting"] = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=250, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1, random_state=45), n_jobs=-1).fit(train_x, train_y)

    gp_models = []
    for floor_index in range(3):
        mask = np.asarray([f == floor_index for _, f in train_keys])
        gp = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=ConstantKernel(1) * RBF(3) + WhiteKernel(.2), alpha=.03, normalize_y=True, optimizer=None))
        gp.fit(train_x[mask], train_y[mask]); gp_models.append(gp)

    initial = lambda keys: np.asarray([indoor[o - 1, f] if o else indoor[0, f] for o, f in keys])[:, None]
    txs, vxs, exs = np.c_[train_x, initial(train_keys)], np.c_[val_x, initial(val_keys)], np.c_[test_x, initial(test_keys)]
    state_models = {}
    _, state_models["narx_ridge"] = choose([(str(a), make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (.01, .1, 1, 10)], txs, train_y, vxs, val_y)
    _, state_models["narx_mlp"] = choose([(str(h), make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, alpha=.03, early_stopping=True, max_iter=1200, random_state=46))) for h in ((8,), (16,), (32,), (16, 8))], txs, train_y, vxs, val_y)
    state_models["narx_extra_trees"] = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.8, random_state=47, n_jobs=-1).fit(txs, train_y)

    weather_state_model = None
    weather_exs = None
    if weather is not None:
        train_weather = weather_windows(weather, train_keys)
        val_weather = weather_windows(weather, val_keys)
        test_weather = weather_windows(weather, test_keys)
        weather_txs, weather_vxs, weather_exs = np.c_[txs, train_weather], np.c_[vxs, val_weather], np.c_[exs, test_weather]
        _, weather_state_model = choose(
            [(str(a), make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (.1, 1, 10, 100, 1000)],
            weather_txs, train_y, weather_vxs, val_y,
        )

    predictions = {"kernel": ktest}
    for name, model in models.items(): predictions[name] = ktest + model.predict(test_x) if name == "kernel_plus_mlp_residual" else model.predict(test_x)
    gp_pred = np.empty_like(test_y)
    for floor_index, gp in enumerate(gp_models):
        mask = np.asarray([f == floor_index for _, f in test_keys]); gp_pred[mask] = gp.predict(test_x[mask])
    predictions["gaussian_process"] = gp_pred
    for name, model in state_models.items(): predictions[name] = model.predict(exs)
    if weather_state_model is not None and weather_exs is not None:
        predictions["narx_ridge_weather"] = weather_state_model.predict(weather_exs)
    params_2r2c = [fit_2r2c(outdoor[:train_end], indoor[:train_end, j]) for j in range(3)]
    predictions["2r2c"] = np.asarray([predict_2r2c(params_2r2c[f], indoor[o - 1, f], outdoor[o:o + 24]) for o, f in test_keys])

    results = {}
    for name, pred in predictions.items():
        per_floor = {}
        for f, floor in enumerate(floors):
            mask = np.asarray([key[1] == f for key in test_keys]); per_floor[floor] = metrics(test_y[mask].reshape(-1), pred[mask].reshape(-1))
        results[name] = {"requires_initial_corridor_temperature": name.startswith("narx") or name == "2r2c", "mean_floor_rmse_c": float(np.mean([v["rmse_c"] for v in per_floor.values()])), "floors": per_floor}
    ranking = sorted(results, key=lambda name: results[name]["mean_floor_rmse_c"])
    result = {"data_quality": qa, "weather_data_quality": weather_qa, "split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, len(frame)]}, "test_windows_per_floor": len(test_keys) // 3, "ranking": ranking, "models": results, "kernel_alpha": best_alpha, "two_r2c_parameters": {f: p.tolist() for f, p in zip(floors, params_2r2c)}, "caveats": ["One building and 17.5 days only.", "Windows overlap.", "State-aware models read corridor temperature once before the forecast window."]}
    output = PROJECT_ROOT / "outputs" / "baseline_model_benchmark"; output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"rank": i + 1, "model": n, "mean_floor_rmse_c": results[n]["mean_floor_rmse_c"], "requires_initial_corridor_temperature": results[n]["requires_initial_corridor_temperature"]} for i, n in enumerate(ranking)]).to_csv(output / "ranking.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    result = run_benchmark(parser.parse_args().config)
    for i, name in enumerate(result["ranking"], 1):
        value = result["models"][name]
        print(f"{i:2d}. {name:26s} RMSE={value['mean_floor_rmse_c']:.3f} C initial_state={value['requires_initial_corridor_temperature']}")


if __name__ == "__main__": main()
