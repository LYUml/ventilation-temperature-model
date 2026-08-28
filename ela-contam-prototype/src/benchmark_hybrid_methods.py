from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares
from .benchmark_baseline_models import fit_2r2c, predict_2r2c
from .temperature_model import PROJECT_ROOT, causal_ewma, fit_linear, metrics, predict_linear, validate_data
from .temperature_rc_model import fit_first_order_rc
from .validate_mlp_nsga import make_windows


def fit_ridge(train_x, train_y, val_x, val_y):
    scored = []
    for alpha in (.01, .1, 1, 10, 100, 1000):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)
        scored.append((math.sqrt(mean_squared_error(val_y, model.predict(val_x))), alpha, model))
    return min(scored, key=lambda item: item[0])[1:]


def rc_windows(outdoor, indoor, keys, coefficients):
    result = []
    for origin, floor in keys:
        gain, response = coefficients[floor]
        current = float(indoor[origin - 1, floor]) if origin else float(indoor[0, floor])
        values = []
        for step in range(24):
            current += gain + response * (outdoor[origin + step - 1] - current if origin + step else outdoor[0] - current)
            values.append(current)
        result.append(values)
    return np.asarray(result)


def shared_rc_windows(outdoor, mean_indoor, keys, coefficients):
    gain, response = coefficients; result = []
    for origin, _ in keys:
        current = float(mean_indoor[origin - 1]) if origin else float(mean_indoor[0]); values = []
        for step in range(24):
            current += gain + response * (outdoor[origin + step - 1] - current if origin + step else outdoor[0] - current)
            values.append(current)
        result.append(values)
    return np.asarray(result)


def run_hybrid_benchmark(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    frame = pd.read_csv((PROJECT_ROOT / config["input_csv"]).resolve())
    ts, outdoor_name, floors = config["timestamp_column"], config["outdoor_column"], list(config["floor_columns"])
    frame[ts] = pd.to_datetime(frame[ts], errors="raise")
    qa = validate_data(frame, ts, [outdoor_name, *floors])
    outdoor, indoor, hours = frame[outdoor_name].to_numpy(float), frame[floors].to_numpy(float), frame[ts].dt.hour.to_numpy()
    train_end, val_end = int(len(frame) * .6), int(len(frame) * .8)
    train_x, train_y, train_keys = make_windows(outdoor, indoor, hours, 1, train_end)
    val_x, val_y, val_keys = make_windows(outdoor, indoor, hours, train_end, val_end)
    test_x, test_y, test_keys = make_windows(outdoor, indoor, hours, val_end, len(frame))

    # Original kernel benchmark.
    inner = int(train_end * .7); best_alpha, best_error = None, float("inf")
    for alpha in np.linspace(.5, .995, 100):
        state = causal_ewma(outdoor, float(alpha))
        error = np.mean([metrics(indoor[inner:train_end, j], predict_linear(state[inner:train_end], fit_linear(state[:inner], indoor[:inner, j])))["rmse_c"] for j in range(3)])
        if error < best_error: best_alpha, best_error = float(alpha), float(error)
    state = causal_ewma(outdoor, best_alpha); coefs = [fit_linear(state[:train_end], indoor[:train_end, j]) for j in range(3)]
    kernel_val = np.asarray([predict_linear(state[o:o + 24], coefs[f]) for o, f in val_keys])
    kernel_test = np.asarray([predict_linear(state[o:o + 24], coefs[f]) for o, f in test_keys])

    # Exact SysIdentPy polynomial NARX with automatically selected terms.
    sys_models = []
    for floor in range(3):
        model = FROLS(ylag=2, xlag=6, order_selection=True, n_info_values=10, info_criteria="aic", estimator=LeastSquares(), basis_function=Polynomial(degree=2))
        model.fit(X=outdoor[:train_end, None], y=indoor[:train_end, floor, None]); sys_models.append(model)
    sys_pred = []
    for origin, floor in test_keys:
        x = outdoor[origin - 6:origin + 24, None]; yseed = indoor[origin - 6:origin, floor, None]
        sys_pred.append(sys_models[floor].predict(X=x, y=yseed, forecast_horizon=len(x))[6:, 0])
    sys_pred = np.asarray(sys_pred)

    # DarkGreyBox-style RC structure search: 1R1C versus 2R2C selected on validation.
    rc1 = [fit_first_order_rc(indoor[:, j], outdoor, train_end) for j in range(3)]
    rc2 = [fit_2r2c(outdoor[:train_end], indoor[:train_end, j]) for j in range(3)]
    rc1_val, rc1_test = rc_windows(outdoor, indoor, val_keys, rc1), rc_windows(outdoor, indoor, test_keys, rc1)
    rc2_val = np.asarray([predict_2r2c(rc2[f], indoor[o - 1, f], outdoor[o:o + 24]) for o, f in val_keys])
    rc2_test = np.asarray([predict_2r2c(rc2[f], indoor[o - 1, f], outdoor[o:o + 24]) for o, f in test_keys])
    chosen = []
    for floor in range(3):
        mask = np.asarray([f == floor for _, f in val_keys])
        e1, e2 = mean_squared_error(val_y[mask], rc1_val[mask]), mean_squared_error(val_y[mask], rc2_val[mask])
        chosen.append("1R1C" if e1 <= e2 else "2R2C")
    greybox_test = np.asarray([rc1_test[i] if chosen[f] == "1R1C" else rc2_test[i] for i, (_, f) in enumerate(test_keys)])

    # ORNL-style hierarchy: shared RC predicts building mean; Ridge predicts floor deviation.
    mean_indoor = indoor.mean(axis=1); mean_rc = fit_first_order_rc(mean_indoor, outdoor, train_end)
    shared_train, shared_val, shared_test = (shared_rc_windows(outdoor, mean_indoor, keys, mean_rc) for keys in (train_keys, val_keys, test_keys))
    delta_train = train_y - np.asarray([mean_indoor[o:o + 24] for o, _ in train_keys])
    delta_val = val_y - np.asarray([mean_indoor[o:o + 24] for o, _ in val_keys])
    initial_delta_train = np.asarray([indoor[o - 1, f] - mean_indoor[o - 1] for o, f in train_keys])[:, None]
    initial_delta_val = np.asarray([indoor[o - 1, f] - mean_indoor[o - 1] for o, f in val_keys])[:, None]
    initial_delta_test = np.asarray([indoor[o - 1, f] - mean_indoor[o - 1] for o, f in test_keys])[:, None]
    _, delta_model = fit_ridge(np.c_[train_x, initial_delta_train], delta_train, np.c_[val_x, initial_delta_val], delta_val)
    ornl_test = shared_test + delta_model.predict(np.c_[test_x, initial_delta_test])

    # Proposed method: floor RC trajectory + regularized NARX residual correction.
    floor_rc_train, floor_rc_val, floor_rc_test = rc_windows(outdoor, indoor, train_keys, rc1), rc1_val, rc1_test
    residual_train, residual_val = train_y - floor_rc_train, val_y - floor_rc_val
    hybrid_train_x, hybrid_val_x, hybrid_test_x = np.c_[train_x, floor_rc_train], np.c_[val_x, floor_rc_val], np.c_[test_x, floor_rc_test]
    alpha, residual_model = fit_ridge(hybrid_train_x, residual_train, hybrid_val_x, residual_val)
    proposed_val = floor_rc_val + residual_model.predict(hybrid_val_x)
    proposed_test = floor_rc_test + residual_model.predict(hybrid_test_x)

    # Proposed adaptive fusion: validation-selected convex blend of stable kernel and RC-NARX.
    fusion_weights = []
    fused_test = np.empty_like(test_y)
    for floor in range(3):
        val_mask = np.asarray([f == floor for _, f in val_keys])
        test_mask = np.asarray([f == floor for _, f in test_keys])
        choices = []
        for weight in np.linspace(0.0, 1.0, 21):
            prediction = weight * kernel_val[val_mask] + (1.0 - weight) * proposed_val[val_mask]
            choices.append((mean_squared_error(val_y[val_mask], prediction), float(weight)))
        weight = min(choices)[1]; fusion_weights.append(weight)
        fused_test[test_mask] = weight * kernel_test[test_mask] + (1.0 - weight) * proposed_test[test_mask]

    predictions = {"kernel": kernel_test, "sysidentpy_polynomial_narx": sys_pred, "greybox_rc_structure_search": greybox_test, "ornl_shared_rc_floor_delta": ornl_test, "proposed_hierarchical_rc_narx_ridge": proposed_test, "proposed_adaptive_kernel_rc_narx_fusion": fused_test}
    results = {}
    for name, pred in predictions.items():
        per_floor = {}
        for floor, label in enumerate(floors):
            mask = np.asarray([f == floor for _, f in test_keys]); per_floor[label] = metrics(test_y[mask].reshape(-1), pred[mask].reshape(-1))
        results[name] = {"mean_floor_rmse_c": float(np.mean([v["rmse_c"] for v in per_floor.values()])), "floors": per_floor}
    ranking = sorted(results, key=lambda n: results[n]["mean_floor_rmse_c"])
    result = {"data_quality": qa, "split": {"train": [0, train_end], "validation": [train_end, val_end], "test": [val_end, len(frame)]}, "test_windows_per_floor": len(test_keys) // 3, "ranking": ranking, "models": results, "proposed_method": {"name": "Adaptive Kernel–RC–NARX Fusion Model", "equation": "T_hat_f = w_f*T_kernel_f + (1-w_f)*(T_RC_f + Ridge residual_f)", "selected_ridge_alpha": alpha, "kernel_weights_by_floor": dict(zip(floors, fusion_weights))}, "greybox_selected_structure": dict(zip(floors, chosen)), "caveats": ["State-aware hybrid methods read the corridor temperature immediately before each 24-hour window.", "Only one building and 17.5 days are available.", "Sliding windows overlap."]}
    output = PROJECT_ROOT / "outputs" / "hybrid_method_benchmark"; output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"rank": i + 1, "model": n, "mean_floor_rmse_c": results[n]["mean_floor_rmse_c"]} for i, n in enumerate(ranking)]).to_csv(output / "ranking.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path)
    result = run_hybrid_benchmark(parser.parse_args().config)
    for i, name in enumerate(result["ranking"], 1): print(f"{i}. {name}: {result['models'][name]['mean_floor_rmse_c']:.3f} C")


if __name__ == "__main__": main()
