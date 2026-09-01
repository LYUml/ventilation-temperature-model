from __future__ import annotations

import json

import numpy as np

from model.common.data import ROOT, error_metrics, load_config, load_inputs, origins, split_indices
from model.common.rdf import extract_building
from model.rdf_kernel_narx_ridge.run import rdf_boundaries
from model.rc_narx_ridge_optimization.run import (
    HORIZON,
    _fit_rc,
    _fit_residual_models,
    _make_features,
    _one_step_residuals,
    _predict,
    _rc_windows,
)


def run() -> dict:
    config = load_config(); frame, qa = load_inputs(config)
    floors = list(config["floor_columns"])
    outdoor = frame[config["outdoor_column"]].to_numpy(float)
    indoor = frame[floors].to_numpy(float)
    hours = frame[config["timestamp_column"]].dt.hour.to_numpy()
    building = extract_building(ROOT / config["rdf_file"])
    boundaries, topology = rdf_boundaries(building, frame, floors)
    train_end, val_end = split_indices(len(frame))
    origin_sets = (origins(24, train_end), origins(train_end, val_end), origins(val_end, len(frame)))
    key_sets = [[(origin, floor) for origin in starts for floor in range(3)] for starts in origin_sets]
    actual_sets = [np.asarray([indoor[o:o + HORIZON, f] for o, f in keys]) for keys in key_sets]

    kind = "1r1c_one_step"
    parameters = [_fit_rc(indoor[:, floor], outdoor, train_end, kind) for floor in range(3)]
    rc_sets = [_rc_windows(outdoor, indoor, keys, parameters, kind) for keys in key_sets]
    rc_residual = _one_step_residuals(outdoor, indoor, parameters, kind)
    x_sets = [
        np.c_[_make_features(outdoor, indoor, hours, boundaries, topology, rc_residual, keys, "residual_lags_rdf"), rc]
        for keys, rc in zip(key_sets, rc_sets)
    ]
    residual_targets = [actual - rc for actual, rc in zip(actual_sets, rc_sets)]
    validation_rmse, alpha, models = _fit_residual_models(
        x_sets[0], residual_targets[0], x_sets[1], residual_targets[1], key_sets[0], key_sets[1], "shared"
    )
    prediction = rc_sets[2] + _predict(models, x_sets[2], key_sets[2], "shared")
    floor_metrics = {}
    for floor, label in enumerate(floors):
        mask = np.asarray([key[1] == floor for key in key_sets[2]])
        floor_metrics[label] = error_metrics(actual_sets[2][mask], prediction[mask])
    result = {
        "method": "RDF-informed RC NARX-Ridge",
        "data_quality": qa,
        "test_windows_per_floor": len(origin_sets[2]),
        "selected_ridge_alpha": alpha,
        "validation_rmse_c": validation_rmse,
        "mean_floor_rmse_c": float(np.mean([value["rmse_c"] for value in floor_metrics.values()])),
        "floors": floor_metrics,
        "fitted_rc": {label: {"gain_c_per_h": parameters[i][0], "response_per_h": parameters[i][1]} for i, label in enumerate(floors)},
        "caveat": "Uses only past RDF-selected room/floor temperatures, but the candidate was identified in a multi-configuration experiment on 421 hours and needs external validation.",
    }
    destination = ROOT / "results" / "rdf_rc_narx_ridge"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
