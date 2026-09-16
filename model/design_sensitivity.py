"""One-at-a-time 4F412 design-model sensitivity analysis."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np

from .base_ta import _timestamped
from .common.data import ROOT, error_metrics, load_config, load_inputs, split_indices
from .common.rdf import extract_building
from .mz5r1c import RdfMz5r1cModel
from .mz5r1c_validation import WEATHER_COLUMNS


def _metrics(actual, prediction, train_end, val_end):
    return {
        "validation": error_metrics(actual[train_end:val_end], prediction[train_end:val_end]),
        "locked_test": error_metrics(actual[val_end:], prediction[val_end:]),
        "locked_test_predicted_std_c": float(np.std(prediction[val_end:])),
        "locked_test_actual_std_c": float(np.std(actual[val_end:])),
    }


def _simulate_scaled(model, rdf, frame, target, factor_name, factor):
    frame = _timestamped(frame, "weather", model.timestamp_column)
    building = extract_building(rdf, target_names=[])
    spaces = {
        n: s for n, s in building["spaces"].items()
        if (s.get("floor_area_m2") or 0) > 0 and (s.get("volume_m3") or 0) > 0
    }
    names = sorted(spaces)
    data = model._assemble(names, spaces, building, set(model.baseline_spaces))
    data = deepcopy(data)
    if factor_name == "thermal_capacitance":
        data["cm"] *= factor
    elif factor_name == "infiltration":
        data["hve"] *= factor
    elif factor_name == "exterior_ua":
        data["hem"] *= factor
        data["hw"] *= factor
    elif factor_name == "interzone_ua":
        data["coupling"] *= factor
    else:
        raise ValueError(f"Unknown factor {factor_name}")
    result, _, _ = model._run(
        data,
        frame,
        frame["TEM"].to_numpy(float),
        frame["diffuse"].to_numpy(float),
        frame["direct"].to_numpy(float),
    )
    return result[:, names.index(target)]


def run():
    config = load_config()
    frame, qa = load_inputs(config)
    train_end, val_end = split_indices(len(frame))
    target = "4F412"
    actual = frame[target].to_numpy(float)
    rdf = ROOT / config["rdf_file"]
    common = {
        "outdoor_column": "TEM",
        "parameters": {"allow_non_project_assumptions": True},
        "baseline_spaces": config["design_baseline_spaces"],
    }
    model_1c = RdfMz5r1cModel(**common)
    factors = {
        "thermal_capacitance": [0.5, 1.0, 2.0, 4.0],
        "infiltration": [0.0, 0.2, 0.6, 1.0, 1.4],
        "exterior_ua": [0.8, 1.0, 1.2],
        "interzone_ua": [0.8, 1.0, 1.2],
    }
    sensitivity = {}
    for name, levels in factors.items():
        trials = []
        for factor in levels:
            prediction = _simulate_scaled(model_1c, rdf, frame[WEATHER_COLUMNS], target, name, factor)
            trials.append({"factor": factor, **_metrics(actual, prediction, train_end, val_end)})
        sensitivity[name] = {
            "trials": trials,
            "selected_by_validation_factor": min(
                trials, key=lambda item: item["validation"]["rmse_c"]
            )["factor"],
        }

    result = {
        "method": "4F412 one-at-a-time sensitivity analysis",
        "data_quality": qa,
        "split": {"train_end": train_end, "validation_end": val_end, "rows": len(frame)},
        "selection_rule": "Directions are selected by validation RMSE only; locked test is never used for selection.",
        "sensitivity": sensitivity,
        "caveats": [
            "Sensitivity factors are uncertainty scenarios, not fitted project parameters.",
            "Only 421 hourly rows from one May measurement period are available.",
        ],
    }
    output = ROOT / "results" / "design_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
