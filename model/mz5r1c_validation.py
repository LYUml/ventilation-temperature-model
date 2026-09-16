"""Evaluate design-stage MZ5R1C outputs against measurements.

Measurements are read only after simulation and are never passed to the
design model. Missing measurements are reported rather than guessed or
silently aliased to a differently named sensor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common.data import ROOT, error_metrics, load_config, load_inputs, split_indices
from .mz5r1c import RdfMz5r1cModel


WEATHER_COLUMNS = [
    "Timestamp",
    "TEM",
    "PRS",
    "RHU",
    "PRE_1h",
    "WIN_S_Avg_2mi",
    "WIN_D_Avg_2mi",
    "diffuse",
    "direct",
]


def run(config_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    frame, qa = load_inputs(config)
    targets = list(config.get("design_validation_spaces", config["floor_columns"]))
    baselines = list(config.get("design_baseline_spaces", ()))
    model = RdfMz5r1cModel(
        timestamp_column=config["timestamp_column"],
        outdoor_column="TEM",
        parameters={"allow_non_project_assumptions": True},
        baseline_spaces=baselines,
    )
    prediction = model.simulate(
        ROOT / config["rdf_file"], frame[WEATHER_COLUMNS], targets
    )

    _, test_start = split_indices(len(frame))
    measured = [name for name in targets if name in frame.columns]
    missing = [name for name in targets if name not in frame.columns]
    metrics = {
        name: error_metrics(
            frame[name].to_numpy(dtype=float)[test_start:],
            np.asarray(prediction[name], dtype=float)[test_start:],
        )
        for name in measured
    }
    result = {
        "method": "RDF-driven simultaneous multi-zone ISO 13790 5R1C",
        "evaluation_role": "Measurements are validation-only and are never model inputs.",
        "data_quality": qa,
        "targets": targets,
        "baseline_spaces": baselines,
        "evaluation_split": {
            "rule": "last 20 percent locked chronological test segment",
            "start_index": test_start,
            "rows": len(frame) - test_start,
            "start_timestamp": frame[config["timestamp_column"]].iloc[test_start].isoformat(),
            "end_timestamp": frame[config["timestamp_column"]].iloc[-1].isoformat(),
        },
        "evaluated_spaces": measured,
        "missing_measurement_spaces": missing,
        "metrics": metrics,
        "mean_rmse_c": (
            float(np.mean([value["rmse_c"] for value in metrics.values()]))
            if metrics else None
        ),
        "model_metadata": model.last_metadata,
    }
    output = ROOT / "results" / "mz5r1c_validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    print(json.dumps(run(parser.parse_args().config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
