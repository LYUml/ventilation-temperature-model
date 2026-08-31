from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..common.data import ROOT, error_metrics, fit_kernel, load_config, load_inputs, origins, split_indices, windows


def run(config_path: Path | None = None) -> dict:
    config = load_config(config_path); frame, qa = load_inputs(config)
    floors = config["floor_columns"]; indoor = frame[floors].to_numpy(float); outdoor = frame[config["outdoor_column"]].to_numpy(float)
    train_end, validation_end = split_indices(len(frame)); test_origins = origins(validation_end, len(frame))
    prediction, alpha = fit_kernel(outdoor, indoor, train_end); actual = windows(indoor, test_origins); predicted = windows(prediction, test_origins)
    floor_results = {name: error_metrics(actual[:, :, index].reshape(-1), predicted[:, :, index].reshape(-1)) for index, name in enumerate(floors)}
    result = {"method": "causal_kernel_baseline", "data_quality": qa, "split": [train_end, validation_end, len(frame)], "test_windows_per_floor": len(test_origins), "selected_alpha": alpha, "mean_floor_rmse_c": float(np.mean([value["rmse_c"] for value in floor_results.values()])), "floors": floor_results}
    output = ROOT / "results" / "kernel"; output.mkdir(parents=True, exist_ok=True); (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path); result = run(parser.parse_args().config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
