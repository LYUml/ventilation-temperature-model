from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.temperature_model import PROJECT_ROOT, run_temperature_model


class TemperatureModelTests(unittest.TestCase):
    def test_time_split_and_prediction_quality(self) -> None:
        result = run_temperature_model(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["data_quality"]["row_count"], 421)
        self.assertLess(result["train"]["end"], result["test"]["start"])
        for floor in ("2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR"):
            self.assertLess(result["floors"][floor]["test_metrics"]["rmse_c"], 1.0)
        metrics_path = PROJECT_ROOT / "outputs" / "temperature_model" / "temperature_metrics.json"
        self.assertTrue(metrics_path.exists())
        json.loads(metrics_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
