from __future__ import annotations

import unittest

from src.temperature_model import PROJECT_ROOT
from src.temperature_rc_model import run_rc_model


class TemperatureRcModelTests(unittest.TestCase):
    def test_rc_model_is_physical_and_compared_with_persistence(self) -> None:
        result = run_rc_model(PROJECT_ROOT / "configs" / "temperature_rc_model.json")
        self.assertEqual(result["data_quality"]["row_count"], 421)
        for values in result["floors"].values():
            self.assertGreater(values["thermal_response_per_h"], 0.0)
            self.assertLess(values["thermal_response_per_h"], 1.0)
            self.assertIn("persistence_test_metrics", values)
            self.assertIn("rollout_test_metrics", values)
            self.assertIn("neighbor_model", values)
            self.assertGreaterEqual(values["neighbor_model"]["neighbor_response_per_h"], 0.0)


if __name__ == "__main__":
    unittest.main()
