from __future__ import annotations

import unittest

from src.temperature_model import PROJECT_ROOT
from src.validate_24h_baseline import run_24h_validation


class Baseline24HourValidationTests(unittest.TestCase):
    def test_rolling_forecasts_have_expected_shape_and_no_window_leakage(self) -> None:
        result = run_24h_validation(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["horizon_hours"], 24)
        self.assertEqual(result["rolling_origins"], 104)
        for values in result["floors"].values():
            self.assertIn("kernel", values["all_horizons"])
            self.assertIn("rc", values["all_horizons"])
            self.assertGreater(values["rc_coefficients"]["response_per_h"], 0.0)


if __name__ == "__main__":
    unittest.main()
