from __future__ import annotations

import unittest

from src.temperature_model import PROJECT_ROOT
from src.validate_mlp_nsga import run_experiment


class MlpNsgaValidationTests(unittest.TestCase):
    def test_locked_test_split_and_outputs(self) -> None:
        result = run_experiment(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["split"]["test_rows"], 85)
        self.assertEqual(result["test_windows_per_floor"], 62)
        self.assertGreater(result["selected_parameter_count"], 0)
        self.assertEqual(set(result["floors"]), {"2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR"})


if __name__ == "__main__":
    unittest.main()
