from __future__ import annotations

import json
import unittest

from src.run_three_floor_case import run_three_floor_case
from src.temperature_model import PROJECT_ROOT


class ThreeFloorWorkflowTests(unittest.TestCase):
    def test_hypothetical_three_floor_case(self) -> None:
        result = run_three_floor_case(PROJECT_ROOT / "configs" / "three_floor_smoke_test.json")
        self.assertEqual(len(result["floors"]), 3)
        self.assertTrue(result["checks"]["all_floor_mass_balances_below_0_1_percent"])
        self.assertTrue(result["checks"]["all_floors_have_inflow_and_outflow"])
        for floor in result["floors"]:
            self.assertGreater(floor["infiltration_ach_1_per_h"], 0.0)

    def test_isolated_equal_temperature_floors_have_equal_ach(self) -> None:
        source = PROJECT_ROOT / "configs" / "three_floor_smoke_test.json"
        config = json.loads(source.read_text(encoding="utf-8"))
        config["case_name"] = "three_floor_equal_temperature_test"
        for floor in config["floors"]:
            floor["indoor_temperature_c"] = 25.0
        temporary_config = PROJECT_ROOT / "outputs" / "three_floor_equal_temperature_config.json"
        temporary_config.write_text(json.dumps(config), encoding="utf-8")
        result = run_three_floor_case(temporary_config)
        values = [floor["infiltration_ach_1_per_h"] for floor in result["floors"]]
        self.assertLess(max(values) - min(values), 1e-10)


if __name__ == "__main__":
    unittest.main()
