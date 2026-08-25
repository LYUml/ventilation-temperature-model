from __future__ import annotations

import unittest

from src.run_stair_connected_case import run_case
from src.temperature_model import PROJECT_ROOT


class StairConnectedCaseTests(unittest.TestCase):
    def test_all_connection_scenarios_solve_and_balance(self) -> None:
        config = PROJECT_ROOT / "configs" / "rdf_stair_connected.json"
        flows = []
        for name in ("closed", "partial", "open"):
            result = run_case(config, name)
            self.assertEqual(len(result["zones"]), 6)
            self.assertTrue(result["checks"]["all_zones_below_0_1_percent"])
            vertical = [abs(path["mass_flow_kg_s"]) for path in result["paths"] if path["kind"] == "vertical_stair"]
            self.assertEqual(len(vertical), 2)
            flows.append(max(vertical))
        self.assertLess(flows[0], flows[1])
        self.assertLess(flows[1], flows[2])


if __name__ == "__main__":
    unittest.main()
