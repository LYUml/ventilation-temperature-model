from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.run_case import PROJECT_ROOT, load_config, run_case


CONFIG_PATH = PROJECT_ROOT / "configs" / "smoke_test.json"


class ElaContamIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = load_config(CONFIG_PATH)
        cls.temp_root = tempfile.TemporaryDirectory(prefix="ela-contam-tests-")
        cls.root = Path(cls.temp_root.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_root.cleanup()

    def run_variant(self, name: str, update) -> dict:
        config = copy.deepcopy(self.base)
        config["case_name"] = name
        update(config)
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return run_case(path, self.root / name)

    def test_smoke_case_and_mass_balance(self) -> None:
        summary = run_case(CONFIG_PATH, self.root / "smoke")
        self.assertEqual(summary["execution"]["exit_code"], 0)
        self.assertLess(summary["results"]["relative_mass_imbalance"], 0.001)
        self.assertGreater(summary["results"]["infiltration_ach_1_per_h"], 0.0)

    def test_equal_temperatures_remove_stack_flow(self) -> None:
        summary = self.run_variant(
            "equal_temperature",
            lambda config: config["outdoor"].update(
                temperature_c=config["room"]["indoor_temperature_c"]
            ),
        )
        self.assertLess(abs(summary["results"]["inflow_mass_kg_s"]), 1.0e-9)

    def test_zero_ela_is_numerically_zero(self) -> None:
        summary = self.run_variant(
            "zero_ela",
            lambda config: config["leakage"].update(normalized_ela_cm2_per_m2=0.0),
        )
        self.assertEqual(summary["physical_input"]["total_ela_cm2"], 0.0)
        self.assertLess(summary["results"]["inflow_mass_kg_s"], 1.0e-9)

    def test_double_ela_approximately_doubles_flow(self) -> None:
        base = self.run_variant("base_for_ratio", lambda config: None)
        doubled = self.run_variant(
            "double_ela",
            lambda config: config["leakage"].update(
                normalized_ela_cm2_per_m2=(
                    2.0 * config["leakage"]["normalized_ela_cm2_per_m2"]
                )
            ),
        )
        ratio = (
            doubled["results"]["inflow_mass_kg_s"]
            / base["results"]["inflow_mass_kg_s"]
        )
        self.assertGreaterEqual(ratio, 1.8)
        self.assertLessEqual(ratio, 2.2)

    def test_repeatability(self) -> None:
        first = self.run_variant("repeat_a", lambda config: None)
        second = self.run_variant("repeat_b", lambda config: None)
        self.assertAlmostEqual(
            first["results"]["inflow_mass_kg_s"],
            second["results"]["inflow_mass_kg_s"],
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
