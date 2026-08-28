import unittest

from src.temperature_model import PROJECT_ROOT
from src.validate_rdf_2r2c import run_validation


class Rdf2R2cValidationTests(unittest.TestCase):
    def test_parameters_are_physical_and_test_is_locked(self):
        result = run_validation(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["test_windows_per_floor"], 62)
        for item in result["floors"].values():
            self.assertGreater(item["rdf"]["exterior_ua_w_k"], 0.0)
            self.assertGreaterEqual(item["parameters"]["ceff_multiplier"], 110 / 165)
            self.assertLessEqual(item["parameters"]["ceff_multiplier"], 260 / 165)
            self.assertGreater(item["parameters"]["g_mass_air_w_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
