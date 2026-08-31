import unittest
from src.temperature_model import PROJECT_ROOT
from src.validate_ceff_state_space import TOTAL_CEFF_J_K, run_validation


class CeffStateSpaceTests(unittest.TestCase):
    def test_fixed_ceff_and_locked_test(self):
        result = run_validation(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertAlmostEqual(TOTAL_CEFF_J_K, 8_840_700.0)
        self.assertEqual(result["test_windows_per_floor"], 62)
        for values in result["floors"].values():
            self.assertGreaterEqual(values["h_out_w_k"], 0.0)
            self.assertGreaterEqual(values["h_mass_w_k"], 0.0)


if __name__ == "__main__": unittest.main()
