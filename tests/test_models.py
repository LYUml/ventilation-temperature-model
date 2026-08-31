import unittest

from model.kernel.run import run as run_kernel
from model.rdf_msts.run import run as run_rdf_msts


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = run_kernel(); cls.rdf = run_rdf_msts()

    def test_locked_test_is_shared(self):
        self.assertEqual(self.kernel["test_windows_per_floor"], 62)
        self.assertEqual(self.rdf["test_windows_per_floor"], 62)

    def test_rdf_is_really_consumed(self):
        self.assertTrue(self.rdf["rdf"]["source"].endswith("building.rdf"))
        self.assertNotIn("4F411", self.rdf["rdf"]["topology"][2]["measured_spaces"])

    def test_original_unified_model_beats_kernel(self):
        self.assertLess(self.rdf["slow_state_model"]["mean_floor_rmse_c"], self.kernel["mean_floor_rmse_c"])
        self.assertLess(self.rdf["slow_state_model"]["selected"]["spectral_radius"], 1.0)


if __name__ == "__main__": unittest.main()
