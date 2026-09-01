import unittest

from model.kernel.run import run as run_kernel
from model.rc_narx_ridge.run import run as run_rc_narx_ridge
from model.rdf_rc_narx_ridge.run import run as run_rdf_rc_narx_ridge
from model.rdf_kernel_narx_ridge.run import run as run_rdf_kernel_narx_ridge


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = run_kernel(); cls.rc = run_rc_narx_ridge(); cls.rdf_rc = run_rdf_rc_narx_ridge(); cls.rdf_kernel = run_rdf_kernel_narx_ridge()

    def test_locked_test_is_shared(self):
        self.assertEqual(self.kernel["test_windows_per_floor"], 62)
        self.assertEqual(self.rdf_kernel["test_windows_per_floor"], 62)
        self.assertEqual(self.rdf_rc["test_windows_per_floor"], 62)
        self.assertEqual(self.rc["test_windows_per_floor"], 62)

    def test_rc_narx_ridge_uses_fitted_effective_rc(self):
        for values in self.rc["fitted_rc"].values():
            self.assertGreater(values["response_per_h"], 0.0)
            self.assertLess(values["response_per_h"], 1.0)

    def test_rdf_is_really_consumed(self):
        self.assertTrue(self.rdf_kernel["rdf"]["source"].endswith("building.rdf"))
        self.assertNotIn("4F411", self.rdf_kernel["rdf"]["topology"][2]["measured_spaces"])

    def test_original_unified_model_beats_kernel(self):
        self.assertLess(self.rdf_kernel["slow_state_model"]["mean_floor_rmse_c"], self.kernel["mean_floor_rmse_c"])
        self.assertLess(self.rdf_kernel["slow_state_model"]["selected"]["spectral_radius"], 1.0)

    def test_rdf_rc_narx_ridge_beats_rc(self):
        self.assertLess(self.rdf_rc["mean_floor_rmse_c"], self.rc["mean_floor_rmse_c"])


if __name__ == "__main__": unittest.main()
