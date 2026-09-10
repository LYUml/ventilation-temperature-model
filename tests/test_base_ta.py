import tempfile
import unittest
from pathlib import Path

import numpy as np

from base_ta import BaseTaModel, RdfMz5r1cModel, calculateBaseTa
from model.common.data import ROOT, load_config, load_inputs


class BaseTaApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        frame, _ = load_inputs(cls.config)
        cls.weather_columns = [
            "Timestamp",
            "TEM",
            "PRS",
            "RHU",
            "PRE_1h",
            "WIN_S_Avg_2mi",
            "WIN_D_Avg_2mi",
            "diffuse",
            "direct",
        ]
        cls.indoor_columns = ["Timestamp", *cls.config["floor_columns"]]
        cls.history = frame.iloc[:-24].reset_index(drop=True)
        cls.forecast = frame.iloc[-24:].reset_index(drop=True)

    def _fit(self, method):
        return BaseTaModel(method=method).fit(
            ROOT / self.config["rdf_file"],
            self.history[self.weather_columns],
            self.history[self.indoor_columns],
            self.config["floor_columns"],
        )

    def test_both_methods_predict_every_requested_space(self):
        for method in ("rdf_rc_narx_ridge", "rdf_kernel_narx_ridge"):
            with self.subTest(method=method):
                result = self._fit(method).predict(self.forecast[self.weather_columns])
                self.assertEqual(list(result), self.config["floor_columns"])
                for values in result.values():
                    self.assertEqual(len(values), 24)
                    self.assertTrue(np.isfinite(values).all())

    def test_saved_model_reproduces_prediction(self):
        model = self._fit("rdf_rc_narx_ridge")
        expected = model.predict(self.forecast[self.weather_columns])
        with tempfile.TemporaryDirectory() as directory:
            restored = BaseTaModel.load(model.save(Path(directory) / "base_ta.pkl"))
            actual = restored.predict(self.forecast[self.weather_columns])
        for space in expected:
            np.testing.assert_allclose(actual[space], expected[space])

    def test_one_call_interface_keeps_histories_separate(self):
        result = calculateBaseTa(
            ROOT / self.config["rdf_file"],
            self.history[self.weather_columns],
            self.history[self.indoor_columns],
            self.forecast[self.weather_columns],
            ["2FCORRIDOR"],
            method="rdf_rc_narx_ridge",
        )
        self.assertEqual(list(result), ["2FCORRIDOR"])
        self.assertEqual(len(result["2FCORRIDOR"]), 24)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            BaseTaModel(method="unknown")

    def test_mz5r1c_is_design_stage_and_finite(self):
        model = RdfMz5r1cModel(parameters={"allow_non_project_assumptions": True})
        result = model.simulate(
            ROOT / self.config["rdf_file"], self.forecast[self.weather_columns], ["2FCORRIDOR"]
        )
        self.assertFalse(model.last_metadata["requires_measured_indoor_history"])
        self.assertTrue(model.last_metadata["warmup"]["converged"])
        self.assertEqual(len(result["2FCORRIDOR"]), 24)
        self.assertTrue(np.isfinite(result["2FCORRIDOR"]).all())

    def test_design_model_rejects_silent_assumptions_by_default(self):
        rdf = ROOT / self.config["rdf_file"]
        weather = self.forecast[self.weather_columns]
        with self.assertRaisesRegex(ValueError, "Strict project-data mode"):
            RdfMz5r1cModel().simulate(rdf, weather, ["2FCORRIDOR"])


if __name__ == "__main__":
    unittest.main()
