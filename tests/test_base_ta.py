import tempfile
import unittest
from pathlib import Path

import numpy as np

from base_ta import BaseTaModel, RdfMz5r1cModel, calculateBaseTa
from model.common.data import ROOT, load_config, load_inputs
from rdf_weather_temperature import simulateIndoorTemperature


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
        model = RdfMz5r1cModel(
            parameters={"allow_non_project_assumptions": True},
            baseline_spaces=["4F412"],
        )
        result = model.simulate(
            ROOT / self.config["rdf_file"], self.forecast[self.weather_columns],
            ["2FCORRIDOR", "4F412"],
        )
        self.assertFalse(model.last_metadata["requires_measured_indoor_history"])
        self.assertEqual(model.last_metadata["explicit_baseline_spaces"], ["4F412"])
        self.assertTrue(model.last_metadata["warmup"]["converged"])
        self.assertEqual(len(result["2FCORRIDOR"]), 24)
        self.assertTrue(np.isfinite(result["2FCORRIDOR"]).all())
        self.assertEqual(len(result["4F412"]), 24)
        self.assertTrue(np.isfinite(result["4F412"]).all())

    def test_baseline_space_rejects_unknown_rdf_name(self):
        model = RdfMz5r1cModel(
            parameters={"allow_non_project_assumptions": True},
            baseline_spaces=["4F_NOT_A_SPACE"],
        )
        with self.assertRaisesRegex(ValueError, "RDF is missing baseline spaces"):
            model.simulate(
                ROOT / self.config["rdf_file"],
                self.forecast[self.weather_columns],
                ["2FCORRIDOR"],
            )

    def test_design_model_rejects_silent_assumptions_by_default(self):
        rdf = ROOT / self.config["rdf_file"]
        weather = self.forecast[self.weather_columns]
        with self.assertRaisesRegex(ValueError, "Strict project-data mode"):
            RdfMz5r1cModel().simulate(rdf, weather, ["2FCORRIDOR"])

    def test_compact_rdf_weather_interface_returns_all_spaces_by_default(self):
        result = simulateIndoorTemperature(
            ROOT / self.config["rdf_file"],
            ROOT / "data/weather_54399.csv",
        )
        self.assertIsInstance(result, dict)
        self.assertIn("4F412", result)
        self.assertIn("2FCORRIDOR", result)
        for values in result.values():
            self.assertEqual(len(values), 736)
            self.assertTrue(np.isfinite(values).all())

    def test_compact_interface_accepts_canonical_weather_csv(self):
        weather = self.forecast
        canonical = weather[["Timestamp", "TEM", "diffuse", "direct"]].rename(
            columns={
                "Timestamp": "timestamp",
                "TEM": "outdoor_temperature_c",
                "diffuse": "diffuse_solar_w_m2",
                "direct": "direct_solar_w_m2",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.csv"
            canonical.to_csv(path, index=False)
            result = simulateIndoorTemperature(
                ROOT / self.config["rdf_file"], path, "4F412"
            )
        self.assertEqual(list(result), ["4F412"])
        self.assertEqual(len(result["4F412"]), len(canonical))
        self.assertTrue(np.isfinite(result["4F412"]).all())

    def test_reusable_interface_accepts_another_rdf_space(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.csv"
            self.forecast[["Timestamp", "TEM", "diffuse", "direct"]].rename(
                columns={
                    "Timestamp": "timestamp",
                    "TEM": "outdoor_temperature_c",
                    "diffuse": "diffuse_solar_w_m2",
                    "direct": "direct_solar_w_m2",
                }
            ).to_csv(path, index=False)
            result = simulateIndoorTemperature(
                ROOT / self.config["rdf_file"], path, "2FCORRIDOR"
            )
        self.assertEqual(list(result), ["2FCORRIDOR"])
        self.assertEqual(len(result["2FCORRIDOR"]), len(self.forecast))
        self.assertTrue(np.isfinite(result["2FCORRIDOR"]).all())

if __name__ == "__main__":
    unittest.main()
