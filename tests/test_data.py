import unittest

from model.common.data import ROOT, load_config, load_inputs


class DataTests(unittest.TestCase):
    def test_only_declared_real_sources_are_loaded(self):
        config = load_config()
        self.assertEqual(
            set(config) - {
                "timestamp_column", "outdoor_column", "floor_columns",
                "design_validation_spaces", "design_baseline_spaces",
            },
            {"temperature_csv", "weather_csv", "rdf_file"},
        )
        self.assertEqual(
            config["design_validation_spaces"],
            ["2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR", "4F412"],
        )
        self.assertEqual(config["design_baseline_spaces"], ["4F412"])
        for key in ("temperature_csv", "weather_csv", "rdf_file"):
            self.assertTrue((ROOT / config[key]).is_file())
        frame, qa = load_inputs(config)
        self.assertEqual(len(frame), 421)
        self.assertIn("4F412", frame.columns)
        self.assertNotIn("4F411", frame.columns)
        self.assertFalse(frame["4F412"].isna().any())
        self.assertEqual(qa["weather_temperature_rmse_c"], 0.0)


if __name__ == "__main__": unittest.main()
