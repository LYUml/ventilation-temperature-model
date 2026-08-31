import unittest

import pandas as pd

from src.temperature_model import PROJECT_ROOT
from src.weather_data import WEATHER_COLUMNS, merge_hourly_weather


class WeatherDataTests(unittest.TestCase):
    def test_station_data_align_exactly(self):
        frame = pd.read_csv(PROJECT_ROOT.parent / "data" / "TEMPERATURE-rev.csv")
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
        merged, qa = merge_hourly_weather(frame, "Timestamp", PROJECT_ROOT.parent / "data" / "543990(1).csv")
        self.assertEqual(qa["matched_rows"], 421)
        self.assertAlmostEqual(qa["station_vs_building_outdoor_rmse_c"], 0.0)
        self.assertFalse(merged[WEATHER_COLUMNS].isna().any().any())


if __name__ == "__main__":
    unittest.main()
