from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.temperature_model import PROJECT_ROOT
from src.weather_input import build_weather_input


class WeatherInputTests(unittest.TestCase):
    def test_zero_wind_is_explicit_and_time_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = build_weather_input(
                PROJECT_ROOT.parent / "data" / "TEMPERATURE-rev.csv",
                root / "weather.csv",
                root / "metadata.json",
            )
            frame = pd.read_csv(root / "weather.csv")
            self.assertEqual(len(frame), 421)
            self.assertTrue((frame["wind_speed_m_s"] == 0.0).all())
            self.assertIsNone(metadata["wind_source"])


if __name__ == "__main__":
    unittest.main()
