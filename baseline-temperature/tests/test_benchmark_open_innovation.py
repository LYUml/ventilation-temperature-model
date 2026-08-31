import unittest

from src.benchmark_open_innovation import run_benchmark
from src.temperature_model import PROJECT_ROOT


class OpenInnovationBenchmarkTests(unittest.TestCase):
    def test_locked_comparison_and_original_model(self):
        result = run_benchmark(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["test_windows_per_floor"], 62)
        for name in ("kernel", "arx_weather_rooms", "mimo_n4sid", "thermal_innovation", "organic_fusion"):
            self.assertIn(name, result["models"])
        for weights in result["selected"]["fusion_weights"].values():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        self.assertLess(result["models"]["organic_fusion"]["mean_floor_rmse_c"], result["models"]["kernel"]["mean_floor_rmse_c"])


if __name__ == "__main__":
    unittest.main()
