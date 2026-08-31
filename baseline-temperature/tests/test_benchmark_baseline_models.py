import unittest
from src.benchmark_baseline_models import run_benchmark
from src.temperature_model import PROJECT_ROOT


class BaselineModelBenchmarkTests(unittest.TestCase):
    def test_all_models_are_reported_on_locked_test(self):
        result = run_benchmark(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["test_windows_per_floor"], 62)
        for name in ("kernel", "2r2c", "gaussian_process", "gradient_boosting", "narx_ridge"):
            self.assertIn(name, result["models"])


if __name__ == "__main__": unittest.main()
