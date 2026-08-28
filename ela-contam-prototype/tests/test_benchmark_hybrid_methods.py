import unittest
from src.benchmark_hybrid_methods import run_hybrid_benchmark
from src.temperature_model import PROJECT_ROOT


class HybridMethodBenchmarkTests(unittest.TestCase):
    def test_open_source_and_proposed_methods_are_compared(self):
        result = run_hybrid_benchmark(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["test_windows_per_floor"], 62)
        self.assertIn("sysidentpy_polynomial_narx", result["models"])
        self.assertIn("proposed_hierarchical_rc_narx_ridge", result["models"])


if __name__ == "__main__": unittest.main()
