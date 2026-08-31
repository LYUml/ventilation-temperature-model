import unittest

from src.temperature_model import PROJECT_ROOT
from src.validate_rdf_msts import run_validation


class RdfMstsTests(unittest.TestCase):
    def test_single_stable_transition_and_locked_test(self):
        result = run_validation(PROJECT_ROOT / "configs" / "temperature_model.json")
        self.assertEqual(result["test_windows_per_floor"], 62)
        self.assertEqual(len(result["state_definition"]), 41)
        self.assertLessEqual(result["selected"]["transition_spectral_radius"], 1.02)
        self.assertIn("One RDF-masked companion-form latent thermal transition system", result["unification"])
        self.assertTrue(result["rdf"]["source"].endswith("NBuilding.rdf"))
        self.assertEqual(result["rdf"]["space_count"], 37)
        floor4 = result["rdf"]["floor_constraints"][2]
        self.assertIn("5F510", floor4["measured_adjacent_spaces"])
        self.assertNotIn("4F411", floor4["measured_adjacent_spaces"])


if __name__ == "__main__":
    unittest.main()
