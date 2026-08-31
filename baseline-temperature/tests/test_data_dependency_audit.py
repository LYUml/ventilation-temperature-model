import json
import unittest

from src.rdf_building import extract_building
from src.temperature_model import PROJECT_ROOT


class DataDependencyAuditTests(unittest.TestCase):
    def test_declared_sources_exist_and_rdf_topology_is_real(self):
        config = json.loads((PROJECT_ROOT / "configs" / "temperature_model.json").read_text(encoding="utf-8"))
        for key in ("input_csv", "weather_csv", "rdf_file"):
            self.assertIn(key, config)
            self.assertTrue((PROJECT_ROOT / config[key]).resolve().is_file(), key)
        building = extract_building((PROJECT_ROOT / config["rdf_file"]).resolve())
        floor4_neighbours = {edge["space"] for edge in building["corridors"]["4FCORRIDOR"]["adjacent_spaces"]}
        self.assertIn("5F510", floor4_neighbours)
        self.assertNotIn("4F411", floor4_neighbours)
        self.assertIn("3FCORRIDOR", floor4_neighbours)


if __name__ == "__main__":
    unittest.main()
