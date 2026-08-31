import unittest

from model.common.data import ROOT, load_config
from model.common.rdf import extract_building


class RdfTests(unittest.TestCase):
    def test_real_shared_elements_define_topology(self):
        config = load_config(); building = extract_building(ROOT / config["rdf_file"])
        self.assertEqual(building["space_count"], 37)
        neighbours = {edge["space"] for edge in building["corridors"]["4FCORRIDOR"]["adjacent_spaces"]}
        self.assertIn("3FCORRIDOR", neighbours)
        self.assertIn("5F510", neighbours)
        self.assertNotIn("4F411", neighbours)


if __name__ == "__main__": unittest.main()
