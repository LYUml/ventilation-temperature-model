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

    def test_design_settings_and_exterior_windows_are_extracted(self):
        config = load_config(); building = extract_building(ROOT / config["rdf_file"])
        corridor = building["spaces"]["2FCORRIDOR"]
        self.assertEqual(corridor["settings"]["type"], "OFFICE")
        self.assertEqual(corridor["settings"]["zone_infiltration"], 0.5)
        self.assertEqual(corridor["settings"]["zone_c_temp"], 26.0)
        self.assertLess(corridor["operable_window_area_m2"], 15.109)

    def test_real_window_geometry_and_hourly_schedules_are_extracted(self):
        config = load_config(); building = extract_building(ROOT / config["rdf_file"])
        office = building["spaces"]["2F215"]
        window = office["exterior_windows"][0]
        self.assertEqual(office["north_direction_deg"], 0.0)
        self.assertEqual(window["normal_xyz"], (1.0, 0.0, 0.0))
        self.assertEqual(window["shgc"], 0.3)
        weekly = building["weekly_schedules"][office["settings"]["zone_equipment"]]
        monday = building["daily_schedules"][weekly["days"]["monday"]]
        self.assertEqual(monday["unit"], "W/m2")
        self.assertEqual(len(monday["values"]), 24)
        self.assertAlmostEqual(monday["values"][10], 13.08411215)


if __name__ == "__main__": unittest.main()
