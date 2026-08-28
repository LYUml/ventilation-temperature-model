from __future__ import annotations

import unittest

from src.rdf_building import extract_building
from src.temperature_model import PROJECT_ROOT


class RdfBuildingTests(unittest.TestCase):
    def test_extracts_real_corridor_geometry(self) -> None:
        result = extract_building(PROJECT_ROOT.parent / "data" / "NBuilding.rdf")
        self.assertEqual(result["explicit_door_interface_count"], 0)
        for index, name in enumerate(("2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR")):
            corridor = result["corridors"][name]
            self.assertAlmostEqual(corridor["floor_area_m2"], 53.58)
            self.assertAlmostEqual(corridor["volume_m3"], 214.32)
            self.assertAlmostEqual(corridor["height_m"], 4.0)
            self.assertAlmostEqual(corridor["base_height_m"], index * 4.0)
            self.assertGreater(corridor["exterior_wall_area_m2"], 0.0)
            self.assertGreater(corridor["exterior_ua_w_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
