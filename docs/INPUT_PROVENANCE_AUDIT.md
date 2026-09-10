# Design-model input provenance audit

This audit is the acceptance gate for claims that a result uses only project data. A numerical value being physically reasonable or defined by a standard does not make it project data.

| Model input | Present in supplied project files? | Currently consumed correctly? | Classification / action |
|---|---:|---:|---|
| Hourly outdoor temperature, diffuse and direct radiation | Yes, weather CSV | Yes | Project-supplied measured weather series |
| Space floor area and volume | Yes, RDF | Yes | Project RDF data |
| Construction area and U-value | Yes, RDF | Yes | Project RDF data |
| Window area and U-value | Yes, RDF | Yes | Project RDF data |
| Window SHGC | Yes, at construction and template levels | **Resolved by RDF specificity** | Element value 0.3 is used for each actual window; zone-template value 0.45 is retained only as a program/compliance parameter. This matches the existing element-over-template treatment of window U-value (1.8 vs 2.2) |
| Background infiltration ACH | Yes, RDF | Yes | Project RDF design value; not an infiltration-test measurement |
| Heating/cooling setpoints and work hours | Yes, RDF | Yes | Project RDF design values |
| Hourly occupancy/equipment/lighting schedules | Yes, RDF schedule objects and `hourlyValuesJson` | **Yes** | All 36 non-zero space schedule references resolve to complete seven-day, 24-value schedules; incomplete references now raise an error |
| Window/surface 3-D geometry and normal vectors | Yes, RDF `geo:asWKT` and normal-vector fields | Partly | Per-window normals are parsed; WKT is verified present but is not yet needed by the heat model beyond area/normal |
| Building north direction | Yes, RDF `bes:hasNorthDirection_deg` | Parsed, not yet applied | Must be combined with surface normals and verified radiation semantics before facade solar is valid |
| Site latitude/longitude and solar-time convention | Not established by current parser/input contract | No | Required for project-specific sun position |
| Material-layer thickness, conductivity, density and specific heat | Not found in the current parsed input | No | Required to calculate project thermal capacitance and dynamic layer response |
| Door opening/airflow coefficients and schedules | Not established | No | Required for corridor/inter-zone air exchange |
| Mechanical ventilation and HVAC airflow/capacity schedules | Not fully established | No | Setpoint-only ideal control is not a complete project HVAC boundary |
| `0.5` vertical-window diffuse factor | Derived from an assumed 90° tilt | Yes in research mode | Standard geometric derivation, **not project data** until RDF surface tilt is consumed |
| `0.25` direct exposure factor | No | Yes in research mode | Research assumption, **not project data** |
| `165000 J/(m² K)` capacitance | No | Yes in research mode | ISO archetype default, **not project data** |
| Effective mass/internal-area factors `2.5` and `4.5` | No | Yes in research mode | ISO archetype defaults, **not project data** |

## Enforcement

`RdfMz5r1cModel` defaults to strict project-data mode. It raises an exception while any listed non-project input remains. Reproducing assumption-based research trials requires the explicit opt-in `allow_non_project_assumptions=True`; such output retains `project_data_compliant: false` in its metadata and must not be reported as an all-real-data result.

The previously reported MZ-5R1C mean RMSE of 0.852 °C used archetype load magnitudes. Replacing those with real RDF hourly schedules produced 0.988 °C. Applying the construction-over-template RDF precedence for SHGC then produces 0.738 °C. All remain non-compliant because the remaining solar, thermal-mass and HVAC/airflow boundary issues are unresolved; only 0.738 °C is the current partial-project-data research result.
