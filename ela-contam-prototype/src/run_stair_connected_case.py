from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from .ela import air_density_kg_m3, cm2_per_m2_to_m2_per_m2, contam_leak3_coefficients
from .run_case import PROJECT_ROOT, TEMPLATE_PATH
from .run_three_floor_case import _replace_section


def _flow_element_line(number: int, name: str, normalized_ela: float, leakage: dict[str, Any]) -> str:
    clam, cturb = contam_leak3_coefficients(
        normalized_ela,
        float(leakage["reference_pressure_pa"]),
        float(leakage["discharge_coefficient"]),
        float(leakage["flow_exponent"]),
    )
    return (
        f"{number} 23 plr_leak3 {name}\n{name}\n"
        f" {clam:.12g} {cturb:.12g} {float(leakage['flow_exponent']):.12g} "
        f"{float(leakage['discharge_coefficient']):.12g} "
        f"{float(leakage['reference_pressure_pa']):.12g} 0 0 {normalized_ela:.12g} 2 2 2 0"
    )


def _validate(config: dict[str, Any], scenario_name: str) -> None:
    if len(config["floors"]) != 3 or len(config["stairs"]) != 3:
        raise ValueError("stair-connected case requires three corridors and three stairs")
    if scenario_name not in config["connection_scenarios"]:
        raise ValueError(f"unknown connection scenario: {scenario_name}")
    scenario = config["connection_scenarios"][scenario_name]
    for key in ("corridor_stair_ela_m2", "vertical_stair_ela_m2"):
        if float(scenario[key]) <= 0:
            raise ValueError(f"{key} must be positive")


def render_project(config: dict[str, Any], scenario_name: str) -> tuple[str, list[dict[str, Any]]]:
    _validate(config, scenario_name)
    floors = config["floors"]
    stairs = config["stairs"]
    leakage = config["leakage"]
    scenario = config["connection_scenarios"][scenario_name]
    normalized = cm2_per_m2_to_m2_per_m2(float(leakage["normalized_ela_cm2_per_m2"]))
    text = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "ELA one-zone automated smoke test", f"RDF stair-connected scenario: {scenario_name}"
    )

    elements = [
        "2 ! flow elements:",
        _flow_element_line(1, "ext_ela", normalized, leakage),
        _flow_element_line(2, "int_ela", 1.0, leakage),
        "-999\n",
    ]
    text = _replace_section(text, "1 ! flow elements:\n", "0 ! duct elements:", "\n".join(elements))

    zones = [
        "6 ! zones:",
        "! Z#  f  s#  c#  k#  l#  relHt    Vol  T0  P0  name  clr uH uT uP uV axs cdvf <cdvfName> cfd <cfdName> <1dData:>",
    ]
    zone_numbers: dict[str, int] = {}
    for number, zone in enumerate([*floors, *stairs], start=1):
        zone_numbers[zone["name"]] = number
        zones.append(
            f"   {number}  3   0   0   0   1   {float(zone['base_height_m']):.6g} "
            f"{float(zone['volume_m3']):.6g} {float(zone['indoor_temperature_c']) + 273.15:.8g} 0 "
            f"{zone['name']} -1 0 2 0 0 0 0 0"
        )
    zones.append("-999\n")
    text = _replace_section(text, "1 ! zones:\n", "0 ! initial zone concentrations:", "\n".join(zones))

    definitions: list[dict[str, Any]] = []
    for floor in floors:
        for item in leakage["paths"]:
            definitions.append(
                {
                    "kind": "exterior",
                    "from": "OUTDOOR",
                    "to": floor["name"],
                    "from_zone": -1,
                    "to_zone": zone_numbers[floor["name"]],
                    "element": 1,
                    "height_m": float(floor["base_height_m"]) + float(item["relative_height_m"]),
                    "multiplier": float(floor["exterior_wall_area_m2"]) * float(item["wall_area_fraction"]),
                    "ela_m2": normalized * float(floor["exterior_wall_area_m2"]) * float(item["wall_area_fraction"]),
                }
            )
    for floor, stair in zip(floors, stairs):
        definitions.append(
            {
                "kind": "corridor_stair",
                "from": floor["name"],
                "to": stair["name"],
                "from_zone": zone_numbers[floor["name"]],
                "to_zone": zone_numbers[stair["name"]],
                "element": 2,
                "height_m": float(floor["base_height_m"]) + float(floor["height_m"]) / 2,
                "multiplier": float(scenario["corridor_stair_ela_m2"]),
                "ela_m2": float(scenario["corridor_stair_ela_m2"]),
            }
        )
    for lower, upper in zip(stairs[:-1], stairs[1:]):
        definitions.append(
            {
                "kind": "vertical_stair",
                "from": lower["name"],
                "to": upper["name"],
                "from_zone": zone_numbers[lower["name"]],
                "to_zone": zone_numbers[upper["name"]],
                "element": 2,
                "height_m": float(upper["base_height_m"]),
                "multiplier": float(scenario["vertical_stair_ela_m2"]),
                "ela_m2": float(scenario["vertical_stair_ela_m2"]),
            }
        )
    paths = [
        f"{len(definitions)} ! flow paths:",
        "! P#    f  n#  m#  e#  f#  w#  a#  s#  c#  l#    X       Y      relHt  mult wPset wPmod wazm Fahs Xmax Xmin icn dir u[4] cdvf <cdvfName> cfd <cfdData[4]>",
    ]
    for number, item in enumerate(definitions, start=1):
        paths.append(
            f"   {number}    0  {item['from_zone']}   {item['to_zone']}   {item['element']}   0   0   0   0   0   1   "
            f"0.000   0.000 {item['height_m']:.6g} {item['multiplier']:.12g} 0 0 90 0 0 0  23  5 -1 0 0 0 0 0 0"
        )
    paths.append("-999\n")
    text = _replace_section(text, "2 ! flow paths:\n", "0 ! duct junctions:", "\n".join(paths))

    outdoor = config["outdoor"]
    replacements = {
        "INDOOR_K": float(floors[0]["indoor_temperature_c"]) + 273.15,
        "OUTDOOR_K": float(outdoor["temperature_c"]) + 273.15,
        "PRESSURE_PA": float(outdoor["barometric_pressure_pa"]),
        "WIND_SPEED": float(outdoor["wind_speed_m_s"]),
        "WIND_DIRECTION": float(outdoor["wind_direction_deg"]),
        "ROOM_HEIGHT_M": max(float(s["base_height_m"]) + float(s["height_m"]) for s in stairs),
    }
    for key, value in replacements.items():
        text = text.replace("{" + key + "}", f"{value:.12g}")
    for key in (
        "CLAM", "CTURB", "FLOW_EXPONENT", "DISCHARGE_COEFFICIENT", "REFERENCE_PRESSURE_PA",
        "NORMALIZED_ELA_M2_PER_M2", "ROOM_VOLUME_M3", "PATH1_HEIGHT_M", "PATH2_HEIGHT_M",
        "PATH1_WALL_AREA_M2", "PATH2_WALL_AREA_M2",
    ):
        text = text.replace("{" + key + "}", "0")
    if "{" in text or "}" in text:
        raise RuntimeError("unresolved placeholder in generated project")
    return text, definitions


def run_case(config_path: Path, scenario_name: str) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    project_text, definitions = render_project(config, scenario_name)
    output_dir = PROJECT_ROOT / config.get("output_root", "outputs") / f"{config['case_name']}_{scenario_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.glob("generated.*"):
        item.unlink()
    project = output_dir / "generated.prj"
    project.write_text(project_text, encoding="utf-8", newline="\n")
    process = subprocess.run(
        [config["contam_executable"], str(project)], cwd=output_dir, capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    (output_dir / "contam_stdout.log").write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"ContamX failed with exit code {process.returncode}")
    database = output_dir / "generated.sqlite3"
    if not database.exists():
        raise RuntimeError("ContamX did not produce a database")
    if (output_dir / "generated.xlog").exists():
        shutil.copyfile(output_dir / "generated.xlog", output_dir / "contam_xlog.log")
    with sqlite3.connect(database) as connection:
        flows = connection.execute("SELECT PathNumber, dP, Flow0 FROM LINKPATHDATA ORDER BY PathNumber").fetchall()
        zones = connection.execute("SELECT ZoneNumber, Pressure, Density FROM NODEZONEFLOWDATA ORDER BY ZoneNumber").fetchall()
        ambient = connection.execute("SELECT Tambt, Pambt FROM AMBIENTCONDITIONS ORDER BY TimeID LIMIT 1").fetchone()
    if len(flows) != len(definitions) or len(zones) != 6 or ambient is None:
        raise RuntimeError("unexpected stair-connected result shape")
    ambient_density = air_density_kg_m3(float(ambient[0]), float(ambient[1]))
    density = {int(number): float(value) for number, _, value in zones}
    rows = []
    for definition, (number, dp, mass) in zip(definitions, flows):
        mass = float(mass)
        source_density = ambient_density if definition["from_zone"] == -1 or mass >= 0 else density[definition["to_zone"]]
        rows.append({
            "path_number": int(number), "kind": definition["kind"], "from": definition["from"], "to": definition["to"],
            "ela_m2": definition["ela_m2"], "height_m": definition["height_m"], "pressure_difference_pa": float(dp),
            "mass_flow_kg_s": mass, "volume_flow_m3_s": mass / source_density,
            "positive_direction": f"{definition['from']} -> {definition['to']}",
        })
    with (output_dir / "path_flows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    net = {number: 0.0 for number in range(1, 7)}
    for definition, row in zip(definitions, rows):
        mass = row["mass_flow_kg_s"]
        if definition["from_zone"] != -1:
            net[definition["from_zone"]] -= mass
        net[definition["to_zone"]] += mass
    scale = max(max(abs(row["mass_flow_kg_s"]) for row in rows), 1e-15)
    summary = {
        "case_name": config["case_name"], "scenario": scenario_name,
        "status": "RDF geometry with assumed connection ELA",
        "assumptions": config["connection_scenarios"][scenario_name],
        "zones": [*config["floors"], *config["stairs"]],
        "paths": rows,
        "checks": {
            "contam_exit_code": process.returncode,
            "maximum_relative_zone_mass_imbalance": max(abs(value) for value in net.values()) / scale,
            "all_zones_below_0_1_percent": all(abs(value) / scale < 0.001 for value in net.values()),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RDF stair-connected CONTAM scenarios")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", choices=["closed", "partial", "open", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    names = list(config["connection_scenarios"]) if args.scenario == "all" else [args.scenario]
    for name in names:
        result = run_case(args.config, name)
        vertical = [abs(p["mass_flow_kg_s"]) for p in result["paths"] if p["kind"] == "vertical_stair"]
        print(f"{name}: max vertical flow={max(vertical):.6g} kg/s, balance={result['checks']['maximum_relative_zone_mass_imbalance']:.3e}")


if __name__ == "__main__":
    main()
