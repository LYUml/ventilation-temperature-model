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


def _validate(config: dict[str, Any]) -> None:
    floors = config["floors"]
    if len(floors) != 3:
        raise ValueError("workflow test requires exactly three floors")
    names = [floor["name"] for floor in floors]
    if len(set(names)) != len(names):
        raise ValueError("floor names must be unique")
    if any(float(floor[key]) <= 0 for floor in floors for key in ("height_m", "volume_m3", "exterior_wall_area_m2")):
        raise ValueError("floor height, volume and wall area must be positive")
    paths = config["leakage"]["paths"]
    if len(paths) != 2 or abs(sum(float(p["wall_area_fraction"]) for p in paths) - 1.0) > 1e-9:
        raise ValueError("two path fractions must add to one")


def _replace_section(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


def render_three_floor_project(config: dict[str, Any]) -> str:
    _validate(config)
    floors = config["floors"]
    outdoor = config["outdoor"]
    leakage = config["leakage"]
    paths = leakage["paths"]
    normalized = cm2_per_m2_to_m2_per_m2(float(leakage["normalized_ela_cm2_per_m2"]))
    clam, cturb = contam_leak3_coefficients(
        normalized,
        float(leakage["reference_pressure_pa"]),
        float(leakage["discharge_coefficient"]),
        float(leakage["flow_exponent"]),
    )
    text = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "ELA one-zone automated smoke test", "ELA three-floor hypothetical workflow test"
    )
    zone_lines = [
        "3 ! zones:",
        "! Z#  f  s#  c#  k#  l#  relHt    Vol  T0  P0  name  clr uH uT uP uV axs cdvf <cdvfName> cfd <cfdName> <1dData:>",
    ]
    for zone_number, floor in enumerate(floors, start=1):
        zone_lines.append(
            f"   {zone_number}  3   0   0   0   1   {float(floor['base_height_m']):.6g} "
            f"{float(floor['volume_m3']):.6g} {float(floor['indoor_temperature_c']) + 273.15:.8g} 0 "
            f"{floor['name']} -1 0 2 0 0 0 0 0"
        )
    zone_lines.append("-999\n")
    text = _replace_section(text, "1 ! zones:\n", "0 ! initial zone concentrations:", "\n".join(zone_lines))

    path_lines = [
        "6 ! flow paths:",
        "! P#    f  n#  m#  e#  f#  w#  a#  s#  c#  l#    X       Y      relHt  mult wPset wPmod wazm Fahs Xmax Xmin icn dir u[4] cdvf <cdvfName> cfd <cfdData[4]>",
    ]
    path_number = 1
    for zone_number, floor in enumerate(floors, start=1):
        for path in paths:
            absolute_height = float(floor["base_height_m"]) + float(path["relative_height_m"])
            represented_area = float(floor["exterior_wall_area_m2"]) * float(path["wall_area_fraction"])
            path_lines.append(
                f"   {path_number}    0  -1   {zone_number}   1   0   0   0   0   0   1   "
                f"0.000   0.000 {absolute_height:.6g} {represented_area:.6g} 0 0 90 0 0 0  23  5 -1 0 0 0 0 0 0"
            )
            path_number += 1
    path_lines.append("-999\n")
    text = _replace_section(text, "2 ! flow paths:\n", "0 ! duct junctions:", "\n".join(path_lines))

    building_height = max(float(f["base_height_m"]) + float(f["height_m"]) for f in floors)
    replacements = {
        "INDOOR_K": float(floors[0]["indoor_temperature_c"]) + 273.15,
        "OUTDOOR_K": float(outdoor["temperature_c"]) + 273.15,
        "PRESSURE_PA": float(outdoor["barometric_pressure_pa"]),
        "WIND_SPEED": float(outdoor["wind_speed_m_s"]),
        "WIND_DIRECTION": float(outdoor["wind_direction_deg"]),
        "ROOM_HEIGHT_M": building_height,
        "CLAM": clam,
        "CTURB": cturb,
        "FLOW_EXPONENT": float(leakage["flow_exponent"]),
        "DISCHARGE_COEFFICIENT": float(leakage["discharge_coefficient"]),
        "REFERENCE_PRESSURE_PA": float(leakage["reference_pressure_pa"]),
        "NORMALIZED_ELA_M2_PER_M2": max(normalized, 1e-12),
    }
    for key, value in replacements.items():
        text = text.replace("{" + key + "}", f"{value:.12g}")
    unused = ("ROOM_VOLUME_M3", "PATH1_HEIGHT_M", "PATH2_HEIGHT_M", "PATH1_WALL_AREA_M2", "PATH2_WALL_AREA_M2")
    for key in unused:
        text = text.replace("{" + key + "}", "0")
    if "{" in text or "}" in text:
        raise RuntimeError("unresolved placeholder in generated project")
    return text


def _read_results(database: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with sqlite3.connect(database) as connection:
        flows = connection.execute(
            "SELECT PathNumber, dP, Flow0 FROM LINKPATHDATA ORDER BY PathNumber"
        ).fetchall()
        zones = connection.execute(
            "SELECT ZoneNumber, Pressure, Density FROM NODEZONEFLOWDATA ORDER BY ZoneNumber"
        ).fetchall()
        ambient = connection.execute(
            "SELECT Tambt, Pambt FROM AMBIENTCONDITIONS ORDER BY TimeID LIMIT 1"
        ).fetchone()
    if len(flows) != 6 or len(zones) != 3 or ambient is None:
        raise RuntimeError("unexpected three-floor CONTAM result shape")
    ambient_density = air_density_kg_m3(float(ambient[0]), float(ambient[1]))
    density_by_zone = {int(number): float(density) for number, _, density in zones}
    path_rows: list[dict[str, Any]] = []
    floor_rows: list[dict[str, Any]] = []
    normalized = float(config["leakage"]["normalized_ela_cm2_per_m2"])
    for path_index, (number, dp, mass) in enumerate(flows):
        floor_index, local_path_index = divmod(path_index, 2)
        floor = config["floors"][floor_index]
        path = config["leakage"]["paths"][local_path_index]
        zone_number = floor_index + 1
        mass = float(mass)
        density = ambient_density if mass >= 0 else density_by_zone[zone_number]
        volume = mass / density
        area = float(floor["exterior_wall_area_m2"]) * float(path["wall_area_fraction"])
        path_rows.append({
            "path_number": int(number), "floor": floor["name"], "path": path["name"],
            "absolute_height_m": float(floor["base_height_m"]) + float(path["relative_height_m"]),
            "path_ela_cm2": normalized * area, "pressure_difference_pa": float(dp),
            "mass_flow_kg_s": mass, "volume_flow_m3_s": volume,
            "direction": "outdoor_to_floor" if mass >= 0 else "floor_to_outdoor",
        })
    for zone_number, floor in enumerate(config["floors"], start=1):
        selected = [row for row in path_rows if row["floor"] == floor["name"]]
        inflow_mass = sum(max(row["mass_flow_kg_s"], 0.0) for row in selected)
        outflow_mass = sum(max(-row["mass_flow_kg_s"], 0.0) for row in selected)
        inflow_volume = sum(max(row["volume_flow_m3_s"], 0.0) for row in selected)
        scale = max(inflow_mass, outflow_mass, 1e-15)
        pressure = float(zones[zone_number - 1][1])
        floor_rows.append({
            "floor": floor["name"], "base_height_m": float(floor["base_height_m"]),
            "indoor_temperature_c": float(floor["indoor_temperature_c"]),
            "total_ela_cm2": normalized * float(floor["exterior_wall_area_m2"]),
            "zone_pressure_pa_relative": pressure, "inflow_mass_kg_s": inflow_mass,
            "outflow_mass_kg_s": outflow_mass,
            "relative_mass_imbalance": abs(inflow_mass - outflow_mass) / scale,
            "infiltration_volume_flow_m3_s": inflow_volume,
            "infiltration_ach_1_per_h": 3600.0 * inflow_volume / float(floor["volume_m3"]),
        })
    return path_rows, floor_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_three_floor_case(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    _validate(config)
    output_dir = PROJECT_ROOT / config.get("output_root", "outputs") / config["case_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.glob("generated.*"):
        item.unlink()
    project = output_dir / "generated.prj"
    project.write_text(render_three_floor_project(config), encoding="utf-8", newline="\n")
    process = subprocess.run(
        [config["contam_executable"], str(project)], cwd=output_dir, capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    (output_dir / "contam_stdout.log").write_text((process.stdout or "") + (process.stderr or ""), encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"ContamX failed with exit code {process.returncode}")
    database = output_dir / "generated.sqlite3"
    if not database.exists() or not (output_dir / "generated.sim").exists():
        raise RuntimeError("ContamX did not produce expected result files")
    if (output_dir / "generated.xlog").exists():
        shutil.copyfile(output_dir / "generated.xlog", output_dir / "contam_xlog.log")
    path_rows, floor_rows = _read_results(database, config)
    _write_csv(output_dir / "path_flows.csv", path_rows)
    _write_csv(output_dir / "floor_summary.csv", floor_rows)
    summary = {
        "case_name": config["case_name"], "status": "workflow_test_only",
        "assumption_notice": config["assumption_notice"],
        "outdoor": config["outdoor"], "floors": floor_rows,
        "checks": {
            "contam_exit_code": process.returncode,
            "all_floor_mass_balances_below_0_1_percent": all(row["relative_mass_imbalance"] < 0.001 for row in floor_rows),
            "all_floors_have_inflow_and_outflow": all(row["inflow_mass_kg_s"] > 0 and row["outflow_mass_kg_s"] > 0 for row in floor_rows),
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hypothetical three-floor ELA-CONTAM case")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = run_three_floor_case(args.config)
    for floor in summary["floors"]:
        print(f"{floor['floor']}: ACH={floor['infiltration_ach_1_per_h']:.6f} 1/h, imbalance={floor['relative_mass_imbalance']:.3e}")


if __name__ == "__main__":
    main()
