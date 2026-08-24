from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from .ela import (
    air_density_kg_m3,
    cm2_per_m2_to_m2_per_m2,
    contam_leak3_coefficients,
    total_ela_cm2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "one_zone_ela.prj.tpl"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    room = config["room"]
    leakage = config["leakage"]
    paths = leakage["paths"]
    if len(paths) != 2:
        raise ValueError("the minimal model requires exactly two leakage paths")
    if room["volume_m3"] <= 0 or room["exterior_wall_area_m2"] <= 0:
        raise ValueError("room volume and exterior wall area must be positive")
    fractions = [float(path["wall_area_fraction"]) for path in paths]
    if any(value <= 0 for value in fractions):
        raise ValueError("path wall-area fractions must be positive")
    if abs(sum(fractions) - 1.0) > 1.0e-9:
        raise ValueError("path wall-area fractions must add to 1.0")
    heights = [float(path["relative_height_m"]) for path in paths]
    if any(value < 0 or value > room["height_m"] for value in heights):
        raise ValueError("path heights must lie within the room height")


def render_project(config: dict[str, Any]) -> str:
    room = config["room"]
    outdoor = config["outdoor"]
    leakage = config["leakage"]
    paths = leakage["paths"]
    normalized = cm2_per_m2_to_m2_per_m2(
        float(leakage["normalized_ela_cm2_per_m2"])
    )
    clam, cturb = contam_leak3_coefficients(
        normalized,
        float(leakage["reference_pressure_pa"]),
        float(leakage["discharge_coefficient"]),
        float(leakage["flow_exponent"]),
    )
    numerical_normalized = max(normalized, 1.0e-12)
    replacements = {
        "INDOOR_K": float(room["indoor_temperature_c"]) + 273.15,
        "OUTDOOR_K": float(outdoor["temperature_c"]) + 273.15,
        "PRESSURE_PA": float(outdoor["barometric_pressure_pa"]),
        "WIND_SPEED": float(outdoor["wind_speed_m_s"]),
        "WIND_DIRECTION": float(outdoor["wind_direction_deg"]),
        "ROOM_HEIGHT_M": float(room["height_m"]),
        "ROOM_VOLUME_M3": float(room["volume_m3"]),
        "CLAM": clam,
        "CTURB": cturb,
        "FLOW_EXPONENT": float(leakage["flow_exponent"]),
        "DISCHARGE_COEFFICIENT": float(leakage["discharge_coefficient"]),
        "REFERENCE_PRESSURE_PA": float(leakage["reference_pressure_pa"]),
        "NORMALIZED_ELA_M2_PER_M2": numerical_normalized,
        "PATH1_HEIGHT_M": float(paths[0]["relative_height_m"]),
        "PATH2_HEIGHT_M": float(paths[1]["relative_height_m"]),
        "PATH1_WALL_AREA_M2": float(room["exterior_wall_area_m2"])
        * float(paths[0]["wall_area_fraction"]),
        "PATH2_WALL_AREA_M2": float(room["exterior_wall_area_m2"])
        * float(paths[1]["wall_area_fraction"]),
    }
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace("{" + key + "}", f"{value:.12g}")
    if "{" in text or "}" in text:
        raise RuntimeError("unresolved placeholder in CONTAM template")
    return text


def _prepare_output_dir(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        output_dir = override
    else:
        root = PROJECT_ROOT / config.get("output_root", "outputs")
        output_dir = root / config["case_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "generated.prj",
        "generated.sim",
        "generated.sqlite3",
        "generated.xlog",
        "path_flows.csv",
        "run_summary.json",
        "contam_stdout.log",
        "contam_xlog.log",
    ):
        target = output_dir / name
        if target.exists():
            target.unlink()
    return output_dir


def _read_results(
    database_path: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    room = config["room"]
    outdoor = config["outdoor"]
    leakage = config["leakage"]
    paths_cfg = leakage["paths"]
    connection = sqlite3.connect(database_path)
    try:
        flow_rows = connection.execute(
            "SELECT PathNumber, dP, Flow0, Flow1 FROM LINKPATHDATA ORDER BY PathNumber"
        ).fetchall()
        ambient = connection.execute(
            "SELECT Tambt, Pambt FROM AMBIENTCONDITIONS ORDER BY TimeID LIMIT 1"
        ).fetchone()
        zone = connection.execute(
            "SELECT Temperature, Pressure, Density FROM NODEZONEFLOWDATA "
            "WHERE ZoneNumber=1 ORDER BY TimeID LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if len(flow_rows) != 2 or ambient is None or zone is None:
        raise RuntimeError("CONTAM result database does not contain the expected case")

    ambient_density = air_density_kg_m3(float(ambient[0]), float(ambient[1]))
    zone_density = float(zone[2])
    normalized = float(leakage["normalized_ela_cm2_per_m2"])
    path_rows: list[dict[str, Any]] = []
    for index, (path_number, dp_pa, flow0, flow1) in enumerate(flow_rows):
        mass_flow = float(flow0)
        density = ambient_density if mass_flow >= 0 else zone_density
        signed_volume_flow = mass_flow / density
        represented_area = float(room["exterior_wall_area_m2"]) * float(
            paths_cfg[index]["wall_area_fraction"]
        )
        path_rows.append(
            {
                "path_number": int(path_number),
                "path_name": paths_cfg[index]["name"],
                "relative_height_m": float(paths_cfg[index]["relative_height_m"]),
                "represented_wall_area_m2": represented_area,
                "path_ela_cm2": normalized * represented_area,
                "pressure_difference_pa": float(dp_pa),
                "mass_flow_kg_s": mass_flow,
                "secondary_mass_flow_kg_s": float(flow1),
                "volume_flow_m3_s": signed_volume_flow,
                "direction": "outdoor_to_room" if mass_flow >= 0 else "room_to_outdoor",
            }
        )

    inflow_mass = sum(max(row["mass_flow_kg_s"], 0.0) for row in path_rows)
    outflow_mass = sum(max(-row["mass_flow_kg_s"], 0.0) for row in path_rows)
    imbalance = abs(inflow_mass - outflow_mass)
    balance_scale = max(inflow_mass, outflow_mass, 1.0e-15)
    inflow_volume = sum(max(row["volume_flow_m3_s"], 0.0) for row in path_rows)
    summary = {
        "case_name": config["case_name"],
        "solver": "CONTAM ContamX",
        "physical_input": {
            "room_volume_m3": float(room["volume_m3"]),
            "indoor_temperature_c": float(room["indoor_temperature_c"]),
            "outdoor_temperature_c": float(outdoor["temperature_c"]),
            "normalized_ela_cm2_per_m2": normalized,
            "total_ela_cm2": total_ela_cm2(
                normalized, float(room["exterior_wall_area_m2"])
            ),
            "reference_pressure_pa": float(leakage["reference_pressure_pa"]),
            "discharge_coefficient": float(leakage["discharge_coefficient"]),
            "flow_exponent": float(leakage["flow_exponent"]),
        },
        "results": {
            "inflow_mass_kg_s": inflow_mass,
            "outflow_mass_kg_s": outflow_mass,
            "mass_imbalance_kg_s": imbalance,
            "relative_mass_imbalance": imbalance / balance_scale,
            "infiltration_volume_flow_m3_s": inflow_volume,
            "infiltration_ach_1_per_h": inflow_volume
            * 3600.0
            / float(room["volume_m3"]),
            "zone_pressure_pa_relative": float(zone[1]),
            "ambient_density_kg_m3": ambient_density,
            "zone_density_kg_m3": zone_density,
        },
    }
    return path_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_case(config_path: Path, output_dir_override: Path | None = None) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    output_dir = _prepare_output_dir(config, output_dir_override)
    executable = Path(config["contam_executable"])
    if not executable.exists():
        raise FileNotFoundError(f"ContamX executable not found: {executable}")

    project_path = output_dir / "generated.prj"
    project_path.write_text(render_project(config), encoding="utf-8", newline="\n")
    process = subprocess.run(
        [str(executable), str(project_path)],
        cwd=output_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout = (process.stdout or "") + (process.stderr or "")
    (output_dir / "contam_stdout.log").write_text(stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"ContamX failed with exit code {process.returncode}; see contam_stdout.log"
        )
    database_path = output_dir / "generated.sqlite3"
    sim_path = output_dir / "generated.sim"
    if not database_path.exists() or not sim_path.exists():
        raise RuntimeError("ContamX did not create expected SIM and SQLite results")
    xlog = output_dir / "generated.xlog"
    if xlog.exists():
        shutil.copyfile(xlog, output_dir / "contam_xlog.log")

    rows, summary = _read_results(database_path, config)
    summary["execution"] = {
        "exit_code": process.returncode,
        "project_file": str(project_path),
        "sim_file": str(sim_path),
        "sqlite_file": str(database_path),
    }
    _write_csv(output_dir / "path_flows.csv", rows)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one automated ELA-CONTAM case")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_case(args.config, args.output_dir)
    results = summary["results"]
    print(f"case: {summary['case_name']}")
    print(f"ContamX exit code: {summary['execution']['exit_code']}")
    print(f"infiltration ACH: {results['infiltration_ach_1_per_h']:.6f} 1/h")
    print(f"relative mass imbalance: {results['relative_mass_imbalance']:.6e}")


if __name__ == "__main__":
    main()
