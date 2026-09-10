from __future__ import annotations

import argparse
import json
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SUBJECT_BLOCK = re.compile(r"(?ms)^<([^>]+)>\s+(.*?)(?=\n<|\Z)")


def _number(body: str, predicate: str) -> float | None:
    match = re.search(rf"{re.escape(predicate)}\s+([^\s;,]+)", body)
    return float(match.group(1).strip('"')) if match else None


def _setting(body: str, name: str) -> str | float | None:
    """Read Moosas' non-Turtle quoted setting predicates."""
    match = re.search(rf'(?m)^\s*"{re.escape(name)}"\s+([^\s;,]+)', body)
    if not match:
        return None
    raw = match.group(1).strip('"<>')
    try:
        return float(raw)
    except ValueError:
        return raw


def extract_building(
    rdf_path: Path,
    target_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Extract the subset of the Moosas Turtle/RDF needed by this prototype."""
    text = rdf_path.resolve().read_text(encoding="utf-8")
    blocks = {name: body for name, body in SUBJECT_BLOCK.findall(text)}

    levels: dict[str, float] = {}
    space_level: dict[str, float] = {}
    for name, body in blocks.items():
        if "a bot:Storey" not in body:
            continue
        altitude = _number(body, "moosas:altitute")
        if altitude is None:
            continue
        levels[name] = altitude
        for space in re.findall(r"<Space_([^>]+)>", body):
            space_level[space] = altitude

    element_properties = {
        name: {
            "area_m2": _number(body, "bes:hasArea_m2"),
            "u_value_w_m2k": _number(body, "moosas:U_Value"),
            "shgc": _number(body, "moosas:SHGC"),
            "normal_xyz": (
                _number(body, "bes:hasNormalVectorX_m"),
                _number(body, "bes:hasNormalVectorY_m"),
                _number(body, "bes:hasNormalVectorZ_m"),
            ),
        }
        for name, body in blocks.items()
        if "a bot:Element" in body
    }
    interfaces: dict[str, list[dict[str, Any]]] = {}
    element_spaces: dict[str, set[str]] = defaultdict(set)
    door_count = 0
    for name, body in blocks.items():
        if "a bot:Interface" not in body:
            continue
        spaces = re.findall(r"<Space_([^>]+)>", body)
        elements = re.findall(r"<(element_[^>]+)>", body)
        surface_match = re.search(r"bes:surfaceType\s+([^\s;]+)", body)
        surface_type = surface_match.group(1) if surface_match else "unknown"
        if "door" in surface_type.lower() or "door" in name.lower():
            door_count += 1
        for space in spaces:
            if elements:
                element_spaces[elements[0]].add(space)
            interfaces.setdefault(space, []).append(
                {
                    "interface": name,
                    "surface_type": surface_type,
                    "element": elements[0] if elements else None,
                    **(
                        element_properties.get(elements[0], {})
                        if elements
                        else {"area_m2": None, "u_value_w_m2k": None}
                    ),
                }
            )

    daily_schedules: dict[str, dict[str, Any]] = {}
    for name, body in blocks.items():
        if "a bes:DailySchedule" not in body:
            continue
        match = re.search(r'bes:hourlyValuesJson\s+"(\[[^\"]*\])"', body)
        values = json.loads(match.group(1)) if match else None
        daily_schedules[name] = {
            "values": values,
            "time_step_hours": _number(body, "bes:timeStepHours"),
            "value_count": _number(body, "bes:valueCount"),
            "unit": (re.search(r'bes:valueUnit\s+"([^\"]+)"', body) or [None, None])[1],
        }

    weekday_predicates = (
        "mondaySchedule", "tuesdaySchedule", "wednesdaySchedule", "thursdaySchedule",
        "fridaySchedule", "saturdaySchedule", "sundaySchedule",
    )
    weekly_schedules: dict[str, dict[str, Any]] = {}
    for name, body in blocks.items():
        if "a bes:WeeklySchedule" not in body:
            continue
        days = {}
        for predicate in weekday_predicates:
            match = re.search(rf"bes:{predicate}\s+<([^>]+)>", body)
            days[predicate.removesuffix("Schedule")] = match.group(1) if match else None
        weekly_schedules[name] = {"days": days}

    spaces: dict[str, Any] = {}
    for name, body in blocks.items():
        if not name.startswith("Space_") or "a bot:Space" not in body:
            continue
        uid = name.removeprefix("Space_")
        floor_area = _number(body, "bes:hasFloorArea_m2")
        volume = _number(body, "bes:hasVolume_m3")
        height = volume / floor_area if floor_area and volume else None
        items = interfaces.get(uid, [])
        def is_exterior(item: dict[str, Any]) -> bool:
            element = item.get("element")
            return bool(element) and len(element_spaces.get(element, ())) == 1

        exterior_wall_area = sum(
            item["area_m2"] or 0.0
            for item in items
            if item["surface_type"] == "bes:ExteriorWall" and is_exterior(item)
        )
        window_area = sum(
            item["area_m2"] or 0.0
            for item in items
            if item["surface_type"] == "bes:OperableWindow" and is_exterior(item)
        )
        opaque_exterior_ua = sum(
            (item["area_m2"] or 0.0) * (item["u_value_w_m2k"] or 0.0)
            for item in items
            if item["surface_type"] == "bes:ExteriorWall" and is_exterior(item)
        )
        window_ua = sum(
            (item["area_m2"] or 0.0) * (item["u_value_w_m2k"] or 0.0)
            for item in items
            if item["surface_type"] == "bes:OperableWindow" and is_exterior(item)
        )
        exterior_windows = [
            {
                "element": item["element"],
                "area_m2": item["area_m2"],
                "u_value_w_m2k": item["u_value_w_m2k"],
                "shgc": element_properties.get(item["element"], {}).get("shgc"),
                "normal_xyz": element_properties.get(item["element"], {}).get("normal_xyz"),
            }
            for item in items
            if item["surface_type"] == "bes:OperableWindow" and is_exterior(item)
        ]
        other_exterior_ua = sum(
            (item["area_m2"] or 0.0) * (item["u_value_w_m2k"] or 0.0)
            for item in items
            if is_exterior(item)
            and item["surface_type"]
            not in {"bes:ExteriorWall", "bes:OperableWindow", "bes:InteriorWall"}
        )
        settings = {
            key: _setting(body, key)
            for key in (
                "type",
                "standard",
                "zone_infiltration",
                "zone_win_SHGC",
                "zone_h_temp",
                "zone_c_temp",
                "zone_work_start",
                "zone_work_end",
                "zone_ppsm",
                "zone_popheat",
                "zone_lighting",
                "zone_equipment",
            )
        }
        spaces[uid] = {
            "floor_area_m2": floor_area,
            "volume_m3": volume,
            "height_m": height,
            "base_height_m": space_level.get(uid),
            "exterior_wall_area_m2": exterior_wall_area,
            "operable_window_area_m2": window_area,
            "exterior_windows": exterior_windows,
            "opaque_exterior_ua_w_k": opaque_exterior_ua,
            "window_ua_w_k": window_ua,
            "other_exterior_ua_w_k": other_exterior_ua,
            "exterior_ua_w_k": opaque_exterior_ua + window_ua + other_exterior_ua,
            "interface_count": len(items),
            "settings": settings,
            "north_direction_deg": _number(body, "bes:hasNorthDirection_deg"),
        }

    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for element, connected_spaces in element_spaces.items():
        if len(connected_spaces) < 2:
            continue
        properties = element_properties.get(element, {})
        for source in connected_spaces:
            for target in connected_spaces:
                if source == target:
                    continue
                source_item = next((item for item in interfaces.get(source, []) if item["element"] == element), None)
                adjacency[source].append({
                    "space": target,
                    "element": element,
                    "surface_type": source_item["surface_type"] if source_item else "unknown",
                    "area_m2": properties.get("area_m2"),
                    "u_value_w_m2k": properties.get("u_value_w_m2k"),
                    "ua_w_k": (properties.get("area_m2") or 0.0) * (properties.get("u_value_w_m2k") or 0.0),
                })
    for name, space in spaces.items():
        space["adjacent_spaces"] = adjacency.get(name, [])

    requested = list(target_names) if target_names is not None else [
        "2FCORRIDOR",
        "3FCORRIDOR",
        "4FCORRIDOR",
    ]
    missing = [name for name in requested if name not in spaces]
    if missing:
        raise ValueError(f"RDF is missing target spaces: {missing}")
    return {
        "source_rdf": str(rdf_path.resolve()),
        "format": "Turtle RDF exported by Moosas",
        "storeys": levels,
        "space_count": len(spaces),
        "interface_count": text.count("a bot:Interface"),
        "explicit_door_interface_count": door_count,
        "corridors": {name: spaces[name] for name in requested},
        "spaces": spaces,
        "daily_schedules": daily_schedules,
        "weekly_schedules": weekly_schedules,
        "limitations": [
            "No explicit door interfaces were found." if door_count == 0 else "Door interfaces exist but opening schedules still require review.",
            "RDF geometry does not contain measured ELA or pressure-flow test data.",
            "OperableWindow denotes capability, not the measured opening state.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CONTAM geometry inputs from Moosas RDF")
    parser.add_argument("--rdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = extract_building(args.rdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["corridors"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
