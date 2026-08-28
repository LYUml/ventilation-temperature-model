from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SUBJECT_BLOCK = re.compile(r"(?ms)^<([^>]+)>\s+(.*?)(?=\n<|\Z)")


def _number(body: str, predicate: str) -> float | None:
    match = re.search(rf"{re.escape(predicate)}\s+([^\s;,]+)", body)
    return float(match.group(1).strip('"')) if match else None


def extract_building(rdf_path: Path) -> dict[str, Any]:
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
        }
        for name, body in blocks.items()
        if "a bot:Element" in body
    }
    interfaces: dict[str, list[dict[str, Any]]] = {}
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

    spaces: dict[str, Any] = {}
    for name, body in blocks.items():
        if not name.startswith("Space_") or "a bot:Space" not in body:
            continue
        uid = name.removeprefix("Space_")
        floor_area = _number(body, "bes:hasFloorArea_m2")
        volume = _number(body, "bes:hasVolume_m3")
        height = volume / floor_area if floor_area and volume else None
        items = interfaces.get(uid, [])
        exterior_wall_area = sum(
            item["area_m2"] or 0.0
            for item in items
            if item["surface_type"] == "bes:ExteriorWall"
        )
        window_area = sum(
            item["area_m2"] or 0.0
            for item in items
            if item["surface_type"] == "bes:OperableWindow"
        )
        opaque_exterior_ua = sum(
            (item["area_m2"] or 0.0) * (item["u_value_w_m2k"] or 0.0)
            for item in items
            if item["surface_type"] == "bes:ExteriorWall"
        )
        window_ua = sum(
            (item["area_m2"] or 0.0) * (item["u_value_w_m2k"] or 0.0)
            for item in items
            if item["surface_type"] == "bes:OperableWindow"
        )
        spaces[uid] = {
            "floor_area_m2": floor_area,
            "volume_m3": volume,
            "height_m": height,
            "base_height_m": space_level.get(uid),
            "exterior_wall_area_m2": exterior_wall_area,
            "operable_window_area_m2": window_area,
            "opaque_exterior_ua_w_k": opaque_exterior_ua,
            "window_ua_w_k": window_ua,
            "exterior_ua_w_k": opaque_exterior_ua + window_ua,
            "interface_count": len(items),
        }

    target_names = ["2FCORRIDOR", "3FCORRIDOR", "4FCORRIDOR"]
    missing = [name for name in target_names if name not in spaces]
    if missing:
        raise ValueError(f"RDF is missing target corridor spaces: {missing}")
    return {
        "source_rdf": str(rdf_path.resolve()),
        "format": "Turtle RDF exported by Moosas",
        "storeys": levels,
        "space_count": len(spaces),
        "interface_count": text.count("a bot:Interface"),
        "explicit_door_interface_count": door_count,
        "corridors": {name: spaces[name] for name in target_names},
        "spaces": spaces,
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
