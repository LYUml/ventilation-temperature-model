"""Archived 1R1C design-stage prototype without measured indoor history.

The model is a deliberately small, transparent multi-zone RC network.  Its
parameters come from the RDF and documented archetype defaults.  Measured room
temperature is never an input to this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.base_ta import _OUTDOOR_CANDIDATES, _finite_numeric, _timestamped
from model.common.rdf import extract_building

RHO_AIR = 1.2
CP_AIR = 1005.0
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class DesignParameters:
    """Transparent defaults; callers may replace any field."""

    allow_non_project_assumptions: bool = False
    areal_heat_capacity_j_m2k: float = 165_000.0
    infiltration_ach: float = 0.5
    window_shgc: float = 0.45
    office_people_per_m2: float = 0.25
    occupant_sensible_w: float = 75.0
    office_lighting_w_m2: float = 7.0
    office_equipment_w_m2: float = 9.0
    work_start_hour: int = 8
    work_end_hour: int = 18
    solar_direct_exposure_factor: float = 0.25
    initial_temperature_c: float = 23.0
    warmup_tolerance_c: float = 0.1
    max_warmup_days: int = 30


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _parameters(value: DesignParameters | Mapping[str, Any] | None) -> DesignParameters:
    if value is None:
        result = DesignParameters()
    elif isinstance(value, DesignParameters):
        result = value
    elif isinstance(value, Mapping):
        unknown = set(value) - set(DesignParameters.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown design parameters: {sorted(unknown)}")
        result = replace(DesignParameters(), **value)
    else:
        raise TypeError("parameters must be DesignParameters, a mapping, or None")
    _positive(result.areal_heat_capacity_j_m2k, "areal_heat_capacity_j_m2k")
    _positive(result.infiltration_ach, "infiltration_ach", allow_zero=True)
    _positive(result.window_shgc, "window_shgc", allow_zero=True)
    _positive(result.warmup_tolerance_c, "warmup_tolerance_c")
    if result.max_warmup_days < 1:
        raise ValueError("max_warmup_days must be positive")
    return result


def _numeric_setting(space: dict[str, Any], name: str, default: float) -> tuple[float, bool]:
    value = space.get("settings", {}).get(name)
    if isinstance(value, (float, int)) and np.isfinite(value):
        return float(value), False
    return float(default), True


class DesignBaseTaModel:
    """Simulate corridor baseline temperature from design inputs."""

    def __init__(
        self,
        *,
        timestamp_column: str = "Timestamp",
        outdoor_column: str | None = None,
        parameters: DesignParameters | Mapping[str, Any] | None = None,
    ) -> None:
        self.timestamp_column = timestamp_column
        self.outdoor_column = outdoor_column
        self.parameters = _parameters(parameters)
        self.last_metadata: dict[str, Any] | None = None

    def simulate(
        self,
        inputRdf: str | Path,
        weather: pd.DataFrame,
        targetSpaces: Sequence[str] | None = None,
    ) -> dict[str, list[float]]:
        frame = _timestamped(weather, "weather", self.timestamp_column)
        numeric = [
            name
            for name in frame.columns
            if name not in {self.timestamp_column, "_base_ta_timestamp", "year", "mon", "month", "day", "hour"}
            and pd.api.types.is_numeric_dtype(frame[name])
        ]
        outdoor_name = self.outdoor_column or next(
            (name for name in _OUTDOOR_CANDIDATES if name in numeric), None
        )
        if outdoor_name not in numeric:
            raise ValueError("Cannot identify outdoor temperature in weather")
        values = _finite_numeric(frame, numeric, "weather")
        column_index = {name: index for index, name in enumerate(numeric)}
        outdoor = values[:, column_index[outdoor_name]]
        diffuse = values[:, column_index["diffuse"]] if "diffuse" in column_index else np.zeros(len(frame))
        direct = values[:, column_index["direct"]] if "direct" in column_index else np.zeros(len(frame))

        building = extract_building(Path(inputRdf), target_names=[])
        usable = {
            name: space
            for name, space in building["spaces"].items()
            if (space.get("floor_area_m2") or 0) > 0 and (space.get("volume_m3") or 0) > 0
        }
        names = sorted(usable)
        if not names:
            raise ValueError("RDF contains no spaces with positive area and volume")
        targets = list(targetSpaces) if targetSpaces is not None else names
        missing = [name for name in targets if name not in usable]
        if missing:
            raise ValueError(f"RDF is missing usable target spaces: {missing}")

        assumption_audit = [
            "equivalent areal heat capacity is an archetype default rather than a project material-layer calculation",
            "RDF schedule identifiers are not decoded to their project hourly values; archetype load magnitudes are substituted",
            "window WKT geometry exists in RDF but is not yet linked to facade-resolved solar calculation",
            "the direct-solar exposure factor is a research assumption",
            "ideal adjacent-zone HVAC bands are a modeled boundary rather than a complete project HVAC/airflow specification",
        ]
        if not self.parameters.allow_non_project_assumptions:
            raise ValueError(
                "Strict project-data mode rejected DesignBaseTaModel because non-project inputs remain: "
                + "; ".join(assumption_audit)
                + ". Pass parameters={'allow_non_project_assumptions': True} only for an explicitly labelled research run."
            )

        matrix, capacity, exterior, solar_area, gains, assumptions = self._network(names, usable)
        temperatures, warmup = self._run(
            matrix, capacity, exterior, solar_area, gains, frame, outdoor, diffuse, direct
        )
        self.last_metadata = {
            "method": "RDF-parameterized implicit multi-zone 1R1C",
            "mode": "design",
            "requires_measured_indoor_history": False,
            "project_data_compliant": False,
            "non_project_inputs": assumption_audit,
            "baseline_condition": "target corridors free-running; occupied adjacent rooms use RDF design heating/cooling bands; optimized natural-ventilation openings closed",
            "weather_columns_used": [outdoor_name, *(name for name in ("diffuse", "direct") if name in column_index)],
            "parameters": asdict(self.parameters),
            "assumptions": assumptions,
            "warmup": warmup,
            "space_count": len(names),
        }
        indices = {name: index for index, name in enumerate(names)}
        return {name: temperatures[:, indices[name]].tolist() for name in targets}

    def _network(self, names: list[str], spaces: dict[str, dict[str, Any]]):
        p = self.parameters
        index = {name: i for i, name in enumerate(names)}
        count = len(names)
        conductance = np.zeros((count, count))
        capacity = np.empty(count)
        exterior = np.empty(count)
        solar_area = np.empty(count)
        gains: list[tuple[float, float, int, int, float | None, float | None, bool]] = []
        fallbacks: dict[str, list[str]] = {}
        seen: set[tuple[str, str, str]] = set()
        for name, i in index.items():
            space = spaces[name]
            area, volume = float(space["floor_area_m2"]), float(space["volume_m3"])
            capacity[i] = area * p.areal_heat_capacity_j_m2k
            ach, fb_ach = _numeric_setting(space, "zone_infiltration", p.infiltration_ach)
            shgc, fb_shgc = _numeric_setting(space, "zone_win_SHGC", p.window_shgc)
            exterior[i] = float(space.get("exterior_ua_w_k") or 0.0) + RHO_AIR * CP_AIR * ach * volume / 3600.0
            solar_area[i] = float(space.get("operable_window_area_m2") or 0.0) * shgc

            settings = space.get("settings", {})
            zero_people = settings.get("zone_ppsm") == "ALL_ZERO"
            zero_lighting = settings.get("zone_lighting") == "ALL_ZERO"
            zero_equipment = settings.get("zone_equipment") == "ALL_ZERO"
            people, fb_people = _numeric_setting(space, "zone_ppsm", 0.0 if zero_people else p.office_people_per_m2)
            person_w, fb_person_w = _numeric_setting(space, "zone_popheat", p.occupant_sensible_w)
            lighting, fb_light = _numeric_setting(space, "zone_lighting", 0.0 if zero_lighting else p.office_lighting_w_m2)
            equipment, fb_equip = _numeric_setting(space, "zone_equipment", 0.0 if zero_equipment else p.office_equipment_w_m2)
            start, fb_start = _numeric_setting(space, "zone_work_start", p.work_start_hour)
            end, fb_end = _numeric_setting(space, "zone_work_end", p.work_end_hour)
            heating = settings.get("zone_h_temp")
            cooling = settings.get("zone_c_temp")
            has_active_program = not (zero_people and zero_lighting and zero_equipment)
            gains.append((
                area * (people * person_w + lighting + equipment),
                0.0,
                int(start),
                int(end),
                float(heating) if isinstance(heating, (float, int)) else None,
                float(cooling) if isinstance(cooling, (float, int)) else None,
                has_active_program,
            ))
            used = [key for key, flag in (("infiltration", fb_ach), ("SHGC", fb_shgc), ("occupancy", fb_people or fb_person_w), ("lighting", fb_light), ("equipment", fb_equip), ("work hours", fb_start or fb_end)) if flag]
            if used:
                fallbacks[name] = used

            for edge in space.get("adjacent_spaces", []):
                other = edge["space"]
                if other not in index:
                    continue
                key = (*sorted((name, other)), str(edge.get("element")))
                if key in seen:
                    continue
                seen.add(key)
                g = max(0.0, float(edge.get("ua_w_k") or 0.0))
                conductance[i, index[other]] += g
                conductance[index[other], i] += g
        assumptions = [
            "Equivalent thermal mass uses a single areal heat-capacity value for every space.",
            "Solar gain uses exterior glazing area and diffuse + 0.25*direct radiation because facade normals are not yet exposed by the RDF parser.",
            "Background infiltration is sensible-only and optimized operable-window airflow is excluded.",
            "Spaces with active RDF programs use ideal heating/cooling bands during work hours; ALL_ZERO corridors remain free-running.",
        ]
        if fallbacks:
            assumptions.append(f"Archetype defaults replaced missing/non-numeric RDF settings: {fallbacks}")
        return conductance, capacity, exterior, solar_area, gains, assumptions

    def _run(self, conductance, capacity, exterior, solar_area, gains, frame, outdoor, diffuse, direct):
        p = self.parameters
        diagonal = capacity / SECONDS_PER_HOUR + exterior + conductance.sum(axis=1)
        system = -conductance.copy()
        np.fill_diagonal(system, diagonal)
        # The conductance network is constant for a simulation. Factor it once
        # instead of solving the same dense system at every hourly step.
        system_inverse = np.linalg.inv(system)
        timestamps = frame["_base_ta_timestamp"].tolist()

        def step(current: np.ndarray, weather_index: int) -> np.ndarray:
            timestamp = timestamps[weather_index]
            occupied = timestamp.weekday() < 5
            scheduled = np.asarray([
                peak if occupied and start <= timestamp.hour < end else idle
                for peak, idle, start, end, _heating, _cooling, _conditioned in gains
            ])
            irradiance = max(0.0, diffuse[weather_index]) + p.solar_direct_exposure_factor * max(0.0, direct[weather_index])
            rhs = capacity / SECONDS_PER_HOUR * current + exterior * outdoor[weather_index] + scheduled + solar_area * irradiance
            free = system_inverse @ rhs
            controlled: dict[int, float] = {}
            for index, (_peak, _idle, start, end, heating, cooling, conditioned) in enumerate(gains):
                if not conditioned or not occupied or not start <= timestamp.hour < end:
                    continue
                if heating is not None and free[index] < heating:
                    controlled[index] = heating
                elif cooling is not None and free[index] > cooling:
                    controlled[index] = cooling
            if not controlled:
                return free
            constrained = system.copy()
            constrained_rhs = rhs.copy()
            for index, setpoint in controlled.items():
                constrained[index, :] = 0.0
                constrained[index, index] = 1.0
                constrained_rhs[index] = setpoint
            return np.linalg.solve(constrained, constrained_rhs)

        current = np.full(len(capacity), p.initial_temperature_c)
        warm_count = min(24, len(frame))
        converged = False
        delta = float("inf")
        previous_day: np.ndarray | None = None
        for day in range(1, p.max_warmup_days + 1):
            trajectory = np.empty((warm_count, len(capacity)))
            for weather_index in range(warm_count):
                current = step(current, weather_index)
                trajectory[weather_index] = current
            if previous_day is not None:
                delta = float(np.max(np.abs(trajectory - previous_day)))
            if previous_day is not None and delta <= p.warmup_tolerance_c:
                converged = True
                break
            previous_day = trajectory
        if not converged:
            raise RuntimeError(
                f"Design temperature warm-up did not converge after {p.max_warmup_days} days; final delta={delta:.3f} C"
            )
        result = np.empty((len(frame), len(capacity)))
        for weather_index in range(len(frame)):
            current = step(current, weather_index)
            result[weather_index] = current
        return result, {"converged": True, "days": day, "final_delta_c": delta, "tolerance_c": p.warmup_tolerance_c}


def calculateDesignBaseTa(
    inputRdf: str | Path,
    weather: pd.DataFrame,
    targetSpaces: Sequence[str] | None = None,
    *,
    parameters: DesignParameters | Mapping[str, Any] | None = None,
    timestamp_column: str = "Timestamp",
    outdoor_column: str | None = None,
) -> dict[str, list[float]]:
    return DesignBaseTaModel(
        timestamp_column=timestamp_column,
        outdoor_column=outdoor_column,
        parameters=parameters,
    ).simulate(inputRdf, weather, targetSpaces)


calculate_design_base_ta = calculateDesignBaseTa

__all__ = ["DesignBaseTaModel", "DesignParameters", "calculateDesignBaseTa", "calculate_design_base_ta"]
