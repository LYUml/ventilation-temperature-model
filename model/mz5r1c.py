"""RDF-driven multi-zone ISO 13790 5R1C design-stage simulator.

The implementation follows ISO 13790 Annex C's air/surface/mass heat-flow
split and Crank--Nicolson mass update.  RDF shared constructions couple zone
surface nodes by a converged fixed-point solve.  No measured indoor
temperature is accepted by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base_ta import _OUTDOOR_CANDIDATES, _finite_numeric, _timestamped
from .common.rdf import extract_building


@dataclass(frozen=True)
class Mz5r1cParameters:
    allow_non_project_assumptions: bool = False
    thermal_capacitance_j_m2k: float = 165_000.0
    mass_area_factor: float = 2.5
    total_internal_area_factor: float = 4.5
    infiltration_ach: float = 0.5
    window_shgc: float = 0.45
    office_people_per_m2: float = 0.25
    occupant_sensible_w: float = 75.0
    office_lighting_w_m2: float = 7.0
    office_equipment_w_m2: float = 9.0
    work_start_hour: int = 8
    work_end_hour: int = 18
    solar_diffuse_exposure_factor: float = 0.5
    solar_direct_exposure_factor: float = 0.25
    initial_mass_temperature_c: float = 23.0
    warmup_tolerance_c: float = 0.1
    max_warmup_days: int = 45
    coupling_tolerance_c: float = 1e-5
    coupling_max_iterations: int = 80


def _params(value: Mz5r1cParameters | Mapping[str, Any] | None) -> Mz5r1cParameters:
    if value is None:
        result = Mz5r1cParameters()
    elif isinstance(value, Mz5r1cParameters):
        result = value
    elif isinstance(value, Mapping):
        unknown = set(value) - set(Mz5r1cParameters.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown MZ5R1C parameters: {sorted(unknown)}")
        result = replace(Mz5r1cParameters(), **value)
    else:
        raise TypeError("parameters must be Mz5r1cParameters, a mapping, or None")
    for name in ("thermal_capacitance_j_m2k", "mass_area_factor", "total_internal_area_factor",
                 "warmup_tolerance_c", "coupling_tolerance_c"):
        if not np.isfinite(getattr(result, name)) or getattr(result, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (result.infiltration_ach < 0 or result.solar_diffuse_exposure_factor < 0
            or result.solar_direct_exposure_factor < 0 or result.coupling_max_iterations < 1):
        raise ValueError("infiltration and iteration settings are invalid")
    return result


def _number(space: dict[str, Any], key: str, default: float) -> float:
    value = space.get("settings", {}).get(key)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else float(default)


class RdfMz5r1cModel:
    """One simultaneous multi-zone 5R1C state model derived from RDF."""

    def __init__(self, *, timestamp_column: str = "Timestamp", outdoor_column: str | None = None,
                 parameters: Mz5r1cParameters | Mapping[str, Any] | None = None) -> None:
        self.timestamp_column = timestamp_column
        self.outdoor_column = outdoor_column
        self.parameters = _params(parameters)
        self.last_metadata: dict[str, Any] | None = None

    def simulate(self, inputRdf: str | Path, weather: pd.DataFrame,
                 targetSpaces: Sequence[str] | None = None) -> dict[str, list[float]]:
        frame = _timestamped(weather, "weather", self.timestamp_column)
        numeric = [c for c in frame.columns if c not in {self.timestamp_column, "_base_ta_timestamp",
                   "year", "mon", "month", "day", "hour"} and pd.api.types.is_numeric_dtype(frame[c])]
        outdoor_name = self.outdoor_column or next((c for c in _OUTDOOR_CANDIDATES if c in numeric), None)
        if outdoor_name not in numeric:
            raise ValueError("Cannot identify outdoor temperature in weather")
        raw = _finite_numeric(frame, numeric, "weather")
        ci = {c: i for i, c in enumerate(numeric)}
        outdoor = raw[:, ci[outdoor_name]]
        diffuse = raw[:, ci["diffuse"]] if "diffuse" in ci else np.zeros(len(frame))
        direct = raw[:, ci["direct"]] if "direct" in ci else np.zeros(len(frame))

        building = extract_building(Path(inputRdf), target_names=[])
        spaces = {n: s for n, s in building["spaces"].items()
                  if (s.get("floor_area_m2") or 0) > 0 and (s.get("volume_m3") or 0) > 0}
        names = sorted(spaces)
        targets = list(targetSpaces) if targetSpaces is not None else names
        missing = [n for n in targets if n not in spaces]
        if missing:
            raise ValueError(f"RDF is missing usable target spaces: {missing}")
        assumption_audit = [
            "thermal_capacitance_j_m2k is an ISO archetype value, not derived from project material layers",
            "mass_area_factor and total_internal_area_factor are ISO archetype factors, not project surface/material calculations",
            "window normals and north direction are parsed from RDF but not yet used by the solar calculation",
            "solar_direct_exposure_factor is a research assumption because window azimuth is not yet consumed",
            "ideal HVAC temperature-band enforcement is a modeled design boundary, not a measured or fully specified plant/airflow model",
        ]
        if not self.parameters.allow_non_project_assumptions:
            raise ValueError(
                "Strict project-data mode rejected MZ5R1C because non-project inputs remain: "
                + "; ".join(assumption_audit)
                + ". Pass parameters={'allow_non_project_assumptions': True} only for an explicitly labelled research run."
            )
        data = self._assemble(names, spaces, building)
        result, warmup, max_iterations = self._run(data, frame, outdoor, diffuse, direct)
        self.last_metadata = {
            "method": "RDF-driven simultaneous multi-zone ISO 13790 5R1C",
            "mode": "design", "requires_measured_indoor_history": False,
            "baseline_condition": "corridors free-running; programmed adjacent zones use ideal RDF temperature bands",
            "parameters": asdict(self.parameters), "warmup": warmup,
            "project_data_compliant": False,
            "non_project_inputs": assumption_audit,
            "maximum_coupling_iterations": max_iterations,
            "limitations": [
                "Facade normals and north direction are parsed but not yet consumed; solar uses a predeclared direct-exposure factor.",
                "Door opening schedules and pressure-flow measurements are absent; only RDF conduction and background infiltration are represented.",
            ],
        }
        ix = {n: i for i, n in enumerate(names)}
        return {n: result[:, ix[n]].tolist() for n in targets}

    def _assemble(self, names, spaces, building):
        p = self.parameters; n = len(names); ix = {name: i for i, name in enumerate(names)}
        arrays = {key: np.zeros(n) for key in ("af", "cm", "am", "at", "hem", "hw", "hve", "solar")}
        coupling = np.zeros((n, n)); programs = []; seen = set()
        for name, i in ix.items():
            s = spaces[name]; af = float(s["floor_area_m2"]); volume = float(s["volume_m3"])
            arrays["af"][i] = af; arrays["cm"][i] = p.thermal_capacitance_j_m2k * af
            arrays["am"][i] = p.mass_area_factor * af; arrays["at"][i] = p.total_internal_area_factor * af
            arrays["hem"][i] = float(s.get("opaque_exterior_ua_w_k") or 0) + float(s.get("other_exterior_ua_w_k") or 0)
            arrays["hw"][i] = float(s.get("window_ua_w_k") or 0)
            arrays["hve"][i] = 1200.0 * volume * _number(s, "zone_infiltration", p.infiltration_ach) / 3600.0
            windows = s.get("exterior_windows", [])
            missing_window_shgc = [w["element"] for w in windows if not isinstance(w.get("shgc"), (int, float))]
            if missing_window_shgc:
                raise ValueError(f"Exterior windows lack element-level SHGC in {name}: {missing_window_shgc}")
            # Element properties describe the actual modeled windows.  The
            # space-level zone_win_SHGC belongs to its program/template and is
            # retained for compliance comparison, not used to overwrite a
            # more specific construction value.
            arrays["solar"][i] = sum(float(w["area_m2"] or 0) * float(w["shgc"]) for w in windows)
            settings = s.get("settings", {}); zp = settings.get("zone_ppsm") == "ALL_ZERO"
            zl = settings.get("zone_lighting") == "ALL_ZERO"; ze = settings.get("zone_equipment") == "ALL_ZERO"
            programs.append({
                "area": af,
                "people_schedule": settings.get("zone_ppsm"),
                "lighting_schedule": settings.get("zone_lighting"),
                "equipment_schedule": settings.get("zone_equipment"),
                "person_w": _number(s, "zone_popheat", p.occupant_sensible_w),
                "start": int(_number(s, "zone_work_start", p.work_start_hour)),
                "end": int(_number(s, "zone_work_end", p.work_end_hour)),
                "heat": settings.get("zone_h_temp"), "cool": settings.get("zone_c_temp"),
                "active": not (zp and zl and ze),
            })
            for edge in s.get("adjacent_spaces", []):
                other = edge["space"]
                if other not in ix: continue
                key = (*sorted((name, other)), str(edge.get("element")))
                if key in seen: continue
                seen.add(key); g = max(0.0, float(edge.get("ua_w_k") or 0))
                coupling[i, ix[other]] += g; coupling[ix[other], i] += g
        arrays["coupling"] = coupling; arrays["programs"] = programs
        arrays["daily_schedules"] = building["daily_schedules"]
        arrays["weekly_schedules"] = building["weekly_schedules"]
        return arrays

    @staticmethod
    def _schedule_value(reference, timestamp, weekly, daily) -> float:
        if reference == "ALL_ZERO":
            return 0.0
        if not isinstance(reference, str) or reference not in weekly:
            raise ValueError(f"Missing RDF weekly schedule: {reference!r}")
        day_name = timestamp.day_name().lower()
        daily_name = weekly[reference]["days"].get(day_name)
        item = daily.get(daily_name or "")
        if (not item or item.get("time_step_hours") != 1.0 or item.get("value_count") != 24.0
                or not isinstance(item.get("values"), list) or len(item["values"]) != 24):
            raise ValueError(f"Incomplete RDF daily schedule: {daily_name!r}")
        return float(item["values"][timestamp.hour])

    @staticmethod
    def _zone(cm, am, at, hem, hw, hve, g_row, neighbor, tm_prev, tout, gains, solar, hvac):
        # ISO 13790 Annex C.1--C.11, with shared-interface conductance folded
        # into the surface boundary term.
        hms = 9.1 * am; his = 3.45 * at; hb = hw + g_row.sum()
        hve = max(hve, 1e-9); h1 = 1.0 / (1.0 / hve + 1.0 / his); h2 = h1 + hb
        h3 = 1.0 / (1.0 / max(h2, 1e-9) + 1.0 / hms)
        boundary = hw * tout + float(g_row @ neighbor)
        phi_ia = 0.5 * gains + hvac
        shared = 0.5 * gains + solar
        phi_m = am / at * shared
        phi_st = (1.0 - am / at - hw / (9.1 * at)) * shared
        phi_m_tot = phi_m + hem * tout + h3 * (phi_st + boundary + h1 * (phi_ia / hve + tout)) / h2
        tm_next = (tm_prev * (cm / 3600.0 - 0.5 * (h3 + hem)) + phi_m_tot) / (cm / 3600.0 + 0.5 * (h3 + hem))
        tm = 0.5 * (tm_next + tm_prev)
        ts = (hms * tm + phi_st + boundary + h1 * (tout + phi_ia / hve)) / (hms + hb + h1)
        ta = (his * ts + hve * tout + phi_ia) / (his + hve)
        return tm_next, ta

    def _run(self, d, frame, outdoor, diffuse, direct):
        p = self.parameters; n = len(d["af"]); timestamps = frame["_base_ta_timestamp"].tolist(); max_used = 0
        def step(tm_prev, ta_prev, k):
            ts = timestamps[k]; workday = ts.weekday() < 5
            gains = np.array([
                program["area"] * (
                    self._schedule_value(program["people_schedule"], ts, d["weekly_schedules"], d["daily_schedules"])
                    * program["person_w"]
                    + self._schedule_value(program["lighting_schedule"], ts, d["weekly_schedules"], d["daily_schedules"])
                    + self._schedule_value(program["equipment_schedule"], ts, d["weekly_schedules"], d["daily_schedules"])
                )
                for program in d["programs"]
            ])
            # ISO 13790 reference implementations use the isotropic-sky view
            # factor (1 + cos(tilt)) / 2.  RDF windows are known to be facade
            # windows but lack azimuth; for a vertical tilt this factor is 0.5.
            irr = (p.solar_diffuse_exposure_factor * max(0.0, diffuse[k])
                   + p.solar_direct_exposure_factor * max(0.0, direct[k]))
            solar = d["solar"] * irr; neighbor = ta_prev.copy()
            for iteration in range(1, p.coupling_max_iterations + 1):
                tm_new = np.empty(n); ta_new = np.empty(n)
                for i in range(n):
                    args = (d["cm"][i], d["am"][i], d["at"][i], d["hem"][i], d["hw"][i], d["hve"][i],
                            d["coupling"][i], neighbor, tm_prev[i], outdoor[k], gains[i], solar[i])
                    m0, t0 = self._zone(*args, 0.0)
                    program = d["programs"][i]
                    start, end, heat, cool, active = (program["start"], program["end"], program["heat"],
                                                      program["cool"], program["active"])
                    setpoint = None
                    if active and workday and start <= ts.hour < end:
                        if isinstance(heat, (int, float)) and t0 < heat: setpoint = float(heat)
                        elif isinstance(cool, (int, float)) and t0 > cool: setpoint = float(cool)
                    if setpoint is None:
                        tm_new[i], ta_new[i] = m0, t0
                    else:
                        m10, t10 = self._zone(*args, 10.0 * d["af"][i])
                        hvac = 10.0 * d["af"][i] * (setpoint - t0) / (t10 - t0)
                        tm_new[i], ta_new[i] = self._zone(*args, hvac)
                if np.max(np.abs(ta_new - neighbor)) <= p.coupling_tolerance_c: break
                neighbor = 0.5 * neighbor + 0.5 * ta_new
            else:
                raise RuntimeError("MZ5R1C inter-zone coupling did not converge")
            return tm_new, ta_new, iteration

        tm = np.full(n, p.initial_mass_temperature_c); ta = tm.copy(); previous = None; delta = float("inf")
        warm_count = min(24, len(frame))
        for day in range(1, p.max_warmup_days + 1):
            trajectory = []
            for k in range(warm_count):
                tm, ta, used = step(tm, ta, k); max_used = max(max_used, used); trajectory.append(np.r_[tm, ta])
            trajectory = np.asarray(trajectory)
            if previous is not None: delta = float(np.max(np.abs(trajectory - previous)))
            if previous is not None and delta <= p.warmup_tolerance_c: break
            previous = trajectory
        else:
            raise RuntimeError(f"MZ5R1C warm-up did not converge; final delta={delta:.3f} C")
        result = np.empty((len(frame), n))
        for k in range(len(frame)):
            tm, ta, used = step(tm, ta, k); max_used = max(max_used, used); result[k] = ta
        return result, {"converged": True, "days": day, "final_delta_c": delta}, max_used


def calculateRdfMz5r1c(inputRdf: str | Path, weather: pd.DataFrame,
                       targetSpaces: Sequence[str] | None = None, **kwargs) -> dict[str, list[float]]:
    return RdfMz5r1cModel(**kwargs).simulate(inputRdf, weather, targetSpaces)


__all__ = ["Mz5r1cParameters", "RdfMz5r1cModel", "calculateRdfMz5r1c"]
