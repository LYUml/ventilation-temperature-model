"""Experimental multi-zone 5R2C model.

This candidate keeps the five ISO-13790 heat-transfer paths used by the
project's 5R1C implementation and adds zone-air capacitance as a second
dynamic state.  It is an explicitly named research comparison, not a claim
that project material-layer capacitances are known.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from model.base_ta import _OUTDOOR_CANDIDATES, _finite_numeric, _timestamped
from model.common.rdf import extract_building
from model.mz5r1c import Mz5r1cParameters, RdfMz5r1cModel


class RdfMz5r2cModel(RdfMz5r1cModel):
    """5R1C network with a second, physical zone-air capacitance."""

    def __init__(
        self,
        *,
        timestamp_column: str = "Timestamp",
        outdoor_column: str | None = None,
        parameters: Mz5r1cParameters | Mapping[str, Any] | None = None,
        baseline_spaces: Sequence[str] | None = None,
        air_heat_capacity_j_m3k: float = 1200.0,
    ) -> None:
        super().__init__(
            timestamp_column=timestamp_column,
            outdoor_column=outdoor_column,
            parameters=parameters,
            baseline_spaces=baseline_spaces,
        )
        if not np.isfinite(air_heat_capacity_j_m3k) or air_heat_capacity_j_m3k <= 0:
            raise ValueError("air_heat_capacity_j_m3k must be finite and positive")
        self.air_heat_capacity_j_m3k = float(air_heat_capacity_j_m3k)

    def _assemble(self, names, spaces, building, baseline_spaces):
        data = super()._assemble(names, spaces, building, baseline_spaces)
        data["volume"] = np.asarray(
            [float(spaces[name]["volume_m3"]) for name in names], dtype=float
        )
        return data

    def simulate(
        self,
        inputRdf: str | Path,
        weather: pd.DataFrame,
        targetSpaces: Sequence[str] | None = None,
    ) -> dict[str, list[float]]:
        frame = _timestamped(weather, "weather", self.timestamp_column)
        numeric = [
            c for c in frame.columns
            if c not in {self.timestamp_column, "_base_ta_timestamp", "year", "mon", "month", "day", "hour"}
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        outdoor_name = self.outdoor_column or next(
            (c for c in _OUTDOOR_CANDIDATES if c in numeric), None
        )
        if outdoor_name not in numeric:
            raise ValueError("Cannot identify outdoor temperature in weather")
        raw = _finite_numeric(frame, numeric, "weather")
        ci = {c: i for i, c in enumerate(numeric)}
        outdoor = raw[:, ci[outdoor_name]]
        diffuse = raw[:, ci["diffuse"]] if "diffuse" in ci else np.zeros(len(frame))
        direct = raw[:, ci["direct"]] if "direct" in ci else np.zeros(len(frame))

        building = extract_building(Path(inputRdf), target_names=[])
        spaces = {
            n: s for n, s in building["spaces"].items()
            if (s.get("floor_area_m2") or 0) > 0 and (s.get("volume_m3") or 0) > 0
        }
        names = sorted(spaces)
        targets = list(targetSpaces) if targetSpaces is not None else names
        missing = [n for n in [*targets, *self.baseline_spaces] if n not in spaces]
        if missing:
            raise ValueError(f"RDF is missing usable spaces: {sorted(set(missing))}")
        if not self.parameters.allow_non_project_assumptions:
            raise ValueError(
                "Strict project-data mode rejected MZ5R2C because project material-layer "
                "capacitances and complete airflow/solar boundaries remain unavailable."
            )
        data = self._assemble(names, spaces, building, set(self.baseline_spaces))
        result, warmup, max_iterations = self._run_2c(
            data, frame, outdoor, diffuse, direct
        )
        self.last_metadata = {
            "method": "Experimental RDF-driven multi-zone 5R2C",
            "mode": "design",
            "requires_measured_indoor_history": False,
            "second_capacitance": "zone air: volume * air_heat_capacity_j_m3k",
            "air_heat_capacity_j_m3k": self.air_heat_capacity_j_m3k,
            "explicit_baseline_spaces": list(self.baseline_spaces),
            "warmup": warmup,
            "maximum_coupling_iterations": max_iterations,
            "project_data_compliant": False,
        }
        ix = {name: i for i, name in enumerate(names)}
        return {name: result[:, ix[name]].tolist() for name in targets}

    @staticmethod
    def _implicit_zone(
        ca, cm, am, at, hem, hw, hve, g_row, neighbor, ta_prev, tm_prev,
        tout, gains, solar, hvac,
    ):
        hms = 9.1 * am
        his = 3.45 * at
        hb = hw + g_row.sum()
        denominator = his + hms + hb
        boundary = hw * tout + float(g_row @ neighbor)
        phi_ia = 0.5 * gains + hvac
        shared = 0.5 * gains + solar
        phi_m = am / at * shared
        phi_st = (1.0 - am / at - hw / (9.1 * at)) * shared
        constant_surface = (boundary + phi_st) / denominator
        a_ta = his / denominator
        a_tm = hms / denominator

        dt = 3600.0
        matrix = np.array([
            [ca / dt + hve + his * (1.0 - a_ta), -his * a_tm],
            [-hms * a_ta, cm / dt + hem + hms * (1.0 - a_tm)],
        ])
        rhs = np.array([
            ca / dt * ta_prev + hve * tout + his * constant_surface + phi_ia,
            cm / dt * tm_prev + hem * tout + hms * constant_surface + phi_m,
        ])
        ta, tm = np.linalg.solve(matrix, rhs)
        return float(tm), float(ta)

    def _run_2c(self, d, frame, outdoor, diffuse, direct):
        p = self.parameters
        n = len(d["af"])
        timestamps = frame["_base_ta_timestamp"].tolist()
        ca = self.air_heat_capacity_j_m3k * d["volume"]
        max_used = 0

        def step(tm_prev, ta_prev, k):
            ts = timestamps[k]
            workday = ts.weekday() < 5
            gains = np.array([
                program["area"] * (
                    self._schedule_value(program["people_schedule"], ts, d["weekly_schedules"], d["daily_schedules"])
                    * program["person_w"]
                    + self._schedule_value(program["lighting_schedule"], ts, d["weekly_schedules"], d["daily_schedules"])
                    + self._schedule_value(program["equipment_schedule"], ts, d["weekly_schedules"], d["daily_schedules"])
                )
                for program in d["programs"]
            ])
            irr = (
                p.solar_diffuse_exposure_factor * max(0.0, diffuse[k])
                + p.solar_direct_exposure_factor * max(0.0, direct[k])
            )
            solar = d["solar"] * irr
            neighbor = ta_prev.copy()
            for iteration in range(1, p.coupling_max_iterations + 1):
                tm_new = np.empty(n)
                ta_new = np.empty(n)
                for i in range(n):
                    args = (
                        ca[i], d["cm"][i], d["am"][i], d["at"][i], d["hem"][i],
                        d["hw"][i], d["hve"][i], d["coupling"][i], neighbor,
                        ta_prev[i], tm_prev[i], outdoor[k], gains[i], solar[i],
                    )
                    m0, t0 = self._implicit_zone(*args, 0.0)
                    program = d["programs"][i]
                    active = program["active"] and workday and program["start"] <= ts.hour < program["end"]
                    setpoint = None
                    if active and isinstance(program["heat"], (int, float)) and t0 < program["heat"]:
                        setpoint = float(program["heat"])
                    elif active and isinstance(program["cool"], (int, float)) and t0 > program["cool"]:
                        setpoint = float(program["cool"])
                    if setpoint is None:
                        tm_new[i], ta_new[i] = m0, t0
                    else:
                        m10, t10 = self._implicit_zone(*args, 10.0 * d["af"][i])
                        hvac = 10.0 * d["af"][i] * (setpoint - t0) / (t10 - t0)
                        tm_new[i], ta_new[i] = self._implicit_zone(*args, hvac)
                if np.max(np.abs(ta_new - neighbor)) <= p.coupling_tolerance_c:
                    break
                neighbor = 0.5 * neighbor + 0.5 * ta_new
            else:
                raise RuntimeError("MZ5R2C inter-zone coupling did not converge")
            return tm_new, ta_new, iteration

        tm = np.full(n, p.initial_mass_temperature_c)
        ta = tm.copy()
        previous = None
        delta = float("inf")
        warm_count = min(24, len(frame))
        for day in range(1, p.max_warmup_days + 1):
            trajectory = []
            for k in range(warm_count):
                tm, ta, used = step(tm, ta, k)
                max_used = max(max_used, used)
                trajectory.append(np.r_[tm, ta])
            trajectory = np.asarray(trajectory)
            if previous is not None:
                delta = float(np.max(np.abs(trajectory - previous)))
            if previous is not None and delta <= p.warmup_tolerance_c:
                break
            previous = trajectory
        else:
            raise RuntimeError(f"MZ5R2C warm-up did not converge; final delta={delta:.3f} C")
        result = np.empty((len(frame), n))
        for k in range(len(frame)):
            tm, ta, used = step(tm, ta, k)
            max_used = max(max_used, used)
            result[k] = ta
        return result, {"converged": True, "days": day, "final_delta_c": delta}, max_used


__all__ = ["RdfMz5r2cModel"]
