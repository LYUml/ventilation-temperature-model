from __future__ import annotations

import copy
import csv
import contextlib
import json
import os
from pathlib import Path
from typing import Any
import networkx as nx
from networkx.readwrite import json_graph

import numpy as np

def evaluate_lpg_fitness(
    model: Any,
    graph: Any,
    base_network_dict: dict,
    config: Any,
    mute_std: bool = False,
    trace_root: str | None = None,
    trace_context: dict[str, Any] | None = None,
) -> float:
    """
    Single evaluation entrypoint.
    Input: LPG graph + simulation config.
    Output: scalar fitness (total_energy_vent), lower is better.
    """

    @contextlib.contextmanager
    def _silence_fds():
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            os.close(devnull_fd)

    network_dict = sync_lpg_structure_to_network_dict(
        model=model,
        graph=graph,
        base_network_dict=base_network_dict,
        default_edge_attrs=config.default_edge_attrs,
    )
    network_dict = model.lpg_to_networkDict(graph, base_networkDict=network_dict)

    for path in network_dict.get("paths", {}).values():
        path_id = str(path.get("userName", ""))
        if path_id and path_id not in model.pathRadIntensity:
            model.pathRadIntensity[path_id] = np.zeros(8760, dtype=float)

    if mute_std:
        with _silence_fds():
            with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    result = model.coupledTask(
                        energyDict=network_dict,
                        timestep=config.timestep,
                        iteration=config.iteration,
                        mode=config.mode,
                        preheat=config.preheat,
                        k=config.k,
                        sigma=config.sigma,
                        start_hoy=config.start_hoy,
                        end_hoy=config.end_hoy,
                    )
    else:
        result = model.coupledTask(
            energyDict=network_dict,
            timestep=config.timestep,
            iteration=config.iteration,
            mode=config.mode,
            preheat=config.preheat,
            k=config.k,
            sigma=config.sigma,
            start_hoy=config.start_hoy,
            end_hoy=config.end_hoy,
        )

    ok_result, _ = _validate_coupled_result(result)
    if not ok_result:
        if trace_root:
            _append_case_trace(
                trace_root,
                graph=graph,
                base_network_dict=base_network_dict,
                network_dict=network_dict,
                result=result,
                trace_context=trace_context or {},
                fitness=float(config.fail_penalty),
                comfort_mean=float("nan"),
                comfort_min=float("nan"),
                comfort_max=float("nan"),
                comfort_count=0,
                valid=False,
                error="invalid_coupled_result",
            )
        return float(config.fail_penalty)

    fitness = _sum_total_energy_vent(result)
    if not np.isfinite(fitness):
        if trace_root:
            _append_case_trace(
                trace_root,
                graph=graph,
                base_network_dict=base_network_dict,
                network_dict=network_dict,
                result=result,
                trace_context=trace_context or {},
                fitness=float(config.fail_penalty),
                comfort_mean=float("nan"),
                comfort_min=float("nan"),
                comfort_max=float("nan"),
                comfort_count=0,
                valid=False,
                error="non_finite_fitness",
            )
        return float(config.fail_penalty)

    comfort_mean, comfort_min, comfort_max, comfort_count = _summarize_comfort(result)
    if trace_root:
        _append_case_trace(
            trace_root,
            graph=graph,
            base_network_dict=base_network_dict,
            network_dict=network_dict,
            result=result,
            trace_context=trace_context or {},
            fitness=float(fitness),
            comfort_mean=comfort_mean,
            comfort_min=comfort_min,
            comfort_max=comfort_max,
            comfort_count=comfort_count,
            valid=True,
            error="",
        )
    return float(fitness)

def evaluate_candidate(
    graph: nx.MultiDiGraph, metadata: dict[str, Any]
) -> tuple[float, str]:
    model = metadata["model"]
    base_network_dict = metadata["base_network_dict"]
    config = metadata["config"]

    try:
        j = evaluate_lpg_fitness(
            model=model,
            graph=graph,
            base_network_dict=base_network_dict,
            config=config,
            mute_std=True,
            trace_root=metadata.get("trace_root"),
            trace_context=metadata.get("trace_context"),
        )
    except Exception as exc:
        trace_root = metadata.get("trace_root")
        if trace_root:
            _append_case_trace(
                trace_root,
                graph=graph,
                base_network_dict=base_network_dict,
                network_dict=base_network_dict,
                result={},
                trace_context=metadata.get("trace_context") or {},
                fitness=float(config.fail_penalty),
                comfort_mean=float("nan"),
                comfort_min=float("nan"),
                comfort_max=float("nan"),
                comfort_count=0,
                valid=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return float(config.fail_penalty), f"{type(exc).__name__}: {exc}"
    if not np.isfinite(j):
        trace_root = metadata.get("trace_root")
        if trace_root:
            _append_case_trace(
                trace_root,
                graph=graph,
                base_network_dict=base_network_dict,
                network_dict=base_network_dict,
                result={},
                trace_context=metadata.get("trace_context") or {},
                fitness=float(config.fail_penalty),
                comfort_mean=float("nan"),
                comfort_min=float("nan"),
                comfort_max=float("nan"),
                comfort_count=0,
                valid=False,
                error="non_finite_fitness",
            )
        return float(config.fail_penalty), "non_finite_fitness"
    return float(j), ""


def _summarize_comfort(coupled_result: dict) -> tuple[float, float, float, int]:
    comfort_values: list[float] = []
    for zone_data in coupled_result.values():
        comfort_arr = np.asarray(zone_data.get("Comfort", []), dtype=float)
        comfort_arr = comfort_arr[np.isfinite(comfort_arr)]
        if comfort_arr.size > 0:
            comfort_values.extend(comfort_arr.tolist())
    if not comfort_values:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.asarray(comfort_values, dtype=float)
    return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr)), int(arr.size)


def _append_case_trace(
    trace_root: str,
    graph: nx.MultiDiGraph,
    base_network_dict: dict,
    network_dict: dict,
    result: dict,
    trace_context: dict[str, Any],
    *,
    fitness: float,
    comfort_mean: float,
    comfort_min: float,
    comfort_max: float,
    comfort_count: int,
    valid: bool,
    error: str,
) -> None:
    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    path = root / f"pid_{pid}.csv"
    case_id = _case_id_from_context(trace_context, pid)
    base_row: dict[str, Any] = {
        "pid": pid,
        "case_id": case_id,
        "fitness(total_energy_vent)": fitness,
        "comfort_mean": comfort_mean,
        "comfort_min": comfort_min,
        "comfort_max": comfort_max,
        "comfort_count": comfort_count,
        "valid": int(bool(valid)),
        "error": error,
        "case_file": str(case_dir / f"{case_id}.json"),
    }
    trace_fields = [
        "city",
        "station_id",
        "outer_iter",
        "inner_iter",
        "structure_attempt",
        "eval_index",
        "valid_structure_sample_index",
    ]
    row: dict[str, Any] = dict(base_row)
    for field in trace_fields:
        row[field] = trace_context.get(field, "")
    write_header = not path.exists()
    fieldnames = list(base_row.keys()) + trace_fields
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    payload = {
        "pid": pid,
        "case_id": case_id,
        "trace_context": trace_context,
        "summary": row,
        "graph": json_graph.node_link_data(graph),
        "network_diff": _network_dict_diff(base_network_dict, network_dict),
        "result": result,
    }
    _write_json(case_dir / f"{case_id}.json", payload)


def _sum_total_energy_vent(coupled_result: dict) -> float:
    total = 0.0
    has_valid = False
    for zone_data in coupled_result.values():
        arr = np.asarray(zone_data.get("total_energy_vent", []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        has_valid = True
        total += float(np.sum(arr))
    if not has_valid:
        return float("inf")
    return float(total)


def _validate_coupled_result(coupled_result: dict) -> tuple[bool, str]:
    ach_finite: list[float] = []
    comfort_values: list[float] = []
    for zone_data in coupled_result.values():
        ach_arr = np.asarray(zone_data.get("ACH", []), dtype=float)
        ach_arr = ach_arr[np.isfinite(ach_arr)]
        if ach_arr.size > 0:
            ach_finite.extend(ach_arr.tolist())

        comfort_arr = np.asarray(zone_data.get("Comfort", []), dtype=float)
        comfort_arr = comfort_arr[np.isfinite(comfort_arr)]
        if comfort_arr.size > 0:
            comfort_values.extend(comfort_arr.tolist())

    if len(ach_finite) == 0:
        return False, "all_nan_ach"
    if float(np.max(np.abs(np.asarray(ach_finite, dtype=float)))) <= 1e-12:
        return False, "all_zero_ach"

    if len(comfort_values) == 0:
        return False, "all_nan_comfort"
    if float(np.max(np.asarray(comfort_values, dtype=float))) <= 1e-12:
        return False, "all_zero_comfort"

    return True, ""


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonify(v) for v in value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def _case_id_from_context(trace_context: dict[str, Any], pid: int) -> str:
    outer_i = trace_context.get("outer_iter", "na")
    inner_i = trace_context.get("inner_iter", "na")
    attempt = trace_context.get("structure_attempt", "na")
    eval_index = trace_context.get("eval_index", "na")
    sample_idx = trace_context.get("valid_structure_sample_index", "na")
    return f"pid{pid}_o{outer_i}_i{inner_i}_a{attempt}_e{eval_index}_s{sample_idx}"


def _network_dict_diff(base_dict: dict, candidate_dict: dict) -> dict[str, list[dict]]:
    def _section_diff(section_name: str) -> list[dict]:
        base_section = dict(base_dict.get(section_name, {}))
        cand_section = dict(candidate_dict.get(section_name, {}))
        rows: list[dict] = []
        for item_key in sorted(set(base_section) | set(cand_section), key=str):
            before = base_section.get(item_key, {})
            after = cand_section.get(item_key, {})
            all_fields = set(before) | set(after)
            for field in sorted(all_fields, key=str):
                b = _jsonify(before.get(field))
                a = _jsonify(after.get(field))
                if b != a:
                    rows.append(
                        {
                            "item_key": str(item_key),
                            "field": str(field),
                            "before": b,
                            "after": a,
                        }
                    )
        return rows

    return {
        "zone_changes": _section_diff("zones"),
        "path_changes": _section_diff("paths"),
    }

def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def choose_path_template(paths: dict[str, dict]) -> dict:
    return copy.deepcopy(max(paths.values(), key=lambda p: len(p.keys())))


def build_node_to_prj_index(network_dict: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for zone_key, zone in network_dict["zones"].items():
        node_id = str(zone.get("userName", zone_key))
        mapping[node_id] = str(zone.get("prjIndex"))
    return mapping


def _edge_position_tuple(path_dict: dict) -> tuple[float, float, float] | None:
    try:
        return (
            float(path_dict.get("position_x")),
            float(path_dict.get("position_y")),
            float(path_dict.get("position_z")),
        )
    except Exception:
        return None


def _zone_position_tuple(zone_dict: dict) -> tuple[float, float, float] | None:
    try:
        return (
            float(zone_dict.get("position_x")),
            float(zone_dict.get("position_y")),
            float(zone_dict.get("position_z")),
        )
    except Exception:
        return None


def _mean_xyz(points: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    if not points:
        return None
    arr = np.asarray(points, dtype=float)
    return float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])), float(np.mean(arr[:, 2]))


def _collect_zone_connected_path_points(paths: dict, zone_prj: str) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for p in paths.values():
        fz = str(p.get("fromZone", ""))
        tz = str(p.get("toZone", ""))
        if fz != zone_prj and tz != zone_prj:
            continue
        xyz = _edge_position_tuple(p)
        if xyz is not None:
            pts.append(xyz)
    return pts


def _infer_new_edge_xyz(
    zones: dict,
    paths: dict,
    u_prj: str | None,
    v_prj: str | None,
    is_outside_edge: bool,
) -> tuple[float, float, float]:
    if is_outside_edge:
        if u_prj is None:
            raise ValueError("Outside edge must have an indoor source zone.")
        pts = _collect_zone_connected_path_points(paths, str(u_prj))
        mean_xyz = _mean_xyz(pts)
        if mean_xyz is not None:
            return mean_xyz
        for z in zones.values():
            if str(z.get("prjIndex")) == str(u_prj):
                zxyz = _zone_position_tuple(z)
                if zxyz is not None:
                    return zxyz
        raise ValueError(f"Cannot infer xyz for new outside edge from zone prjIndex={u_prj}.")

    if u_prj is None or v_prj is None:
        raise ValueError("Indoor edge requires two indoor zone endpoints.")
    pts = _collect_zone_connected_path_points(paths, str(u_prj))
    pts.extend(_collect_zone_connected_path_points(paths, str(v_prj)))
    mean_xyz = _mean_xyz(pts)
    if mean_xyz is not None:
        return mean_xyz

    u_xyz, v_xyz = None, None
    for z in zones.values():
        if str(z.get("prjIndex")) == str(u_prj):
            u_xyz = _zone_position_tuple(z)
        elif str(z.get("prjIndex")) == str(v_prj):
            v_xyz = _zone_position_tuple(z)
    if u_xyz is None or v_xyz is None:
        raise ValueError(
            f"Cannot infer xyz for new indoor edge with zone prjIndex={u_prj}->{v_prj}."
        )
    return (
        float((u_xyz[0] + v_xyz[0]) * 0.5),
        float((u_xyz[1] + v_xyz[1]) * 0.5),
        float((u_xyz[2] + v_xyz[2]) * 0.5),
    )


def sync_lpg_structure_to_network_dict(
    model: Any,
    graph: Any,
    base_network_dict: dict,
    default_edge_attrs: dict,
) -> dict:
    network_dict = copy.deepcopy(base_network_dict)
    zones = network_dict["zones"]
    paths = network_dict["paths"]
    outside = model.LPG_OUTSIDE_NODE

    zone_node_to_key = {str(z.get("userName", k)): k for k, z in zones.items()}
    for node_id, attrs in graph.nodes(data=True):
        if node_id == outside:
            continue
        zone_key = zone_node_to_key.get(str(node_id))
        if zone_key is None:
            continue
        for field in model.LPG_ZONE_FIELDS:
            if field in attrs:
                zones[zone_key][field] = attrs[field]

    path_id_to_key: dict[str, str] = {}
    for path_key, path_value in paths.items():
        path_id_to_key[str(path_key)] = path_key
        if path_value.get("userName") is not None:
            path_id_to_key[str(path_value["userName"])] = path_key

    node_to_prj = build_node_to_prj_index(network_dict)
    path_template = choose_path_template(paths)
    prj_indices = [
        int(_to_float(p.get("prjIndex"), 0))
        for p in paths.values()
        if str(p.get("prjIndex", "")).strip() != ""
    ]
    next_prj = (max(prj_indices) + 1) if prj_indices else 1

    def _finalize_new_path_dict(
        edge_id: str,
        from_prj: str,
        to_prj: str,
        attrs: dict,
        xyz: tuple[float, float, float],
        is_outside_edge: bool,
    ) -> dict:
        nonlocal next_prj
        new_path = copy.deepcopy(path_template)
        new_path["userName"] = str(edge_id)
        new_path["prjIndex"] = str(next_prj)
        next_prj += 1
        new_path["fromZone"] = str(from_prj)
        new_path["toZone"] = str(to_prj)
        new_path["position_x"] = str(float(xyz[0]))
        new_path["position_y"] = str(float(xyz[1]))
        new_path["position_z"] = str(float(xyz[2]))
        new_path["element"] = ""
        new_path["operable"] = str(float(default_edge_attrs.get("operable", 0.5)))
        if new_path.get("orientation") in (None, "", "None"):
            new_path["orientation"] = "Vector(0.00,0.00,1.00)"
        if is_outside_edge:
            new_path["winType"] = "0"
        elif new_path.get("winType") in (None, "", "None"):
            new_path["winType"] = "1"
        for field in model.LPG_PATH_FIELDS:
            value = attrs.get(field, default_edge_attrs[field])
            if field == "pressure" and not is_outside_edge:
                value = 0.0
            new_path[field] = str(float(value))
        return new_path

    current_indoor_edge_ids: set[str] = set()
    current_outside_edge_ids: set[str] = set()
    for u, v, edge_key, attrs in graph.edges(keys=True, data=True):
        edge_key = str(edge_key)
        is_outside = (u == outside or v == outside)
        path_key = path_id_to_key.get(edge_key)

        if is_outside:
            if str(u) == outside and path_key is None:
                raise ValueError(f"Illegal OUTSIDE->zone edge detected: {(u, v, edge_key)}")
            current_outside_edge_ids.add(edge_key)
            if path_key is None:
                from_prj = str(node_to_prj[str(u)])
                to_prj = "-1"
                xyz = _infer_new_edge_xyz(
                    zones=zones,
                    paths=paths,
                    u_prj=from_prj,
                    v_prj=None,
                    is_outside_edge=True,
                )
                paths[edge_key] = _finalize_new_path_dict(
                    edge_id=edge_key,
                    from_prj=from_prj,
                    to_prj=to_prj,
                    attrs=attrs,
                    xyz=xyz,
                    is_outside_edge=True,
                )
                path_id_to_key[edge_key] = edge_key
            else:
                for field in model.LPG_PATH_FIELDS:
                    if field in attrs:
                        value = attrs[field]
                        if field == "pressure":
                            paths[path_key][field] = str(float(value))
                        else:
                            paths[path_key][field] = value
            continue

        current_indoor_edge_ids.add(edge_key)
        if path_key is None:
            from_prj = str(node_to_prj[str(u)])
            to_prj = str(node_to_prj[str(v)])
            xyz = _infer_new_edge_xyz(
                zones=zones,
                paths=paths,
                u_prj=from_prj,
                v_prj=to_prj,
                is_outside_edge=False,
            )
            paths[edge_key] = _finalize_new_path_dict(
                edge_id=edge_key,
                from_prj=from_prj,
                to_prj=to_prj,
                attrs=attrs,
                xyz=xyz,
                is_outside_edge=False,
            )
            path_id_to_key[edge_key] = edge_key
        else:
            paths[path_key]["fromZone"] = str(node_to_prj[str(u)])
            paths[path_key]["toZone"] = str(node_to_prj[str(v)])
            for field in model.LPG_PATH_FIELDS:
                if field in attrs:
                    value = attrs[field]
                    if field == "pressure":
                        paths[path_key][field] = "0.0"
                    else:
                        paths[path_key][field] = value

    to_delete: list[str] = []
    for path_key, p in paths.items():
        from_zone = str(p.get("fromZone", ""))
        to_zone = str(p.get("toZone", ""))
        edge_id = str(p.get("userName", path_key))
        if from_zone == "-1" or to_zone == "-1":
            if edge_id.startswith("cem_mech_") and edge_id not in current_outside_edge_ids:
                to_delete.append(path_key)
            continue
        if edge_id not in current_indoor_edge_ids:
            to_delete.append(path_key)
    for key in to_delete:
        del paths[key]

    return network_dict
