from __future__ import annotations

import argparse
import copy
import csv
import json
import contextlib
import os
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

try:
    from . import constant as C
    from .evaluation import evaluate_candidate
    from .structure import AttributeSpec, EdgeOption
except ImportError:
    import constant as C
    from evaluation import evaluate_candidate
    from structure import AttributeSpec, EdgeOption


@dataclass
class Candidate:
    """Optimization candidate container.

    Attributes
    ----------
    z : np.ndarray
        Binary structure decision vector for candidate edges.
    x : np.ndarray
        Continuous attribute decision vector.
    J : float
        Scalar fitness value (lower is better).
    graph : nx.MultiDiGraph | None
        Materialized candidate graph. ``None`` means the candidate is invalid.
    error : str, optional
        Error or invalidation reason. Empty string means valid/success.
    """
    z: np.ndarray
    x: np.ndarray
    J: float
    graph: nx.MultiDiGraph | None
    error: str = ""


@dataclass
class BiLevelCEMConfig:
    """Configuration for bi-level CEM and simulator runtime.

    Notes
    -----
    This object combines:
    - CEM hyper-parameters (sampling counts, elite ratio, smoothing factors)
    - structural constraints (max edges, sampling repair limits)
    - simulation runtime parameters consumed by evaluation
    - worker count for optional in-loop parallel evaluation
    """
    seed: int = C.SEED
    outer_iters: int = C.OUTER_ITERS
    structure_samples: int = C.STRUCTURE_SAMPLES
    inner_iters: int = C.INNER_ITERS
    attribute_samples: int = C.ATTRIBUTE_SAMPLES
    elite_ratio: float = C.ELITE_RATIO
    alpha_s: float = C.ALPHA_S
    alpha_a: float = C.ALPHA_A
    p_min: float = C.P_MIN
    p_max: float = C.P_MAX
    sigma_floor_ratio: float = C.SIGMA_FLOOR_RATIO
    max_indoor_edges: int | None = C.MAX_INDOOR_EDGES
    fail_penalty: float = C.FAIL_PENALTY
    mode: str = C.MODE
    timestep: int = C.TIMESTEP
    iteration: int = C.ITERATION
    preheat: int = C.PREHEAT
    k: float = C.K
    sigma: float = C.SIGMA
    start_hoy: int = C.START_HOY
    end_hoy: int = C.END_HOY
    objective: str = "total_energy_vent"
    max_new_edges_per_pair: int = C.MAX_NEW_EDGES_PER_INDOOR_PAIR
    max_new_outside_edges_per_zone: int = C.MAX_NEW_OUTSIDE_EDGES_PER_ZONE
    default_edge_attrs: dict | None = None
    structure_resample_multiplier: int = C.STRUCTURE_RESAMPLE_MULTIPLIER
    workers: int = C.WORKERS


@dataclass
class LPGSchema:
    """Search-space schema and immutable base-graph context.

    Attributes
    ----------
    outside_node : str
        Identifier of the outside/reference node.
    zone_nodes : list[str]
        Indoor zone node identifiers participating in optimization.
    locked_outside_edges : list[tuple[str, str, str]]
        Outside-related edges that must remain present.
    candidate_edges : list[EdgeOption]
        All structural edge options (existing and creatable).
    z_base : np.ndarray
        Baseline binary structure vector aligned with ``candidate_edges``.
    attr_specs : list[AttributeSpec]
        Continuous attribute decision definitions with bounds and initials.
    base_graph : nx.MultiDiGraph
        Immutable source graph used for reconstruction and comparisons.
    """
    outside_node: str
    zone_nodes: list[str]
    locked_outside_edges: list[tuple[str, str, str]]
    candidate_edges: list[EdgeOption]
    z_base: np.ndarray
    attr_specs: list[AttributeSpec]
    base_graph: nx.MultiDiGraph


_WORKER_METADATA: dict[str, Any] = {}


def _rewrite_station_artifacts(
    rdf_file: str | Path,
    station_id: str,
    output_dir: str | Path,
) -> Path:
    """Create a station-specific RDF copy without mutating source files."""
    station_id = str(station_id).strip()
    if not station_id:
        raise ValueError("station_id must not be empty.")

    rdf_path = Path(rdf_file)
    temp_dir = Path(output_dir) / "_station_inputs" / station_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    rdf_text = rdf_path.read_text(encoding="utf-8")

    rdf_text = rdf_text.replace("545110.csv", f"{station_id}.csv")
    rdf_text = rdf_text.replace("cumsky_545110.csv", f"cumsky_{station_id}.csv")
    rdf_text = rdf_text.replace('"545110"', f'"{station_id}"')
    rdf_text = rdf_text.replace("<545110>", f"<{station_id}>")
    rdf_text = re.sub(
        r'bes:stationId\s+"[^"]+"\s*\.',
        f'bes:stationId "{station_id}" .',
        rdf_text,
    )

    station_rdf = temp_dir / rdf_path.name
    station_rdf.write_text(rdf_text, encoding="utf-8")
    return station_rdf


def _init_opt_worker(
    rdf_file: str,
    cfg_payload: dict[str, Any],
    trace_root: str | None = None,
) -> None:
    """Initialize a process worker with model/config metadata.

    Parameters
    ----------
    rdf_file : str
        RDF model file path.
    cfg_payload : dict[str, Any]
        Serialized :class:`BiLevelCEMConfig`.
    """
    global _WORKER_METADATA
    with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model_obj = C.loadModel(str(Path(rdf_file)))
    model = C.heatLoadModel(model_obj)
    base_network_dict = copy.deepcopy(model.networkDict)
    config = BiLevelCEMConfig(**cfg_payload)
    _WORKER_METADATA = {
        "model": model,
        "base_network_dict": base_network_dict,
        "config": config,
        "trace_root": trace_root,
    }


def _evaluate_graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one serialized graph payload in a worker process.

    Parameters
    ----------
    payload : dict[str, Any]
        Payload containing a case index and node-link graph json.

    Returns
    -------
    dict[str, Any]
        Evaluation result with ``idx``, ``J`` and ``error``.
    """
    graph = json_graph.node_link_graph(payload["graph_payload"], directed=True, multigraph=True)
    metadata = dict(_WORKER_METADATA)
    if "trace_context" in payload:
        metadata["trace_context"] = payload["trace_context"]
    j, err = evaluate_candidate(graph, metadata)
    return {"idx": int(payload["idx"]), "J": float(j), "error": str(err)}


class StructureCEM:
    """Bernoulli CEM updater for discrete structural decisions.

    The class maintains a Bernoulli parameter vector ``p`` and supports:
    - sampling a binary structure vector
    - validating/repairing edge-count constraints
    - updating ``p`` from elite samples
    """

    def __init__(
        self,
        p0: np.ndarray,
        p_min: float,
        p_max: float,
        alpha: float,
        max_indoor_edges: int | None = None,
    ):
        """Initialize structure sampler state.

        Parameters
        ----------
        p0 : np.ndarray
            Initial Bernoulli probabilities for each structural variable.
        p_min : float
            Lower clip bound for probabilities during updates.
        p_max : float
            Upper clip bound for probabilities during updates.
        alpha : float
            Exponential smoothing factor for parameter updates.
        max_indoor_edges : int | None, optional
            Maximum allowed number of active indoor edges.
        """
        self.p = np.asarray(p0, dtype=float)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.alpha = float(alpha)
        self.max_indoor_edges = max_indoor_edges

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a feasible structure vector.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator used for Bernoulli sampling.

        Returns
        -------
        np.ndarray
            Binary vector ``z`` aligned with schema candidate edges.

        Notes
        -----
        The sampler retries up to 100 times for a valid sample, then falls back
        to thresholding and repair.
        """
        for _ in range(100):
            z = rng.binomial(1, self.p).astype(int)
            if self.is_valid(z):
                return z
        z = (self.p >= 0.5).astype(int)
        return self.repair(z)

    def is_valid(self, z: np.ndarray) -> bool:
        """Check whether a structure satisfies edge-count constraint.

        Parameters
        ----------
        z : np.ndarray
            Binary structure vector.

        Returns
        -------
        bool
            ``True`` when ``z`` satisfies max-indoor-edge constraint.
        """
        if self.max_indoor_edges is None:
            return True
        return int(np.sum(z)) <= int(self.max_indoor_edges)

    def repair(self, z: np.ndarray) -> np.ndarray:
        """Repair an invalid structure via greedy edge removal.

        Parameters
        ----------
        z : np.ndarray
            Potentially invalid binary structure vector.

        Returns
        -------
        np.ndarray
            Repaired vector satisfying structural count limit.
        """
        if self.max_indoor_edges is None:
            return z
        z = z.copy()
        while int(np.sum(z)) > int(self.max_indoor_edges):
            on_idx = np.where(z == 1)[0]
            if len(on_idx) == 0:
                break
            drop = on_idx[np.argmin(self.p[on_idx])]
            z[drop] = 0
        return z

    def update(self, elite_z: list[np.ndarray]) -> None:
        """Update Bernoulli probabilities from elite samples.

        Parameters
        ----------
        elite_z : list[np.ndarray]
            Elite binary vectors selected by current iteration fitness.
        """
        arr = np.asarray(elite_z, dtype=float)
        p_mle = np.mean(arr, axis=0)
        self.p = (1.0 - self.alpha) * self.p + self.alpha * p_mle
        self.p = np.clip(self.p, self.p_min, self.p_max)


class AttributeCEM:
    """Gaussian CEM updater for continuous attribute decisions.

    The class maintains ``mu`` and ``sigma`` for each continuous variable and
    clips sampled values to variable bounds.
    """

    def __init__(
        self,
        attr_specs: list[AttributeSpec],
        alpha: float,
        sigma_floor_ratio: float,
    ):
        """Initialize attribute sampler state.

        Parameters
        ----------
        attr_specs : list[AttributeSpec]
            Attribute variable definitions and bounds.
        alpha : float
            Exponential smoothing factor for ``mu``/``sigma`` updates.
        sigma_floor_ratio : float
            Minimum ``sigma`` ratio relative to variable range.
        """
        self.attr_specs = attr_specs
        self.alpha = float(alpha)
        self.mu = np.asarray([s.init for s in attr_specs], dtype=float)
        self.sigma = np.asarray(
            [max((s.upper - s.lower) * 0.25, 1e-6) for s in attr_specs], dtype=float
        )
        self.sigma_floor = np.asarray(
            [max((s.upper - s.lower) * sigma_floor_ratio, 1e-6) for s in attr_specs],
            dtype=float,
        )

    def copy(self) -> "AttributeCEM":
        """Create an independent copy of the current sampler state.

        Returns
        -------
        AttributeCEM
            New sampler with copied ``mu``, ``sigma`` and ``sigma_floor``.
        """
        other = AttributeCEM(self.attr_specs, self.alpha, 0.01)
        other.mu = self.mu.copy()
        other.sigma = self.sigma.copy()
        other.sigma_floor = self.sigma_floor.copy()
        return other

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample one bounded attribute vector.

        Parameters
        ----------
        rng : np.random.Generator
            Random generator used for Gaussian sampling.

        Returns
        -------
        np.ndarray
            Sampled and bound-clipped continuous vector ``x``.
        """
        x = rng.normal(self.mu, self.sigma)
        lower = np.asarray([s.lower for s in self.attr_specs], dtype=float)
        upper = np.asarray([s.upper for s in self.attr_specs], dtype=float)
        return np.clip(x, lower, upper)

    def update(self, elite_x: list[np.ndarray]) -> None:
        """Update Gaussian parameters from elite attribute samples.

        Parameters
        ----------
        elite_x : list[np.ndarray]
            Elite continuous vectors selected by current iteration fitness.

        Notes
        -----
        ``sigma`` is lower-bounded by ``sigma_floor`` to avoid premature collapse.
        """
        arr = np.asarray(elite_x, dtype=float)
        mu_mle = np.mean(arr, axis=0)
        sigma_mle = np.std(arr, axis=0)
        self.mu = (1.0 - self.alpha) * self.mu + self.alpha * mu_mle
        self.sigma = (1.0 - self.alpha) * self.sigma + self.alpha * sigma_mle
        self.sigma = np.maximum(self.sigma, self.sigma_floor)


def _to_float(value: Any, default: float) -> float:
    """Convert value to float with fallback default."""
    try:
        return float(value)
    except Exception:
        return float(default)


def outside_edge_signature(
    graph: nx.MultiDiGraph, outside_node: str
) -> tuple[tuple[str, str, str], ...]:
    """Build canonical signature for all edges incident to outside node."""
    sig = []
    for u, v, k in graph.edges(keys=True):
        if u == outside_node or v == outside_node:
            sig.append((str(u), str(v), str(k)))
    return tuple(sorted(sig))


def validate_outside_edges_locked(
    graph: nx.MultiDiGraph, outside_node: str, locked_signature: tuple[tuple[str, str, str], ...]
) -> bool:
    """Validate lock constraints for outside-related edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Candidate graph to validate.
    outside_node : str
        Outside node id.
    locked_signature : tuple[tuple[str, str, str], ...]
        Immutable outside edge signature from base graph.

    Returns
    -------
    bool
        ``True`` if constraints are satisfied.
    """
    current = outside_edge_signature(graph, outside_node)
    locked_set = set(locked_signature)
    current_set = set(current)
    if not locked_set.issubset(current_set):
        return False
    for u, v, k in current:
        if (u, v, k) in locked_set:
            continue
        # Only allow newly added mechanical edges as zone -> OUTSIDE.
        if u == outside_node:
            return False
        if v != outside_node:
            return False
    return True


def validate_structure_graph(graph: nx.MultiDiGraph, outside_node: str) -> tuple[bool, str]:
    """Validate basic structural feasibility/connectivity of candidate graph.

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, reason)`` where ``reason`` is empty when valid.
    """
    zone_nodes = [n for n in graph.nodes if n != outside_node]
    if not zone_nodes:
        return False, "no_zone_nodes"

    undir = nx.Graph()
    undir.add_nodes_from(graph.nodes)
    for u, v, _k in graph.edges(keys=True):
        undir.add_edge(u, v)

    for zn in zone_nodes:
        if undir.degree(zn) <= 0:
            return False, "isolated_zone_node"

    if outside_node in undir.nodes:
        for zn in zone_nodes:
            if not nx.has_path(undir, zn, outside_node):
                return False, "zone_not_connected_to_outside"
    else:
        comps = list(nx.connected_components(undir.subgraph(zone_nodes)))
        if len(comps) > 1:
            return False, "multi_indoor_components_without_outside"

    return True, ""



def build_LPGschema(
    rdf_file: str | Path,
    config: BiLevelCEMConfig | None = None,
) -> tuple[LPGSchema, dict[str, Any]]:
    """Build schema and metadata as the workflow-switching boundary.

    Parameters
    ----------
    rdf_file : str | Path
        Model file used to create simulation object.
    config : BiLevelCEMConfig | None, optional
        Optimization config. Defaults to project defaults.

    Returns
    -------
    tuple[LPGSchema, dict[str, Any]]
        ``schema`` defines search variables and base graph; ``metadata`` contains
        evaluation/runtime context passed to evaluation layer.
    """
    def extract_schema(
    model: C.heatLoadModel,
    base_graph: nx.MultiDiGraph,
    base_network_dict: dict,
    config: BiLevelCEMConfig,
    ) -> LPGSchema:
        """Derive candidate edges and attribute variables from base graph."""
        outside = model.LPG_OUTSIDE_NODE
        zone_nodes = sorted([n for n in base_graph.nodes if n != outside], key=str)

        locked_outside_edges: list[tuple[str, str, str]] = []
        existing_indoor: list[EdgeOption] = []
        existing_pair_to_count: dict[tuple[str, str], int] = {}

        for u, v, k, attrs in base_graph.edges(keys=True, data=True):
            edge_key = str(k)
            if u == outside or v == outside:
                locked_outside_edges.append((str(u), str(v), edge_key))
            else:
                existing_indoor.append(
                    EdgeOption(
                        u=str(u),
                        v=str(v),
                        key=edge_key,
                        exists_in_base=True,
                        base_attrs={f: attrs.get(f) for f in C.LPG_PATH_FIELDS},
                    )
                )
                pair = (str(u), str(v))
                existing_pair_to_count[pair] = existing_pair_to_count.get(pair, 0) + 1

        candidate_edges: list[EdgeOption] = list(existing_indoor)
        for u in zone_nodes:
            for v in zone_nodes:
                if u == v:
                    continue
                pair = (u, v)
                if pair in existing_pair_to_count:
                    continue
                for idx in range(1, int(config.max_new_edges_per_pair) + 1):
                    key = f"cem_indoor_{u}_to_{v}_{idx}"
                    candidate_edges.append(
                        EdgeOption(
                            u=u,
                            v=v,
                            key=key,
                            exists_in_base=False,
                            base_attrs={**config.default_edge_attrs},
                        )
                    )
        for u in zone_nodes:
            for idx in range(1, int(config.max_new_outside_edges_per_zone) + 1):
                key = f"cem_mech_{u}_to_outside_{idx}"
                candidate_edges.append(
                    EdgeOption(
                        u=u,
                        v=outside,
                        key=key,
                        exists_in_base=False,
                        base_attrs={**config.default_edge_attrs},
                    )
                )

        z_base = np.asarray([1 if e.exists_in_base else 0 for e in candidate_edges], dtype=int)

        attr_specs: list[AttributeSpec] = []
        for n in zone_nodes:
            attrs = base_graph.nodes[n]
            for field, (lower, upper) in C.ZONE_BOUNDS.items():
                init = _to_float(attrs.get(field), (lower + upper) * 0.5)
                init = float(np.clip(init, lower, upper))
                attr_specs.append(
                    AttributeSpec("node", str(n), field, float(lower), float(upper), init)
                )

        seen_edge_owner = set()
        for u, v, k, attrs in base_graph.edges(keys=True, data=True):
            owner_id = (str(u), str(v), str(k))
            if owner_id in seen_edge_owner:
                continue
            seen_edge_owner.add(owner_id)
            for field, (lower, upper) in C.PATH_BOUNDS.items():
                if field == "pressure" and not (u == outside or v == outside):
                    continue
                init = _to_float(attrs.get(field), config.default_edge_attrs[field])
                init = float(np.clip(init, lower, upper))
                attr_specs.append(
                    AttributeSpec("edge", owner_id, field, float(lower), float(upper), init)
                )

        for edge_option in candidate_edges:
            if edge_option.exists_in_base:
                continue
            owner_id = (edge_option.u, edge_option.v, edge_option.key)
            for field, (lower, upper) in C.PATH_BOUNDS.items():
                if field == "pressure" and edge_option.v != outside:
                    continue
                if field == "pressure":
                    lower, upper = getattr(C, "NEW_EDGE_PRESSURE_BOUNDS", (lower, upper))
                init = _to_float(edge_option.base_attrs.get(field), config.default_edge_attrs[field])
                init = float(np.clip(init, lower, upper))
                attr_specs.append(
                    AttributeSpec("edge", owner_id, field, float(lower), float(upper), init)
                )

        return LPGSchema(
            outside_node=outside,
            zone_nodes=zone_nodes,
            locked_outside_edges=sorted(locked_outside_edges),
            candidate_edges=candidate_edges,
            z_base=z_base,
            attr_specs=attr_specs,
            base_graph=base_graph,
        )
    cfg = config or BiLevelCEMConfig(default_edge_attrs=dict(C.DEFAULT_EDGE_ATTRS))
    if cfg.default_edge_attrs is None:
        cfg.default_edge_attrs = dict(C.DEFAULT_EDGE_ATTRS)

    model_obj = C.loadModel(str(Path(rdf_file)))
    model = C.heatLoadModel(model_obj)
    base_network_dict = copy.deepcopy(model.networkDict)
    base_graph = model.networkDict_to_lpg(copy.deepcopy(base_network_dict))
    schema = extract_schema(model, base_graph, base_network_dict, cfg)

    metadata: dict[str, Any] = {
        "model": model,
        "base_network_dict": base_network_dict,
        "config": cfg,
        "rdf_file": str(Path(rdf_file)),
        "outside_node": schema.outside_node,
    }
    return schema, metadata


def assemble_graph(
    base_graph: nx.MultiDiGraph,
    schema: LPGSchema,
    z: np.ndarray,
    x: np.ndarray,
) -> nx.MultiDiGraph:
    """Assemble a concrete graph instance from structure and attribute vectors.

    Parameters
    ----------
    base_graph : nx.MultiDiGraph
        Immutable base graph carrying original node/edge attrs.
    schema : LPGSchema
        Search-space schema.
    z : np.ndarray
        Binary edge-selection vector.
    x : np.ndarray
        Continuous attribute vector.

    Returns
    -------
    nx.MultiDiGraph
        Candidate graph for evaluation.
    """
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((n, copy.deepcopy(attrs)) for n, attrs in base_graph.nodes(data=True))
    for u, v, k, attrs in base_graph.edges(keys=True, data=True):
        if u == schema.outside_node or v == schema.outside_node:
            graph.add_edge(u, v, key=k, **copy.deepcopy(attrs))

    for i, edge_option in enumerate(schema.candidate_edges):
        if int(z[i]) != 1:
            continue
        attrs = {}
        for field in C.LPG_PATH_FIELDS:
            attrs[field] = edge_option.base_attrs.get(field, C.DEFAULT_EDGE_ATTRS[field])
        graph.add_edge(edge_option.u, edge_option.v, key=edge_option.key, **attrs)

    for value, spec in zip(x, schema.attr_specs):
        clipped = float(np.clip(value, spec.lower, spec.upper))
        if spec.owner_type == "node":
            if spec.owner_id in graph.nodes:
                graph.nodes[spec.owner_id][spec.field] = clipped
        else:
            u, v, k = spec.owner_id
            if graph.has_edge(u, v, key=k):
                graph[u][v][k][spec.field] = clipped
    return graph

def select_elite(candidates: list[Candidate], elite_ratio: float) -> list[Candidate]:
    """Select top-performing valid candidates by elite ratio."""
    valid = [
        c
        for c in candidates
        if np.isfinite(c.J)
        and c.graph is not None
    ]
    if not valid:
        return []
    valid.sort(key=lambda c: c.J)
    n = max(1, int(np.ceil(float(elite_ratio) * len(valid))))
    return valid[:n]


def bernoulli_entropy(p: np.ndarray) -> float:
    """Compute mean Bernoulli entropy for structure distribution monitoring."""
    q = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    h = -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))
    return float(np.mean(h))


def bilevel_cem_optimize(
    config: BiLevelCEMConfig,
    schema: LPGSchema,
    metadata: dict[str, Any],
) -> tuple[Candidate, list[dict], LPGSchema, nx.MultiDiGraph]:
    """Run bi-level CEM optimization with optional in-loop parallel evaluation.

    Parameters
    ----------
    config : BiLevelCEMConfig
        Optimization/simulation configuration.
    schema : LPGSchema
        Search-space schema produced by :func:`build_LPGschema`.
    metadata : dict[str, Any]
        Evaluation runtime context produced by :func:`build_LPGschema`.

    Returns
    -------
    tuple[Candidate, list[dict], LPGSchema, nx.MultiDiGraph]
        Best candidate, iteration history, input schema and base graph.
    """

    rng = np.random.default_rng(config.seed)
    base_graph = schema.base_graph
    locked_sig = outside_edge_signature(base_graph, schema.outside_node)

    p0 = np.asarray(
        [0.97 if e.exists_in_base else 0.03 for e in schema.candidate_edges], dtype=float
    )
    base_indoor_edges = int(np.sum(schema.z_base))
    max_indoor_edges = config.max_indoor_edges
    if max_indoor_edges is None:
        max_indoor_edges = base_indoor_edges + max(2, int(0.1 * base_indoor_edges))
    struct_cem = StructureCEM(
        p0=p0,
        p_min=config.p_min,
        p_max=config.p_max,
        alpha=config.alpha_s,
        max_indoor_edges=max_indoor_edges,
    )
    attr_cem_global = AttributeCEM(
        attr_specs=schema.attr_specs,
        alpha=config.alpha_a,
        sigma_floor_ratio=config.sigma_floor_ratio,
    )

    base_x = np.asarray([s.init for s in schema.attr_specs], dtype=float)
    base_candidate_graph = assemble_graph(
        base_graph=base_graph, schema=schema, z=schema.z_base, x=base_x
    )
    base_eval_metadata = dict(metadata)
    base_eval_metadata["trace_context"] = {
        "city": metadata.get("city", ""),
        "station_id": metadata.get("station_id", ""),
        "outer_iter": "baseline",
        "inner_iter": "baseline",
        "structure_attempt": "baseline",
        "eval_index": "baseline",
        "valid_structure_sample_index": "baseline",
    }
    base_j, base_err = evaluate_candidate(base_candidate_graph, base_eval_metadata)
    if base_err:
        raise RuntimeError(f"Baseline candidate invalid: {base_err}")
    best = Candidate(
        z=schema.z_base.copy(),
        x=base_x.copy(),
        J=base_j,
        graph=base_candidate_graph,
    )

    history: list[dict] = []
    round_snapshots: list[dict[str, Any]] = []
    workers = max(1, int(metadata.get("workers", config.workers)))
    can_parallel = workers > 1 and "rdf_file" in metadata
    executor_ctx = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_opt_worker,
        initargs=(
            str(metadata["rdf_file"]),
            asdict(config),
            str(metadata.get("trace_root")) if metadata.get("trace_root") else None,
        ),
    ) if can_parallel else contextlib.nullcontext()

    with executor_ctx as executor:
        for outer_i in range(config.outer_iters):
            structure_records: list[Candidate] = []
            fail_count = 0
            invalid_structure_count = 0
            invalid_result_count = 0
            simulation_exception_count = 0
            attempts = 0
            total_eval_batches = 0
            target_valid_structures = int(config.structure_samples)
            max_attempts = max(1, target_valid_structures * int(config.structure_resample_multiplier))
            while len(structure_records) < target_valid_structures and attempts < max_attempts:
                attempts += 1
                z = struct_cem.sample(rng)
                attr_cem = attr_cem_global.copy()
                local_best: Candidate | None = None

                for inner_i in range(config.inner_iters):
                    batch: list[Candidate] = []
                    eval_graphs: list[tuple[int, np.ndarray, np.ndarray, nx.MultiDiGraph]] = []
                    for _ in range(config.attribute_samples):
                        sample_index = total_eval_batches
                        total_eval_batches += 1
                        x = attr_cem.sample(rng)
                        graph = assemble_graph(base_graph, schema, z, x)
                        if not validate_outside_edges_locked(graph, schema.outside_node, locked_sig):
                            batch.append(
                                Candidate(
                                    z=z.copy(),
                                    x=x.copy(),
                                    J=float(config.fail_penalty),
                                    graph=None,
                                    error="outside_edges_unlocked",
                                )
                            )
                            fail_count += 1
                            invalid_structure_count += 1
                            continue
                        ok_struct, struct_reason = validate_structure_graph(graph, schema.outside_node)
                        if not ok_struct:
                            batch.append(
                                Candidate(
                                    z=z.copy(),
                                    x=x.copy(),
                                    J=float(config.fail_penalty),
                                    graph=None,
                                    error=f"invalid_structure:{struct_reason}",
                                )
                            )
                            fail_count += 1
                            invalid_structure_count += 1
                            continue
                        eval_graphs.append((sample_index, z.copy(), x.copy(), graph))

                    if executor is not None:
                        payloads = [
                            {
                                "idx": idx,
                                "graph_payload": json_graph.node_link_data(item[3]),
                                "trace_context": {
                                    "city": metadata.get("city", ""),
                                    "station_id": metadata.get("station_id", ""),
                                    "outer_iter": outer_i,
                                    "inner_iter": inner_i,
                                    "structure_attempt": attempts,
                                    "eval_index": item[0],
                                    "valid_structure_sample_index": len(structure_records),
                                },
                            }
                            for idx, item in enumerate(eval_graphs)
                        ]
                        futures = [executor.submit(_evaluate_graph_payload, p) for p in payloads]
                        for fut in futures:
                            try:
                                res = fut.result()
                                _, zz, xx, gg = eval_graphs[int(res["idx"])]
                                eval_err = str(res["error"])
                                batch.append(
                                    Candidate(z=zz, x=xx, J=float(res["J"]), graph=gg, error=eval_err)
                                )
                                if eval_err:
                                    invalid_result_count += 1
                            except Exception as exc:
                                batch.append(
                                    Candidate(
                                        z=z.copy(),
                                        x=x.copy(),
                                        J=float(config.fail_penalty),
                                        graph=None,
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                )
                                fail_count += 1
                                simulation_exception_count += 1
                    else:
                        for sample_index, zz, xx, gg in eval_graphs:
                            try:
                                eval_metadata = dict(metadata)
                                eval_metadata["trace_context"] = {
                                    "city": metadata.get("city", ""),
                                    "station_id": metadata.get("station_id", ""),
                                    "outer_iter": outer_i,
                                    "inner_iter": inner_i,
                                    "structure_attempt": attempts,
                                    "eval_index": sample_index,
                                    "valid_structure_sample_index": len(structure_records),
                                }
                                j, eval_err = evaluate_candidate(gg, eval_metadata)
                                batch.append(Candidate(z=zz, x=xx, J=j, graph=gg, error=eval_err))
                                if eval_err:
                                    invalid_result_count += 1
                            except Exception as exc:
                                batch.append(
                                    Candidate(
                                        z=zz,
                                        x=xx,
                                        J=float(config.fail_penalty),
                                        graph=None,
                                        error=f"{type(exc).__name__}: {exc}",
                                    )
                                )
                                fail_count += 1
                                simulation_exception_count += 1

                    elite = select_elite(batch, config.elite_ratio)
                    if elite:
                        attr_cem.update([c.x for c in elite])
                        best_batch = min(elite, key=lambda c: c.J)
                        if local_best is None or best_batch.J < local_best.J:
                            local_best = best_batch

                if local_best is None:
                    fail_count += 1
                    continue
                structure_records.append(local_best)

            if len(structure_records) < target_valid_structures:
                raise RuntimeError(
                    "Failed to collect enough valid structure samples. "
                    f"valid={len(structure_records)} target={target_valid_structures} attempts={attempts}"
                )

            structure_records.sort(key=lambda c: c.J)
            elite_struct = select_elite(structure_records, config.elite_ratio)
            if elite_struct:
                struct_cem.update([c.z for c in elite_struct])
                attr_cem_global.update([c.x for c in elite_struct])

            round_best = structure_records[0]
            if round_best.J < best.J:
                best = round_best

            round_snapshots.append(
                {
                    "history": {
                        "iter": outer_i,
                        "best_J": float(best.J),
                        "round_best_J": float(round_best.J),
                        "mean_J": float(np.mean([c.J for c in structure_records])),
                        "p_entropy": bernoulli_entropy(struct_cem.p),
                        "indoor_edge_count": int(np.sum(round_best.z)),
                        "valid_structure_samples": int(len(structure_records)),
                        "structure_attempts": int(attempts),
                        "fail_rate": float(
                            fail_count
                            / max(1, total_eval_batches)
                        ),
                        "invalid_structure_count": int(invalid_structure_count),
                        "invalid_result_count": int(invalid_result_count),
                        "simulation_exception_count": int(simulation_exception_count),
                    },
                    "round_best_snapshot": _candidate_snapshot(round_best, base_graph=base_graph),
                }
            )

            history.append(
                round_snapshots[-1]["history"]
            )
            print(
                f"[outer={outer_i}] best={best.J:.6f} round_best={round_best.J:.6f} "
                f"mean={history[-1]['mean_J']:.6f} p_entropy={history[-1]['p_entropy']:.6f}"
            )

    return best, history, schema, base_graph


def edge_diff(base_graph: nx.MultiDiGraph, best_graph: nx.MultiDiGraph) -> list[dict]:
    """Generate edge-level added/removed diff table between base and best graph."""
    def edge_ids(graph: nx.MultiDiGraph) -> set[str]:
        ids = set()
        for u, v, k in graph.edges(keys=True):
            ids.add(f"{u}->{v}::{k}")
        return ids

    base_ids = edge_ids(base_graph)
    best_ids = edge_ids(best_graph)
    rows = []
    for k in sorted(base_ids - best_ids):
        rows.append({"edge_id": k, "change": "removed"})
    for k in sorted(best_ids - base_ids):
        rows.append({"edge_id": k, "change": "added"})
    return rows


def attribute_diff(
    base_graph: nx.MultiDiGraph, best_graph: nx.MultiDiGraph
) -> list[dict]:
    """Generate node/edge numeric attribute diff rows between two graphs."""
    rows: list[dict] = []
    for node in best_graph.nodes:
        if node not in base_graph.nodes:
            continue
        b0_attrs = base_graph.nodes[node]
        b1_attrs = best_graph.nodes[node]
        for field in set(b0_attrs) | set(b1_attrs):
            b0 = _to_float(b0_attrs.get(field), np.nan)
            b1 = _to_float(b1_attrs.get(field), np.nan)
            if np.isfinite(b0) and np.isfinite(b1) and abs(b1 - b0) > 1e-12:
                rows.append(
                    {
                        "owner_type": "node",
                        "owner_id": str(node),
                        "field": field,
                        "before": b0,
                        "after": b1,
                    }
                )

    base_edge_values: dict[tuple[str, str, str], dict] = {}
    for u, v, k, attrs in base_graph.edges(keys=True, data=True):
        base_edge_values[(str(u), str(v), str(k))] = attrs
    for u, v, k, attrs in best_graph.edges(keys=True, data=True):
        owner = (str(u), str(v), str(k))
        base_attrs = base_edge_values.get(owner)
        if base_attrs is None:
            continue
        for field in set(base_attrs) | set(attrs):
            b0 = _to_float(base_attrs.get(field), np.nan)
            b1 = _to_float(attrs.get(field), np.nan)
            if np.isfinite(b0) and np.isfinite(b1) and abs(b1 - b0) > 1e-12:
                rows.append(
                    {
                        "owner_type": "edge",
                        "owner_id": f"{owner[0]}->{owner[1]}::{owner[2]}",
                        "field": field,
                        "before": b0,
                        "after": b1,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows as CSV; writes empty file when rows are empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


def _candidate_snapshot(candidate: Candidate, base_graph: nx.MultiDiGraph | None = None) -> dict[str, Any]:
    snapshot = {
        "fitness": float(candidate.J),
        "error": str(candidate.error),
        "indoor_edge_count": int(np.sum(candidate.z)),
        "z": _jsonify(candidate.z),
        "x": _jsonify(candidate.x),
        "graph": _jsonify(json_graph.node_link_data(candidate.graph)) if candidate.graph is not None else None,
    }
    if base_graph is not None and candidate.graph is not None:
        snapshot["edge_diff"] = edge_diff(base_graph, candidate.graph)
        snapshot["attribute_diff"] = attribute_diff(base_graph, candidate.graph)
    return snapshot


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)


def _resolve_run_dir(base_dir: str | Path, stamp: str) -> Path:
    base_path = Path(base_dir)
    if base_path.name == stamp:
        return base_path
    return base_path / stamp


def parse_args() -> argparse.Namespace:
    """Parse optimization CLI arguments."""
    parser = argparse.ArgumentParser(description="Bi-level CEM optimization entrypoint.")
    parser.add_argument("--rdf", type=str, default=str(C.BASE_RDF_PATH))
    parser.add_argument("--output-dir", type=str, default=str(C.OUTPUT_DIR))
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=int(C.WORKERS))
    parser.add_argument("--station_id", type=str, default=None)
    parser.add_argument("--mini", action="store_true", help="Run user-requested minimal CEM case.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BiLevelCEMConfig:
    """Build :class:`BiLevelCEMConfig` from CLI args (including mini mode)."""
    cfg = BiLevelCEMConfig(default_edge_attrs=dict(C.DEFAULT_EDGE_ATTRS))
    cfg.workers = int(args.workers)
    if args.mini:
        # Older checkouts do not define dedicated mini constants.
        cfg.mode = getattr(C, "MINI_MODE", C.MODE)
        cfg.start_hoy = getattr(C, "MINI_START_HOY", C.START_HOY)
        cfg.end_hoy = getattr(C, "MINI_END_HOY", C.END_HOY)
        cfg.outer_iters = getattr(C, "MINI_OUTER_ITERS", 1)
        cfg.structure_samples = getattr(C, "MINI_STRUCTURE_SAMPLES", 2)
        cfg.inner_iters = getattr(C, "MINI_INNER_ITERS", 1)
        cfg.attribute_samples = getattr(C, "MINI_ATTRIBUTE_SAMPLES", 2)
    return cfg


def main() -> None:
    """CLI entrypoint: build schema, run optimization, and persist artifacts."""
    args = parse_args()
    run_stamp = Path(args.log_dir).name if args.log_dir else datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    out_dir = _resolve_run_dir(args.output_dir, run_stamp)
    log_base = Path(args.log_dir) if args.log_dir else Path(C.LOG_DIR)
    log_dir = _resolve_run_dir(log_base, run_stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    trace_root = log_dir / "case_metrics"
    trace_root.mkdir(parents=True, exist_ok=True)
    rdf_file = Path(args.rdf)
    if args.station_id:
        rdf_file = _rewrite_station_artifacts(
            rdf_file=rdf_file,
            station_id=str(args.station_id),
            output_dir=out_dir,
        )
        print(f"[LOAD] station_id={args.station_id} -> rdf={rdf_file}")

    config = build_config(args)
    run_manifest = {
        "rdf": str(rdf_file),
        "output_dir": str(out_dir),
        "log_dir": str(log_dir),
        "workers": int(args.workers),
        "station_id": str(args.station_id) if args.station_id else None,
        "mini": bool(args.mini),
        "trace_root": str(trace_root),
        "config": asdict(config),
    }
    _write_json(log_dir / "run_manifest.json", run_manifest)

    print("[LOAD] model and LPG schema")
    schema, metadata = build_LPGschema(
        rdf_file=rdf_file,
        config=config,
    )
    metadata["workers"] = int(args.workers)
    metadata["trace_root"] = str(trace_root)
    if args.station_id:
        metadata["station_id"] = str(args.station_id)

    print("[RUN] bilevel cem optimization")
    best, history, schema, base_graph = bilevel_cem_optimize(config, schema, metadata)
    if best.graph is None:
        raise RuntimeError("Optimization finished without a valid candidate.")

    print(f"[DONE] best fitness={best.J:.6f}")
    write_csv(out_dir / "history.csv", history)
    with (out_dir / "best_lpg.json").open("w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(best.graph), f, ensure_ascii=False, indent=2)
    _write_json(
        out_dir / "best_metrics.json",
        {
            "best_fitness": float(best.J),
            "best_indoor_edge_count": int(np.sum(best.z)),
            "history_iterations": int(len(history)),
            "trace_root": str(trace_root),
        },
    )
    _write_json(out_dir / "best_snapshot.json", _candidate_snapshot(best, base_graph=base_graph))

    edge_rows = edge_diff(base_graph, best.graph)
    write_csv(out_dir / "edge_diff.csv", edge_rows)
    write_csv(out_dir / "structure_diff.csv", edge_rows)
    
    attr_rows = attribute_diff(base_graph, best.graph)
    write_csv(out_dir / "attribute_diff.csv", attr_rows)
    write_csv(out_dir / "parameter_diff.csv", attr_rows)

    round_root = out_dir / "round_metrics"
    round_root.mkdir(parents=True, exist_ok=True)
    for idx, snapshot in enumerate(round_snapshots):
        _write_json(round_root / f"iter_{idx:03d}.json", snapshot)


if __name__ == "__main__":
    main()
