from dataclasses import dataclass
from typing import Any

import numpy as np


# Node (zone) perturbation bounds
ZONE_BOUNDS = {
    "zone_wallU": (0.15, 0.8),
    "zone_winU": (1.3, 2.3),
    "zone_win_SHGC": (0.3, 0.8),
}

# Edge (path) perturbation bounds
PATH_BOUNDS = {
    "pathHeight": (0.50, 3.0),
    "pathWidth": (0.50, 5.0),
    "pressure": (-0.01, 0.01),
}

NEW_EDGE_PRESSURE_BOUNDS = (-0.01, 0.01)

DEFAULT_EDGE_ATTRS = {
    "operable": 0.5,
    "pathHeight": 0.5,
    "pathWidth": 1.5,
    "pressure": 0.0,
}


@dataclass
class AttributeSpec:
    owner_type: str  # node or edge
    owner_id: Any
    field: str
    lower: float
    upper: float
    init: float


@dataclass
class EdgeOption:
    u: str
    v: str
    key: str
    exists_in_base: bool
    base_attrs: dict


@dataclass
class LPGSchema:
    outside_node: str
    zone_nodes: list[str]
    locked_outside_edges: list[tuple[str, str, str]]
    candidate_edges: list[EdgeOption]
    z_base: np.ndarray
    attr_specs: list[AttributeSpec]
