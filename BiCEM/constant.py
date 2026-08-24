from pathlib import Path
import json

from MoosasPy import loadModel as MOOSAS_LOAD_MODEL
from MoosasPy.energy import heatLoadModel as MOOSAS_HEATLOAD_MODEL
from structure import DEFAULT_EDGE_ATTRS, NEW_EDGE_PRESSURE_BOUNDS, PATH_BOUNDS, ZONE_BOUNDS


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"
LOG_DIR = REPO_ROOT / "logs"

# Default input artifacts packed with this dist project
BASE_RDF_PATH = REPO_ROOT / "data" / "test_v5.rdf"
BASE_NETWORKDICT_PATH = REPO_ROOT / "data" / "heatModel.json"
LPG_PATH_FIELDS = tuple(PATH_BOUNDS.keys())

MAX_NEW_EDGES_PER_INDOOR_PAIR = 1
MAX_NEW_OUTSIDE_EDGES_PER_ZONE = 1

SEED = 44
OUTER_ITERS = 10
STRUCTURE_SAMPLES = 15
INNER_ITERS = 1
ATTRIBUTE_SAMPLES = 20
WORKERS = 6
ELITE_RATIO = 0.2
ALPHA_S = 0.5
ALPHA_A = 0.4
P_MIN = 0.02
P_MAX = 0.98
SIGMA_FLOOR_RATIO = 0.02
MAX_INDOOR_EDGES = None
FAIL_PENALTY = 1e9
STRUCTURE_RESAMPLE_MULTIPLIER = 80

MODE = "ping-pong"
TIMESTEP = 1
ITERATION = 2
PREHEAT = 3
K = 0.5
SIGMA = 1
START_HOY = 4128
END_HOY = 4151

loadModel = MOOSAS_LOAD_MODEL
heatLoadModel = MOOSAS_HEATLOAD_MODEL


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
