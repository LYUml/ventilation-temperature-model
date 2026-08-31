from __future__ import annotations

import json

from .kernel.run import run as run_kernel
from .rdf_msts.run import run as run_rdf_msts


def main() -> None:
    kernel = run_kernel(); rdf = run_rdf_msts()
    rows = {
        "kernel": kernel["mean_floor_rmse_c"],
        "rdf_graph_state": rdf["model_mean_floor_rmse_c"],
        "rdf_msts": rdf["slow_state_model"]["mean_floor_rmse_c"],
    }
    print(json.dumps(dict(sorted(rows.items(), key=lambda item: item[1])), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
