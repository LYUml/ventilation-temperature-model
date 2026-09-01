from __future__ import annotations

import json

from .kernel.run import run as run_kernel
from .rc_narx_ridge.run import run as run_rc_narx_ridge
from .rdf_rc_narx_ridge.run import run as run_rdf_rc_narx_ridge
from .rdf_kernel_narx_ridge.run import run as run_rdf_kernel_narx_ridge


def main() -> None:
    kernel = run_kernel(); rc = run_rc_narx_ridge(); rdf_rc = run_rdf_rc_narx_ridge(); rdf_kernel = run_rdf_kernel_narx_ridge()
    rows = {
        "kernel": kernel["mean_floor_rmse_c"],
        "rc_narx_ridge": rc["mean_floor_rmse_c"],
        "rdf_rc_narx_ridge": rdf_rc["mean_floor_rmse_c"],
        "rdf_graph_state_ablation": rdf_kernel["model_mean_floor_rmse_c"],
        "rdf_kernel_narx_ridge": rdf_kernel["slow_state_model"]["mean_floor_rmse_c"],
    }
    print(json.dumps(dict(sorted(rows.items(), key=lambda item: item[1])), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
