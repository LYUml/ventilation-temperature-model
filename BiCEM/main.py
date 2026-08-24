import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"


def run_optimization(tag: str, debug: bool = False) -> Path:
    cmd = [sys.executable, str(HERE / "run.py"), "--tag", tag]
    if debug:
        cmd.append("--debug")
    subprocess.run(cmd, cwd=str(HERE), check=True)
    csv_path = OUTPUT_DIR / f"bicem_cases_{tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected result csv not found: {csv_path}")
    return csv_path


def analyze_and_plot(case_csv: Path, tag: str) -> tuple[Path, Path]:
    rows = []
    with case_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            row = {"iterations": int(r["iterations"]), "cases": r["cases"]}
            try:
                row["fitness(total_energy_vent)"] = float(r["fitness(total_energy_vent)"])
            except Exception:
                row["fitness(total_energy_vent)"] = float("nan")
            rows.append(row)

    valid = [r for r in rows if math.isfinite(r["fitness(total_energy_vent)"])]
    if not valid:
        raise RuntimeError("No valid rows to analyze.")

    by_iter: dict[int, list[dict]] = {}
    for r in valid:
        by_iter.setdefault(r["iterations"], []).append(r)

    iters = sorted(by_iter.keys())
    round_best = [min(by_iter[it], key=lambda x: x["fitness(total_energy_vent)"]) for it in iters]

    kept_best = []
    global_best = None
    for rb in round_best:
        if global_best is None or rb["fitness(total_energy_vent)"] < global_best["fitness(total_energy_vent)"]:
            global_best = dict(rb)
        kept_best.append(dict(global_best))

    base = kept_best[0]

    def pct(v: float, b: float) -> float:
        if (not math.isfinite(v)) or (not math.isfinite(b)) or abs(b) < 1e-12:
            return float("nan")
        return v / b * 100.0

    summary = []
    for it, rb, kb in zip(iters, round_best, kept_best):
        rec = {
            "iterations": it,
            "round_case": rb["cases"],
            "kept_case": kb["cases"],
            "round_fitness(total_energy_vent)": rb["fitness(total_energy_vent)"],
            "kept_fitness(total_energy_vent)": kb["fitness(total_energy_vent)"],
        }
        rec["round_fitness_pct"] = pct(
            rec["round_fitness(total_energy_vent)"], base["fitness(total_energy_vent)"]
        )
        rec["kept_fitness_pct"] = pct(
            rec["kept_fitness(total_energy_vent)"], base["fitness(total_energy_vent)"]
        )
        summary.append(rec)

    summary_csv = OUTPUT_DIR / f"bicem_{tag}_iteration_best_fitness_with_carry.csv"
    fields = [
        "iterations",
        "round_case",
        "kept_case",
        "round_fitness(total_energy_vent)",
        "kept_fitness(total_energy_vent)",
        "round_fitness_pct",
        "kept_fitness_pct",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(summary)

    x = np.array([r["iterations"] for r in summary], dtype=float)
    y_round = np.array([r["round_fitness_pct"] for r in summary], dtype=float)
    y_kept = np.array([r["kept_fitness_pct"] for r in summary], dtype=float)

    plt.figure(figsize=(13, 8))
    plt.plot(x, y_kept, color="#d62728", linewidth=2.6, marker="o", label="Fitness kept-best (solid)")
    plt.plot(
        x,
        y_round,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        marker="x",
        alpha=0.9,
        label="Fitness round-best (dashed)",
    )

    plt.axhline(100, color="gray", linestyle="--", linewidth=1)
    plt.xticks(x)
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Value (%)")
    plt.title(f"BiCEM {tag}: Fitness Kept-Best (Solid) vs Round-Best (Dashed)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / f"bicem_{tag}_fitness_kept_vs_round.png"
    plt.savefig(fig_path, dpi=220)
    return summary_csv, fig_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible BiCEM workflow: run optimization, summarize fitness, and plot."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=datetime.now().strftime("%m%d%H%M"),
        help="Output tag used in filenames.",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_csv = run_optimization(tag=args.tag, debug=args.debug)
    summary_csv, fig_path = analyze_and_plot(case_csv=case_csv, tag=args.tag)
    print(case_csv)
    print(summary_csv)
    print(fig_path)


if __name__ == "__main__":
    main()
