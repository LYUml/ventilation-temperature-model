import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
MOOSAS_ROOT = Path(r"E:\PycharmProjects\moosas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrapper entrypoint for optimization.py")
    parser.add_argument("--model", type=str, default=str(HERE / "data" / "test_v5.rdf"))
    parser.add_argument("--output-dir", type=str, default=str(HERE / "outputs"))
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--station_id", type=str, default=None)
    parser.add_argument("--mini", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_base = Path(args.log_dir) if args.log_dir else HERE / "logs"
    out_base = Path(args.output_dir) if args.output_dir else HERE / "outputs"
    log_dir = log_base / run_stamp
    out_dir = out_base / run_stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HERE / "optimization.py"),
        "--rdf",
        args.model,
        "--output-dir",
        str(out_dir),
        "--log-dir",
        str(log_dir),
        "--workers",
        str(args.workers),
    ]
    if args.station_id:
        cmd += ["--station_id", str(args.station_id)]
    if args.mini:
        cmd.append("--mini")
    env = os.environ.copy()
    pythonpath_parts = [str(MOOSAS_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    subprocess.run(cmd, cwd=str(HERE), check=True, env=env)


if __name__ == "__main__":
    main()
