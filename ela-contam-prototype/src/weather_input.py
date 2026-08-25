from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_weather_input(temperature_csv: Path, output_csv: Path, metadata_json: Path) -> dict:
    frame = pd.read_csv(temperature_csv.resolve())
    if "Timestamp" not in frame or "OUTDOOR" not in frame:
        raise ValueError("temperature source must contain Timestamp and OUTDOOR")
    weather = pd.DataFrame(
        {
            "Timestamp": frame["Timestamp"],
            "outdoor_temperature_c": frame["OUTDOOR"],
            "wind_speed_m_s": 0.0,
            "wind_direction_deg": 0.0,
            "barometric_pressure_pa": 101325.0,
            "wind_data_status": "assumed_zero_missing_observation",
        }
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_csv, index=False)
    metadata = {
        "rows": len(weather),
        "start": str(weather["Timestamp"].iloc[0]),
        "end": str(weather["Timestamp"].iloc[-1]),
        "temperature_source": str(temperature_csv.resolve()),
        "wind_source": None,
        "wind_assumption": "0 m/s at all timestamps because no site or station observation is available",
        "pressure_assumption": "constant standard atmosphere, 101325 Pa",
        "allowed_use": "controlled no-wind stack-effect baseline",
        "prohibited_interpretation": "must not be described as observed weather or full real-building infiltration",
        "replacement_contract": {
            "required_columns": ["Timestamp", "wind_speed_m_s", "wind_direction_deg"],
            "units": {"wind_speed_m_s": "m/s", "wind_direction_deg": "degrees clockwise from north"},
            "time_alignment": "hourly Asia/Shanghai timestamps matching the temperature CSV",
        },
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an explicit zero-wind weather baseline")
    parser.add_argument("--temperature-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    result = build_weather_input(args.temperature_csv, args.output_csv, args.metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
