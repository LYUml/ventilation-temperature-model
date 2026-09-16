# Model and parameter audit

## Design-stage model

`RdfMz5r1cModel` is the only public design-stage API. It uses no measured indoor-temperature history and does not fit target-building parameters. It simultaneously solves RDF zones with ISO 13790 air, surface and thermal-mass states, and numerically warms up hidden mass temperatures.

The model consumes project RDF geometry/U-values, construction-level SHGC, hourly occupancy/lighting/equipment schedules, setpoints and shared interfaces. Strict mode rejects execution while project-specific thermal mass, verified facade solar inputs and complete HVAC/airflow boundaries remain unavailable.

With explicit research assumptions enabled, the former RDF-schedule scenario produced 0.561, 0.591 and 1.062 °C for 2F, 3F and 4F (mean 0.738 °C). The new validation scenario retains those corridors and adds 4F412 as an explicitly free-running, zero-internal-gain space. Because 4F412 is coupled to adjacent zones, its schedule override also changes corridor predictions; the old aggregate must not be reused for the new scenario. Neither scenario is an all-project-data result. Superseded design experiments and scores are preserved under `archive/`.

After the temperature sensor label was confirmed and corrected from `4F411` to `4F412`, the new locked-test RMSE values are 0.597, 0.709, 1.275 and 1.325 °C for 2F corridor, 3F corridor, 4F corridor and 4F412 respectively (four-target mean 0.976 °C). Measurements remain evaluation-only inputs.

The 4F412 one-at-a-time uncertainty experiment identifies effective thermal capacitance as the dominant tested driver. Relative capacitance factors 2.0 and 4.0 reduce validation RMSE from 0.920 °C to 0.506 and 0.333 °C, and locked-test RMSE from 1.325 °C to 0.771 and 0.512 °C. These are uncertainty scenarios rather than fitted project values. An experimental MZ5R2C variant that adds physical zone-air capacitance reduces locked-test RMSE only to 1.292 °C; a second air state alone does not replace missing material-layer capacitance.

## Authoritative inputs

Only these three files are treated as factual model inputs:

- `data/temperature.csv`: 421 continuous hourly observations.
- `data/weather_54399.csv`: observed station weather; `TEM` exactly matches `OUTDOOR`.
- `data/building.rdf`: building spaces, shared interfaces, areas, volumes and U-values.

## Retained methods

| Method | Role | Mean 24 h RMSE |
|---|---|---:|
| Kernel | stable baseline | 0.344 °C |
| RDF graph state | topology ablation | 0.397 °C |
| RDF-Kernel NARX-Ridge | Kernel + RDF-informed Ridge correction | 0.319 °C |

## Removed forecast assumptions

The final models do not use the following earlier assumptions:

- fixed `Ceff = 165 kJ/(m²·K)`;
- assumed air-capacity multipliers or infiltration bounds;
- synthetic periodic heat gains in watts;
- assumed station coordinates or facade azimuths;
- manually declared room–corridor adjacency;
- the erroneous `4F411` boundary;
- MLP, NSGA-II, N4SID, DarkGreyBox-style or RC outputs as production inputs.

These forecast experiments remain recoverable from Git commit `b1cc336`; separate design-stage assumptions are tracked in `INPUT_PROVENANCE_AUDIT.md`.

## Fitted—not measured—quantities

The following are selected only from training/validation data and must not be described as measured physics:

- Kernel smoothing coefficient;
- autoregressive lag length;
- Ridge regularization strength;
- learned transition coefficients;
- latent slow-state residuals.

RDF interface UA values are factual values from the supplied RDF, but their effective participation in measured short-term heat transfer is not independently validated.
