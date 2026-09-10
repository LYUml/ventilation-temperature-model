# Model and parameter audit

## Design-stage model

`RdfMz5r1cModel` is the only public design-stage API. It uses no measured indoor-temperature history and does not fit target-building parameters. It simultaneously solves RDF zones with ISO 13790 air, surface and thermal-mass states, and numerically warms up hidden mass temperatures.

The model consumes project RDF geometry/U-values, construction-level SHGC, hourly occupancy/lighting/equipment schedules, setpoints and shared interfaces. Strict mode rejects execution while project-specific thermal mass, verified facade solar inputs and complete HVAC/airflow boundaries remain unavailable.

With explicit research assumptions enabled, the current partial-project-data result is 0.561, 0.591 and 1.062 °C for 2F, 3F and 4F (mean 0.738 °C). It is not an all-project-data result. Superseded design experiments and scores are preserved under `archive/`.

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
