# Model and parameter audit

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
| RDF-MSTS | proposed unified model | 0.319 °C |

## Removed assumptions

The final models do not use the following earlier assumptions:

- fixed `Ceff = 165 kJ/(m²·K)`;
- assumed air-capacity multipliers or infiltration bounds;
- synthetic periodic heat gains in watts;
- assumed station coordinates or facade azimuths;
- manually declared room–corridor adjacency;
- the erroneous `4F411` boundary;
- MLP, NSGA-II, N4SID, DarkGreyBox-style or RC outputs as production inputs.

These experiments remain recoverable from Git commit `b1cc336`, but they are not part of the cleaned working model.

## Fitted—not measured—quantities

The following are selected only from training/validation data and must not be described as measured physics:

- Kernel smoothing coefficient;
- autoregressive lag length;
- Ridge regularization strength;
- learned transition coefficients;
- latent slow-state residuals.

RDF interface UA values are factual values from the supplied RDF, but their effective participation in measured short-term heat transfer is not independently validated.
