# Archived design-stage baseline-temperature benchmark

## Question and leakage rule

The deployment question is whether a not-yet-built target can be simulated without measured target-building indoor-temperature history. Results are therefore separated by information regime. A method that fits the target corridor's earlier temperatures is a forecast benchmark, not a valid design-stage competitor.

The locked evaluation interval follows the repository convention: rows 0–251 train historical forecast models, rows 252–335 are validation, and 62 rolling 24-hour windows start in rows 336–397.

## Results

| Information regime | Method | Target indoor history | Mean floor RMSE / °C | Interpretation |
|---|---|---:|---:|---|
| Historical forecast | Causal Kernel | Required for coefficients | 0.344 | Existing benchmark; unavailable for an unbuilt target |
| Historical forecast | RDF-Kernel NARX-Ridge | Required | 0.319 | Best historical score; not a design-stage result |
| Strict design | RDF multi-zone 1R1C, all spaces free-running | None | 5.51 over full series | Incorrect operating boundary for occupied adjacent rooms |
| Strict design | RDF multi-zone 1R1C with RDF HVAC bands | None | 3.120 on locked windows | Corrected boundary, but overly lumped thermal/solar physics |
| Strict design trial | Independent-zone ISO 13790 5R1C, direct factor 0.25 | None | 1.338 on locked windows | Earlier boundary sensitivity trial; adjacent RDF zones were not simultaneously solved |
| Strict design sensitivity | Independent-zone ISO 13790 5R1C, diffuse-only | None | 0.986 on locked windows | Sensitivity bound, not a selectable production result |
| Superseded research assumptions | RDF simultaneous MZ-5R1C with archetype load magnitudes | None | 0.852 on locked windows | Invalid as an all-project-data claim; superseded after RDF schedules were decoded |
| Design research assumptions | RDF simultaneous MZ-5R1C with real RDF hourly loads, assumed solar/mass | None | **0.988 on locked windows** | Current partial-project-data experiment; still not compliant |
| Design research assumptions | Above + construction-level RDF SHGC 0.3 | None | **0.738 on locked windows** | Most project-specific current experiment; solar/mass/HVAC gaps remain |
| Strict design experiment | RDF graph 2R2C (zone air + thermal mass) | None | 2.70 best locked-window trial | A second state alone does not repair missing boundary conditions |
| Transfer proxy | Ridge / tree ensembles trained on other measured rooms | No target-corridor history | 1.85–2.15 over full series | Same-building proxy; still unavailable for a wholly unbuilt building |
| Hybrid transfer proxy | Physical result plus residual learned from other rooms | No target-corridor history | 1.66–2.23 over full series | Does not beat pure 5R1C and lacks cross-building evidence |

The independent-zone ISO 5R1C trials used the current `architecture-building-systems/RC_BuildingSimulator` implementation at Git revision `97c1cdd`, with RDF geometry/U-values, 0.5 ACH background infiltration, 165 kJ/(m²·K) medium thermal mass, numerical warm-up, and no measured target temperature. They did not simultaneously propagate every RDF adjacent-zone state and are therefore retained only as sensitivity evidence. The diffuse-only row is not selectable: choosing it because it scores best on this test set would be test leakage.

The implemented `RdfMz5r1cModel` ports the Annex C air/surface/mass heat-flow split, couples RDF shared interfaces, applies RDF design temperature bands, and converges both inter-zone temperatures and the numerical warm-up. Its first run incorrectly applied horizontal diffuse irradiance to a vertical window with a factor of 1.0; the ISO isotropic-sky projection gives `(1 + cos(90°))/2 = 0.5`. That physical correction is part of the model definition, not a coefficient selected from target-temperature test error.

After that correction plus the earlier archetype load assumptions, locked-window RMSE was 0.961, 0.824 and 0.770 °C for the 2F, 3F and 4F corridors (mean 0.852 °C). This row is now superseded. The RDF contains complete hourly schedule objects; after decoding all 36 active references, RMSE becomes 1.068, 1.043 and 0.853 °C (mean **0.988 °C**). The worse but more truthful score replaces the earlier result. Strict mode still refuses to run because solar orientation, thermal mass and HVAC/airflow boundaries remain unresolved.

The apparent SHGC conflict is resolved semantically rather than by validation error. Each actual glazing `bot:Element` specifies SHGC 0.3 and U-value 1.8, whereas its space program/template specifies `zone_win_SHGC=0.45` and `zone_winU=2.2`. The heat model already uses construction-level U-values, so it now consistently uses construction-level SHGC and retains template values for compliance comparison. With real RDF schedules and construction SHGC, locked RMSE is 0.561, 0.591 and 1.062 °C (mean **0.738 °C**). This remains a research-assumption result until the remaining solar, thermal-mass and HVAC/airflow boundaries are closed.

## Decision

No strict design-stage candidate beats the historical Kernel, and the scores must not be presented as though they have the same information. The next production candidate should be an RDF-derived ISO 5R1C model with facade-resolved solar gains, parsed RDF schedules, and explicit adjacent-zone operating states. Its acceptance criterion is improvement over the current strict-design model without fitting or selecting parameters on the locked test interval.

The graph 2R2C experiment was a single simultaneous model, not an ensemble: every RDF space had an air-temperature state and a thermal-mass state, and shared RDF interfaces formed the only inter-zone conductance edges. A small sweep over standard, predeclared heat-capacity and air–mass coupling values reduced the best mean RMSE only to 2.70 °C. This falsifies the hypothesis that the current error is mainly caused by using only one thermal state.

## Single unified model research direction

The defensible learned successor is one **RDF-conditioned graph thermal state-space model**, not an average of separate predictors:

1. RDF spaces are graph nodes; exterior and shared constructions are typed conductance edges.
2. Each node carries physically meaningful air, surface and mass states. Weather, orientation-resolved solar radiation, schedules and design HVAC commands are exogenous inputs.
3. An implicit energy-balance step advances all zones together. A small shared neural constitutive function may adjust infiltration, gains and effective conductance, but it lives inside that one state equation and is constrained to non-negative, reciprocal heat flows.
4. Shared parameters are trained on a corpus of varied EnergyPlus/ISO 52016 simulations. The target building contributes only RDF static features and weather; it contributes no measured indoor-temperature labels.
5. Validation and test splits hold out entire buildings, climates and operating strategies. Time splits within the same building do not establish design-stage transfer.

This project currently has only one measured building, so it cannot identify or validate that universal correction function honestly. The immediate implementation target remains the deterministic RDF-derived 5R1C solver. The graph model becomes justified after generating or obtaining a multi-building simulation corpus.

Relevant implementations and evidence:

- ISO 13790 5R1C reference implementation: https://github.com/architecture-building-systems/RC_BuildingSimulator
- ISO 52016 design-stage implementation: https://github.com/EURAC-EEBgroup/pyBuildingEnergy
- Thermal-circuit to state-space reference: https://github.com/cghiaus/dm4bem
- Physics-informed multi-zone Neural ODE code (requires thermostat training data): https://github.com/GabSabb/Physics-Informed-NODEs
- Physics-constrained structured multi-zone dynamics: https://arxiv.org/abs/2011.05987
- Zero-shot physics-informed thermal transformer (simulated residential evidence): https://arxiv.org/abs/2605.01364
- Synthetic EnergyPlus pretraining plus target fine-tuning: https://doi.org/10.1016/j.jobe.2025.113341
