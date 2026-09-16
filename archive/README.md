# Archive

This directory preserves superseded or non-production experiments so the main
package stays small and unambiguous.

- `models/design_base_ta_1r1c.py`: superseded RDF multi-zone 1R1C design prototype.
- `models/mz5r2c.py`: experimental MZ5R2C air-capacitance extension; locked-test RMSE improved only from 1.325 to 1.292 °C and did not address missing material-layer capacitance.
- `docs/DESIGN_MODEL_BENCHMARK.md`: chronological research results, including superseded scores and provenance limitations.

Archived code is retained for auditability and is not imported by `base_ta.py`
or the `model` package.
