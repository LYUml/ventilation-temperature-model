# biCEM

A structured BiCEM framework for graph-based optimization with a pluggable evaluation entrypoint.

## Layout

- `main.py`: workflow entrypoint (`run.py` + fitness trend analysis/plot).
- `run.py`: thin wrapper CLI that maps arguments and calls `optimization.py`.
- `optimization.py`: bi-level CEM optimization on LPG topology/attributes.
- `structure.py`: graph structure definitions and perturbation bounds (node/edge).
- `evaluation.py`: independent LPG evaluation entrypoint (input LPG + config, output fitness).
- `constant.py`: default paths/config and MoosasPy import bridge.
- `data/test_v3.rdf`: base RDF input.
- `data/office.sch`: schedule file.
- `data/heatModel.json`: base network input.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

With explicit model/schedule:

```bash
python run.py --model data/test_v3.rdf --sch data/office.sch --workers 4
```

## Evaluation Interface

`evaluation.py` exposes a single-entry evaluation API:

- Input: LPG graph + base network dict + simulation config
- Output: scalar fitness only (`total_energy_vent`, lower is better)

This separation is intended to make it easier to plug in:

- other graph structures
- other optimization algorithms
- other objective wrappers (by adding new evaluation adapters)

## Workflow Switching

The framework is designed so that workflow switching is concentrated at two boundaries:

1. `optimization.build_LPGschema(...)`
- Defines how an external workflow is converted into:
  - `LPGSchema` (optimization search space + base graph)
  - `metadata` (evaluation runtime context)

2. `evaluation.evaluate_candidate(graph, metadata)`
- Defines how a candidate graph is scored.
- Returns only scalar fitness and error string.

Current optimization loop (`bilevel_cem_optimize`) only consumes `config`, `schema`, and `metadata`.
So for most workflow migrations, you only need to adapt:
- schema/metadata construction in `build_LPGschema`
- scoring behavior in `evaluate_candidate`

Note:
- If your new workflow uses different edge field names than current path fields,
  update `constant.LPG_PATH_FIELDS` accordingly.

## Outputs

- `outputs/bicem_cases_<tag>.csv`: case-level fitness records.
- `outputs/bicem_<tag>_iteration_best_fitness_with_carry.csv`: round-best vs kept-best fitness summary.
- `outputs/bicem_<tag>_fitness_kept_vs_round.png`: fitness trend plot.
