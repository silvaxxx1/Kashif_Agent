# Kashif — كاشف

> The working application for the Kashif tabular ML agent.

![Kashif Logo](assets/Kashif3.png)

Kashif combines a deterministic sklearn-based tabular ML pipeline with an LLM-driven feature engineering loop. The model training stack stays fixed. The LLM is limited to writing `engineer_features(df) -> df`, and the system keeps or discards those changes using cross-validation results.

---

## What lives here

- `core/` — the engine: profiling, execution, training, LLM orchestration, narration, and reporting
- `cli/` — the Typer command-line interface
- `tests/` — unit and integration tests for the engine
- `scripts/` — smoke-test entry points for local verification
- `data/` — local sample datasets
- `outputs/` — generated reports, artifacts, and run outputs
- `program.md` — domain guidance injected into every LLM round
- `config.yaml` — runtime configuration, including LLM provider selection

---

## Core idea

Kashif keeps the parts of tabular ML that should be deterministic:

- data cleaning
- dtype-aware preprocessing
- model selection
- cross-validation
- artifact generation

It delegates the part that benefits from reasoning:

- feature engineering
- reflection over prior rounds
- narrative reporting

That boundary is enforced in code. The LLM does not alter the training harness. It produces feature logic, the executor runs it safely, and the trainer scores it inside the same leak-free pipeline.

---

## Architecture

```text
CSV -> profiler -> fe_agent -> executor -> trainer -> reporter/narrator -> outputs
                    ^                             |
                    |--------- round history -----|
```

More concretely:

- `core/profiler.py` reads the dataset and produces structured profiling output.
- `core/fe_agent.py` builds prompts from the profile, `program.md`, and prior rounds.
- `core/executor.py` runs generated feature code with a guarded fallback path.
- `core/trainer.py` evaluates the resulting features inside an sklearn pipeline.
- `core/reporter.py` and `core/narrator.py` turn run history into human-readable output.

---

## Current modules

```text
core/
  trainer.py     static ML pipeline, CV, model registry
  profiler.py    dataset profiling and task detection
  executor.py    guarded execution of generated feature code
  fe_agent.py    feature-engineering loop and round control
  reporter.py    markdown report generation
  narrator.py    run summary/narrative generation
  llm/
    base.py
    groq.py
    anthropic.py
cli/
  main.py
```

The repository also contains output reports under `outputs/` and sample datasets under `data/` for local testing.

---

## Usage

```bash
cd kashif_core
uv sync

# Static pipeline only
uv run python -m cli.main run --csv data/titanic.csv --target survived --no-agent

# Full loop with the configured provider
uv run python -m cli.main run --csv data/titanic.csv --target survived

# Override provider and round count
uv run python -m cli.main run --csv data/titanic.csv --target survived --llm anthropic --rounds 4
```

Required environment variables depend on `config.yaml`. Typical examples:

```bash
export GROQ_API_KEY=...
export ANTHROPIC_API_KEY=...
```

---

## Tests and smoke checks

```bash
# Full test suite
uv run pytest

# Focused test file
uv run pytest tests/test_trainer.py -v

# Smoke checks
uv run python scripts/smoke_classification.py
uv run python scripts/smoke_regression.py
uv run python scripts/smoke_integration.py
```

The smoke scripts exercise the project against bundled datasets and are the fastest way to verify the end-to-end flow locally.

---

## Notes

- `program.md` is user-editable and should contain domain context, constraints, and hints for the feature-engineering loop.
- `outputs/` contains generated artifacts and example reports; treat it as runtime output, not canonical documentation.
- The root-level docs in the parent directory are the design history for this implementation.
