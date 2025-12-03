# Repository Guidelines

## Project Structure & Module Organization

- `mesa_frames/`: Source package.
  - `abstract/` and `concrete/`: Core APIs and implementations.
  - Key modules: `agents.py`, `agentset.py`, `space.py`, `datacollector.py`, `types_.py`.
- `tests/`: Pytest suite (`test_*.py`) covering public APIs.
- `docs/`: MkDocs and Sphinx content for user and API docs.
- `examples/`: Reproducible demo models and performance scripts.

## Build, Test, and Development Commands

- Install (dev stack): `uv sync` (always use uv)
- Lint & format: `uv run ruff check . --fix && uv run ruff format .`
- Tests (quiet + coverage): `export MESA_FRAMES_RUNTIME_TYPECHECKING=1 && uv run pytest -q --cov=mesa_frames --cov-report=term-missing`
- Pre-commit (all files): `uv run pre-commit run -a`
- Docs preview: `uv run mkdocs serve`

Always run tools via uv: `uv run <command>`.

## Coding Style & Naming Conventions

- Python 3.11+, 4-space indent, type hints required for public APIs.
- Docstrings: NumPy style (validated by Ruff/pydoclint).
- Formatting/linting: Ruff (formatter + lints). Fix on save if your IDE supports it.
- Names: `CamelCase` for classes, `snake_case` for functions/attributes, tests as `test_<unit>.py` with `Test<Class>` groups.
- **Avoid `TYPE_CHECKING` guards for runtime type checking:** In the mesa-frames codebase, avoid using `TYPE_CHECKING` guards for type annotations because the project uses `beartype` for runtime type checking, which requires the actual type objects to be available at runtime. `TYPE_CHECKING` guards would hide these imports from runtime execution, breaking `beartype` compatibility.


## Testing Guidelines

- Framework: Pytest; place tests under `tests/` mirroring module paths.
- Conventions: One test module per feature; name tests `test_<method_or_behavior>`.
- Coverage: Aim to exercise new branches and error paths; keep `--cov=mesa_frames` green.
- Run fast locally: `pytest -q` or `uv run pytest -q`.

## Commit & Pull Request Guidelines

- Commits: Imperative mood, concise subject, meaningful body when needed.
  Example: `Fix AgentSetRegistry.sets copy binding and tests`.
- PRs: Link issues, summarize changes, note API impacts, add/adjust tests and docs.
- CI hygiene: Run `ruff`, `pytest`, and `pre-commit` locally before pushing.

## Security & Configuration Tips

- Never commit secrets; use env vars. Example: `MESA_FRAMES_RUNTIME_TYPECHECKING=1` for stricter dev runs.
- Treat underscored attributes as internal.
