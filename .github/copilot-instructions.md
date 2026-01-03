# Copilot instructions for mlops2025_carla_mariane

Purpose: Help AI coding agents be productive in this repository by describing the
project layout, developer workflows, important conventions, and where to make
safe, discoverable changes.

Key facts (what's actually in the repo)
- Project root contains: `README.md`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml` (currently empty), `uv.lock`, `configs/`, `data/`, `notebooks/`, `scripts/`, `src/`, `tests/`.
- Source package scaffold: `src/mlproject/` with subpackages: `preprocess/`, `features/`, `train/`, `inference/`, `pipelines/`, `utils/`. Most subpackages are empty skeletons.
- CLI scripts present under `scripts/`: `preprocess.py`, `feature_engineering.py`, `train.py`, `batch_inference.py` — these files currently exist but are empty placeholders.

Big-picture architecture (inferred from README and layout)
- Single Python package using src-layout: `src/mlproject`.
- Pipeline stages split into scripts (one per stage): preprocess -> features -> train -> batch_inference.
- Configs are expected under `configs/` (OmegaConf-style YAMLs are suggested by README).
- Containerization: `Dockerfile` + `docker-compose.yml` intended to run `uv`-based commands inside `app` service.

Developer workflows (how to run & test — verified or implied)
- Local CLI (intended): `uv run train`, `uv run inference`. Confirm `pyproject.toml` contains `tool.uv.cli` entry points; currently the file is empty — add entry points before relying on `uv run`.
- Docker: example usage in README: `docker-compose run app train` or `docker-compose run app inference`.
- Tests: `uv run pytest` is the expected command (ensure `pyproject.toml` includes test-deps and `pytest` in `dev-dependencies`).

Project-specific conventions and checks for agents
- Preserve `src/` layout: make runtime imports like `from mlproject...` and avoid relative imports that break packaging.
- CLI & packaging: the package name should be `ml-project` (per README). If adding entry points, place them under `tool.uv` or `project.scripts` in `pyproject.toml` so `uv run <cmd>` works.
- Pipeline outputs: batch inference should write to `outputs/YYYYMMDD_predictions.csv` as specified in README — use `datetime.utcnow().strftime("%Y%m%d")` to generate filename.
- Dockerfile requirements: use Python 3.11 and install dependencies with `uv` (the `Dockerfile` already exists; check it for `uv sync` usage and follow the same pattern).

Where to look first when making edits
- `README.md` — holds course/project constraints and the authoritative target behavior.
- `scripts/` — implement pipeline-stage entrypoints here (these are the script files CI and Docker will call).
- `src/mlproject/` — add implementation (preprocess, features, train, inference, utils).
- `pyproject.toml` — currently empty: add project metadata, dependencies, and CLI entry points before wiring `uv run` commands.

Examples from this repository
- Implement CLI entry points: add a `tool.uv.scripts` or `project.scripts` section in `pyproject.toml` so `uv run train` calls `mlproject.scripts.train:main` (or similar).
- Use the scaffold: `scripts/train.py` should import implemented functions from `src/mlproject/train` rather than duplicating logic.

Safety and minimal-impact guidance for AI edits
- Small, focused changes only: prefer adding implementations under `src/mlproject` and wire the small script wrapper in `scripts/`.
- Don't remove or rename top-level folders (e.g., `src/`, `scripts/`, `configs/`) unless user asks.
- When adding dependencies, update `pyproject.toml` and run `uv sync` locally — but do not modify `uv.lock` without running `uv`.

If something is missing or unclear (ask the user)
- `pyproject.toml` is empty: confirm intended package metadata, package name, and CLI entry names.
- Which evaluation metric and model choices should be used? README requires this be documented.

When you finish a change
- Run unit tests: `uv run pytest` (after `pyproject.toml` is configured).
- Run the pipeline locally: `uv run preprocess`, `uv run features`, `uv run train`, `uv run inference` (or the Docker equivalents).

If you want me to update the repository to make `uv run` functional, say so and provide the desired package name and CLI entry names.

— End
