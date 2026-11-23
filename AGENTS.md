# Repository Guidelines

## Project Structure & Module Organization

This Python document scanner uses OpenCV, NumPy, pandas, and scikit-image for image processing and geometry. Active implementation lives in `src/core/scanner.py`, `math_utils.py`, and `entities.py`. OCR, filters, file utilities, and database integration have placeholder modules in `src/core/`, `src/utils/`, and `src/database/`. `src/main.py` is also empty; there is no working application entry point yet.

Tests live in `test/`. Local images belong in `data/input/`, with `data/processed/` and `data/output/` available for results. `data/` is gitignored. Top-level `docs/`, `models/`, and `database/` directories provide space for supporting resources.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync`: install dependencies using `pyproject.toml` and `uv.lock`.
- `uv run python -m unittest discover -s test -v`: run the complete test suite.
- `uv run python -m unittest test.test_math_utils -v`: run geometry and perspective-transform tests.
- `uv run python -m src.main`: invoke the application module; currently exits without processing anything.

Python 3.12+ is required; `.python-version` pins 3.13.12. No dedicated build, formatter, or lint commands are configured. Direct scanner execution currently encounters mixed-import issues.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for new classes. Preserve existing public names unless a deliberate rename updates all callers and tests. Use explicit package imports for new code and `np.ndarray` for array type annotations. Keep image shape, dtype, and coordinate conventions clear at function boundaries.

## Testing Guidelines

Use standard-library `unittest`, `test_*.py` filenames, and `test_*` methods. Prefer synthetic NumPy images and deterministic geometry examples over private image fixtures. Cover invalid dimensions, empty detections, and expected numerical results with NumPy assertions. No coverage threshold is configured. Existing tests exercise unfinished paths; inspect failures and current code before relying on potentially stale inline `BUG` comments or `CLAUDE.md` notes.

## Commit & Pull Request Guidelines

Follow the history's prefixes: `feat:`, `fix:`, `refactor:`, and `style:`, followed by a concise imperative description. Keep changes focused. Pull requests should explain the behavior change, link relevant issues, and report validation results and remaining failures. Include before/after images when changing visual processing behavior. Keep private scans, credentials, and generated local artifacts out of commits.
