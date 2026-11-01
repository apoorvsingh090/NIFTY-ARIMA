# Contributing to NIFTY-ARIMA

Thanks for taking the time to contribute. This project is a small, focused
forecasting study, so changes that improve correctness, reproducibility or
clarity are especially welcome.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install ruff pytest pytest-cov
   ```

3. Put the dataset at `Intraday 1 Min Data/2021/2021 JAN NIFTY.txt`, or set
   `NIFTY_DATA` to its location:

   ```bash
   set NIFTY_DATA="C:\path\to\2021 JAN NIFTY.txt"   # Windows
   export NIFTY_DATA="/path/to/2021 JAN NIFTY.txt"  # macOS/Linux
   ```

4. Run the checks before pushing:

   ```bash
   ruff check .
   python -m pytest
   ```

## Project layout

```text
.
├── TIMESERIES_NIFTY.ipynb   # narrative walkthrough (outputs are not committed)
├── nifty_arima.py           # data loading, forecasters, metrics, diagnostics
├── tests/                   # pytest suite for nifty_arima.py
├── requirements.txt
└── .github/                 # CI, issue/PR templates, Dependabot, CodeQL
```

Keep reusable logic in `nifty_arima.py` and the notebook focused on explanation.
If you change model logic, update the notebook and re-run it, but **do not commit
cell outputs** — they bloat the repository.

## Reporting issues

- Search existing issues first.
- For bugs, include: what you expected, what happened, the exact command or cell,
  and the versions reported by `pip list | grep -E "statsmodels|pandas|numpy"`.
- For model or results issues, state the split, order/parameters and metric used.

## Branching and pull requests

- Branch from `main` and use a descriptive name:
  `feature/short-description`, `fix/short-description`, `docs/short-description`.
- Keep pull requests small and focused on one change.
- Reference the related issue in the PR description (`Closes #123`).
- Make sure CI is green: linting and tests must pass.
- Update `README.md` and `CHANGELOG.md` when behaviour or results change.

## Coding standards

- Python 3.10+.
- Formatting and linting are handled by [Ruff](https://docs.astral.sh/ruff/)
  with the configuration in `pyproject.toml` (`ruff check .`).
- Prefer small, typed functions with docstrings; avoid duplicating forecast or
  metric code between the notebook and the module.
- Do not bake magic constants into models. If a correction is needed, derive it
  and document why.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat: add seasonal ARIMA baseline
fix: round forecasts once, after AR summation
docs: explain MASE in the README
test: cover tick rounding of ARIMA intervals
```

## Review process

A maintainer reviews pull requests as time allows. Expect feedback on
correctness and reproducibility first, style second. Address review comments by
pushing follow-up commits to the same branch; do not force-push over a review
unless asked.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
