# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `nifty_arima.py` with data loading, forecasters, metrics and diagnostics, so
  reusable logic no longer lives inside notebook cells.
- Persistence, moving-average, AR (`AutoReg`), residual-AR and ARIMA
  walk-forward forecasters.
- Evaluation metrics beyond RMSE: MAE, sMAPE, MASE, directional accuracy and
  95% interval coverage.
- Ljung-Box residual test.
- AIC-based ARIMA order selection using training data only.
- `requirements.txt`, `.gitignore`, `LICENSE`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.
- Issue templates, a pull request template and `CODEOWNERS`.
- GitHub Actions CI (lint + tests on Python 3.10-3.12), CodeQL scanning and
  Dependabot updates.
- `pytest` suite covering loading, rounding, forecasting and metrics.

### Changed

- All models now share one train/test split (`TRAIN_FRACTION = 0.9`) so results
  are comparable.
- Notebook rewritten as a narrative that calls into `nifty_arima.py`; outputs
  are no longer committed (1.4 MB to 12 KB).
- Data is loaded with a real `DatetimeIndex` instead of the meaningless
  `date % 5` pseudo-day column.
- README expanded with setup, method, results, limitations and license.

### Fixed

- AR forecast rounded to the 0.05 tick inside the coefficient loop, corrupting
  every partial sum.
- ARIMA predictions had an unexplained `-0.42` offset.
- Moving-average model was scored on almost the whole series instead of a
  held-out test set.
- Removed statsmodels APIs (`tsa.arima_model.ARIMA`, `tsa.ar_model.AR`) replaced
  with the supported equivalents, so the code runs on statsmodels >= 0.14.

### Removed

- `except: continue` grid search that silently swallowed every failure.
- Unused TensorFlow and scikit-learn preprocessing imports.

## [0.1.0] - 2021-01-31

### Added

- Initial Colab notebook exploring ARIMA on 1-minute NIFTY data for January 2021
  with a persistence baseline.
