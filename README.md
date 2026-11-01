# NIFTY-ARIMA

Forecasting 1-minute NIFTY 50 prices (January 2021) with ARIMA, benchmarked against a random-walk
(persistence) baseline.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apoorvsingh090/NIFTY-ARIMA/blob/main/TIMESERIES_NIFTY.ipynb)

## Contents

| File | Purpose |
| --- | --- |
| `TIMESERIES_NIFTY.ipynb` | Walkthrough: EDA, stationarity tests, baselines, AR/MA/ARIMA, results table |
| `nifty_arima.py` | Data loading, forecasters, metrics and diagnostics used by the notebook |
| `requirements.txt` | Dependencies |

## Data

The notebook reads a headerless intraday file from the Kaggle *Intraday 1 Min Data* dataset:

```
Intraday 1 Min Data/2021/2021 JAN NIFTY.txt
```

Place it at that path, or point the notebook anywhere via the `NIFTY_DATA` environment variable:

```bash
set NIFTY_DATA="C:\path\to\2021 JAN NIFTY.txt"   # Windows
export NIFTY_DATA="/path/to/2021 JAN NIFTY.txt"  # macOS/Linux
```

The file is expected to have 9 columns: `serial, date, time, open, high, low, close, x, y`; the serial
and the two trailing columns are dropped, and a real `DatetimeIndex` is built from `date` + `time`.

## Setup

```bash
git clone https://github.com/apoorvsingh090/NIFTY-ARIMA.git
cd NIFTY-ARIMA
pip install -r requirements.txt
jupyter notebook TIMESERIES_NIFTY.ipynb
```

In Colab, just run the first cell: it clones the repo, installs dependencies and adds the module to
`sys.path`.

## Method

- **One split for every model** (`na.TRAIN_FRACTION = 0.9`) so RMSE is comparable across models.
- **Walk-forward, one-step-ahead forecasts**: each prediction only sees data up to that minute; the
  observed value is appended before the next step. No future leakage.
- **ARIMA order is selected by AIC on the training slice only**; the test period is never used for tuning.
- **Prices are snapped to the 0.05 exchange tick** after forecasting, not inside the model math.
- **Baselines come first**: persistence and a moving average are reported so any model gain is explicit.
- **Metrics**: RMSE, MAE, sMAPE, MASE (vs. the naive forecast), directional accuracy, and 95% interval
  coverage. Ljung-Box checks whether residuals are white noise.

## Results (original notebook run)

Kept for reference. These numbers come from the pre-cleanup notebook, which used **different splits per
model**, so they are not directly comparable; re-run the notebook for a consistent table.

| Model | Test split | RMSE |
| --- | --- | --- |
| Persistence | last 34% | 9.145 |
| Moving average (w=3) | almost the whole series (not held out) | 9.195 |
| AR | last 34% | 9.166 |
| MA (residual AR) | last 34% | 9.160 |
| ARIMA(4,1,2) | last 10%, with a −0.42 offset | 11.197 |

The ARIMA run also placed **7.8% of actuals outside its 95% interval** (5% expected), a sign that the
uncertainty estimate was miscalibrated.

## What changed in this cleanup

- Fixed the AR forecast bug where the 0.05 tick rounding happened inside the coefficient loop.
- Removed the unexplained −0.42 constant offset from ARIMA predictions.
- Migrated removed APIs: `statsmodels.tsa.arima_model.ARIMA` → `statsmodels.tsa.arima.model.ARIMA`,
  `statsmodels.tsa.ar_model.AR` → `AutoReg`.
- Moved the moving-average model onto a real train/test split (it previously scored almost the whole
  series, not held-out data).
- Replaced the `except: continue` grid search with AIC selection on training data only, and made
  failures visible.
- Unified the split, extracted duplicated logic into `nifty_arima.py`, and stripped notebook outputs
  (1.4 MB → 12 KB).
- Added richer metrics (MAE, sMAPE, MASE, directional accuracy, interval coverage) and Ljung-Box.
- Built a `DatetimeIndex` instead of the meaningless `date % 5` "day" column.

## Limitations

- 1-minute equity data is not equally spaced (overnight gaps, weekends, holidays); ARIMA treats the index
  as regular, so calendar effects are not modeled.
- One month of data only — results are illustrative, not generalizable.
- No transaction costs, volatility modeling or intraday seasonality; this is a forecasting exercise, not
  a trading strategy.
