"""Reusable helpers for the NIFTY-ARIMA notebook.

Keeps the notebook focused on narrative while all forecasting, evaluation and
plotting logic lives in one tested place.

Typical use::

    import nifty_arima as na

    df = na.load_intraday("Intraday 1 Min Data/2021/2021 JAN NIFTY.txt")
    close = df["close"]

    baseline = na.persistence_forecast(close)
    result = na.walk_forward_arima(close, order=(2, 1, 1))
    result.summary_row()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

__all__ = [
    "TICK",
    "TRAIN_FRACTION",
    "ForecastResult",
    "adf_summary",
    "load_intraday",
    "ljung_box_test",
    "moving_average_forecast",
    "persistence_forecast",
    "plot_residual_diagnostics",
    "residual_correction_forecast",
    "select_order_by_aic",
    "tick_round",
    "walk_forward_ar",
    "walk_forward_arima",
]

# NIFTY 50 trades in 0.05 point increments, so forecasts are snapped to the tick.
TICK = 0.05

# Single source of truth for the train/test split so every model is comparable.
TRAIN_FRACTION = 0.9

# Cap on autoregressive order when it is selected automatically.
MAX_LAG = 40


def tick_round(values, tick: float = TICK):
    """Snap price(s) to the nearest exchange tick.

    >>> tick_round([13740.07, 13740.03])
    array([13740.05, 13740.05])
    """
    arr = np.asarray(values, dtype=float)
    return np.round(arr / tick) * tick


def _to_datetime(dates: pd.Series, times: pd.Series) -> pd.DatetimeIndex:
    """Combine YYYYMMDD ints and HHMM ints into a DatetimeIndex."""
    date_s = dates.astype("int64").astype(str).str.zfill(8)
    time_s = times.astype("int64").astype(str).str.zfill(4)
    return pd.DatetimeIndex(pd.to_datetime(date_s + time_s, format="%Y%m%d%H%M"))


def load_intraday(path: str) -> pd.DataFrame:
    """Load a headerless NIFTY intraday file and return a time-indexed frame.

    The source files carry a leading serial column and two trailing columns
    that are not needed here, matching the original notebook layout.
    """
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] != 9:
        raise ValueError(
            f"expected 9 columns (serial, date, time, open, high, low, close, x, y), "
            f"found {raw.shape[1]} in {path!r}"
        )

    raw = raw.drop(columns=[0, 7, 8])
    raw.columns = ["date", "time", "open", "high", "low", "close"]

    df = raw[["open", "high", "low", "close"]].copy()
    df.index = _to_datetime(raw["date"], raw["time"])
    df.index.name = "timestamp"
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps in source data")
    close = df["close"]
    if not np.isfinite(close).all():
        raise ValueError("close column contains NaN/inf values")
    return df


def adf_summary(series, label: str = "series") -> dict:
    """Run the Augmented Dickey-Fuller test and print a readable summary."""
    values = pd.Series(series).dropna()
    stat, pvalue, used_lags, nobs, critical, _ = adfuller(values)
    return {"stat": stat, "pvalue": pvalue, "used_lags": used_lags, "critical": critical}


def _auto_lags(series, maxlag: int = MAX_LAG) -> list[int]:
    """Pick autoregressive lags by AIC, falling back to a single lag."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected = ar_select_order(np.asarray(series, dtype=float), maxlag=maxlag, ic="aic")
    lags = list(selected.ar_lags or [1])
    return lags


@dataclass
class ForecastResult:
    """One model's out-of-sample forecasts and the metrics they imply."""

    name: str
    predictions: np.ndarray
    actual: np.ndarray
    order: tuple | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    train: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.predictions = np.asarray(self.predictions, dtype=float)
        self.actual = np.asarray(self.actual, dtype=float)
        for attr in ("ci_lower", "ci_upper"):
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, np.asarray(value, dtype=float))

    @property
    def residuals(self) -> np.ndarray:
        return self.actual - self.predictions

    def metrics(self) -> dict:
        return forecast_metrics(self.actual, self.predictions, self.train)

    def coverage(self, level: float = 0.95) -> float | None:
        if self.ci_lower is None or self.ci_upper is None:
            return None
        inside = (self.actual >= self.ci_lower) & (self.actual <= self.ci_upper)
        return float(inside.mean())

    def summary_row(self) -> dict:
        row = {"model": self.name, "order": str(self.order) if self.order else "-"}
        row.update(self.metrics())
        coverage = self.coverage()
        if coverage is not None:
            row["ci95_coverage"] = coverage
        return row

    def plot(self, zoom: int | None = None, ax=None):
        """Plot actual vs predicted, optionally zoomed to the first `zoom` steps."""
        import matplotlib.pyplot as plt

        actual, predicted = self.actual, self.predictions
        if zoom is not None:
            actual, predicted = actual[:zoom], predicted[:zoom]

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))
        ax.plot(actual, label="actual", linewidth=1)
        ax.plot(predicted, label="forecast", color="crimson", linewidth=1)
        if self.ci_lower is not None and zoom is None:
            ax.fill_between(
                np.arange(len(actual)),
                self.ci_lower[: len(actual)],
                self.ci_upper[: len(actual)],
                color="crimson",
                alpha=0.15,
                label="95% interval",
            )
        ax.set_title(self.name)
        ax.legend(loc="best")
        ax.set_xlabel("test step")
        ax.set_ylabel("price")
        return ax


def _split(values: np.ndarray, train_fraction: float):
    n_train = int(len(values) * train_fraction)
    if n_train < 2 or n_train >= len(values):
        raise ValueError("train_fraction leaves an empty train or test set")
    return values[:n_train], values[n_train:]


def persistence_forecast(
    series,
    train_fraction: float = TRAIN_FRACTION,
    tick: float = TICK,
) -> ForecastResult:
    """Random-walk baseline: tomorrow's price equals the last observed price."""
    values = np.asarray(series, dtype=float)
    train, test = _split(values, train_fraction)
    raw = np.concatenate([[train[-1]], test[:-1]])
    return ForecastResult(
        name="Persistence",
        predictions=tick_round(raw, tick),
        actual=test,
        train=train,
    )


def moving_average_forecast(
    series,
    window: int = 3,
    train_fraction: float = TRAIN_FRACTION,
    tick: float = TICK,
) -> ForecastResult:
    """Walk-forward forecast from the mean of the last `window` observations."""
    values = np.asarray(series, dtype=float)
    train, test = _split(values, train_fraction)
    history = list(train)
    predictions = []
    for obs in test:
        predictions.append(float(tick_round(np.mean(history[-window:]), tick)))
        history.append(float(obs))
    return ForecastResult(
        name=f"Moving average (w={window})",
        predictions=np.asarray(predictions),
        actual=test,
        train=train,
    )


def walk_forward_ar(
    series,
    train_fraction: float = TRAIN_FRACTION,
    tick: float = TICK,
    maxlag: int = MAX_LAG,
) -> ForecastResult:
    """Autoregressive model on price levels, with lag order chosen by AIC."""
    values = np.asarray(series, dtype=float)
    train, test = _split(values, train_fraction)
    lags = _auto_lags(train, maxlag=maxlag)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = AutoReg(train, lags=lags).fit()

    predictions = []
    for obs in test:
        yhat = float(np.asarray(fit.forecast(1)).ravel()[0])
        predictions.append(float(tick_round(yhat, tick)))
        fit = fit.append([obs], refit=False)

    return ForecastResult(
        name="AR (AutoReg)",
        predictions=predictions,
        actual=test,
        train=train,
    )


def residual_correction_forecast(
    series,
    train_fraction: float = TRAIN_FRACTION,
    tick: float = TICK,
    maxlag: int = MAX_LAG,
) -> ForecastResult:
    """Moving-average style model: forecast the persistence error with an AR model."""
    values = np.asarray(series, dtype=float)
    train, test = _split(values, train_fraction)

    train_errors = np.diff(train)
    lags = _auto_lags(train_errors, maxlag=maxlag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = AutoReg(train_errors, lags=lags).fit()

    coefs = np.asarray(fit.params)[1 : 1 + len(lags)]
    error_history = list(train_errors)

    predictions = []
    previous = float(train[-1])
    for obs in test:
        pred_error = float(np.asarray(fit.params)[0])
        for coef, lag in zip(coefs, lags, strict=False):
            pred_error += coef * error_history[-lag]
        predictions.append(float(tick_round(previous + pred_error, tick)))
        error_history.append(float(obs) - previous)
        previous = float(obs)

    return ForecastResult(
        name="MA (residual AR)",
        predictions=predictions,
        actual=test,
        order=None,
        train=train,
    )


def walk_forward_arima(
    series,
    order: tuple,
    train_fraction: float = TRAIN_FRACTION,
    tick: float = TICK,
    refit_every: int = 25,
) -> ForecastResult:
    """Expanding-window one-step-ahead ARIMA forecast.

    The model is fully refit every `refit_every` steps; in between, the fit is
    advanced with `append(..., refit=False)`, which updates the state-space
    Kalman filter without another hill-climb. Pass refit_every=1 for
    per-step refits. No future information leaks into any forecast.
    """
    values = np.asarray(series, dtype=float)
    train, test = _split(values, train_fraction)

    history = [float(x) for x in train]
    fit = None
    predictions, lower, upper = [], [], []

    for step, obs in enumerate(test):
        if fit is None or step % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(np.asarray(history, dtype=float), order=order).fit()

        forecast = fit.get_forecast(steps=1)
        point = float(np.asarray(forecast.predicted_mean).ravel()[0])
        interval = np.asarray(forecast.conf_int())[0]

        predictions.append(float(tick_round(point, tick)))
        lower.append(float(np.floor(interval[0] / tick) * tick))
        upper.append(float(np.ceil(interval[1] / tick) * tick))

        history.append(float(obs))
        if refit_every > 1:
            fit = fit.append([float(obs)], refit=False)

    return ForecastResult(
        name=f"ARIMA{tuple(order)}",
        predictions=np.asarray(predictions),
        actual=test,
        order=tuple(order),
        ci_lower=np.asarray(lower),
        ci_upper=np.asarray(upper),
        train=train,
    )


def select_order_by_aic(series, orders) -> pd.DataFrame:
    """Score candidate ARIMA orders on the supplied (training) data by AIC/BIC.

    Pass only the training slice so the held-out test period stays untouched.
    """
    values = np.asarray(series, dtype=float)
    rows = []
    for order in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(values, order=order).fit()
            rows.append({"order": tuple(order), "aic": fit.aic, "bic": fit.bic})
        except Exception as exc:  # keep the grid honest: log, do not hide
            rows.append({"order": tuple(order), "aic": np.nan, "bic": np.nan, "error": str(exc)})

    table = pd.DataFrame(rows).set_index("order").sort_values("aic")
    if table["aic"].isna().all():
        raise RuntimeError("every candidate order failed to fit; check the series and orders")
    return table


def forecast_metrics(y_true, y_pred, train=None) -> dict:
    """RMSE, MAE, sMAPE, MASE and directional accuracy."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = actual - predicted

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    denom = np.abs(actual) + np.abs(predicted)
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(errors) / denom)) * 100)

    mase = float("nan")
    if train is not None and len(train) > 1:
        scale = float(np.mean(np.abs(np.diff(np.asarray(train, dtype=float)))))
        if scale > 0:
            mase = mae / scale

    direction = float("nan")
    flat_rate = float("nan")
    if len(actual) > 1:
        actual_move = np.diff(actual)
        predicted_move = predicted[1:] - actual[:-1]
        nonzero = actual_move != 0
        if nonzero.any():
            direction = float(np.mean(
                np.sign(actual_move[nonzero]) == np.sign(predicted_move[nonzero])))
        flat_rate = float(1.0 - nonzero.mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "mase": mase,
        # Share of bars with zero move; high values mean the bar set is coarse.
        "flat_rate": flat_rate,
        "dir_acc": direction,  # sign accuracy on NONZERO actual moves only
    }


def ljung_box_test(residuals, lags=(10, 20)) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation left in the residuals."""
    return acorr_ljungbox(np.asarray(residuals, dtype=float), lags=list(lags), return_df=True)


def plot_residual_diagnostics(result: ForecastResult):
    """Histogram, density, ACF and QQ plot of a forecast's residuals."""
    import matplotlib.pyplot as plt

    residuals = pd.Series(result.residuals, name="residual")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    residuals.hist(ax=axes[0, 0], bins=60)
    axes[0, 0].set_title("residuals")
    residuals.plot(kind="kde", ax=axes[0, 1])
    axes[0, 1].set_title("residual density")
    plot_acf(residuals, lags=40, ax=axes[1, 0])
    qqplot(residuals, line="r", ax=axes[1, 1])
    axes[1, 1].set_title("QQ plot")
    fig.tight_layout()
    return fig
