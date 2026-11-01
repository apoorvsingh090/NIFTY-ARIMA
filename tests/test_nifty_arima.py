import numpy as np
import pandas as pd
import pytest

import nifty_arima as na


def write_raw_file(path, rows=420, seed=3):
    """Write a synthetic headerless NIFTY-style file and return the closes."""
    rng = np.random.default_rng(seed)
    prices = 13700 + np.cumsum(rng.normal(0, 1.0, rows))

    records = []
    day, hour, minute = 20210104, 9, 15
    for serial, price in enumerate(prices):
        if serial and serial % 375 == 0:
            day += 1
            hour, minute = 9, 15
        stamp = hour * 100 + minute
        records.append([serial, day, stamp, price - 0.2, price + 0.3, price - 0.4, price, 0, 0])
        minute += 1
        if minute == 60:
            minute = 0
            hour += 1

    pd.DataFrame(records).to_csv(path, header=False, index=False)
    return prices


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "nifty.txt"
    write_raw_file(path)
    return path


@pytest.fixture
def close(raw_file):
    return na.load_intraday(raw_file)["close"]


def test_tick_round_snaps_to_five_paise():
    rounded = na.tick_round([1.02, 1.03, 1.07])
    np.testing.assert_allclose(rounded, [1.0, 1.05, 1.05], atol=1e-9)


def test_load_intraday_builds_sorted_datetime_index(raw_file):
    df = na.load_intraday(raw_file)

    assert list(df.columns) == ["open", "high", "low", "close"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert df.index.name == "timestamp"


def test_load_intraday_rejects_unexpected_shape(tmp_path):
    path = tmp_path / "bad.txt"
    pd.DataFrame(np.zeros((5, 6))).to_csv(path, header=False, index=False)

    with pytest.raises(ValueError, match="expected 9 columns"):
        na.load_intraday(path)


def test_persistence_forecast_repeats_previous_close(close):
    result = na.persistence_forecast(close, train_fraction=0.8)

    assert len(result.predictions) == len(result.actual)
    np.testing.assert_allclose(result.predictions[1:], na.tick_round(result.actual[:-1]))


def test_forecast_metrics_on_perfect_prediction():
    metrics = na.forecast_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["dir_acc"] == pytest.approx(1.0)


def test_forecast_metrics_known_values():
    metrics = na.forecast_metrics([10.0, 12.0], [11.0, 11.0], train=[8.0, 10.0, 12.0])

    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["smape"] == pytest.approx((2 / 21 + 2 / 23) * 100 / 2)
    assert metrics["mase"] == pytest.approx(0.5)
    assert metrics["dir_acc"] == pytest.approx(1.0)


def test_all_models_share_the_same_test_window(close):
    persistence = na.persistence_forecast(close, train_fraction=0.8)
    arima = na.walk_forward_arima(close, order=(1, 1, 0), train_fraction=0.8)

    np.testing.assert_allclose(arima.actual, persistence.actual)
    assert len(arima.predictions) == len(arima.actual)


def test_arima_predictions_and_intervals_are_on_the_tick(close):
    result = na.walk_forward_arima(close, order=(1, 1, 0), train_fraction=0.9)

    for values in (result.predictions, result.ci_lower, result.ci_upper):
        ticks = values / na.TICK
        np.testing.assert_allclose(ticks, np.round(ticks), atol=1e-6)

    coverage = result.coverage()
    assert coverage is not None
    assert 0.0 <= coverage <= 1.0


def test_select_order_by_aic_returns_sorted_table(close):
    table = na.select_order_by_aic(close.iloc[:300], [(0, 1, 0), (1, 1, 0), (2, 1, 0)])

    assert list(table.columns[:2]) == ["aic", "bic"]
    assert table["aic"].is_monotonic_increasing
    assert table["aic"].notna().all()


def test_summary_row_exposes_all_metrics(close):
    row = na.persistence_forecast(close, train_fraction=0.9).summary_row()

    for key in ("model", "order", "rmse", "mae", "smape", "mase", "dir_acc"):
        assert key in row
