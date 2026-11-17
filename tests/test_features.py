import numpy as np
import pandas as pd
import pytest

import forecast


@pytest.fixture()
def hourly():
    idx = pd.date_range("2024-07-01", periods=24 * 20, freq="h", tz="UTC")
    return pd.DataFrame({"load": np.arange(len(idx), dtype=float), "temp": 20.0, "rte_forecast": 1.0}, index=idx)


def test_lags_use_only_the_past(hourly):
    f = forecast.make_features(hourly)
    t = 24 * 10 + 5
    assert f["lag_24"].iloc[t] == hourly["load"].iloc[t - 24]
    assert f["lag_48"].iloc[t] == hourly["load"].iloc[t - 48]
    assert f["lag_168"].iloc[t] == hourly["load"].iloc[t - 168]


def test_same_hour_mean_is_mean_of_previous_7_days(hourly):
    f = forecast.make_features(hourly)
    t = 24 * 10 + 5
    expected = np.mean([hourly["load"].iloc[t - 24 * k] for k in range(1, 8)])
    assert f["mean_same_hour_7d"].iloc[t] == pytest.approx(expected)


def test_no_leakage_from_the_future(hourly):
    """Changing the load AFTER time t must not change the features at time t or before."""
    base = forecast.make_features(hourly)
    changed = hourly.copy()
    cut = 24 * 12
    changed.iloc[cut:, changed.columns.get_loc("load")] += 1_000_000
    new = forecast.make_features(changed)
    cols = ["lag_24", "lag_48", "lag_168", "mean_same_hour_7d"]
    pd.testing.assert_frame_equal(base.iloc[:cut][cols], new.iloc[:cut][cols])


def test_french_holidays_are_flagged(hourly):
    idx = pd.date_range("2024-07-13", periods=72, freq="h", tz="Europe/Paris").tz_convert("UTC")
    f = forecast.make_features(pd.DataFrame({"load": 1.0, "temp": 20.0, "rte_forecast": 1.0}, index=idx))
    local = f.index.tz_convert("Europe/Paris")
    assert f.loc[local.date == pd.Timestamp("2024-07-14").date(), "is_holiday"].eq(1).all()
    assert f.loc[local.date == pd.Timestamp("2024-07-15").date(), "is_holiday"].eq(0).all()
    # the day before a holiday is flagged too
    assert f.loc[local.date == pd.Timestamp("2024-07-13").date(), "holiday_next_day"].eq(1).all()


def test_calendar_features(hourly):
    f = forecast.make_features(hourly)
    assert set(forecast.FEATURES) <= set(f.columns)
    assert f["hour"].between(0, 23).all() and f["dow"].between(0, 6).all()


def test_metrics_perfect_and_known_values():
    y = np.array([100.0, 200.0])
    assert forecast.metrics(y, y) == {"MAE_MW": 0, "RMSE_MW": 0, "MAPE_%": 0.0}
    m = forecast.metrics(y, np.array([110.0, 180.0]))
    assert m["MAE_MW"] == 15
    assert m["MAPE_%"] == pytest.approx(10.0)       # (10% + 10%) / 2


def test_metrics_ignore_missing_predictions():
    m = forecast.metrics(np.array([100.0, 200.0]), np.array([np.nan, 220.0]))
    assert m["MAE_MW"] == 20
