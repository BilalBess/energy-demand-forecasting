"""Day-ahead forecast of the French electricity consumption (hourly).

Setting: every day at midnight we forecast the 24 hours of the day that starts.
So for the hour h of day D we may only use consumption up to D-1 at 23:00.
Temperature is observed (not forecast), see README limitations.

Run:  python src/forecast.py
"""
import json
from pathlib import Path

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

START, END = "2018-01-01", "2025-01-01"
CONSO_URL = ("https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv"
             "?select=date_heure,consommation,prevision_j1&where=date_heure%3E%3D%22{s}%22%20AND%20date_heure%3C%22{e}%22"
             "&delimiter=%2C")
TEMP_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=48.85&longitude=2.35"
            "&start_date=2018-01-01&end_date=2024-12-31&hourly=temperature_2m&timezone=UTC")


def get_data():
    f1, f2 = DATA / "conso_raw.csv", DATA / "temperature_paris.csv"
    if not f1.exists():
        f1.write_bytes(requests.get(CONSO_URL.format(s=START, e=END), timeout=300).content)
    if not f2.exists():
        j = requests.get(TEMP_URL, timeout=120).json()["hourly"]
        pd.DataFrame({"date_heure": j["time"], "temp": j["temperature_2m"]}).to_csv(f2, index=False)

    c = pd.read_csv(f1, encoding="utf-8-sig")
    c["date_heure"] = pd.to_datetime(c["date_heure"], utc=True)
    c = c.set_index("date_heure").sort_index()
    # data is every 15 min but consumption only exists every 30 min -> hourly mean of available points
    hourly = c.resample("1h").mean(numeric_only=True)
    t = pd.read_csv(f2, parse_dates=["date_heure"])
    t["date_heure"] = t["date_heure"].dt.tz_localize("UTC")
    hourly = hourly.join(t.set_index("date_heure"))
    hourly = hourly.rename(columns={"consommation": "load", "prevision_j1": "rte_forecast"})
    return hourly


def make_features(df):
    d = df.copy()
    local = d.index.tz_convert("Europe/Paris")
    d["hour"] = local.hour
    d["dow"] = local.day_of_week
    d["month"] = local.month
    d["doy"] = local.day_of_year
    fr = holidays.France(years=range(2017, 2026))
    d["is_holiday"] = pd.Series(local.date, index=d.index).map(lambda x: x in fr).astype(int)
    # day before / after a holiday (bridge days)
    d["holiday_next_day"] = d["is_holiday"].shift(-24).fillna(0)
    d["holiday_prev_day"] = d["is_holiday"].shift(24).fillna(0)
    # lags that exist at midnight (information up to D-1 23:00)
    d["lag_24"] = d["load"].shift(24)
    d["lag_48"] = d["load"].shift(48)
    d["lag_168"] = d["load"].shift(168)
    # mean of the same hour over the last 7 days (D-1 ... D-7)
    d["mean_same_hour_7d"] = sum(d["load"].shift(24 * k) for k in range(1, 8)) / 7
    d["temp_mean_24h"] = d["temp"].rolling(24, min_periods=12).mean()
    return d


FEATURES = ["hour", "dow", "month", "doy", "is_holiday", "holiday_next_day", "holiday_prev_day",
            "lag_24", "lag_48", "lag_168", "mean_same_hour_7d", "temp", "temp_mean_24h"]


def metrics(y, p):
    ok = ~np.isnan(p) & ~np.isnan(y)
    y, p = y[ok], p[ok]
    return {"MAE_MW": round(float(np.mean(np.abs(y - p)))), "RMSE_MW": round(float(np.sqrt(np.mean((y - p) ** 2)))),
            "MAPE_%": round(float(np.mean(np.abs((y - p) / y)) * 100), 2)}


def main():
    raw = get_data()
    need = ["load", "lag_24", "lag_48", "lag_168", "mean_same_hour_7d", "temp", "temp_mean_24h", "rte_forecast"]
    n0 = len(raw)
    d = make_features(raw).dropna(subset=need)  # same rows for every model, so the comparison is fair
    print("hours dropped because of missing values:", n0 - len(d))
    print("rows:", len(d), "| period:", d.index.min().date(), "->", d.index.max().date())

    y = d["load"]
    train = d.index < "2023-01-01"
    val = (d.index >= "2023-01-01") & (d.index < "2024-01-01")
    test = d.index >= "2024-01-01"
    # 2020 (covid lockdowns) is kept: it is real data, but see README

    # ----- baselines
    res = {}
    res["Naive: same hour last week"] = metrics(y[test].values, d.loc[test, "lag_168"].values)
    res["Naive: same hour yesterday"] = metrics(y[test].values, d.loc[test, "lag_24"].values)
    res["RTE official day-ahead forecast"] = metrics(y[test].values, d.loc[test, "rte_forecast"].values)

    # ----- linear model (calendar one-hot + lags + temperature, with a U-shape on temperature)
    def lin_design(x):
        z = pd.get_dummies(x[["hour", "dow", "month"]].astype(str)).astype(float)
        z["heat"] = np.maximum(x["temp"] - 18, 0)
        z["cold"] = np.maximum(15 - x["temp"], 0)
        for c in ["is_holiday", "holiday_next_day", "holiday_prev_day", "lag_24", "lag_168", "mean_same_hour_7d"]:
            z[c] = x[c]
        return z

    Z = lin_design(d)
    ridge = Ridge(alpha=1.0).fit(Z[d.index < "2024-01-01"], y[d.index < "2024-01-01"])
    res["Ridge (linear)"] = metrics(y[test].values, ridge.predict(Z[test]))

    # ----- gradient boosting: choose a few hyper-parameters on the 2023 validation year
    best, best_mae = None, 1e18
    for lr in [0.05, 0.1]:
        for depth in [4, 6, None]:
            m = HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=400,
                                              l2_regularization=1.0, random_state=0)
            m.fit(d.loc[train, FEATURES], y[train])
            mae = np.mean(np.abs(y[val] - m.predict(d.loc[val, FEATURES])))
            print(f"lr={lr} depth={depth} val MAE={mae:.0f}")
            if mae < best_mae:
                best, best_mae = (lr, depth), mae
    print("best:", best)

    trainval = d.index < "2024-01-01"
    gbm = HistGradientBoostingRegressor(learning_rate=best[0], max_depth=best[1], max_iter=400,
                                        l2_regularization=1.0, random_state=0)
    gbm.fit(d.loc[trainval, FEATURES], y[trainval])
    pred = gbm.predict(d.loc[test, FEATURES])
    res["Gradient boosting (mine)"] = metrics(y[test].values, pred)

    # same model WITHOUT temperature: shows how much of the gain comes from (observed) weather
    no_w = [f for f in FEATURES if not f.startswith("temp")]
    gbm_nw = HistGradientBoostingRegressor(learning_rate=best[0], max_depth=best[1], max_iter=400,
                                           l2_regularization=1.0, random_state=0)
    gbm_nw.fit(d.loc[trainval, no_w], y[trainval])
    res["Gradient boosting, no temperature"] = metrics(y[test].values, gbm_nw.predict(d.loc[test, no_w]))

    table = pd.DataFrame(res).T
    print(table)
    table.to_csv(ROOT / "reports" / "test_results_2024.csv")

    # ----- error analysis
    t = d.loc[test].copy()
    t["pred"] = pred
    t["abs_pct_err"] = np.abs(t["pred"] - t["load"]) / t["load"] * 100
    t["rte_abs_pct_err"] = np.abs(t["rte_forecast"] - t["load"]) / t["load"] * 100
    by_hour = t.groupby("hour")[["abs_pct_err", "rte_abs_pct_err"]].mean()
    by_month = t.groupby("month")[["abs_pct_err", "rte_abs_pct_err"]].mean()
    hol = t.groupby("is_holiday")[["abs_pct_err", "rte_abs_pct_err"]].mean()
    print(hol.round(2))
    print(by_month.round(2))

    # ----- figures
    week = t.loc["2024-01-15":"2024-01-21"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week.index, week["load"], label="Actual", color="black")
    ax.plot(week.index, week["pred"], label="My model", color="#2b6cb0")
    ax.plot(week.index, week["rte_forecast"], label="RTE forecast", color="#c05621", alpha=0.8)
    ax.set_ylabel("Consumption (MW)")
    ax.set_title("One week of January 2024: actual vs forecasts")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "forecast_week.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    by_hour.plot(ax=axes[0], marker="o")
    axes[0].set_title("Mean absolute % error by hour (2024)")
    axes[0].set_ylabel("%")
    by_month.plot(ax=axes[1], marker="o")
    axes[1].set_title("Mean absolute % error by month (2024)")
    axes[1].set_ylabel("%")
    for a in axes:
        a.legend(["My model", "RTE forecast"])
    fig.tight_layout()
    fig.savefig(FIG / "error_by_hour_month.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sample = d.sample(8000, random_state=0)
    ax.scatter(sample["temp"], sample["load"], s=3, alpha=0.3)
    ax.set_xlabel("Temperature in Paris (C)")
    ax.set_ylabel("Consumption (MW)")
    ax.set_title("Consumption vs temperature (2018-2024)")
    fig.tight_layout()
    fig.savefig(FIG / "load_vs_temperature.png", dpi=150)
    plt.close(fig)

    pi = permutation_importance(gbm, d.loc[test, FEATURES].iloc[::3], y[test].iloc[::3],
                                scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#2f855a")
    ax.set_xlabel("Increase of MAE (MW) when the feature is shuffled")
    ax.set_title("Permutation importance")
    fig.tight_layout()
    fig.savefig(FIG / "permutation_importance.png", dpi=150)
    plt.close(fig)
    print(imp.sort_values(ascending=False).round(0).head(8))

    (ROOT / "reports" / "summary.json").write_text(json.dumps(
        {"best_hyperparams": {"learning_rate": best[0], "max_depth": best[1]},
         "holiday_error_%": hol.round(2).to_dict(), "rows": int(len(d)),
         "mean_load_2024_MW": round(float(t["load"].mean()))}, indent=2, default=str))


if __name__ == "__main__":
    main()
