from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = ROOT / "data" / "weather"
PHENOLOGY_FILE = ROOT / "data" / "phenology" / "flower_phenology_source.xlsx"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

CITY_FILES = {
    "西安": "西安.xlsx", "杭州": "杭州.xlsx", "武汉": "武汉.xlsx",
    "郑州": "郑州.xlsx", "毕节": "毕节.xlsx", "吐鲁番": "吐鲁番.xlsx",
}
WEATHER_ALIASES = {
    "temperature": ["T"], "humidity": ["U"], "wind": ["Ff"],
    "weather_code": ["WW", "ww"], "visibility": ["VV"],
    "pressure": ["P0", "Po"],
}


def read_weather(path: Path, city: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    date_col = next((c for c in raw.columns if str(c).startswith("时间")), raw.columns[0])
    dt = pd.to_datetime(raw[date_col], format="%d.%m.%Y %H:%M", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(raw[date_col], errors="coerce")
    out = pd.DataFrame({"datetime": dt, "city": city})
    for target, candidates in WEATHER_ALIASES.items():
        col = next((c for c in candidates if c in raw.columns), None)
        out[target] = pd.to_numeric(raw[col], errors="coerce") if col else np.nan
    out = out.dropna(subset=["datetime"]).copy()
    out = out[out.datetime.dt.month.eq(4) & out.datetime.dt.day.between(4, 6)]
    # The supplied data dictionary defines WW=1 as rain and WW=0 as no rain.
    # Keep unknown / missing codes missing instead of silently treating them as dry.
    out["rain_code_proxy"] = out.weather_code.where(out.weather_code.isin([0, 1]))
    return out


def analyze_weather() -> pd.DataFrame:
    parts = []
    for city, filename in CITY_FILES.items():
        path = WEATHER_DIR / filename
        if path.exists():
            parts.append(read_weather(path, city))
    if not parts:
        raise FileNotFoundError(f"No weather spreadsheets found under {WEATHER_DIR}")
    data = pd.concat(parts, ignore_index=True).drop_duplicates(["city", "datetime"])
    data.to_csv(RESULTS / "清明天气逐时整理.csv", index=False, encoding="utf-8-sig")
    summary = data.groupby("city").agg(
        records=("datetime", "size"), years=("datetime", lambda x: x.dt.year.nunique()),
        mean_temperature=("temperature", "mean"), mean_humidity=("humidity", "mean"),
        rain_code_share=("rain_code_proxy", "mean"),
        rain_code_hours=("rain_code_proxy", "sum"),
    ).reset_index()
    summary.to_csv(RESULTS / "城市清明天气概览.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=summary, x="city", y="rain_code_share", color="#4C78A8")
    plt.ylabel("有雨记录占比")
    plt.xlabel("城市")
    plt.title("清明假期天气代码特征（原始样本汇总）")
    plt.tight_layout()
    plt.savefig(FIGURES / "weather_city_comparison.png", dpi=180)
    plt.close()
    return summary


def parse_month_day(value) -> tuple[int, int] | None:
    if pd.isna(value):
        return None
    try:
        # Excel stores dates such as 4.4 as decimal MM.DD; preserve the source convention.
        v = float(value)
        month = int(v)
        day = int(round((v - month) * 100))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return month, day
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d{1,2})\s*[月./-]\s*(\d{1,2})", str(value))
    return (int(m.group(1)), int(m.group(2))) if m else None


def read_phenology() -> pd.DataFrame:
    raw = pd.read_excel(PHENOLOGY_FILE, header=1, usecols="A:E")
    raw.columns = ["flower", "region", "year", "start_raw", "duration"]
    raw["year"] = pd.to_numeric(raw.year, errors="coerce")
    raw["duration"] = pd.to_numeric(raw.duration, errors="coerce")
    raw["flower"] = raw.flower.where(raw.flower.astype(str).str.strip().ne("品类"))
    raw["region"] = raw.region.where(raw.region.astype(str).str.strip().ne("地区"))
    raw[["flower", "region"]] = raw[["flower", "region"]].ffill()
    rows = []
    for rec in raw.dropna(subset=["year", "duration"]).to_dict("records"):
        md = parse_month_day(rec["start_raw"])
        if not md or not rec["flower"]:
            continue
        month, day = md
        try:
            start = pd.Timestamp(year=int(rec["year"]), month=month, day=day)
        except ValueError:
            continue
        rows.append({"flower": str(rec["flower"]).strip(), "region": str(rec["region"]).strip(),
                     "year": int(rec["year"]), "start_date": start,
                     "start_doy": start.dayofyear, "duration_days": float(rec["duration"])})
    data = pd.DataFrame(rows).drop_duplicates(["flower", "year"])
    if data.empty:
        raise ValueError("No usable phenology observations found in the supplied workbook")
    data.to_csv(RESULTS / "花期历史数据整理.csv", index=False, encoding="utf-8-sig")
    return data


def rolling_origin_metrics(group: pd.DataFrame, target: str) -> dict:
    group = group.sort_values("year").reset_index(drop=True)
    predictions, actuals = [], []
    # Each test point uses only earlier years. A minimum of five train years is required.
    for i in range(5, len(group)):
        train = group.iloc[:i]
        test = group.iloc[[i]]
        x = train[["year"]].to_numpy()
        y = train[target].to_numpy(dtype=float)
        model = LinearRegression().fit(x, y)
        predictions.append(float(model.predict(test[["year"]].to_numpy())[0]))
        actuals.append(float(test[target].iloc[0]))
    if not actuals:
        return {"mae": np.nan, "rmse": np.nan, "n_test": 0}
    return {"mae": mean_absolute_error(actuals, predictions),
            "rmse": mean_squared_error(actuals, predictions) ** 0.5,
            "n_test": len(actuals)}


def analyze_phenology(data: pd.DataFrame) -> pd.DataFrame:
    predictions, metrics = [], []
    for flower, group in data.groupby("flower"):
        group = group.sort_values("year").copy()
        # For short series, a linear trend is more transparent; RF is fit for comparison only.
        model = LinearRegression().fit(group[["year"]], group.start_doy)
        doy = float(model.predict(pd.DataFrame({"year": [2026]}))[0])
        doy = float(np.clip(doy, 1, 365))
        date_2026 = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(round(doy)) - 1)
        duration = float(LinearRegression().fit(group[["year"]], group.duration_days)
                         .predict(pd.DataFrame({"year": [2026]}))[0])
        predictions.append({"flower": flower, "prediction_year": 2026,
                            "predicted_start_date_trend": date_2026.date().isoformat(),
                            "predicted_start_doy_trend": round(doy, 2),
                            "predicted_duration_trend_days": round(max(duration, 1), 2),
                            "training_years": int(group.year.nunique()),
                            "method": "linear trend on year; exploratory baseline"})
        for target in ["start_doy", "duration_days"]:
            metric = rolling_origin_metrics(group, target)
            metrics.append({"flower": flower, "target": target, "method": "linear trend",
                            **metric})
        if len(group) >= 6:
            X = group[["year"]]
            y = group.duration_days
            rf = make_pipeline(SimpleImputer(strategy="median"),
                               RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                     random_state=42))
            rf.fit(X, y)
            rf_pred = float(rf.predict(pd.DataFrame({"year": [2026]}))[0])
            predictions[-1]["predicted_duration_rf_comparison_days"] = round(max(rf_pred, 1), 2)
    pred = pd.DataFrame(predictions)
    pred.to_csv(RESULTS / "花期趋势预测_探索性.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(RESULTS / "花期滚动验证.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))
    for flower, group in data.groupby("flower"):
        ax.plot(group.year, group.start_doy, marker="o", label=flower)
    ax.set_xlabel("年份")
    ax.set_ylabel("始花日期（年积日）")
    ax.set_title("花卉始花日期历史变化")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "phenology_start_trends.png", dpi=180)
    plt.close(fig)
    return pred


def make_travel_window(data: pd.DataFrame) -> pd.DataFrame:
    # Fixed illustrative city-region mapping from the supplied analysis files; kept separate
    # from model-derived phenology predictions because those data are region-level and sparse.
    city_region = {"西安": "西北地区", "杭州": "华东地区", "武汉": "华中地区",
                   "郑州": "华北地区", "毕节": "西南地区", "吐鲁番": "西北地区"}
    cities = []
    for city, region in city_region.items():
        for flower, grp in data[data.region.eq(region)].groupby("flower"):
            starts = []
            for _, row in grp.iterrows():
                dt = row.start_date
                # Circular distance to April 4 avoids comparing Jan/Dec linearly.
                starts.append((dt.month, dt.day, float(row.duration_days)))
            if not starts:
                continue
            best = []
            for year in [2024]:
                holiday_start, holiday_end = pd.Timestamp(year, 4, 4), pd.Timestamp(year, 4, 6)
                for month, day, duration in starts:
                    try:
                        bloom_start = pd.Timestamp(year, month, day)
                        bloom_end = bloom_start + pd.Timedelta(days=max(int(duration) - 1, 0))
                        overlap = max(0, (min(bloom_end, holiday_end) - max(bloom_start, holiday_start)).days + 1)
                        best.append(overlap * 24)
                    except ValueError:
                        continue
            cities.append({"city": city, "region_assumption": region, "flower": flower,
                           "mean_2015_2025_holiday_overlap_hours": round(float(np.mean(best)), 2) if best else 0})
    result = pd.DataFrame(cities)
    result.to_csv(RESULTS / "清明花期重叠分析_区域映射假设.csv", index=False, encoding="utf-8-sig")
    if not result.empty:
        pivot = result.pivot(index="flower", columns="city", values="mean_2015_2025_holiday_overlap_hours")
        plt.figure(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "重叠小时"})
        plt.title("历史花期与清明假期重叠情况（区域对应为分析假设）")
        plt.tight_layout()
        plt.savefig(FIGURES / "holiday_bloom_overlap.png", dpi=180)
        plt.close()
    return result


def main() -> None:
    weather = analyze_weather()
    phenology = read_phenology()
    bloom = analyze_phenology(phenology)
    overlap = make_travel_window(phenology)
    report = {
        "weather_rows": int(weather.records.sum()),
        "weather_cities": int(weather.city.nunique()),
        "phenology_rows": int(len(phenology)),
        "flowers": sorted(phenology.flower.unique().tolist()),
        "bloom_prediction_rows": int(len(bloom)),
        "overlap_rows": int(len(overlap)),
        "outputs": [str(p.relative_to(ROOT)) for p in sorted(RESULTS.rglob("*")) if p.is_file()],
        "notes": [
            "Weather rain-code share is a raw-code proxy, not a validated drizzle-event classifier.",
            "Phenology predictions are exploratory year-trend baselines from short series.",
            "City-to-region mapping in the overlap table is an explicit assumption; it is not city-specific observed phenology.",
        ],
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


