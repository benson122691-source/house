"""Train time-held-out rent and sale price models on New Taipei open data."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = ROOT / "results" / "model_results.json"
SQM_PER_PING = 3.305785
SALE_NUM = ["area_ping", "building_age", "floor_level", "total_floors", "floor_ratio", "bedrooms", "living_rooms", "bathrooms", "trade_year", "trade_month", "parking_price_wan"]
SALE_CAT = ["district", "property_type", "purpose", "management", "elevator", "has_parking"]
RENT_NUM = ["area_ping", "building_age", "floor_level", "total_floors", "floor_ratio", "bedrooms", "living_rooms", "bathrooms", "trade_year", "trade_month"]
RENT_CAT = ["district", "property_type", "purpose", "room_type", "management", "furnished", "elevator"]


def as_roc_date(value):
    if pd.isna(value):
        return pd.NaT
    digits = re.sub(r"\D", "", str(value).split(".")[0])
    if len(digits) < 6:
        return pd.NaT
    digits = digits.zfill(7)[-7:]
    try:
        return pd.Timestamp(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:7]))
    except ValueError:
        return pd.NaT


def chinese_number(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if re.search(r"地下|B\s*\d", s, re.I):
        m = re.search(r"(?:地下|B\s*)(\d+)", s, re.I)
        return -int(m.group(1)) if m else -1
    m = re.search(r"\d+(?:\.\d+)?", s)
    if m:
        return float(m.group(0))
    chars = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if not any(ch in s for ch in chars) and "十" not in s:
        return np.nan
    total, current = 0, 0
    for ch in s:
        if ch in chars:
            current = chars[ch]
        elif ch == "十":
            total += (current or 1) * 10
            current = 0
        elif ch == "百":
            total += (current or 1) * 100
            current = 0
    return float(total + current)


def prepare_frame(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    sale = kind == "sale"
    trade_col = "rps07_yyymmddroc"
    area_col = "rps15_area"
    date = df[trade_col].map(as_roc_date)
    built = df["rps14_yyymmddroc"].map(as_roc_date)
    area = pd.to_numeric(df[area_col], errors="coerce") / SQM_PER_PING
    floor = df["rps09"].map(chinese_number)
    total_floors = df["rps10" if sale else "rps10_quantity"].map(chinese_number)
    age = (date.dt.year - built.dt.year).clip(lower=0, upper=120)
    floor_ratio = (floor / total_floors).where(total_floors > 0).clip(-0.2, 1.5)
    frame = pd.DataFrame({
        "district": df["district"].astype("string").str.strip(),
        "property_type": df["rps11"].astype("string").str.strip(),
        "purpose": df["rps12"].astype("string").str.strip(),
        "area_ping": area,
        "building_age": age,
        "floor_level": floor,
        "total_floors": total_floors,
        "floor_ratio": floor_ratio,
        "bedrooms": pd.to_numeric(df["rps16_quantity"], errors="coerce"),
        "living_rooms": pd.to_numeric(df["rps17_quantity"], errors="coerce"),
        "bathrooms": pd.to_numeric(df["rps18_quantity"], errors="coerce"),
        "management": df["rps20"].astype("string").str.strip(),
        "elevator": df["rps31" if sale else "rps32"].astype("string").str.strip(),
        "trade_year": date.dt.year,
        "trade_month": date.dt.month,
        "trade_date": date,
    })
    if sale:
        total_price = pd.to_numeric(df["rps21_amountsunitdollars"], errors="coerce")
        parking_price = pd.to_numeric(df["rps25_amountsunitdollars"], errors="coerce").fillna(0)
        unit_price_yuan_sqm = pd.to_numeric(df["rps22_amountsunitdollars"], errors="coerce")
        unit_price_wan_ping = (unit_price_yuan_sqm * SQM_PER_PING / 10000)
        fallback = (total_price - parking_price).clip(lower=0) / (area * SQM_PER_PING) * SQM_PER_PING / 10000
        frame["target"] = unit_price_wan_ping.where(unit_price_wan_ping > 0, fallback)
        frame["parking_price_wan"] = parking_price / 10000
        frame["has_parking"] = np.where(parking_price.gt(0), "有車位交易", "無車位/未記載")
    else:
        rent = pd.to_numeric(df["rps22_amountsunitdollars"], errors="coerce")
        frame["target"] = rent / area
        frame["room_type"] = df["rps28"].astype("string").str.strip()
        frame["furnished"] = df["rps21"].astype("string").str.strip()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[frame["district"].notna() & frame["area_ping"].between(3, 150) & frame["target"].gt(0)].copy()
    if sale:
        frame = frame.loc[frame["target"].between(5, 2000)].copy()
        num_cols, cat_cols = SALE_NUM, SALE_CAT
    else:
        frame = frame.loc[frame["target"].between(100, 30000)].copy()
        num_cols, cat_cols = RENT_NUM, RENT_CAT
    frame = frame.loc[frame["trade_date"].notna()].sort_values("trade_date").reset_index(drop=True)
    # Trim only the extreme tails to keep rare data errors from dominating the fitted loss.
    low, high = frame["target"].quantile([0.005, 0.995])
    frame = frame.loc[frame["target"].between(low, high)].reset_index(drop=True)
    return frame


def build_pipeline(model_name, numeric, categorical):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if model_name == "Ridge":
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))])
    prep = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)], remainder="drop")
    if model_name == "Ridge":
        reg = Ridge(alpha=20.0)
    else:
        reg = RandomForestRegressor(n_estimators=180, min_samples_leaf=4, max_features=0.85, n_jobs=-1, random_state=42)
    return Pipeline([("prep", prep), ("model", reg)])


def run_one(kind: str):
    fname = "sales.csv" if kind == "sale" else "rentals.csv"
    raw = pd.read_csv(DATA_DIR / fname, encoding="utf-8-sig", low_memory=False)
    # The source identifier can represent a unit or contract and may legitimately recur;
    # remove only exact duplicate records, not every repeat of the identifier.
    raw = raw.drop_duplicates()
    frame = prepare_frame(raw, kind)
    num, cat = (SALE_NUM, SALE_CAT) if kind == "sale" else (RENT_NUM, RENT_CAT)
    for col in cat:
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    features = num + cat
    cutoff_index = int(len(frame) * 0.80)
    train, test = frame.iloc[:cutoff_index].copy(), frame.iloc[cutoff_index:].copy()
    cutoff = test["trade_date"].min()
    y_train, y_test = train["target"], test["target"]
    rows = []
    median_value = float(y_train.median())
    for name in ["Median baseline", "Ridge", "Random forest"]:
        if name == "Median baseline":
            pred = np.full(len(test), median_value)
        else:
            model = build_pipeline(name, num, cat)
            model.fit(train[features], y_train)
            pred = np.maximum(model.predict(test[features]), 0)
        rows.append({
            "model": name,
            "mae": round(float(mean_absolute_error(y_test, pred)), 3),
            "rmse": round(float(math.sqrt(mean_squared_error(y_test, pred))), 3),
        })
        if name == "Random forest":
            best_model = model
    best = min(rows, key=lambda r: r["mae"])
    # Use an explicit, repeatable case and category modes drawn from 2024+ examples in Banqiao.
    local = frame.loc[(frame["district"] == "板橋區") & (frame["trade_year"] >= 2024)]
    if local.empty:
        local = train
    seed = local.iloc[-1][features].copy()
    for col in cat:
        mode = local[col].mode(dropna=True)
        if not mode.empty:
            seed[col] = mode.iloc[0]
    seed["area_ping"] = 30.0
    seed["building_age"] = 15.0
    seed["floor_level"] = 7.0
    seed["total_floors"] = 14.0
    seed["floor_ratio"] = 0.5
    seed["bedrooms"] = 3.0
    seed["living_rooms"] = 2.0
    seed["bathrooms"] = 2.0
    seed["trade_year"] = 2026.0
    seed["trade_month"] = 10.0
    seed["property_type"] = "住宅大樓(11層含以上有電梯)"
    seed["purpose"] = "住家用"
    seed["management"] = "有"
    seed["elevator"] = "有"
    if kind == "sale":
        seed["parking_price_wan"] = 0.0
        seed["has_parking"] = "無車位/未記載"
    if kind == "rent":
        seed["room_type"] = "整層住家"
        seed["furnished"] = "有"
    pred_model = best_model if best["model"] == "Random forest" else None
    # If the median baseline wins, still use the fitted Random Forest for the concrete demonstration,
    # while metrics retain the genuine best-on-test comparison above.
    if pred_model is None:
        pred_model = build_pipeline("Random forest", num, cat)
        pred_model.fit(train[features], y_train)
    prediction = max(0.0, float(pred_model.predict(pd.DataFrame([seed])[features])[0]))
    model_unit = "萬元/坪" if kind == "sale" else "元/月/坪"
    return {
        "kind": kind,
        "source_rows": int(len(raw)),
        "clean_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_from": cutoff.strftime("%Y-%m-%d"),
        "test_to": frame["trade_date"].max().strftime("%Y-%m-%d"),
        "target_unit": model_unit,
        "features": features,
        "models": rows,
        "best_model": best["model"],
        "random_forest_demo_prediction_unit": round(prediction, 2),
        "demo_features": {key: (None if pd.isna(seed[key]) else str(seed[key]) if key in cat else float(seed[key])) for key in features},
    }


def main():
    results = {
        "city": "新北市",
        "source_files": {"rent": "New Taipei rental real-price records", "sale": "New Taipei sale real-price records"},
        "scikit_learn": sklearn.__version__,
        "split": "按交易日期排序，前80%訓練、最新20%測試；每個模型共用測試集",
        "cleaning": "去重；篩掉無效面積與目標；目標兩端各0.5%極端值；僅保留有交易日期樣本",
        "rent": run_one("rent"),
        "sale": run_one("sale"),
    }
    # Financial comparison is a transparent worked example; all costs are explicit assumptions.
    rent_unit = results["rent"]["random_forest_demo_prediction_unit"]
    sale_unit = results["sale"]["random_forest_demo_prediction_unit"]
    p0 = sale_unit * 30.0  # 萬元
    monthly_rent = rent_unit * 30.0  # 元
    assumptions = {
        "area_ping": 30,
        "sale_tax_now_wan": 45.0,
        "sale_tax_year10_wan": 60.0,
        "sale_cost_rate": 0.04,
        "house_and_land_tax_per_year_wan": 4.0,
        "rental_income_tax_per_year_wan": 2.0,
        "management_and_maintenance_per_year_wan": 6.0,
        "vacancy_rate": 0.05,
        "loan_balance_wan": 0.0,
        "investment_rate": 0.03,
        "years": 10,
        "home_price_growth_scenarios": [0.0, 0.02, 0.04],
        "assumption_note": "示範輸入，不是法定稅額估算；正式試算須改成個別稅單、貸款與稅務估算。",
    }
    annual_gross_rent_wan = monthly_rent * 12 / 10000
    annual_net_rent_wan = annual_gross_rent_wan * (1 - assumptions["vacancy_rate"]) - assumptions["house_and_land_tax_per_year_wan"] - assumptions["rental_income_tax_per_year_wan"] - assumptions["management_and_maintenance_per_year_wan"]
    r, n = assumptions["investment_rate"], assumptions["years"]
    rent_fv = annual_net_rent_wan * (((1 + r) ** n - 1) / r)
    a_now = (p0 * (1 - assumptions["sale_cost_rate"]) - assumptions["sale_tax_now_wan"] - assumptions["loan_balance_wan"]) * (1 + r) ** n
    cases = []
    for g in assumptions["home_price_growth_scenarios"]:
        ending_home_net = p0 * (1 + g) ** n * (1 - assumptions["sale_cost_rate"]) - assumptions["sale_tax_year10_wan"]
        b_hold = rent_fv + ending_home_net
        cases.append({"growth_rate": g, "sell_now_A_wan": round(a_now, 1), "rent_hold_B_wan": round(b_hold, 1), "difference_B_minus_A_wan": round(b_hold - a_now, 1)})
    # Interpolate the root using bisection over a plausible interval.
    lo, hi = -0.2, 0.3
    for _ in range(100):
        mid = (lo + hi) / 2
        diff = rent_fv + p0 * (1 + mid) ** n * (1 - assumptions["sale_cost_rate"]) - assumptions["sale_tax_year10_wan"] - a_now
        if diff >= 0:
            hi = mid
        else:
            lo = mid
    results["comparison"] = {
        "demo_property": "板橋區假設大樓，30坪、屋齡15年、7/14樓、3房2廳2衛、有管理/電梯、無車位、無貸款",
        "model_outputs": {"sale_price_wan": round(p0, 1), "monthly_rent_ntd": round(monthly_rent), "sale_unit_wan_per_ping": round(sale_unit, 2), "rent_unit_ntd_per_ping": round(rent_unit, 2)},
        "assumptions": assumptions,
        "annual_net_rent_wan": round(annual_net_rent_wan, 2),
        "rent_fv_wan": round(rent_fv, 1),
        "sell_now_A_wan": round(a_now, 1),
        "cases": cases,
        "break_even_growth_rate": round((lo + hi) / 2, 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

