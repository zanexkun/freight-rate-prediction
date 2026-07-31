"""
Freight rate prediction.

Trains the model and writes the two submission files:
    validation_predictions.csv
    december-chart-inputs.csv

Run with: python train.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR = "data"
TRAIN_PATH = "train-test.csv"
VALIDATION_PATH = "validation.csv"
DECEMBER_PATH = "december-chart-inputs.csv"

SPLIT_DATE = "2025-09-01"
TRIM_LOW, TRIM_HIGH = 0.01, 0.99

# Not used by the model. They don't exist in the December file, and dropping
# them also made the test MAE better.
DROP_FEATURES = ["market_index", "quote_signal"]


def add_dayofweek(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dayofweek"] = df["date"].dt.dayofweek
    return df


def build_features(df, weight_median, market_median):
    # Both medians are passed in from the training set. Computing them from
    # the validation data would be leakage.
    df = df.copy()

    df["weight"] = df["weight"].fillna(weight_median)
    df["market_index"] = df["market_index"].fillna(df["dayofweek"].map(market_median))

    df = pd.get_dummies(df, columns=["equipment"])
    df = df.astype({col: int for col in df.select_dtypes(bool).columns})

    drop_cols = [c for c in ["load_id", "pickup", "delivery", "date"] if c in df.columns]
    return df.drop(columns=drop_cols)


def main():
    train_raw = pd.read_csv(TRAIN_PATH)
    val_raw = pd.read_csv(VALIDATION_PATH)

    train_raw = add_dayofweek(train_raw)
    val_raw = add_dayofweek(val_raw)

    weight_median = train_raw["weight"].median()
    market_median = train_raw.groupby("dayofweek")["market_index"].median()

    train = build_features(train_raw, weight_median, market_median)
    val = build_features(val_raw, weight_median, market_median)
    val_ids = val_raw["load_id"]

    dates = train_raw["date"]

    # 317 loads are priced above $5/mile when the median is $2.15, and some go
    # up to $14. I checked whether anything separates them from normal loads
    # and nothing does: same equipment mix, same spread across days of the
    # week, distances all over the range. So they can't be predicted, and
    # leaving them in just pushes every similar prediction too high. I cut the
    # top and bottom 1% from training. The test set stays untouched, otherwise
    # the score would be flattering rather than honest.
    rpm = train["posted_rate"] / train["distance"]
    keep = (rpm > rpm.quantile(TRIM_LOW)) & (rpm < rpm.quantile(TRIM_HIGH))
    print(f"Trimmed training rows: {len(train)} -> {keep.sum()}")

    # Training data is Jan to Oct and the real validation set is Nov to Dec,
    # so the split here is by date, not random. Train on Jan to Aug, test on
    # Sep to Oct. A random split would let the model see dates on either side
    # of every test row, which isn't the situation it will actually face.
    is_train_period = dates < SPLIT_DATE

    train_rows = is_train_period & keep
    test_rows = ~is_train_period

    features = [c for c in train.columns if c != "posted_rate" and c not in DROP_FEATURES]

    X_train = train.loc[train_rows, features]
    y_train = train.loc[train_rows, "posted_rate"]
    X_test = train.loc[test_rows, features]
    y_test = train.loc[test_rows, "posted_rate"]

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # I tried XGBoost too and it came out worse (MAE 138.86 against 116.88),
    # including after switching its objective to absolute error. My read is
    # that boosting keeps chasing those unpredictable expensive loads, while
    # averaging independent trees stays calmer when the leftover error is
    # mostly noise.
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"R2:   {r2_score(y_test, preds):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, preds):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, preds) ** 0.5:.2f}")

    validation_predictions = pd.DataFrame({
        "load_id": val_ids,
        "predicted_rate": model.predict(val[features]),
    })
    # The scorer rejects the file for either of these, so check before writing.
    assert len(validation_predictions) == 12_000
    assert (validation_predictions["predicted_rate"] > 0).all()
    validation_predictions.to_csv("validation_predictions.csv", index=False)
    print("Wrote validation_predictions.csv")

    # Every column in the December file is fixed except the date, so whatever
    # shape the chart has comes purely from the day of week. The two cities
    # are always the same, so I pull their coordinates out of the training data.
    december = pd.read_csv(DECEMBER_PATH)
    pickup = train_raw.loc[train_raw["pickup"] == "Lexington", ["pickup_lat", "pickup_lon"]].iloc[0]
    delivery = train_raw.loc[train_raw["delivery"] == "Fort Wayne", ["delivery_lat", "delivery_lon"]].iloc[0]

    december_features = pd.DataFrame({
        "pickup_lat": pickup["pickup_lat"],
        "pickup_lon": pickup["pickup_lon"],
        "delivery_lat": delivery["delivery_lat"],
        "delivery_lon": delivery["delivery_lon"],
        "distance": december["distance"].astype(float),
        "weight": december["weight"].astype(float),
        "dayofweek": pd.to_datetime(december["date"]).dt.dayofweek,
        "equipment_Dry Van": 1,
        "equipment_Flatbed": 0,
        "equipment_Reefer": 0,
    })[features]

    december["predicted_rate"] = model.predict(december_features)
    december.to_csv("december-chart-inputs.csv", index=False)
    print("Wrote december-chart-inputs.csv")


if __name__ == "__main__":
    main()
