import pandas as pd
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

FEATURES = [
    "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon",
    "distance", "weight",
    "month", "day_of_week",
    "lane_rpm",
]
SPLIT_DATE = "2025-09-01"


# cleaning

def cleaning(df):
    df = df.copy()
    df["weight"] = df["weight"].abs()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["lane"] = df["pickup"] + " to " + df["delivery"]
    return df


def remove_odd_loads(df):
    df = df.copy()
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]
    return df[(df["rate_per_mile"] > 1) & (df["rate_per_mile"] < 4)].copy()


train_raw = pd.read_csv("data/train_test.csv")
val_raw = pd.read_csv("data/validation.csv")
dec_raw = pd.read_csv("data/december_chart_inputs.csv")

train = cleaning(train_raw)
val = cleaning(val_raw)
dec = cleaning(dec_raw)

# december has no coordinates, every city has one set so i look them up from training
pickup_coords = train.drop_duplicates("pickup").set_index("pickup")
delivery_coords = train.drop_duplicates("delivery").set_index("delivery")
for col in ["pickup_lat", "pickup_lon"]:
    dec[col] = dec["pickup"].map(pickup_coords[col])
for col in ["delivery_lat", "delivery_lon"]:
    dec[col] = dec["delivery"].map(delivery_coords[col])


# model inputs

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train[["equipment"]])


def make_X(df, weight_median):
    X = df[FEATURES].copy()
    X["weight"] = X["weight"].fillna(weight_median)
    equip = encoder.transform(df[["equipment"]])
    for i, name in enumerate(encoder.categories_[0]):
        X["equip_" + name] = equip[:, i]
    return X


# modelling: random forest and lightgbm, the final prediction is their average

def train_and_predict(train_df, predict_dfs):
    train_df = remove_odd_loads(train_df)
    weight_median = train_df["weight"].median()

    lane_enc = TargetEncoder(target_type="continuous", random_state=42)
    train_df["lane_rpm"] = lane_enc.fit_transform(train_df[["lane"]], train_df["rate_per_mile"])[:, 0]

    X = make_X(train_df, weight_median)
    y = train_df["posted_rate"]
    rf = RandomForestRegressor(random_state=42, n_jobs=-1).fit(X, y)
    lgbm = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1).fit(X, y)

    preds = []
    for df in predict_dfs:
        df = df.copy()
        df["lane_rpm"] = lane_enc.transform(df[["lane"]])[:, 0]
        X_df = make_X(df, weight_median)
        preds.append((rf.predict(X_df) + lgbm.predict(X_df)) / 2)
    return preds


# validation: train on january to august, test on september and october

tr = train[train["date"] < SPLIT_DATE]
te = train[train["date"] >= SPLIT_DATE]

base = remove_odd_loads(tr)["rate_per_mile"].median()
base_pred = base * te["distance"]
print("baseline MAE:", round(mean_absolute_error(te["posted_rate"], base_pred), 2))

[te_pred] = train_and_predict(tr, [te])
print("model MAE:", round(mean_absolute_error(te["posted_rate"], te_pred), 2))
print("model R2:", round(r2_score(te["posted_rate"], te_pred), 4))


# final model: retrain on everything from january to october

val_pred, dec_pred = train_and_predict(train, [val, dec])

val_preds = pd.Series(val_pred, index=val["load_id"])
template = pd.read_csv("data/validation_predictions_template.csv")
template["predicted_rate"] = template["load_id"].map(val_preds)
assert len(template) == 12_000
assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
template[["load_id", "predicted_rate"]].to_csv("validation_predictions.csv", index=False)

dec_out = dec_raw.copy()
dec_out["predicted_rate"] = dec_pred
dec_out.to_csv("data/december_chart_inputs.csv", index=False)

print("saved validation_predictions.csv and data/december_chart_inputs.csv")
