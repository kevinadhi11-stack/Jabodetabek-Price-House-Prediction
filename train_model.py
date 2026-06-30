import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_PATH = "data/harga_rumah.csv"

COLUMN_MAP = {
    "harga": "price_in_rp",
    "luas_tanah": "land_size_m2",
    "luas_bangunan": "building_size_m2",
    "kamar_tidur": "bedrooms",
    "kamar_mandi": "bathrooms",
    "garasi": "garages",
    "kota": "city",
}

def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    rename_dict = {v: k for k, v in COLUMN_MAP.items() if v in df.columns}
    df = df.rename(columns=rename_dict)
    if "kota" in df.columns:
        df["kota"] = df["kota"].astype(str).str.strip()
    required = ["harga","luas_tanah","luas_bangunan","kamar_tidur","kamar_mandi","kota"]
    if "garasi" not in df.columns:
        df["garasi"] = 0
    return df[required + ["garasi"]]

def clean_data(df):
    df = df.dropna(subset=["harga","luas_tanah","luas_bangunan","kamar_tidur","kamar_mandi"])
    df = df[df["harga"] > 0]
    q_low, q_high = df["harga"].quantile(0.01), df["harga"].quantile(0.99)
    df = df[(df["harga"] >= q_low) & (df["harga"] <= q_high)]
    df = df[(df["luas_tanah"] >= 10) & (df["luas_bangunan"] >= 10)]
    df = df[(df["kamar_tidur"] >= 1) & (df["kamar_tidur"] <= 20)]
    df = df[(df["kamar_mandi"] >= 1) & (df["kamar_mandi"] <= 15)]
    df["garasi"] = df["garasi"].fillna(0).clip(0,10)
    return df.reset_index(drop=True)

def engineer_features(df):
    df = df.copy()
    df["rasio_bangunan"] = df["luas_bangunan"] / (df["luas_tanah"] + 1)
    df["total_kamar"] = df["kamar_tidur"] + df["kamar_mandi"]
    df["log_harga"] = np.log1p(df["harga"])
    df["kota"] = df["kota"].str.strip().str.title()
    return df

NUMERIC_FEATURES = ["luas_tanah","luas_bangunan","kamar_tidur","kamar_mandi","garasi","rasio_bangunan","total_kamar"]
CATEGORIC_FEATURES = ["kota"]

if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    X = df[NUMERIC_FEATURES + CATEGORIC_FEATURES]
    y = df["log_harga"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORIC_FEATURES),
    ])
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    y_pred = np.expm1(pipeline.predict(X_test))
    y_true = np.expm1(y_test)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: Rp {mae/1e6:.1f} juta | RMSE: Rp {rmse/1e6:.1f} juta | R2: {r2:.4f}")

    os.makedirs("model", exist_ok=True)
    with open("model/pipeline.pkl","wb") as f: pickle.dump(pipeline,f)
    with open("model/kota_list.pkl","wb") as f: pickle.dump(sorted(df["kota"].unique().tolist()),f)
    meta = {"n_data": len(df), "harga_min": int(df["harga"].min()), "harga_max": int(df["harga"].max()), "harga_median": int(df["harga"].median()), "metrics": {"mae":mae,"rmse":rmse,"r2":r2}, "features": NUMERIC_FEATURES+CATEGORIC_FEATURES}
    with open("model/metadata.pkl","wb") as f: pickle.dump(meta,f)
    print("Selesai! Model tersimpan.")
