import os, json
from pathlib import Path
import numpy as np, pandas as pd, joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

SEQ_LEN = 60
OUT_DIR = Path("compute/app/lstm_models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "production_goldmind_v1.h5"
SCALER_PATH = OUT_DIR / "scalers.joblib"

def load_data(csv_path="gold_data.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")
    prices = df["close"].astype(float).values.reshape(-1, 1)
    return prices, df

def make_dataset(series, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i-seq_len:i, 0])
        y.append(series[i, 0])               # next normalized close
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

def build_model(input_steps=SEQ_LEN, input_features=1):
    m = Sequential([
        Input(shape=(input_steps, input_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)  # regression: next normalized close
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

if __name__ == "__main__":
    raw, df = load_data("gold_data.csv")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)

    # train/val split (time-series safe)
    split = int(len(scaled) * 0.85)
    train, valid = scaled[:split], scaled[split-SEQ_LEN:]  # include overlap for windows

    Xtr, ytr = make_dataset(train, SEQ_LEN)
    Xva, yva = make_dataset(valid, SEQ_LEN)

    model = build_model()
    ckpt = ModelCheckpoint(MODEL_PATH.as_posix(), save_best_only=True, monitor="val_loss", mode="min")
    es   = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=60, batch_size=64,
              callbacks=[ckpt, es], verbose=2)

    # final eval (unscale for RMSE)
    preds = model.predict(Xva, verbose=0)
    yva_un = scaler.inverse_transform(yva)
    p_un   = scaler.inverse_transform(preds)
    rmse = mean_squared_error(yva_un, p_un, squared=False)
    print(f"Validation RMSE: {rmse:,.4f}")

    # save scaler
    joblib.dump(scaler, SCALER_PATH)
    print("Saved artifacts:", MODEL_PATH, SCALER_PATH)

    # optional: record metadata
    (OUT_DIR/"training_history.json").write_text(json.dumps({
        "seq_len": SEQ_LEN, "rmse": float(rmse), "n_samples": int(len(df))
    }, indent=2))
