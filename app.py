# app.py
# ---------------------------------------------
# Batch Stock Price Prediction (Simple LSTM)
# ---------------------------------------------
import warnings, io, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests, certifi
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---- Reproducibility & TF logging ----
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.get_logger().setLevel("ERROR")

@st.cache_data(show_spinner=False)
def _fetch_html(url: str) -> str:
    """Fetch HTML with requests using certifi CA bundle (fixes SSL verify failures on macOS)."""
    resp = requests.get(url, timeout=20, verify=certifi.where())
    resp.raise_for_status()
    return resp.text

# ------------- Index loaders (read_html on fetched HTML) -------------
@st.cache_data(show_spinner=False)
def get_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = _fetch_html(url)
    tables = pd.read_html(html)
    df = tables[0]
    return df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()

@st.cache_data(show_spinner=False)
def get_nasdaq100_stocks():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = _fetch_html(url)
    tables = pd.read_html(html)
    for tbl in tables:
        cols = {str(c).lower(): c for c in tbl.columns}
        if "ticker" in cols:
            return tbl[cols["ticker"]].astype(str).str.replace(".", "-", regex=False).tolist()
    return []

@st.cache_data(show_spinner=False)
def get_dow30_stocks():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    html = _fetch_html(url)
    tables = pd.read_html(html)
    for tbl in tables:
        cols = {str(c).lower(): c for c in tbl.columns}
        if "symbol" in cols and len(tbl) >= 30:
            return tbl[cols["symbol"]].astype(str).str.replace(".", "-", regex=False).tolist()
    return []

def _safe_yf_list(func, default=None):
    try:
        return func()
    except Exception:
        return default or []

def load_universe(index_name: str, uploaded_csv: bytes | None):
    """Return list of stocks from uploaded CSV or chosen index (with fallbacks)."""
    if uploaded_csv:
        df = pd.read_csv(io.BytesIO(uploaded_csv), header=None)
        return df[0].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()

    idx = index_name.lower()
    try:
        if idx == "sp500":
            stocks = get_sp500_stocks()
            if not stocks:
                stocks = _safe_yf_list(yf.tickers_sp500, [])
            return stocks
        if idx == "nasdaq100":
            return get_nasdaq100_stocks()
        if idx == "dow30":
            stocks = get_dow30_stocks()
            if not stocks:
                stocks = _safe_yf_list(yf.tickers_dow, [])
            return stocks
    except Exception as e:
        st.error(f"Failed to load stocks: {e}")
        return []
    return []

# ------------- Market data + modeling -------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_history(stock: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(stock, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    return df

def next_business_day(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + BDay(1)).normalize()

def make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i])
        y.append(arr[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y

def train_and_predict(closes: np.ndarray, lookback: int, train_ratio: float, epochs: int, batch_size: int):
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(closes.reshape(-1, 1)).flatten()
    X, y = make_sequences(scaled, lookback)
    if len(X) < 10:
        raise RuntimeError("Not enough supervised samples.")

    split = int(len(X) * train_ratio)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    # Evaluate (model vs test)
    y_pred_test = model.predict(X_test, verbose=0).reshape(-1, 1)
    y_test_2d = y_test.reshape(-1, 1)
    y_pred_inv = scaler.inverse_transform(y_pred_test).flatten()
    y_test_inv = scaler.inverse_transform(y_test_2d).flatten()
    rmse_model = float(np.sqrt(np.mean((y_pred_inv - y_test_inv) ** 2)))

    # Naive baseline: "next = last value in window"
    naive_scaled = X_test[:, -1, 0].reshape(-1, 1)
    naive_inv = scaler.inverse_transform(naive_scaled).flatten()
    rmse_naive = float(np.sqrt(np.mean((naive_inv - y_test_inv) ** 2)))

    # Next-day prediction
    last_window = scaled[-lookback:].reshape(1, lookback, 1)
    next_scaled = model.predict(last_window, verbose=0)[0, 0]
    next_price = float(scaler.inverse_transform([[next_scaled]])[0, 0])

    return next_price, rmse_model, rmse_naive

def predict_one(stock: str, period: str, lookback: int, train_ratio: float, epochs: int, batch_size: int):
    try:
        df = load_history(stock, period=period)
        if df.empty or "Close" not in df.columns:
            return {"stock": stock, "status": "no_data"}
        closes = df["Close"].astype(float).values
        if len(closes) < (lookback + 100):
            return {"stock": stock, "status": f"insufficient_data({len(closes)})"}
        next_price, rmse, rmse_naive = train_and_predict(
            closes, lookback=lookback, train_ratio=train_ratio, epochs=epochs, batch_size=batch_size
        )
        last_close = float(df["Close"].iloc[-1])
        last_date = pd.to_datetime(df.index[-1])
        pred_date = next_business_day(last_date)
        return {
            "stock": stock,
            "status": "ok",
            "last_date": str(last_date.date()),
            "last_close": last_close,
            "pred_date": str(pred_date.date()),
            "pred_close": next_price,
            "delta": next_price - last_close,
            "abs_pct_delta": abs((next_price - last_close) / last_close) * 100.0,
            "rmse_model": rmse,
            "rmse_naive": rmse_naive,
        }
    except Exception as e:
        return {"stock": stock, "status": f"error: {type(e).__name__}: {e}"}

# ------------- Streamlit UI -------------
st.set_page_config(page_title="Batch Stock LSTM (Demo)", page_icon="📈", layout="wide")
st.title("📈 Batch Stock Price Prediction")
st.caption("**This is not a Trading Tool**. Predicts next-day close with a lightweight LSTM trained on past closing prices.")

with st.sidebar:
    st.header("Batch settings")
    index_name = st.selectbox("Universe", ["sp500", "nasdaq100", "dow30"], index=0)
    uploaded = st.file_uploader("Or upload CSV (one stock per line)", type=["csv"])
    period = st.selectbox("History period", ["2y", "5y", "10y"], index=1)
    lookback = st.slider("Lookback window (days)", 30, 120, 60, 5)
    train_ratio = st.slider("Train split", 0.5, 0.95, 0.8, 0.05)
    epochs = st.slider("Epochs per stock", 3, 20, 6, 1)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)
    max_stocks = st.number_input("Max stocks (0 = all)", min_value=0, value=25, step=5)
    # Stocks list will populate after we load the universe (below)
    run_btn = st.button("Run")

# Load the full universe list
universe = load_universe(index_name, uploaded.read() if uploaded else None)

# Sidebar multiselect for stocks (first 10 preselected)
preselect_n = min(10, len(universe))
stocks_selected = st.sidebar.multiselect(
    "Stocks (searchable)",
    options=universe,
    default=universe[:preselect_n] if preselect_n > 0 else [],
    help="Pick specific stocks to run. If left empty, the app will use the first N based on 'Max stocks'."
)

if stocks_selected:
    stocks_to_run = stocks_selected
else:
    stocks_to_run = universe[:max_stocks] if (max_stocks and max_stocks > 0) else universe

st.write(f"**Universe size:** {len(universe)} stocks  |  **Selected to run:** {len(stocks_to_run)}")
st.divider()

if run_btn:
    if not stocks_to_run:
        st.error("No stocks selected or available. Choose a universe or upload a CSV.")
        st.stop()

    results = []
    prog = st.progress(0, text="Starting…")
    status_box = st.empty()
    start = time.time()

    for i, s in enumerate(stocks_to_run, start=1):
        status_box.write(f"Running {s} ({i}/{len(stocks_to_run)}) …")
        res = predict_one(
            s, period=period, lookback=lookback,
            train_ratio=train_ratio, epochs=epochs, batch_size=batch_size
        )
        results.append(res)
        prog.progress(i / len(stocks_to_run), text=f"{i}/{len(stocks_to_run)} done")

    took = time.time() - start
    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy()

    st.success(f"Done in {took:.1f}s — ok={len(ok)} / total={len(df)}")

    # Show summary & download
    if not ok.empty:
        st.subheader("Top 15 by |%Δ| (pred vs last)")
        leaderboard = ok.sort_values("abs_pct_delta", ascending=False).head(15)[
            ["stock", "last_date", "last_close", "pred_date", "pred_close", "delta", "abs_pct_delta", "rmse_model", "rmse_naive"]
        ]
        st.dataframe(leaderboard, use_container_width=True)

        st.subheader("All results")
        st.dataframe(ok.sort_values("stock"), use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("No successful predictions (all skipped or errored). Try a longer period, smaller lookback, or different universe.")

st.caption("Tips: start small (e.g., 10–25 stocks), compare RMSE vs the naive baseline, and add features beyond closes for anything serious.")
