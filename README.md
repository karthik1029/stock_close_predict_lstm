# stock_close_predict_lstm

Predicts the **next trading day’s close** using a simple **LSTM** trained on historical **closing prices**. Built with **Streamlit**, **TensorFlow (CPU)**, and **yfinance**. Includes batch runs (S&P 500 / Nasdaq-100 / Dow 30).

---

## Features
- **Batch predictions** over a stock universe (S&P 500, Nasdaq-100, Dow 30) or your **own CSV** of stocks.
- **Requests + certifi** for index scraping (avoids macOS SSL issues).
- **YFinance** daily, auto-adjusted OHLC history.
- **Naive baseline** comparison (predict “next = last close”) with RMSE vs model.
- **Leaderboard** of largest absolute % deltas and **CSV download**.
- Dockerfile for one-command containerized runs.


---

## 📦 Project structure
stock_close_predict_lstm/
├── app.py # Streamlit app (batch mode, “stocks” wording)
├── requirements.txt # Pinned deps (TF CPU, pandas, etc.)
├── Dockerfile # Containerized Streamlit app
└── .dockerignore # Optional


---

## 🚀 Quick start (local)

**Python 3.11** recommended.

```bash
# create & activate a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate

# install deps
pip install -r requirements.txt

# run the app
streamlit run app.py
```
Open http://localhost:8501


## 🐳 Run with Docker

### Build and run
```bash
docker build -t stock_close_predict_lstm:latest .
docker run --rm -p 8501:8501 stock_close_predict_lstm:latest
```

Hot-reload (mount your code)
```bash
docker run --rm -p 8501:8501 -v "$PWD":/app stock_close_predict_lstm:latest
```
Apple Silicon: build an x86 image
```bash
docker buildx build --platform linux/amd64 -t stock_close_predict_lstm:amd64 --load .
```

## 🧑‍💻 Using the app

- **Universe:** choose **S&P 500**, **Nasdaq-100**, **Dow 30**, or upload a **CSV**.
- **Stocks:** pick specific stocks in the **sidebar multiselect** (first 10 preselected), or use **Max stocks**.
- **Settings:** set **History period** (2y/5y/10y), **Lookback** (days), **Train split**, **Epochs**, **Batch size**.
- **Run** → You’ll see:
  - **Top 15 by \|%Δ\|** (predicted vs last close)
  - **Full table** (stock, last/pred date, last/pred close, Δ, %Δ, RMSE model vs naive)
  - **CSV download**

### 📄 CSV format (custom universe)

```text
AAPL
MSFT
NVDA
# one symbol per line; dots become dashes for Yahoo (e.g., BRK.B -> BRK-B)
```


## ⚙️ Design notes

- **Sources:** Wikipedia via `requests` + `certifi` → parsed with `pandas.read_html`; `yfinance` fallbacks for S&P/Dow.
- **Data:** Daily **auto-adjusted** closes from **yfinance** (Yahoo Finance).
- **Model:** 2-layer **LSTM** → `Dense(1)`, MinMax scaling, sliding-window supervision.
- **Baseline:** compares test **RMSE** vs “next = last value in window”.

---

## 🧪 Troubleshooting

| Problem | Fix |
| --- | --- |
| SSL error fetching Wikipedia | Mitigated via `requests + certifi`. Ensure `certifi` is installed/updated. |
| Port already in use | Map another port, e.g., `-p 8502:8501` → open `http://localhost:8502`. |
| Docker “blob … input/output error” | Docker Desktop → **Troubleshoot** → **Clean / Purge data**, then rebuild. |
| Behind corporate proxy | Set `HTTP_PROXY` / `HTTPS_PROXY` env vars when running. |
| Batch run slow | Start with **Max stocks = 10–25**; shard externally for large universes. |

---

## ⚠️ Limitations

- Uses **only closing prices**; no volume/factors/regime info.  
- **One model per stock**; no cross-sectional learning.  
- Simple holdout; **no walk-forward CV**.  
- Predictions are **noisy**; not suitable for trading.

---

## 🛡️ Disclaimer

This is an **educational demo** and **not financial advice**.  
Do not use the outputs for financial decisions.
