# stock_close_predict_lstm

Educational demo. **Not investment advice.**  
Predicts the **next trading day’s close** using a simple **LSTM** trained on historical **closing prices**. Built with **Streamlit**, **TensorFlow (CPU)**, and **yfinance**. Includes batch runs (S&P 500 / Nasdaq-100 / Dow 30) and Docker support.

---

## Features
- **Batch predictions** over a stock universe (S&P 500, Nasdaq-100, Dow 30) or your **own CSV** of stocks.
- **Requests + certifi** for index scraping (avoids macOS SSL issues).
- **YFinance** daily, auto-adjusted OHLC history.
- **Naive baseline** comparison (predict “next = last close”) with RMSE vs model.
- **Leaderboard** of largest absolute % deltas and **CSV download**.
- Dockerfile for one-command containerized runs.

> For learning only — do not trade on this.

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


# Running the App with Docker

This guide covers two options for running the app using **Docker Desktop (GUI)** or **docker-compose**.

---

## **Option 1: Docker Desktop (GUI)**

1. **Build the Image**
   - Open **Docker Desktop**.
   - Go to **Builds** → **Build with Dockerfile**.
   - **Context:** Set it to your **project folder** (the folder containing the `Dockerfile`).
   - Click **Build**.

2. **Run the Container**
   - Go to **Images** → find your newly built image.
   - Click **Run**.
   - Under **Ports**, map:
     - `8501:8501` (default), **or**
     - `8502:8501` if port `8501` is already in use.
   - Start the container.

---

## **Option 2: Using docker-compose (Optional)**

1. **Create `docker-compose.yml` in the project root:**

   ```yaml
   services:
     app:
       build: .
       ports:
         - "8501:8501"
       volumes:
         - .:/app  # optional: hot-reload local code
       environment:
         - PORT=8501
   ```
