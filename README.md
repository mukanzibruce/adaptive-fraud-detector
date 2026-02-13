<div align="center">

# 🛡️ Adaptive Fraud Detector

**Real-time fraud detection that learns from every transaction — no retraining needed.**

Online learning · Concept drift detection · Auto-adaptation · Live dashboard

[![CI](https://github.com/mukanzibruce/adaptive-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/mukanzibruce/adaptive-fraud-detector/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![River ML](https://img.shields.io/badge/built%20with-River%20ML-blue)](https://riverml.xyz)

</div>

---

## The Problem

Traditional fraud detection models are trained on historical data and deployed as static systems. When fraud patterns evolve — and they always do — these models degrade silently until someone notices and manually retrains them.

**This project solves that.** The model processes transactions one at a time, learns from each one, detects when fraud patterns change (concept drift), and automatically adapts. Zero downtime. Zero manual retraining.

## How It Works

```
Transaction → Predict → Return Result → Learn from Truth → Detect Drift → Adapt
     ↑                                                                      |
     └──────────────────────── continuous loop ─────────────────────────────┘
```

The core pattern is **test-then-train**:

1. **Predict** — score the transaction (fraud probability)
2. **Learn** — when the true label arrives, update the model incrementally
3. **Monitor** — check if the error pattern has changed (concept drift)
4. **Adapt** — if drift detected, the model self-adjusts

This means the model after processing 100K transactions is fundamentally different (and better) than the model after 1K transactions.

## Features

| Feature | Description |
|---------|-------------|
| 🔄 **Online Learning** | Model updates with every transaction — no batch retraining |
| 📊 **3 Model Types** | Logistic Regression, Hoeffding Adaptive Tree, Adaptive Random Forest |
| ⚠️ **Concept Drift Detection** | ADWIN, DDM, EDDM, Page-Hinkley — multiple detectors in parallel |
| 🏁 **Model Comparison** | Run all models side-by-side on the same stream |
| 📈 **Live Dashboard** | Streamlit dashboard with real-time metric charts |
| 🌐 **REST API** | FastAPI server for real-time predictions |
| 📦 **Batch Baseline** | Compare against traditional RandomForest to show online learning advantages |
| 💉 **Drift Injection** | Artificially inject concept drift to test detector sensitivity |
| 🧪 **Test Suite** | Comprehensive unit tests with pytest |

## Quick Start

### 1. Install

```bash
git clone https://github.com/mukanzibruce/adaptive-fraud-detector.git
cd adaptive-fraud-detector
pip install -e ".[dev]"
```

### 2. Get the Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place `creditcard.csv` in the `data/` directory.

### 3. Run the Simulation

```bash
# Single model — full stream
python -m fraud_detector.simulate --data data/creditcard.csv --model hoeffding_tree

# Compare all models
python -m fraud_detector.simulate --data data/creditcard.csv --model compare

# With artificial concept drift injection
python -m fraud_detector.simulate --data data/creditcard.csv --model compare --inject-drift

# With batch baseline comparison
python -m fraud_detector.simulate --data data/creditcard.csv --model compare --batch-baseline
```

### 4. Launch the Dashboard

```bash
streamlit run src/fraud_detector/dashboard/app.py
```

### 5. Start the API Server

```bash
python -m fraud_detector.api.server
# or
fraud-api
```

Then send transactions:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.35, "V2": -0.07, "V3": 2.53, "Amount": 149.62}'
```

## Architecture

```
src/fraud_detector/
├── pipeline.py          # Core online learning pipeline (test-then-train)
├── data.py              # Dataset loading and transaction streaming
├── drift_detection.py   # Multi-detector concept drift monitoring
├── comparison.py        # Side-by-side model comparison engine
├── simulate.py          # CLI simulation runner
├── api/
│   └── server.py        # FastAPI REST endpoint
└── dashboard/
    └── app.py           # Streamlit live visualization
```

## Online Learning Models

### Logistic Regression (Online)
- Updates weights with each transaction via Adam optimizer
- Fast, interpretable, good baseline
- Best for: linearly separable fraud patterns

### Hoeffding Adaptive Tree
- Decision tree that grows incrementally from streaming data
- Built-in ADWIN change detection replaces branches on drift
- Best for: non-linear patterns with concept drift

### Adaptive Random Forest
- Ensemble of adaptive Hoeffding trees
- Each tree has its own drift detector
- Best for: complex fraud patterns, highest accuracy

## Concept Drift Detection

Fraud patterns change over time. Drift detectors monitor the model's error rate and alert when the distribution shifts:

| Detector | Method | Best For |
|----------|--------|----------|
| **ADWIN** | Adaptive windowing with statistical testing | Gradual + sudden drift |
| **DDM** | Error rate + standard deviation monitoring | Sudden drift |
| **EDDM** | Distance between classification errors | Gradual drift |
| **Page-Hinkley** | Sequential mean-shift detection | Mean shifts |

Multiple detectors run simultaneously — when they agree, confidence is high.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Score a transaction |
| POST | `/feedback` | Provide true label (model learns) |
| GET | `/status` | Model metrics and state |
| GET | `/drift` | Drift events log |
| POST | `/threshold` | Adjust fraud threshold |
| POST | `/reset` | Reset model weights |
| GET | `/health` | Health check |

## Dataset

Uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset:
- **284,807** transactions over 2 days
- **492** fraud cases (0.172% — highly imbalanced)
- 28 PCA-transformed features (V1-V28) + Amount
- Real European cardholder transactions from September 2013

## Tech Stack

- **[River](https://riverml.xyz)** — Online machine learning in Python
- **[FastAPI](https://fastapi.tiangolo.com)** — REST API
- **[Streamlit](https://streamlit.io)** — Live dashboard
- **[Plotly](https://plotly.com)** — Interactive charts
- **[scikit-learn](https://scikit-learn.org)** — Batch baseline comparison
- **[Pydantic](https://docs.pydantic.dev)** — Data validation

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

## Contributing

Contributions welcome! See areas for improvement:

- Additional online learning models (e.g., online gradient boosting)
- More sophisticated drift response strategies
- WebSocket streaming for the dashboard
- Docker deployment
- Prometheus metrics export

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">
  <a href="https://github.com/mukanzibruce/adaptive-fraud-detector">⭐ Star this repo</a> if you find it useful!
</div>
