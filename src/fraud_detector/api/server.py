"""FastAPI server for real-time fraud detection.

Provides REST endpoints for:
- Real-time fraud predictions on incoming transactions
- Model status and metrics
- Drift detection status
- Threshold adjustment
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detector.pipeline import AdaptiveFraudDetector

logger = structlog.get_logger()

# Global detector instance
detector: AdaptiveFraudDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the fraud detector on startup."""
    global detector
    detector = AdaptiveFraudDetector(
        model_name="hoeffding_tree",
        threshold=0.5,
        drift_detector_name="adwin",
    )
    logger.info("fraud_detector_ready", model=detector.model_name)
    yield
    logger.info("fraud_detector_shutdown")


app = FastAPI(
    title="Adaptive Fraud Detector API",
    description=(
        "Real-time fraud detection with online learning. "
        "The model learns from every transaction and adapts to concept drift."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class TransactionInput(BaseModel):
    """A single transaction to classify."""

    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(0.0, ge=0.0)


class FeedbackInput(BaseModel):
    """Ground truth label for a previously predicted transaction."""

    transaction: TransactionInput
    is_fraud: bool


class ThresholdInput(BaseModel):
    """New threshold value."""

    threshold: float = Field(..., ge=0.0, le=1.0)


def _get_detector() -> AdaptiveFraudDetector:
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return detector


def _transaction_to_features(txn: TransactionInput) -> dict[str, float]:
    return {f"V{i}": getattr(txn, f"V{i}") for i in range(1, 29)} | {
        "Amount": txn.Amount
    }


@app.post("/predict")
async def predict_fraud(txn: TransactionInput) -> dict[str, Any]:
    """Predict whether a transaction is fraudulent.

    Returns fraud probability, binary prediction, and model metadata.
    """
    d = _get_detector()
    features = _transaction_to_features(txn)
    result = d.predict(features)
    return {
        "transaction_id": result.transaction_id,
        "fraud_probability": result.fraud_probability,
        "is_fraud": result.is_fraud_prediction,
        "threshold": result.threshold,
        "model": result.model_name,
        "latency_ms": result.latency_ms,
        "transactions_processed": result.transactions_seen,
    }


@app.post("/feedback")
async def provide_feedback(feedback: FeedbackInput) -> dict[str, Any]:
    """Provide ground truth label so the model can learn.

    This is the key differentiator: the model improves from every
    labeled transaction without needing retraining.
    """
    d = _get_detector()
    features = _transaction_to_features(feedback.transaction)
    label = int(feedback.is_fraud)
    learn_result = d.learn(features, label)
    return {
        "learned": True,
        "drift_detected": learn_result["drift_detected"],
        "rolling_f1": learn_result["rolling_f1"],
        "transactions_seen": learn_result["transactions_seen"],
    }


@app.get("/status")
async def get_status() -> dict[str, Any]:
    """Get current model status, metrics, and drift info."""
    return _get_detector().get_status()


@app.get("/drift")
async def get_drift_events() -> dict[str, Any]:
    """Get all concept drift events detected so far."""
    d = _get_detector()
    return {
        "drift_events": [
            {
                "index": e["transaction_index"],
                "timestamp": e["timestamp"],
                "detector": e["detector"],
            }
            for e in d.drift_events
        ],
        "total": len(d.drift_events),
    }


@app.post("/threshold")
async def update_threshold(body: ThresholdInput) -> dict[str, Any]:
    """Dynamically adjust the fraud detection threshold."""
    d = _get_detector()
    old = d.threshold
    d.update_threshold(body.threshold)
    return {"old_threshold": old, "new_threshold": body.threshold}


@app.post("/reset")
async def reset_model() -> dict[str, str]:
    """Reset the model (fresh start, useful after severe drift)."""
    d = _get_detector()
    d.reset_model()
    return {"status": "model_reset", "model": d.model_name}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


def main():
    """Entry point for the API server."""
    uvicorn.run(
        "fraud_detector.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
