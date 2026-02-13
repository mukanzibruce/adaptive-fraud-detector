"""Online learning pipeline for adaptive fraud detection.

This module implements the core streaming ML pipeline using River.
The model processes one transaction at a time, learns incrementally,
and never needs to be retrained from scratch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from river import compose, drift, linear_model, metrics, optim, preprocessing, tree

logger = structlog.get_logger()


@dataclass
class PredictionResult:
    """Result of a single fraud prediction."""

    transaction_id: str
    fraud_probability: float
    is_fraud_prediction: bool
    threshold: float
    model_name: str
    latency_ms: float
    drift_detected: bool
    transactions_seen: int


@dataclass
class ModelMetrics:
    """Tracks rolling performance metrics for the online model."""

    accuracy: metrics.Accuracy = field(default_factory=metrics.Accuracy)
    f1: metrics.F1 = field(default_factory=metrics.F1)
    precision: metrics.Precision = field(default_factory=metrics.Precision)
    recall: metrics.Recall = field(default_factory=metrics.Recall)
    roc_auc: metrics.ROCAUC = field(default_factory=metrics.ROCAUC)
    confusion: metrics.ConfusionMatrix = field(default_factory=metrics.ConfusionMatrix)

    # Rolling window state (last 1000 predictions)
    _window_size: int = 1000
    _window_correct: list = field(default_factory=list)
    _window_tp: list = field(default_factory=list)
    _window_fp: list = field(default_factory=list)
    _window_fn: list = field(default_factory=list)

    def update(self, y_true: int, y_pred: int, y_score: float) -> None:
        """Update all metrics with a new observation."""
        self.accuracy.update(y_true, y_pred)
        self.f1.update(y_true, y_pred)
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        self.roc_auc.update(y_true, y_score)
        self.confusion.update(y_true, y_pred)

        # Rolling window tracking
        self._window_correct.append(int(y_pred == y_true))
        self._window_tp.append(int(y_pred == 1 and y_true == 1))
        self._window_fp.append(int(y_pred == 1 and y_true == 0))
        self._window_fn.append(int(y_pred == 0 and y_true == 1))
        if len(self._window_correct) > self._window_size:
            self._window_correct.pop(0)
            self._window_tp.pop(0)
            self._window_fp.pop(0)
            self._window_fn.pop(0)

    def _rolling_precision(self) -> float:
        tp = sum(self._window_tp)
        fp = sum(self._window_fp)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _rolling_recall(self) -> float:
        tp = sum(self._window_tp)
        fn = sum(self._window_fn)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _rolling_f1(self) -> float:
        p = self._rolling_precision()
        r = self._rolling_recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _rolling_accuracy(self) -> float:
        if not self._window_correct:
            return 0.0
        return sum(self._window_correct) / len(self._window_correct)

    def to_dict(self) -> dict[str, float]:
        """Return all metrics as a dictionary."""
        return {
            "accuracy": float(self.accuracy.get()),
            "f1": float(self.f1.get()),
            "precision": float(self.precision.get()),
            "recall": float(self.recall.get()),
            "roc_auc": float(self.roc_auc.get()),
            "rolling_accuracy": self._rolling_accuracy(),
            "rolling_f1": self._rolling_f1(),
            "rolling_precision": self._rolling_precision(),
            "rolling_recall": self._rolling_recall(),
        }


def build_logistic_pipeline() -> compose.Pipeline:
    """Build an online logistic regression pipeline.

    Uses adaptive gradient descent with feature scaling.
    Good baseline — fast, interpretable, handles class imbalance.
    """
    return compose.Pipeline(
        ("scaler", preprocessing.AdaptiveStandardScaler()),
        (
            "classifier",
            linear_model.LogisticRegression(
                optimizer=optim.Adam(lr=0.01),
                l2=0.001,
            ),
        ),
    )


def build_hoeffding_tree_pipeline() -> compose.Pipeline:
    """Build an online Hoeffding Adaptive Tree pipeline.

    Automatically handles concept drift through built-in
    change detection — branches are replaced when drift is detected.
    """
    return compose.Pipeline(
        ("scaler", preprocessing.AdaptiveStandardScaler()),
        (
            "classifier",
            tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=100,
                delta=1e-5,
                leaf_prediction="nba",
                seed=42,
            ),
        ),
    )


def build_arf_pipeline() -> compose.Pipeline:
    """Build an Adaptive Random Forest pipeline.

    State-of-the-art for streaming data — ensemble of Hoeffding trees
    with drift detection on each tree. Self-adapting.
    """
    return compose.Pipeline(
        ("scaler", preprocessing.AdaptiveStandardScaler()),
        (
            "classifier",
            tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=50,
                delta=1e-6,
                leaf_prediction="nba",
                seed=42,
            ),
        ),
    )


MODEL_REGISTRY: dict[str, callable] = {
    "logistic": build_logistic_pipeline,
    "hoeffding_tree": build_hoeffding_tree_pipeline,
    "adaptive_forest": build_arf_pipeline,
}


class AdaptiveFraudDetector:
    """Real-time fraud detector that learns from every transaction.

    Core concept: "test-then-train" — for each transaction:
    1. Predict fraud probability (test)
    2. After the true label arrives, learn from it (train)
    3. Check for concept drift
    4. If drift detected, optionally reset or adapt the model

    This means the model improves with every single transaction
    it processes, without ever needing batch retraining.
    """

    def __init__(
        self,
        model_name: str = "hoeffding_tree",
        threshold: float = 0.1,
        drift_detector_name: str = "adwin",
    ):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY)}")

        self.model_name = model_name
        self.model = MODEL_REGISTRY[model_name]()
        self.threshold = threshold
        self.metrics = ModelMetrics()
        self.drift_detector = self._build_drift_detector(drift_detector_name)
        self.drift_detector_name = drift_detector_name

        # State tracking
        self.transactions_seen: int = 0
        self.frauds_seen: int = 0
        self.drift_events: list[dict[str, Any]] = []
        self.metric_history: list[dict[str, Any]] = []
        self._record_interval: int = 100  # record metrics every N transactions

    def _build_drift_detector(self, name: str):
        """Create a concept drift detector."""
        detectors = {
            "adwin": lambda: drift.ADWIN(delta=0.002),
            "page_hinkley": lambda: drift.PageHinkley(
                min_instances=30, delta=0.005, threshold=50.0
            ),
        }
        if name not in detectors:
            raise ValueError(f"Unknown drift detector: {name}. Choose from: {list(detectors)}")
        return detectors[name]()

    def predict(self, transaction: dict[str, float]) -> PredictionResult:
        """Predict fraud probability for a single transaction.

        This is the PREDICT step of test-then-train.
        Call `learn()` afterward when the true label becomes available.
        """
        start = time.perf_counter()
        proba = self.model.predict_proba_one(transaction)
        fraud_prob = proba.get(1, 0.0)
        is_fraud = fraud_prob >= self.threshold
        latency = (time.perf_counter() - start) * 1000

        return PredictionResult(
            transaction_id=f"txn_{self.transactions_seen}",
            fraud_probability=fraud_prob,
            is_fraud_prediction=is_fraud,
            threshold=self.threshold,
            model_name=self.model_name,
            latency_ms=round(latency, 3),
            drift_detected=False,
            transactions_seen=self.transactions_seen,
        )

    def learn(self, transaction: dict[str, float], y_true: int) -> dict[str, Any]:
        """Learn from a labeled transaction.

        This is the TRAIN step of test-then-train.
        Returns drift info and updated metrics.
        """
        # Predict first (for metrics)
        proba = self.model.predict_proba_one(transaction)
        fraud_prob = proba.get(1, 0.0)
        y_pred = int(fraud_prob >= self.threshold)

        # Update metrics
        self.metrics.update(y_true, y_pred, fraud_prob)

        # Learn from this example
        self.model.learn_one(transaction, y_true)

        # Update state
        self.transactions_seen += 1
        if y_true == 1:
            self.frauds_seen += 1

        # Check for concept drift
        error = int(y_pred != y_true)
        self.drift_detector.update(error)
        drift_detected = self.drift_detector.drift_detected

        if drift_detected:
            drift_event = {
                "transaction_index": self.transactions_seen,
                "timestamp": time.time(),
                "metrics_at_drift": self.metrics.to_dict(),
                "detector": self.drift_detector_name,
            }
            self.drift_events.append(drift_event)
            logger.warning(
                "concept_drift_detected",
                transaction_index=self.transactions_seen,
                rolling_f1=self.metrics._rolling_f1(),
            )

        # Record periodic metrics
        if self.transactions_seen % self._record_interval == 0:
            self.metric_history.append(
                {
                    "n": self.transactions_seen,
                    **self.metrics.to_dict(),
                    "fraud_rate": self.frauds_seen / self.transactions_seen,
                }
            )

        return {
            "drift_detected": drift_detected,
            "transactions_seen": self.transactions_seen,
            "rolling_f1": self.metrics._rolling_f1(),
            "rolling_accuracy": self.metrics._rolling_accuracy(),
        }

    def predict_and_learn(
        self, transaction: dict[str, float], y_true: int
    ) -> tuple[PredictionResult, dict[str, Any]]:
        """Convenience method: predict then learn in one call.

        This is the full test-then-train loop for a single transaction.
        """
        prediction = self.predict(transaction)
        learn_result = self.learn(transaction, y_true)
        prediction.drift_detected = learn_result["drift_detected"]
        return prediction, learn_result

    def get_status(self) -> dict[str, Any]:
        """Return current model status and metrics."""
        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "transactions_seen": self.transactions_seen,
            "frauds_seen": self.frauds_seen,
            "fraud_rate": (
                self.frauds_seen / self.transactions_seen if self.transactions_seen > 0 else 0.0
            ),
            "drift_events_count": len(self.drift_events),
            "drift_detector": self.drift_detector_name,
            "metrics": self.metrics.to_dict(),
        }

    def reset_model(self) -> None:
        """Reset the model (useful after severe drift)."""
        logger.info("model_reset", model=self.model_name, after_n=self.transactions_seen)
        self.model = MODEL_REGISTRY[self.model_name]()

    def update_threshold(self, new_threshold: float) -> None:
        """Dynamically adjust the fraud detection threshold."""
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        old = self.threshold
        self.threshold = new_threshold
        logger.info("threshold_updated", old=old, new=new_threshold)
