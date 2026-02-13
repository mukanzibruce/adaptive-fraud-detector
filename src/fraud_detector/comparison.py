"""Model comparison engine.

Compares multiple online learning models side-by-side on the same
transaction stream. Also benchmarks against batch ML models to
demonstrate the advantages of online learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from fraud_detector.pipeline import MODEL_REGISTRY, AdaptiveFraudDetector

logger = structlog.get_logger()


class ModelComparison:
    """Run multiple online models side-by-side on the same data stream.

    This is useful for:
    - Finding the best model for your fraud pattern
    - Showing how different models react to concept drift
    - A/B testing models in production
    """

    def __init__(self, model_names: list[str] | None = None, threshold: float = 0.1):
        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        self.detectors: dict[str, AdaptiveFraudDetector] = {}
        for name in model_names:
            self.detectors[name] = AdaptiveFraudDetector(
                model_name=name, threshold=threshold
            )

        self.comparison_history: list[dict[str, Any]] = []
        self._record_interval = 500

    def process_transaction(
        self, features: dict[str, float], label: int
    ) -> dict[str, dict[str, Any]]:
        """Process one transaction through all models.

        Returns results keyed by model name.
        """
        results = {}
        for name, detector in self.detectors.items():
            prediction, learn_info = detector.predict_and_learn(features, label)
            results[name] = {
                "fraud_prob": prediction.fraud_probability,
                "correct": int(prediction.is_fraud_prediction == bool(label)),
                "latency_ms": prediction.latency_ms,
                "drift_detected": learn_info["drift_detected"],
                "rolling_f1": learn_info["rolling_f1"],
            }

        # Record periodic comparison
        n = list(self.detectors.values())[0].transactions_seen
        if n % self._record_interval == 0 and n > 0:
            snapshot = {"n": n}
            for name, detector in self.detectors.items():
                m = detector.metrics.to_dict()
                snapshot[f"{name}_f1"] = m["f1"]
                snapshot[f"{name}_rolling_f1"] = m["rolling_f1"]
                snapshot[f"{name}_roc_auc"] = m["roc_auc"]
                snapshot[f"{name}_drift_events"] = len(detector.drift_events)
            self.comparison_history.append(snapshot)

        return results

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Return models ranked by rolling F1 score."""
        board = []
        for name, detector in self.detectors.items():
            status = detector.get_status()
            board.append(
                {
                    "model": name,
                    "rolling_f1": status["metrics"]["rolling_f1"],
                    "roc_auc": status["metrics"]["roc_auc"],
                    "rolling_precision": status["metrics"]["rolling_precision"],
                    "rolling_recall": status["metrics"]["rolling_recall"],
                    "drift_events": status["drift_events_count"],
                    "transactions": status["transactions_seen"],
                }
            )
        board.sort(key=lambda x: x["rolling_f1"], reverse=True)
        return board

    def get_summary(self) -> dict[str, Any]:
        """Return full comparison summary."""
        return {
            "leaderboard": self.get_leaderboard(),
            "history": self.comparison_history,
            "models": {
                name: detector.get_status() for name, detector in self.detectors.items()
            },
        }


def run_batch_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Run a batch XGBoost/RandomForest baseline for comparison.

    This shows what a traditional batch model achieves on the same data,
    highlighting the tradeoffs of online vs batch learning.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    # Random Forest baseline
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    return {
        "model": "RandomForest (batch)",
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "note": "Static model — does NOT adapt to new patterns after training",
    }
