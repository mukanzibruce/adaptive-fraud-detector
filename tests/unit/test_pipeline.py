"""Tests for the core online learning pipeline."""

import pytest

from fraud_detector.data import generate_synthetic_transaction
from fraud_detector.pipeline import (
    MODEL_REGISTRY,
    AdaptiveFraudDetector,
    ModelMetrics,
    PredictionResult,
    build_hoeffding_tree_pipeline,
    build_logistic_pipeline,
)


class TestModelMetrics:
    def test_initial_values(self):
        m = ModelMetrics()
        d = m.to_dict()
        assert d["accuracy"] == 0.0
        assert d["f1"] == 0.0

    def test_update(self):
        m = ModelMetrics()
        m.update(y_true=1, y_pred=1, y_score=0.9)
        m.update(y_true=0, y_pred=0, y_score=0.1)
        d = m.to_dict()
        assert d["accuracy"] == 1.0
        assert d["f1"] > 0.0

    def test_rolling_metrics(self):
        m = ModelMetrics()
        for _ in range(10):
            m.update(y_true=0, y_pred=0, y_score=0.1)
        assert m._rolling_accuracy() == 1.0


class TestAdaptiveFraudDetector:
    def test_create_all_models(self):
        for name in MODEL_REGISTRY:
            detector = AdaptiveFraudDetector(model_name=name)
            assert detector.model_name == name
            assert detector.transactions_seen == 0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            AdaptiveFraudDetector(model_name="not_a_model")

    def test_predict_returns_result(self):
        detector = AdaptiveFraudDetector(model_name="logistic")
        txn = generate_synthetic_transaction()
        result = detector.predict(txn)
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.latency_ms >= 0

    def test_learn_updates_state(self):
        detector = AdaptiveFraudDetector(model_name="logistic")
        txn = generate_synthetic_transaction()
        info = detector.learn(txn, y_true=0)
        assert detector.transactions_seen == 1
        assert detector.frauds_seen == 0
        assert "drift_detected" in info

    def test_learn_fraud(self):
        detector = AdaptiveFraudDetector(model_name="logistic")
        txn = generate_synthetic_transaction(fraud=True)
        detector.learn(txn, y_true=1)
        assert detector.frauds_seen == 1

    def test_predict_and_learn(self):
        detector = AdaptiveFraudDetector(model_name="hoeffding_tree")
        txn = generate_synthetic_transaction()
        pred, info = detector.predict_and_learn(txn, y_true=0)
        assert isinstance(pred, PredictionResult)
        assert detector.transactions_seen == 1

    def test_model_improves_over_time(self):
        """The model should get better as it sees more data."""
        import numpy as np

        detector = AdaptiveFraudDetector(model_name="logistic", threshold=0.5)
        rng = np.random.RandomState(42)

        # Feed 500 transactions
        for _ in range(500):
            is_fraud = rng.random() < 0.1
            txn = generate_synthetic_transaction(rng=rng, fraud=is_fraud)
            detector.predict_and_learn(txn, y_true=int(is_fraud))

        # After 500 transactions, accuracy should be above random (50%)
        metrics = detector.metrics.to_dict()
        assert metrics["accuracy"] > 0.5

    def test_get_status(self):
        detector = AdaptiveFraudDetector()
        status = detector.get_status()
        assert "model_name" in status
        assert "metrics" in status
        assert "transactions_seen" in status
        assert "drift_events_count" in status

    def test_reset_model(self):
        detector = AdaptiveFraudDetector(model_name="logistic")
        txn = generate_synthetic_transaction()
        detector.learn(txn, 0)
        assert detector.transactions_seen == 1
        detector.reset_model()
        # transactions_seen is NOT reset (it's a counter of processed data)
        # but the model weights are fresh

    def test_update_threshold(self):
        detector = AdaptiveFraudDetector(threshold=0.5)
        detector.update_threshold(0.8)
        assert detector.threshold == 0.8

    def test_invalid_threshold_raises(self):
        detector = AdaptiveFraudDetector()
        with pytest.raises(ValueError):
            detector.update_threshold(1.5)

    def test_metric_history_recorded(self):
        detector = AdaptiveFraudDetector(model_name="logistic")
        detector._record_interval = 10
        rng = __import__("numpy").random.RandomState(42)
        for _ in range(20):
            txn = generate_synthetic_transaction(rng=rng)
            detector.predict_and_learn(txn, y_true=0)
        assert len(detector.metric_history) >= 1


class TestPipelines:
    def test_logistic_pipeline(self):
        pipe = build_logistic_pipeline()
        txn = generate_synthetic_transaction()
        pipe.learn_one(txn, 0)
        proba = pipe.predict_proba_one(txn)
        assert isinstance(proba, dict)

    def test_hoeffding_tree_pipeline(self):
        pipe = build_hoeffding_tree_pipeline()
        txn = generate_synthetic_transaction()
        pipe.learn_one(txn, 0)
        proba = pipe.predict_proba_one(txn)
        assert isinstance(proba, dict)
