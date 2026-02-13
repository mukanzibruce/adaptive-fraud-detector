"""Tests for the drift detection module."""

import pytest

from fraud_detector.drift_detection import DRIFT_DETECTORS, DriftMonitor


class TestDriftMonitor:
    def test_create_default(self):
        monitor = DriftMonitor()
        assert "adwin" in monitor.detectors
        assert "page_hinkley" in monitor.detectors

    def test_create_custom(self):
        monitor = DriftMonitor(["adwin", "page_hinkley"])
        assert len(monitor.detectors) == 2

    def test_invalid_detector_raises(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            DriftMonitor(["not_real"])

    def test_update_increments_count(self):
        monitor = DriftMonitor(["adwin"])
        monitor.update(0)
        monitor.update(1)
        assert monitor.update_count == 2

    def test_no_drift_on_clean_data(self):
        monitor = DriftMonitor(["adwin"])
        for _ in range(100):
           monitor.update(0)
        assert len(monitor.events) == 0

    def test_drift_on_error_spike(self):
        """ADWIN should detect drift when error rate changes sharply."""
        monitor = DriftMonitor(["adwin"])

        # Phase 1: low error rate
        for _ in range(500):
            monitor.update(0)  # All correct

        # Phase 2: sudden high error rate
        detected = False
        for _ in range(500):
            results = monitor.update(1)  # All wrong
            if results.get("adwin", False):
                detected = True
                break

        assert detected, "ADWIN should detect drift on sudden error spike"

    def test_consensus_drift(self):
        monitor = DriftMonitor(["adwin", "page_hinkley"])
        # Feed clean data
        for _ in range(100):
            monitor.update(0)
        # No consensus drift on clean data
        assert not monitor.consensus_drift

    def test_get_summary(self):
        monitor = DriftMonitor(["adwin"])
        for _ in range(10):
            monitor.update(0)
        summary = monitor.get_summary()
        assert "updates" in summary
        assert "detectors" in summary
        assert summary["updates"] == 10

    def test_reset_detector(self):
        monitor = DriftMonitor(["adwin"])
        for _ in range(50):
            monitor.update(1)
        monitor.reset_detector("adwin")
        # After reset, no immediate drift
        results = monitor.update(0)
        assert not results.get("adwin", False)

    def test_reset_all(self):
        monitor = DriftMonitor(["adwin", "page_hinkley"])
        monitor.reset_all()
        assert monitor.update_count == 0  # count is NOT reset
        # But detectors are fresh


class TestDriftDetectorRegistry:
    def test_all_detectors_exist(self):
        expected = {"adwin", "kswin", "page_hinkley"}
        assert expected == set(DRIFT_DETECTORS.keys())

    def test_all_have_factory(self):
        for name, info in DRIFT_DETECTORS.items():
            detector = info["factory"]()
            assert hasattr(detector, "update")
            assert hasattr(detector, "drift_detected")

    def test_all_have_description(self):
        for name, info in DRIFT_DETECTORS.items():
            assert len(info["description"]) > 10
            assert len(info["best_for"]) > 5
