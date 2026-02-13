"""Concept drift detection and analysis.

Implements multiple drift detection strategies and provides
analysis tools to understand when and why drift occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from river import drift

logger = structlog.get_logger()


@dataclass
class DriftEvent:
    """Records a single drift detection event."""

    index: int
    detector_name: str
    severity: str  # "warning" or "drift"
    metrics_snapshot: dict[str, float]
    description: str = ""


DRIFT_DETECTORS = {
    "adwin": {
        "description": "Adaptive Windowing — detects change by maintaining a variable-length "
        "window of recent errors and comparing sub-windows statistically.",
        "best_for": "Gradual and sudden drift",
        "factory": lambda: drift.ADWIN(delta=0.002),
    },
    "kswin": {
        "description": "Kolmogorov-Smirnov Windowing — uses the KS statistical test to "
        "compare distributions in a sliding window, detecting distribution shifts.",
        "best_for": "Sudden drift (abrupt distribution changes)",
        "factory": lambda: drift.KSWIN(alpha=0.005, window_size=100, stat_size=30),
    },
    "page_hinkley": {
        "description": "Page-Hinkley test — sequential analysis method that detects "
        "changes in the average of a sequence.",
        "best_for": "Detecting mean shifts in error rate",
        "factory": lambda: drift.PageHinkley(
            min_instances=30,
            delta=0.005,
            threshold=50.0,
        ),
    },
}


class DriftMonitor:
    """Monitors for concept drift using multiple detectors simultaneously.

    Running multiple detectors in parallel provides:
    - Higher confidence when multiple detectors agree
    - Different sensitivity profiles (fast vs slow drift)
    - Better understanding of drift characteristics
    """

    def __init__(self, detector_names: list[str] | None = None):
        if detector_names is None:
            detector_names = ["adwin", "page_hinkley"]

        self.detectors: dict[str, Any] = {}
        for name in detector_names:
            if name not in DRIFT_DETECTORS:
                raise ValueError(
                    f"Unknown detector: {name}. Choose from: {list(DRIFT_DETECTORS)}"
                )
            self.detectors[name] = DRIFT_DETECTORS[name]["factory"]()

        self.events: list[DriftEvent] = []
        self.update_count: int = 0
        self._last_drift_index: dict[str, int] = {name: -1 for name in detector_names}

    def update(self, error: int, metrics_snapshot: dict[str, float] | None = None) -> dict[str, bool]:
        """Feed an error signal (0=correct, 1=error) to all detectors.

        Returns:
            Dict mapping detector name to whether drift was detected.
        """
        self.update_count += 1
        results = {}

        for name, detector in self.detectors.items():
            detector.update(error)
            detected = detector.drift_detected
            results[name] = detected

            if detected:
                # Avoid duplicate events for the same drift
                if self.update_count - self._last_drift_index[name] > 10:
                    event = DriftEvent(
                        index=self.update_count,
                        detector_name=name,
                        severity="drift",
                        metrics_snapshot=metrics_snapshot or {},
                        description=DRIFT_DETECTORS[name]["description"],
                    )
                    self.events.append(event)
                    self._last_drift_index[name] = self.update_count
                    logger.warning(
                        "drift_detected",
                        detector=name,
                        index=self.update_count,
                    )

        return results

    @property
    def any_drift(self) -> bool:
        """Check if any detector fired on the last update."""
        return any(d.drift_detected for d in self.detectors.values())

    @property
    def consensus_drift(self) -> bool:
        """Check if majority of detectors agree on drift."""
        fired = sum(1 for d in self.detectors.values() if d.drift_detected)
        return fired > len(self.detectors) / 2

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of drift detection status."""
        return {
            "updates": self.update_count,
            "total_drift_events": len(self.events),
            "detectors": {
                name: {
                    "description": DRIFT_DETECTORS[name]["description"],
                    "best_for": DRIFT_DETECTORS[name]["best_for"],
                    "events_triggered": sum(
                        1 for e in self.events if e.detector_name == name
                    ),
                }
                for name in self.detectors
            },
            "recent_events": [
                {
                    "index": e.index,
                    "detector": e.detector_name,
                    "severity": e.severity,
                }
                for e in self.events[-10:]  # Last 10 events
            ],
        }

    def reset_detector(self, name: str) -> None:
        """Reset a specific detector (useful after model adaptation)."""
        if name in self.detectors:
            self.detectors[name] = DRIFT_DETECTORS[name]["factory"]()
            logger.info("detector_reset", name=name)

    def reset_all(self) -> None:
        """Reset all detectors."""
        for name in list(self.detectors.keys()):
            self.reset_detector(name)
