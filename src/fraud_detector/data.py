"""Data loading and streaming utilities.

Loads the Kaggle Credit Card Fraud dataset and simulates
a real-time transaction stream for the online learning pipeline.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

# Dataset download instructions
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def find_dataset() -> Path | None:
    """Look for creditcard.csv in common locations."""
    search_paths = [
        DATA_DIR / "creditcard.csv",
        Path("data/creditcard.csv"),
        Path("creditcard.csv"),
        Path.home() / "Downloads" / "creditcard.csv",
        Path.home() / ".kaggle" / "datasets" / "creditcard.csv",
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the credit card fraud dataset.

    Args:
        path: Path to creditcard.csv. If None, searches common locations.

    Returns:
        DataFrame sorted by Time with features and target.
    """
    if path is None:
        path = find_dataset()
    if path is None:
        raise FileNotFoundError(
            f"creditcard.csv not found. Download it from:\n"
            f"  {DATASET_URL}\n"
            f"Then place it in the data/ directory."
        )

    path = Path(path)
    logger.info("loading_dataset", path=str(path))
    df = pd.read_csv(path)

    # Sort by time to preserve temporal ordering
    df = df.sort_values("Time").reset_index(drop=True)

    logger.info(
        "dataset_loaded",
        total_transactions=len(df),
        fraud_count=int(df[TARGET_COLUMN].sum()),
        fraud_rate=f"{df[TARGET_COLUMN].mean():.4%}",
    )
    return df


def stream_transactions(
    df: pd.DataFrame,
    speed: float = 0.0,
    shuffle: bool = False,
    inject_drift: bool = False,
    drift_at: float = 0.6,
    seed: int = 42,
) -> Generator[tuple[dict[str, float], int], None, None]:
    """Stream transactions one at a time, simulating real-time arrival.

    Args:
        df: The full dataset.
        speed: Delay between transactions (seconds). 0 = full speed.
        shuffle: Whether to shuffle (breaks temporal order — use for testing).
        inject_drift: If True, artificially inject concept drift by flipping
            fraud labels after `drift_at` fraction of data. This lets us
            test how well the drift detector catches distribution changes.
        drift_at: Fraction of data after which to inject drift.
        seed: Random seed.

    Yields:
        (features_dict, label) tuples one at a time.
    """
    rng = np.random.RandomState(seed)
    indices = df.index.tolist()
    if shuffle:
        rng.shuffle(indices)

    n = len(indices)
    drift_point = int(n * drift_at)

    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        features = {col: float(row[col]) for col in FEATURE_COLUMNS}
        label = int(row[TARGET_COLUMN])

        # Inject artificial concept drift: flip some labels after drift_point
        if inject_drift and i >= drift_point:
            if rng.random() < 0.3:  # 30% of labels flip
                label = 1 - label

        if speed > 0:
            time.sleep(speed)

        yield features, label


def generate_synthetic_transaction(
    rng: np.random.RandomState | None = None,
    fraud: bool = False,
) -> dict[str, float]:
    """Generate a single synthetic transaction for testing.

    Args:
        rng: Random state for reproducibility.
        fraud: Whether to generate a fraudulent transaction.

    Returns:
        Feature dictionary.
    """
    if rng is None:
        rng = np.random.RandomState()

    features = {}
    for i in range(1, 29):
        if fraud:
            # Fraudulent transactions tend to have more extreme V values
            features[f"V{i}"] = rng.normal(0, 3.0)
        else:
            features[f"V{i}"] = rng.normal(0, 1.0)

    # Amount: fraud tends to be higher
    if fraud:
        features["Amount"] = rng.exponential(500)
    else:
        features["Amount"] = rng.exponential(80)

    return features


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return summary statistics about the dataset."""
    return {
        "total_transactions": len(df),
        "fraud_count": int(df[TARGET_COLUMN].sum()),
        "legitimate_count": int((1 - df[TARGET_COLUMN]).sum()),
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "amount_mean": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "amount_max": float(df["Amount"].max()),
        "time_span_hours": float((df["Time"].max() - df["Time"].min()) / 3600),
        "features": FEATURE_COLUMNS,
    }
