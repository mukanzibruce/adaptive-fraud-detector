"""Simulation runner — streams the full dataset through the detector.

This is the main experiment script. It:
1. Loads the credit card fraud dataset
2. Streams transactions one-by-one through the online model
3. Tracks metrics, drift events, and model performance over time
4. Compares multiple online models side-by-side
5. Benchmarks against a batch ML baseline
6. Saves results for the dashboard to visualize
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from fraud_detector.comparison import ModelComparison, run_batch_baseline
from fraud_detector.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_dataset_stats,
    load_dataset,
    stream_transactions,
)
from fraud_detector.pipeline import AdaptiveFraudDetector

logger = structlog.get_logger()


def run_single_model(
    df: pd.DataFrame,
    model_name: str = "hoeffding_tree",
    threshold: float = 0.1,
    inject_drift: bool = False,
) -> dict:
    """Run a single model through the entire dataset."""
    detector = AdaptiveFraudDetector(
        model_name=model_name,
        threshold=threshold,
        drift_detector_name="adwin",
    )

    n = len(df)
    start = time.time()
    predictions = []

    print(f"\n🚀 Running {model_name} on {n:,} transactions...")
    print(f"   Threshold: {threshold}")
    print(f"   Drift injection: {'ON' if inject_drift else 'OFF'}")
    print()

    for i, (features, label) in enumerate(
        stream_transactions(df, inject_drift=inject_drift)
    ):
        pred, info = detector.predict_and_learn(features, label)

        if info["drift_detected"]:
            print(f"   ⚠️  Drift detected at transaction {i:,}! (rolling F1: {info['rolling_f1']:.4f})")

        if (i + 1) % 50_000 == 0 or i == n - 1:
            status = detector.get_status()
            m = status["metrics"]
            print(
                f"   [{i + 1:>7,}/{n:,}] "
                f"F1={m['rolling_f1']:.4f}  "
                f"Prec={m['rolling_precision']:.4f}  "
                f"Rec={m['rolling_recall']:.4f}  "
                f"AUC={m['roc_auc']:.4f}  "
                f"Drifts={status['drift_events_count']}"
            )

    elapsed = time.time() - start
    final_status = detector.get_status()

    print(f"\n✅ Done in {elapsed:.1f}s ({n / elapsed:,.0f} txn/sec)")
    print(f"   Final rolling F1: {final_status['metrics']['rolling_f1']:.4f}")
    print(f"   Total drift events: {final_status['drift_events_count']}")

    return {
        "model": model_name,
        "elapsed_seconds": elapsed,
        "throughput_per_sec": n / elapsed,
        "status": final_status,
        "metric_history": detector.metric_history,
        "drift_events": detector.drift_events,
    }


def run_comparison(df: pd.DataFrame, inject_drift: bool = False) -> dict:
    """Compare all online models side-by-side."""
    comparison = ModelComparison(threshold=0.5)
    n = len(df)
    start = time.time()

    print(f"\n🏁 Comparing models on {n:,} transactions...")
    print(f"   Models: {list(comparison.detectors.keys())}")
    print()

    for i, (features, label) in enumerate(
        stream_transactions(df, inject_drift=inject_drift)
    ):
        comparison.process_transaction(features, label)

        if (i + 1) % 50_000 == 0 or i == n - 1:
            board = comparison.get_leaderboard()
            leader = board[0]
            print(
                f"   [{i + 1:>7,}/{n:,}] "
                f"Leader: {leader['model']} "
                f"(F1={leader['rolling_f1']:.4f}, AUC={leader['roc_auc']:.4f})"
            )

    elapsed = time.time() - start
    summary = comparison.get_summary()

    print(f"\n📊 Final Leaderboard ({elapsed:.1f}s):")
    print(f"   {'Model':<20} {'Rolling F1':>10} {'ROC AUC':>10} {'Drifts':>8}")
    print(f"   {'─' * 50}")
    for entry in summary["leaderboard"]:
        print(
            f"   {entry['model']:<20} "
            f"{entry['rolling_f1']:>10.4f} "
            f"{entry['roc_auc']:>10.4f} "
            f"{entry['drift_events']:>8}"
        )

    return {
        "elapsed_seconds": elapsed,
        "comparison": summary,
    }


def run_batch_comparison(df: pd.DataFrame) -> dict:
    """Compare online model against batch baseline."""
    print("\n📦 Running batch baseline (RandomForest)...")

    # Split: first 80% train, last 20% test
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    batch_result = run_batch_baseline(X_train, y_train, X_test, y_test)

    print(f"   Batch RF — F1: {batch_result['f1']:.4f}, AUC: {batch_result['roc_auc']:.4f}")
    print("   ⚠️  Note: Batch model is STATIC — it cannot adapt to new fraud patterns")

    return batch_result


def main():
    parser = argparse.ArgumentParser(description="Run fraud detection simulation")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to creditcard.csv"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hoeffding_tree",
        choices=["logistic", "hoeffding_tree", "adaptive_forest", "compare"],
        help="Model to use, or 'compare' for all models",
    )
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Inject artificial concept drift at 60%% of data",
    )
    parser.add_argument(
        "--batch-baseline",
        action="store_true",
        help="Also run batch ML baseline for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    # Load data
    df = load_dataset(args.data)
    stats = get_dataset_stats(df)
    print(f"\n📂 Dataset: {stats['total_transactions']:,} transactions")
    print(f"   Fraud: {stats['fraud_count']:,} ({stats['fraud_rate']:.4%})")
    print(f"   Time span: {stats['time_span_hours']:.1f} hours")

    results = {"dataset": stats}

    # Run experiment
    if args.model == "compare":
        results["comparison"] = run_comparison(df, inject_drift=args.inject_drift)
    else:
        results["single_model"] = run_single_model(
            df, model_name=args.model, inject_drift=args.inject_drift
        )

    # Optional batch baseline
    if args.batch_baseline:
        results["batch_baseline"] = run_batch_comparison(df)

    # Save results
    output_path = Path(args.output)

    # Convert non-serializable values
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    output_path.write_text(json.dumps(results, indent=2, default=clean))
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()
