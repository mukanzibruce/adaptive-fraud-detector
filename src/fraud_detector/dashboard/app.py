"""Streamlit dashboard for the Adaptive Fraud Detector.

Visualizes real-time model performance, concept drift events,
and model comparisons as transactions stream through.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fraud_detector.comparison import ModelComparison
from fraud_detector.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_dataset_stats,
    load_dataset,
    stream_transactions,
)
from fraud_detector.pipeline import AdaptiveFraudDetector


def main():
    st.set_page_config(
        page_title="Adaptive Fraud Detector",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Adaptive Fraud Detector")
    st.markdown(
        "**Real-time online learning** — the model learns from every transaction "
        "and adapts to changing fraud patterns without retraining."
    )

    # Sidebar config
    st.sidebar.header("⚙️ Configuration")
    data_path = st.sidebar.text_input("Dataset path", value="data/creditcard.csv")
    model_name = st.sidebar.selectbox(
        "Online Model",
        ["hoeffding_tree", "logistic", "adaptive_forest"],
        index=0,
    )
    threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.05)
    inject_drift = st.sidebar.checkbox("Inject Artificial Drift", value=False)
    speed_limit = st.sidebar.number_input(
        "Transactions to process", min_value=1000, max_value=300000, value=50000, step=5000
    )
    compare_mode = st.sidebar.checkbox("Compare All Models", value=False)

    if st.sidebar.button("🚀 Start Simulation", type="primary"):
        run_dashboard(
            data_path=data_path,
            model_name=model_name,
            threshold=threshold,
            inject_drift=inject_drift,
            speed_limit=speed_limit,
            compare_mode=compare_mode,
        )


def run_dashboard(
    data_path: str,
    model_name: str,
    threshold: float,
    inject_drift: bool,
    speed_limit: int,
    compare_mode: bool,
):
    """Run the simulation and update the dashboard in real-time."""
    # Load data
    try:
        df = load_dataset(data_path)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    stats = get_dataset_stats(df)
    n = min(speed_limit, len(df))

    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{stats['total_transactions']:,}")
    col2.metric("Fraud Cases", f"{stats['fraud_count']:,}")
    col3.metric("Fraud Rate", f"{stats['fraud_rate']:.3%}")
    col4.metric("Processing", f"{n:,} txns")

    st.divider()

    if compare_mode:
        _run_comparison_mode(df, n, threshold, inject_drift)
    else:
        _run_single_model_mode(df, n, model_name, threshold, inject_drift)


def _run_single_model_mode(
    df: pd.DataFrame,
    n: int,
    model_name: str,
    threshold: float,
    inject_drift: bool,
):
    """Run a single model and show live metrics."""
    detector = AdaptiveFraudDetector(
        model_name=model_name, threshold=threshold, drift_detector_name="adwin"
    )

    # Live metrics row
    st.subheader(f"📈 Live Metrics — {model_name}")
    metric_cols = st.columns(5)
    m_f1 = metric_cols[0].empty()
    m_prec = metric_cols[1].empty()
    m_rec = metric_cols[2].empty()
    m_auc = metric_cols[3].empty()
    m_drift = metric_cols[4].empty()

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    chart_f1 = chart_col1.empty()
    chart_drift = chart_col2.empty()

    # Progress
    progress = st.progress(0, text="Processing transactions...")
    status_text = st.empty()

    # Collect data for charts
    history_n = []
    history_f1 = []
    history_precision = []
    history_recall = []
    drift_points = []

    start_time = time.time()

    for i, (features, label) in enumerate(
        stream_transactions(df, inject_drift=inject_drift)
    ):
        if i >= n:
            break

        pred, info = detector.predict_and_learn(features, label)

        if info["drift_detected"]:
            drift_points.append(i)

        # Update UI every 500 transactions
        if (i + 1) % 500 == 0 or i == n - 1:
            metrics = detector.metrics.to_dict()
            elapsed = time.time() - start_time
            tps = (i + 1) / elapsed

            m_f1.metric("Rolling F1", f"{metrics['rolling_f1']:.4f}")
            m_prec.metric("Rolling Precision", f"{metrics['rolling_precision']:.4f}")
            m_rec.metric("Rolling Recall", f"{metrics['rolling_recall']:.4f}")
            m_auc.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
            m_drift.metric("Drift Events", len(detector.drift_events))

            history_n.append(i + 1)
            history_f1.append(metrics["rolling_f1"])
            history_precision.append(metrics["rolling_precision"])
            history_recall.append(metrics["rolling_recall"])

            # F1 chart
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(x=history_n, y=history_f1, name="F1", line=dict(color="#10b981")))
            fig_f1.add_trace(go.Scatter(x=history_n, y=history_precision, name="Precision", line=dict(color="#3b82f6", dash="dot")))
            fig_f1.add_trace(go.Scatter(x=history_n, y=history_recall, name="Recall", line=dict(color="#f59e0b", dash="dot")))

            for dp in drift_points:
                fig_f1.add_vline(x=dp, line_dash="dash", line_color="red", opacity=0.5)

            fig_f1.update_layout(
                title="Rolling Metrics Over Time",
                xaxis_title="Transactions",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                height=400,
                template="plotly_dark",
            )
            chart_f1.plotly_chart(fig_f1, use_container_width=True)

            # Drift events chart
            if drift_points:
                fig_drift = go.Figure()
                fig_drift.add_trace(go.Scatter(
                    x=list(range(len(drift_points))),
                    y=drift_points,
                    mode="markers+lines",
                    name="Drift Events",
                    marker=dict(color="red", size=10),
                ))
                fig_drift.update_layout(
                    title="Concept Drift Events",
                    xaxis_title="Event #",
                    yaxis_title="Transaction Index",
                    height=400,
                    template="plotly_dark",
                )
                chart_drift.plotly_chart(fig_drift, use_container_width=True)
            else:
                chart_drift.info("No drift events detected yet.")

            progress.progress((i + 1) / n, text=f"Processing... {i + 1:,}/{n:,} ({tps:,.0f} txn/sec)")

    elapsed = time.time() - start_time
    status_text.success(f"✅ Processed {n:,} transactions in {elapsed:.1f}s ({n / elapsed:,.0f} txn/sec)")

    # Final summary
    st.divider()
    st.subheader("📋 Final Results")
    final = detector.get_status()
    results_df = pd.DataFrame([final["metrics"]]).T
    results_df.columns = ["Score"]
    st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

    if detector.drift_events:
        st.subheader("⚠️ Drift Events Log")
        drift_df = pd.DataFrame([
            {"Transaction #": e["transaction_index"], "Detector": e["detector"]}
            for e in detector.drift_events
        ])
        st.dataframe(drift_df, use_container_width=True)


def _run_comparison_mode(
    df: pd.DataFrame,
    n: int,
    threshold: float,
    inject_drift: bool,
):
    """Compare all models side-by-side."""
    comparison = ModelComparison(threshold=threshold)
    model_names = list(comparison.detectors.keys())

    st.subheader("🏁 Model Comparison")

    # Live leaderboard
    leaderboard_placeholder = st.empty()
    chart_placeholder = st.empty()
    progress = st.progress(0, text="Comparing models...")

    history = {name: {"n": [], "f1": []} for name in model_names}
    start_time = time.time()

    for i, (features, label) in enumerate(
        stream_transactions(df, inject_drift=inject_drift)
    ):
        if i >= n:
            break

        comparison.process_transaction(features, label)

        if (i + 1) % 1000 == 0 or i == n - 1:
            board = comparison.get_leaderboard()

            # Update leaderboard
            board_df = pd.DataFrame(board)
            board_df = board_df.rename(columns={
                "model": "Model",
                "rolling_f1": "Rolling F1",
                "roc_auc": "ROC AUC",
                "rolling_precision": "Precision",
                "rolling_recall": "Recall",
                "drift_events": "Drifts",
            })
            leaderboard_placeholder.dataframe(
                board_df.style.format({
                    "Rolling F1": "{:.4f}",
                    "ROC AUC": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                }),
                use_container_width=True,
            )

            # Update chart
            for entry in board:
                history[entry["model"]]["n"].append(i + 1)
                history[entry["model"]]["f1"].append(entry["rolling_f1"])

            fig = go.Figure()
            colors = {"logistic": "#3b82f6", "hoeffding_tree": "#10b981", "adaptive_forest": "#f59e0b"}
            for name in model_names:
                fig.add_trace(go.Scatter(
                    x=history[name]["n"],
                    y=history[name]["f1"],
                    name=name,
                    line=dict(color=colors.get(name, "#888")),
                ))
            fig.update_layout(
                title="Rolling F1 — Model Comparison",
                xaxis_title="Transactions",
                yaxis_title="Rolling F1",
                yaxis_range=[0, 1],
                height=450,
                template="plotly_dark",
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            elapsed = time.time() - start_time
            tps = (i + 1) / elapsed
            progress.progress((i + 1) / n, text=f"Comparing... {i + 1:,}/{n:,} ({tps:,.0f} txn/sec)")

    st.success(f"✅ Comparison complete!")


if __name__ == "__main__":
    main()
