"""Tests for data loading and streaming utilities."""

import numpy as np
import pytest

from fraud_detector.data import FEATURE_COLUMNS, generate_synthetic_transaction


class TestSyntheticTransactions:
    def test_generates_all_features(self):
        txn = generate_synthetic_transaction()
        for col in FEATURE_COLUMNS:
            assert col in txn

    def test_legitimate_transaction(self):
        rng = np.random.RandomState(42)
        txn = generate_synthetic_transaction(rng=rng, fraud=False)
        assert txn["Amount"] >= 0

    def test_fraud_transaction(self):
        rng = np.random.RandomState(42)
        txn = generate_synthetic_transaction(rng=rng, fraud=True)
        assert txn["Amount"] >= 0

    def test_fraud_has_higher_amounts_on_average(self):
        rng = np.random.RandomState(42)
        legit_amounts = [
            generate_synthetic_transaction(rng=rng, fraud=False)["Amount"]
            for _ in range(100)
        ]
        fraud_amounts = [
            generate_synthetic_transaction(rng=rng, fraud=True)["Amount"]
            for _ in range(100)
        ]
        assert np.mean(fraud_amounts) > np.mean(legit_amounts)

    def test_reproducible_with_seed(self):
        rng1 = np.random.RandomState(42)
        txn1 = generate_synthetic_transaction(rng=rng1)
        rng2 = np.random.RandomState(42)
        txn2 = generate_synthetic_transaction(rng=rng2)
        assert txn1 == txn2

    def test_29_features(self):
        txn = generate_synthetic_transaction()
        assert len(txn) == 29  # V1-V28 + Amount
