"""
Unit tests validating mathematical correctness of Information Retrieval metrics.
Compares implementation outputs against analytically calculated ground truth values.
"""

import math
import pytest
from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr_at_k,
    dcg_at_k,
    ndcg_at_k,
)


def test_recall_at_k_exact_values():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2", "d4", "d6"}  # Total 3 relevant items

    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2.0 / 3.0)
    # k larger than retrieved list
    assert recall_at_k(retrieved, relevant, k=10) == pytest.approx(2.0 / 3.0)


def test_recall_empty_or_zero():
    assert recall_at_k(["d1", "d2"], [], k=5) == 0.0
    assert recall_at_k([], {"d1"}, k=5) == 0.0
    assert recall_at_k(["d1"], {"d1"}, k=0) == 0.0


def test_precision_at_k_exact_values():
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2", "d4", "d6"}

    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)  # 1/2
    assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(0.5)  # 2/4
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)  # 2/5


def test_mrr_at_k_first_hit():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d2", "d4"}

    # First hit is at rank 2
    assert mrr_at_k(retrieved, relevant, k=1) == pytest.approx(0.0)
    assert mrr_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)  # 1/2
    assert mrr_at_k(retrieved, relevant, k=4) == pytest.approx(0.5)  # still 1/2

    # First hit at rank 1
    assert mrr_at_k(["d2", "d1"], relevant, k=1) == pytest.approx(1.0)

    # No hits in top K
    assert mrr_at_k(["d9", "d8"], relevant, k=2) == pytest.approx(0.0)


def test_ndcg_at_k_hand_calculated():
    # Setup test fixture
    # Gold scores: d1:1, d2:2, d4:2
    relevance_scores = {"d1": 1, "d2": 2, "d4": 2}
    retrieved = ["d1", "d2", "d3", "d5"]  # d3 and d5 have 0 relevance

    # Hand calculation:
    # Actual DCG@4:
    # rank 1 (d1): (2^1 - 1) / log2(2) = 1.0 / 1.0 = 1.0
    # rank 2 (d2): (2^2 - 1) / log2(3) = 3.0 / 1.5849625 = 1.892789
    # rank 3 (d3): 0.0
    # rank 4 (d5): 0.0
    # Actual DCG = 2.892789
    expected_dcg = 1.0 + (3.0 / math.log2(3))
    assert dcg_at_k(retrieved, relevance_scores, k=4) == pytest.approx(expected_dcg, rel=1e-5)

    # Ideal DCG@4:
    # ideal scores: [2, 2, 1]
    # rank 1: (2^2 - 1) / log2(2) = 3.0
    # rank 2: (2^2 - 1) / log2(3) = 3.0 / log2(3) = 1.892789
    # rank 3: (2^1 - 1) / log2(4) = 1.0 / 2.0 = 0.5
    # IDCG = 3.0 + (3.0 / log2(3)) + 0.5 = 5.392789
    expected_idcg = 3.0 + (3.0 / math.log2(3)) + 0.5
    expected_ndcg = expected_dcg / expected_idcg

    assert ndcg_at_k(retrieved, relevance_scores, k=4) == pytest.approx(expected_ndcg, rel=1e-5)


def test_ndcg_perfect_ranking_equals_one():
    relevance_scores = {"d1": 3, "d2": 2, "d3": 1}
    perfect_retrieved = ["d1", "d2", "d3"]

    assert ndcg_at_k(perfect_retrieved, relevance_scores, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(perfect_retrieved, relevance_scores, k=1) == pytest.approx(1.0)


def test_ndcg_zero_relevant_items():
    assert ndcg_at_k(["d1", "d2"], {}, k=5) == 0.0
    assert ndcg_at_k(["d1", "d2"], {"d3": 1}, k=2) == 0.0
