"""
Unit tests validating dense vector indexing and cosine retrieval ranking logic.
"""

import numpy as np
import pytest
from src.evaluation.retriever import DenseRetriever


def test_retriever_ranking_exact_similarity():
    doc_ids = ["doc_x", "doc_y", "doc_z"]
    # Unit vectors
    doc_embeddings = np.array([
        [1.0, 0.0],   # doc_x: 0 degrees
        [0.0, 1.0],   # doc_y: 90 degrees
        [0.7071, 0.7071],  # doc_z: 45 degrees
    ], dtype=np.float32)

    retriever = DenseRetriever(doc_ids=doc_ids, doc_embeddings=doc_embeddings)

    # Query aligned with doc_x
    query_embeddings = np.array([
        [1.0, 0.0],
    ], dtype=np.float32)

    top_ids = retriever.retrieve_top_k(query_embeddings, k=3)
    assert len(top_ids) == 1
    # Expected ordering: doc_x (dot=1.0), doc_z (dot=0.7071), doc_y (dot=0.0)
    assert top_ids[0] == ["doc_x", "doc_z", "doc_y"]


def test_retriever_evaluation_macro_averages():
    doc_ids = ["doc_a", "doc_b", "doc_c"]
    doc_embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ], dtype=np.float32)

    retriever = DenseRetriever(doc_ids=doc_ids, doc_embeddings=doc_embeddings)

    query_ids = ["q1"]
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    # Gold qrels: doc_a is highly relevant (2), doc_z is relevant (1)
    qrels = {
        "q1": {"doc_a": 2, "doc_c": 1}
    }

    report = retriever.evaluate(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        qrels=qrels,
        top_k_values=[1, 3],
        model_name="test_model",
    )

    assert report.model_name == "test_model"
    assert report.num_queries_evaluated == 1
    assert report.metrics["MRR@1"] == 1.0
    assert report.metrics["Recall@1"] == 0.5  # 1 out of 2 relevant docs
    assert report.metrics["Recall@3"] == 1.0  # both relevant docs retrieved
