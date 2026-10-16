"""
Unit test for model wrapper: instantiates all-MiniLM-L6-v2, tests encoding shape and normalization.
"""

import numpy as np
import pytest
from src.models.loader import EmbeddingModelWrapper


def test_model_wrapper_encoding_and_normalization():
    wrapper = EmbeddingModelWrapper(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        name="all-MiniLM-L6-v2",
        device="cpu",  # Test on CPU for deterministic CI/local run
    )
    profile = wrapper.load()

    assert profile.name == "all-MiniLM-L6-v2"
    assert profile.dimension == 384
    assert profile.parameter_count > 20_000_000
    assert profile.memory_footprint_mb > 0.0

    # Test single query encoding
    queries = ["What are the nutritional benefits of green tea?"]
    embeddings = wrapper.encode_queries(queries, batch_size=1, normalize_embeddings=True)

    assert embeddings.shape == (1, 384)
    # Verify L2 normalization: norm should equal 1.0
    norm = np.linalg.norm(embeddings[0])
    assert pytest.approx(1.0, abs=1e-4) == norm


def test_model_wrapper_prefix_handling():
    wrapper = EmbeddingModelWrapper(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        name="all-MiniLM-L6-v2",
        query_prefix="Represent this sentence for searching relevant passages: ",
        device="cpu",
    )
    wrapper.load()
    queries = ["heart disease risk factors"]
    embeddings = wrapper.encode_queries(queries, batch_size=1)
    assert embeddings.shape == (1, 384)
