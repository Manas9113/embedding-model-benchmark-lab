"""
Unified model loading and profiling wrapper for text embedding architectures.
Encapsulates SentenceTransformer models, handles device mapping (CPU/CUDA),
manages instruction prefixes, and extracts model architectural metadata.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger("model_loader")


@dataclass
class ModelProfile:
    """Stores architectural and system characteristics of a loaded embedding model."""

    model_id: str
    name: str
    architecture: str
    dimension: int
    max_seq_length: int
    parameter_count: int
    memory_footprint_mb: float
    device: str
    load_time_seconds: float
    query_prefix: str
    doc_prefix: str


class EmbeddingModelWrapper:
    """Wrapper around SentenceTransformer with standardized inference and profiling APIs."""

    def __init__(
        self,
        model_id: str,
        name: Optional[str] = None,
        architecture: str = "Transformer Bi-Encoder",
        device: str = "auto",
        max_seq_length: int = 256,
        query_prefix: str = "",
        doc_prefix: str = "",
    ):
        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.architecture = architecture
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

        # Determine target device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[SentenceTransformer] = None
        self.profile: Optional[ModelProfile] = None

    def load(self) -> ModelProfile:
        """Loads model into memory/device, timing the load duration and profiling parameters."""
        logger.info("Loading model '%s' on %s ...", self.model_id, self.device)
        start_time = time.perf_counter()

        # Load SentenceTransformer
        self.model = SentenceTransformer(self.model_id, device=self.device)
        self.model.max_seq_length = self.max_seq_length

        load_time = time.perf_counter() - start_time

        # Calculate parameter count and footprint
        param_count = sum(p.numel() for p in self.model.parameters())
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        memory_mb = (param_bytes + buffer_bytes) / (1024 * 1024)

        # Get actual output dimension
        sample_vec = self.model.encode("Sanity check dimension", normalize_embeddings=True)
        dimension = int(sample_vec.shape[0])

        self.profile = ModelProfile(
            model_id=self.model_id,
            name=self.name,
            architecture=self.architecture,
            dimension=dimension,
            max_seq_length=self.max_seq_length,
            parameter_count=param_count,
            memory_footprint_mb=round(memory_mb, 2),
            device=self.device,
            load_time_seconds=round(load_time, 4),
            query_prefix=self.query_prefix,
            doc_prefix=self.doc_prefix,
        )

        logger.info(
            "Model '%s' loaded in %.3fs | Dim: %d | Params: %.1fM | Memory: %.2fMB | Device: %s",
            self.name,
            load_time,
            dimension,
            param_count / 1e6,
            memory_mb,
            self.device,
        )
        return self.profile

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        """Encodes queries, automatically applying model-specific query prefixes if configured."""
        if self.model is None:
            raise RuntimeError("Model must be loaded before calling encode_queries()")

        # Apply prefix if needed (e.g., BGE or E5 instruction prefixes)
        if self.query_prefix:
            prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        else:
            prefixed_queries = queries

        return self.model.encode(
            prefixed_queries,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

    def encode_corpus(
        self,
        documents: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        """Encodes corpus passages, applying document prefixes if configured."""
        if self.model is None:
            raise RuntimeError("Model must be loaded before calling encode_corpus()")

        if self.doc_prefix:
            prefixed_docs = [f"{self.doc_prefix}{d}" for d in documents]
        else:
            prefixed_docs = documents

        return self.model.encode(
            prefixed_docs,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
