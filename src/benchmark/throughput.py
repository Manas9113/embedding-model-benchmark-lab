"""
Throughput benchmarking harness across various batch sizes.
Quantifies document embedding speed (documents/second) and batch scaling efficiency.
"""

import time
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch

from src.models.loader import EmbeddingModelWrapper
from src.utils.logger import setup_logger

logger = setup_logger("benchmark_throughput")


@dataclass
class BatchThroughputResult:
    """Throughput performance metrics for a specific batch size."""

    batch_size: int
    total_documents: int
    elapsed_time_seconds: float
    docs_per_second: float
    ms_per_batch: float


@dataclass
class ThroughputReport:
    """Overall multi-batch throughput evaluation summary."""

    model_name: str
    device: str
    results_by_batch_size: Dict[int, BatchThroughputResult]
    peak_throughput_docs_sec: float
    optimal_batch_size: int


class ThroughputBenchmark:
    """Evaluates embedding generation throughput across parameterized batch sizes."""

    def __init__(self, batch_sizes: List[int] | None = None, warmup_batches: int = 2):
        self.batch_sizes = batch_sizes or [8, 16, 32, 64]
        self.warmup_batches = warmup_batches

    def measure(
        self,
        model_wrapper: EmbeddingModelWrapper,
        documents: List[str],
        sample_size: int = 512,
    ) -> ThroughputReport:
        """Measures throughput on a slice of corpus documents across defined batch sizes.

        Args:
            model_wrapper: Loaded instance of EmbeddingModelWrapper.
            documents: List of text passages.
            sample_size: Number of documents to encode per batch size test.

        Returns:
            ThroughputReport with docs/sec and optimal batch size.
        """
        if not documents:
            raise ValueError("documents list cannot be empty")

        test_docs = documents[: min(len(documents), sample_size)]
        is_cuda = model_wrapper.device.startswith("cuda") and torch.cuda.is_available()

        logger.info(
            "Starting throughput benchmark for '%s' across batch sizes %s on %d documents ...",
            model_wrapper.name,
            self.batch_sizes,
            len(test_docs),
        )

        results: Dict[int, BatchThroughputResult] = {}

        for bs in self.batch_sizes:
            # 1. Warm-up with a couple of batches
            warmup_slice = test_docs[: min(len(test_docs), bs * self.warmup_batches)]
            if warmup_slice:
                _ = model_wrapper.encode_corpus(warmup_slice, batch_size=bs, show_progress_bar=False)
                if is_cuda:
                    torch.cuda.synchronize()

            # 2. Benchmark timed pass
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model_wrapper.encode_corpus(test_docs, batch_size=bs, show_progress_bar=False)

            if is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            docs_sec = len(test_docs) / elapsed
            num_batches = int(np.ceil(len(test_docs) / bs))
            ms_batch = (elapsed / num_batches) * 1000.0

            result = BatchThroughputResult(
                batch_size=bs,
                total_documents=len(test_docs),
                elapsed_time_seconds=round(elapsed, 4),
                docs_per_second=round(docs_sec, 2),
                ms_per_batch=round(ms_batch, 2),
            )
            results[bs] = result

            logger.info(
                "Batch size %2d: %7.1f docs/sec | %6.2f ms/batch | Total %.2fs",
                bs,
                docs_sec,
                ms_batch,
                elapsed,
            )

        optimal_bs = max(results.keys(), key=lambda k: results[k].docs_per_second)
        peak_throughput = results[optimal_bs].docs_per_second

        report = ThroughputReport(
            model_name=model_wrapper.name,
            device=model_wrapper.device,
            results_by_batch_size=results,
            peak_throughput_docs_sec=peak_throughput,
            optimal_batch_size=optimal_bs,
        )

        logger.info(
            "Throughput summary for '%s': Peak = %.1f docs/sec at batch_size = %d",
            model_wrapper.name,
            peak_throughput,
            optimal_bs,
        )
        return report
