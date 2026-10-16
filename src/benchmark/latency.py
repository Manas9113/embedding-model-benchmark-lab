"""
High-precision latency benchmarking harness for text embedding models.
Measures single-query (batch_size=1) latency distributions (mean, std, p50, p95, p99),
explicitly handling warm-up iterations and GPU kernel synchronization.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch

from src.models.loader import EmbeddingModelWrapper
from src.utils.logger import setup_logger

logger = setup_logger("benchmark_latency")


@dataclass
class LatencyReport:
    """Statistical summary of query latency measurements."""

    model_name: str
    device: str
    num_trials: int
    warmup_runs: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    raw_latencies_ms: List[float]


class LatencyBenchmark:
    """Profiles query encoding latency with rigorous warm-up and percentile statistics."""

    def __init__(self, warmup_runs: int = 5, num_trials: int = 50):
        self.warmup_runs = warmup_runs
        self.num_trials = num_trials

    def measure(
        self,
        model_wrapper: EmbeddingModelWrapper,
        sample_queries: List[str],
    ) -> LatencyReport:
        """Measures single-query encoding latency across trials.

        Args:
            model_wrapper: Loaded instance of EmbeddingModelWrapper.
            sample_queries: List of representative queries to cycle through.

        Returns:
            LatencyReport with mean, median, p95, p99 latencies in milliseconds.
        """
        if not sample_queries:
            raise ValueError("sample_queries cannot be empty")

        is_cuda = model_wrapper.device.startswith("cuda") and torch.cuda.is_available()

        logger.info(
            "Starting latency benchmark for '%s' (%d warmups, %d trials on %s) ...",
            model_wrapper.name,
            self.warmup_runs,
            self.num_trials,
            model_wrapper.device,
        )

        # 1. Warm-up runs (discarded to eliminate JIT/caching/allocation artifacts)
        for i in range(self.warmup_runs):
            query = [sample_queries[i % len(sample_queries)]]
            _ = model_wrapper.encode_queries(query, batch_size=1, show_progress_bar=False)
            if is_cuda:
                torch.cuda.synchronize()

        # 2. Timed trials
        latencies_ms: List[float] = []

        for i in range(self.num_trials):
            query = [sample_queries[i % len(sample_queries)]]

            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model_wrapper.encode_queries(query, batch_size=1, show_progress_bar=False)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)

        # 3. Statistical aggregation
        arr = np.array(latencies_ms)
        report = LatencyReport(
            model_name=model_wrapper.name,
            device=model_wrapper.device,
            num_trials=self.num_trials,
            warmup_runs=self.warmup_runs,
            mean_ms=round(float(np.mean(arr)), 3),
            std_ms=round(float(np.std(arr)), 3),
            p50_ms=round(float(np.median(arr)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            min_ms=round(float(np.min(arr)), 3),
            max_ms=round(float(np.max(arr)), 3),
            raw_latencies_ms=[round(x, 3) for x in latencies_ms],
        )

        logger.info(
            "Latency for '%s': Mean=%.2fms (±%.2fms) | p50=%.2fms | p95=%.2fms | p99=%.2fms",
            model_wrapper.name,
            report.mean_ms,
            report.std_ms,
            report.p50_ms,
            report.p95_ms,
            report.p99_ms,
        )
        return report
