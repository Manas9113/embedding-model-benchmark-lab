"""
Memory consumption profiling for CPU RAM and GPU VRAM.
Tracks baseline, post-load, and peak inference memory deltas.
"""

import gc
import os
from dataclasses import dataclass
from typing import Optional
import psutil
import torch

from src.utils.logger import setup_logger

logger = setup_logger("benchmark_memory")


@dataclass
class MemorySnapshot:
    """Snapshot of process memory usage in megabytes (MB)."""

    ram_rss_mb: float
    ram_vms_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_mb: float


class MemoryTracker:
    """Monitors system RAM and GPU memory before and during benchmark operations."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.is_cuda = torch.cuda.is_available()

    def snapshot(self) -> MemorySnapshot:
        """Captures current RAM and GPU memory usage."""
        mem_info = self.process.memory_info()
        ram_rss = mem_info.rss / (1024 * 1024)
        ram_vms = mem_info.vms / (1024 * 1024)

        gpu_alloc = 0.0
        gpu_res = 0.0
        gpu_peak = 0.0

        if self.is_cuda:
            gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_res = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return MemorySnapshot(
            ram_rss_mb=round(ram_rss, 2),
            ram_vms_mb=round(ram_vms, 2),
            gpu_allocated_mb=round(gpu_alloc, 2),
            gpu_reserved_mb=round(gpu_res, 2),
            gpu_peak_mb=round(gpu_peak, 2),
        )

    def reset_gpu_peak(self):
        """Resets the peak memory tracker on GPU."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

    def force_garbage_collection(self):
        """Forces Python garbage collection and empties PyTorch CUDA cache."""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
