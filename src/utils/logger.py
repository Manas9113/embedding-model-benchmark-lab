"""
Standardized logging utility for the Embedding Model Benchmark Lab.
Provides consistent formatting, timestamping, and dual-output (stdout + file).
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "benchmark_lab",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a thread-safe, formatted logger.

    Args:
        name: Name of the logger instance.
        log_file: Optional file path to persist log messages.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
