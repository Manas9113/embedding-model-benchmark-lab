"""
Sanity tests to verify Python environment, package versions, and configuration parsing.
"""

import sys
from pathlib import Path
import yaml
import pytest


def test_python_version():
    """Verify Python is 3.10 or newer."""
    assert sys.version_info >= (3, 10), f"Python version too old: {sys.version}"


def test_critical_imports():
    """Ensure core libraries import without C-extension or DLL conflicts."""
    import torch
    import numpy
    import pandas
    import sentence_transformers
    import transformers

    assert torch.__version__ is not None
    assert numpy.__version__ is not None
    assert pandas.__version__ is not None
    assert sentence_transformers.__version__ is not None


def test_config_parsing():
    """Ensure config/benchmark_config.yaml exists and contains required keys."""
    config_path = Path("config/benchmark_config.yaml")
    assert config_path.exists(), "config/benchmark_config.yaml does not exist!"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "models" in config, "Missing 'models' section in config"
    assert len(config["models"]) == 3, f"Expected 3 models, found {len(config['models'])}"
    assert "dataset" in config, "Missing 'dataset' section in config"
    assert "benchmark" in config, "Missing 'benchmark' section in config"


def test_directory_structure():
    """Ensure all required directories exist."""
    required_dirs = [
        "config",
        "src/models",
        "src/data",
        "src/evaluation",
        "src/benchmark",
        "src/utils",
        "scripts",
        "notebooks",
        "tests",
        "results/raw",
        "results/tables",
        "results/figures",
        "docs",
    ]
    for d in required_dirs:
        assert Path(d).is_dir(), f"Required directory '{d}' is missing!"
