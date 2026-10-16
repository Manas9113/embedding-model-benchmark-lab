"""
Standalone script to re-generate comparison figures from existing benchmark JSON files
without re-running the neural network inference pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.utils.visualization import (
    plot_pareto_frontier,
    plot_retrieval_metrics_comparison,
    plot_throughput_scaling,
)

logger = setup_logger("generate_visualizations")


def generate_figures(results_json_path: str, output_dir: str = "results/figures"):
    json_path = Path(results_json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Results JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations for %d models from %s ...", len(results), json_path.name)
    plot_pareto_frontier(results, str(out_path / "pareto_frontier.png"))
    plot_throughput_scaling(results, str(out_path / "throughput_scaling.png"))
    plot_retrieval_metrics_comparison(results, str(out_path / "retrieval_comparison.png"))
    logger.info("Visualizations successfully saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures from benchmark JSON results")
    parser.add_argument(
        "--results",
        default=None,
        help="Path to specific benchmark_results_*.json file. Defaults to latest in results/raw/",
    )
    parser.add_argument(
        "--output_dir",
        default="results/figures",
        help="Directory to save generated chart figures",
    )
    args = parser.parse_args()

    target_json = args.results
    if not target_json:
        # Find latest JSON in results/raw
        raw_dir = PROJECT_ROOT / "results" / "raw"
        json_files = sorted(raw_dir.glob("benchmark_results_*.json"))
        if not json_files:
            logger.error("No benchmark results JSON found in %s! Run 'python scripts/run_benchmark.py' first.", raw_dir)
            sys.exit(1)
        target_json = str(json_files[-1])

    generate_figures(target_json, output_dir=args.output_dir)
