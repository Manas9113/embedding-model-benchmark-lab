"""
Visualization utilities for Embedding Model Benchmark Lab.
Generates publication-quality charts for Pareto frontiers, throughput scaling,
retrieval accuracy comparisons, and memory footprints using Matplotlib and Seaborn.
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils.logger import setup_logger

logger = setup_logger("visualization")

# Apply clean, modern research styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_pareto_frontier(
    results: List[Dict[str, Any]],
    output_path: str = "results/figures/pareto_frontier.png",
):
    """Plots the Latency vs. nDCG@10 Pareto Frontier trade-off.

    Each model is plotted with:
    - X-axis: Single-query latency p50 (ms) [Lower is better]
    - Y-axis: nDCG@10 [Higher is better]
    - Bubble size: Embedding dimension (e.g., 384 vs 768)
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    model_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for i, res in enumerate(results):
        name = res["model_name"]
        latency = res["latency_p50_ms"]
        ndcg = res["ndcg_10"]
        dim = res.get("dimension", 384)
        color = model_colors[i % len(model_colors)]

        # Plot point with bubble size scaled by dimension
        scatter = ax.scatter(
            latency,
            ndcg,
            s=dim * 0.7,
            color=color,
            alpha=0.85,
            edgecolors="#333333",
            linewidth=1.2,
            label=f"{name} ({dim}d)",
            zorder=3,
        )

        # Annotate model name
        ax.annotate(
            f" {name}\n (nDCG: {ndcg:.3f}, {latency:.1f}ms)",
            xy=(latency, ndcg),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xlabel("Single-Query Latency p50 (ms) [Lower is Better]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Retrieval Quality (nDCG@10) [Higher is Better]", fontsize=11, fontweight="bold")
    ax.set_title("Pareto Trade-off: Query Latency vs. Retrieval Quality", fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Models & Dimensions", loc="lower right", frameon=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved Pareto frontier plot to %s", output_path)


def plot_throughput_scaling(
    results: List[Dict[str, Any]],
    output_path: str = "results/figures/throughput_scaling.png",
):
    """Plots document encoding throughput (docs/sec) across batch sizes."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = ["#2b5c8f", "#d95f02", "#7570b3"]
    markers = ["o", "s", "^"]

    for i, res in enumerate(results):
        name = res["model_name"]
        batch_sizes = sorted(res["throughput_by_batch"].keys())
        throughputs = [res["throughput_by_batch"][bs] for bs in batch_sizes]

        ax.plot(
            batch_sizes,
            throughputs,
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=7,
            color=palette[i % len(palette)],
            label=name,
        )

    ax.set_xlabel("Batch Size", fontsize=11, fontweight="bold")
    ax.set_ylabel("Throughput (Documents / Second)", fontsize=11, fontweight="bold")
    ax.set_title("Encoding Throughput Scaling by Batch Size", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(batch_sizes)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved throughput scaling plot to %s", output_path)


def plot_retrieval_metrics_comparison(
    results: List[Dict[str, Any]],
    output_path: str = "results/figures/retrieval_comparison.png",
):
    """Bar chart comparing Recall@1, Recall@5, Recall@10, MRR@10, and nDCG@10."""
    metrics_to_plot = ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]

    model_names = [res["model_name"] for res in results]
    x = np.arange(len(metrics_to_plot))
    bar_width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    palette = ["#437299", "#d6604d", "#7b3294"]

    for i, res in enumerate(results):
        scores = [res["metrics"].get(m, 0.0) for m in metrics_to_plot]
        offset = (i - len(model_names) / 2.0 + 0.5) * bar_width
        rects = ax.bar(
            x + offset,
            scores,
            bar_width,
            label=res["model_name"],
            color=palette[i % len(palette)],
            edgecolor="#333333",
            linewidth=0.6,
        )

        # Add value label on bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Information Retrieval Quality Metrics Comparison (NFCorpus)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info("Saved retrieval metrics comparison plot to %s", output_path)
