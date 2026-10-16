"""
Main executable benchmarking pipeline for Embedding Model Benchmark Lab.
Automates model loading, profiling, latency measurement, throughput measurement,
corpus indexing, semantic search retrieval evaluation, and results export.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.benchmark.latency import LatencyBenchmark
from src.benchmark.memory import MemoryTracker
from src.benchmark.throughput import ThroughputBenchmark
from src.data.dataset_loader import DatasetLoader
from src.evaluation.retriever import DenseRetriever
from src.models.loader import EmbeddingModelWrapper
from src.utils.logger import setup_logger
from src.utils.visualization import (
    plot_pareto_frontier,
    plot_retrieval_metrics_comparison,
    plot_throughput_scaling,
)

logger = setup_logger("benchmark_runner")


def load_config(config_path: str = "config/benchmark_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_benchmark(config_path: str = "config/benchmark_config.yaml", device_override: str | None = None):
    config = load_config(config_path)

    device = device_override or config["experiment"].get("device", "auto")
    logger.info("Initializing Embedding Model Benchmark Lab (Device: %s)", device)

    # 1. Ingest Dataset
    dataset_cfg = config["dataset"]
    loader = DatasetLoader(
        data_dir=dataset_cfg["data_dir"],
        download_url=dataset_cfg.get("download_url"),
    )
    dataset = loader.load_dataset(split="test", auto_download=True)
    logger.info("Dataset ready: %d docs, %d test queries", dataset.num_documents, dataset.num_queries)

    # Prepare corpus documents & queries
    corpus_ids = list(dataset.corpus.keys())
    corpus_texts = [
        f"{doc['title']} {doc['text']}".strip() for doc in dataset.corpus.values()
    ]

    query_ids = list(dataset.queries.keys())
    query_texts = [dataset.queries[qid] for qid in query_ids]

    # Initialize benchmark harnesses
    bench_cfg = config["benchmark"]
    latency_bench = LatencyBenchmark(
        warmup_runs=bench_cfg["latency"]["warmup_runs"],
        num_trials=bench_cfg["latency"]["num_trials"],
    )
    throughput_bench = ThroughputBenchmark(
        batch_sizes=bench_cfg["throughput"]["batch_sizes"],
        warmup_batches=bench_cfg["throughput"]["warmup_batches"],
    )
    memory_tracker = MemoryTracker()

    all_model_results: List[Dict[str, Any]] = []

    # 2. Iterate through each model
    for m_cfg in config["models"]:
        model_id = m_cfg["id"]
        model_name = m_cfg["name"]
        logger.info("=================================================================")
        logger.info("BENCHMARKING MODEL: %s (%s)", model_name, model_id)
        logger.info("=================================================================")

        memory_tracker.force_garbage_collection()
        mem_before = memory_tracker.snapshot()

        # Load and profile model
        wrapper = EmbeddingModelWrapper(
            model_id=model_id,
            name=model_name,
            architecture=m_cfg.get("architecture", "Transformer"),
            device=device,
            max_seq_length=m_cfg.get("max_seq_length", 256),
            query_prefix=m_cfg.get("query_prefix", ""),
            doc_prefix=m_cfg.get("doc_prefix", ""),
        )
        profile = wrapper.load()
        mem_after_load = memory_tracker.snapshot()

        # A. Latency Benchmark (Single-query p50, p95, p99)
        sample_queries = query_texts[:10]  # Representative subset for repeated latency trials
        latency_report = latency_bench.measure(wrapper, sample_queries=sample_queries)

        # B. Throughput Benchmark (Documents/second across batch sizes)
        throughput_report = throughput_bench.measure(wrapper, corpus_texts, sample_size=512)

        # C. Full Corpus Indexing
        logger.info("Encoding entire corpus (%d documents) with %s ...", len(corpus_texts), model_name)
        t0_encode = time.perf_counter()
        corpus_embeddings = wrapper.encode_corpus(
            corpus_texts,
            batch_size=throughput_report.optimal_batch_size,
            show_progress_bar=True,
        )
        corpus_encode_time = time.perf_counter() - t0_encode
        corpus_docs_sec = len(corpus_texts) / corpus_encode_time
        logger.info(
            "Corpus encoded in %.2fs (%.1f docs/sec) | Shape: %s",
            corpus_encode_time,
            corpus_docs_sec,
            corpus_embeddings.shape,
        )

        # D. Query Encoding
        logger.info("Encoding %d test queries with %s ...", len(query_texts), model_name)
        query_embeddings = wrapper.encode_queries(
            query_texts,
            batch_size=32,
            show_progress_bar=False,
        )

        # E. Information Retrieval Evaluation
        logger.info("Building DenseRetriever and scoring against gold qrels ...")
        retriever = DenseRetriever(doc_ids=corpus_ids, doc_embeddings=corpus_embeddings)
        eval_report = retriever.evaluate(
            query_ids=query_ids,
            query_embeddings=query_embeddings,
            qrels=dataset.qrels,
            top_k_values=bench_cfg["evaluation"]["top_k_values"],
            model_name=model_name,
        )

        mem_peak = memory_tracker.snapshot()

        # Compile structured result record
        model_result = {
            "model_id": model_id,
            "model_name": model_name,
            "architecture": profile.architecture,
            "dimension": profile.dimension,
            "parameter_count": profile.parameter_count,
            "memory_footprint_mb": profile.memory_footprint_mb,
            "load_time_seconds": profile.load_time_seconds,
            "device": profile.device,
            "latency_mean_ms": latency_report.mean_ms,
            "latency_p50_ms": latency_report.p50_ms,
            "latency_p95_ms": latency_report.p95_ms,
            "latency_p99_ms": latency_report.p99_ms,
            "peak_throughput_docs_sec": throughput_report.peak_throughput_docs_sec,
            "optimal_batch_size": throughput_report.optimal_batch_size,
            "throughput_by_batch": {
                bs: res.docs_per_second
                for bs, res in throughput_report.results_by_batch_size.items()
            },
            "corpus_encode_time_seconds": round(corpus_encode_time, 2),
            "corpus_docs_per_sec": round(corpus_docs_sec, 1),
            "ndcg_10": eval_report.metrics.get("nDCG@10", 0.0),
            "mrr_10": eval_report.metrics.get("MRR@10", 0.0),
            "recall_10": eval_report.metrics.get("Recall@10", 0.0),
            "precision_10": eval_report.metrics.get("Precision@10", 0.0),
            "metrics": eval_report.metrics,
            "ram_rss_delta_mb": round(mem_peak.ram_rss_mb - mem_before.ram_rss_mb, 2),
            "gpu_allocated_mb": mem_peak.gpu_allocated_mb,
            "gpu_peak_mb": mem_peak.gpu_peak_mb,
        }
        all_model_results.append(model_result)

        # Cleanup model from memory
        del wrapper
        del corpus_embeddings
        del query_embeddings
        del retriever
        memory_tracker.force_garbage_collection()

    # 3. Export Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_raw_dir = Path(config["paths"]["results_raw"])
    results_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = results_raw_dir / f"benchmark_results_{timestamp}.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, indent=2)
    logger.info("Saved raw JSON results to %s", raw_json_path)

    # 4. Generate Tabular Summaries (Markdown & CSV)
    results_tables_dir = Path(config["paths"]["results_tables"])
    results_tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for r in all_model_results:
        summary_rows.append({
            "Model": r["model_name"],
            "Params (M)": round(r["parameter_count"] / 1e6, 1),
            "Dim": r["dimension"],
            "Load (s)": r["load_time_seconds"],
            "Latency p50 (ms)": r["latency_p50_ms"],
            "Latency p95 (ms)": r["latency_p95_ms"],
            "Throughput (docs/s)": r["peak_throughput_docs_sec"],
            "nDCG@10": r["ndcg_10"],
            "MRR@10": r["mrr_10"],
            "Recall@10": r["recall_10"],
            "Precision@10": r["precision_10"],
        })

    df = pd.DataFrame(summary_rows)
    csv_path = results_tables_dir / f"benchmark_summary_{timestamp}.csv"
    latest_csv_path = results_tables_dir / "benchmark_summary_latest.csv"
    md_path = results_tables_dir / "benchmark_summary_latest.md"

    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv_path, index=False)

    try:
        md_table = df.to_markdown(index=False)
    except Exception:
        # Fallback to standard dataframe string if tabulate is unavailable
        md_table = "```\n" + df.to_string(index=False) + "\n```"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary Table\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} on `{device}`\n\n")
        f.write(md_table + "\n")

    logger.info("Saved summary table to %s and %s", latest_csv_path, md_path)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE - SUMMARY RESULTS:")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    # 5. Generate Comparison Visualizations
    results_figures_dir = Path(config["paths"]["results_figures"])
    results_figures_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto_frontier(all_model_results, str(results_figures_dir / "pareto_frontier.png"))
    plot_throughput_scaling(all_model_results, str(results_figures_dir / "throughput_scaling.png"))
    plot_retrieval_metrics_comparison(all_model_results, str(results_figures_dir / "retrieval_comparison.png"))
    logger.info("All comparison charts generated successfully!")

    return all_model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Embedding Model Benchmark Lab")
    parser.add_argument("--config", default="config/benchmark_config.yaml", help="Path to config YAML")
    parser.add_argument("--device", default=None, help="Device override ('cpu' or 'cuda')")
    args = parser.parse_args()

    run_benchmark(config_path=args.config, device_override=args.device)
