# Embedding Model Benchmark Lab - Project Walkthrough

## Overview
We built the **Embedding Model Benchmark Lab**, an independent, reproducible benchmarking suite for comparing open-source dense text embedding models on semantic search and information retrieval tasks.

---

## 1. Environment & Modular Architecture

### Hardware Environment
- **Platform**: Windows 11 (64-bit)
- **CPU**: 16 Physical/Logical Cores
- **RAM**: 16 GB Total System Memory
- **GPU Detected**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Interpreter**: Python 3.14.7 within isolated `.venv`
- **Execution Mode**: Deterministic CPU execution (FP32) for universal reproducibility

### Project Layout
```
Embedding-Model-Benchmark-Lab/
├── .gitignore                          <- Clean VCS rules
├── README.md                           <- Production-grade repository presentation
├── requirements.txt                    <- Pinned dependencies
├── config/
│   └── benchmark_config.yaml           <- Central experiment configuration
├── src/
│   ├── models/loader.py                <- Uniform model wrapper, memory profiler & prompt handler
│   ├── data/dataset_loader.py          <- Automated BEIR NFCorpus downloader, parser, & validator
│   ├── evaluation/metrics.py           <- Analytical Recall@K, Precision@K, MRR@K, nDCG@K
│   ├── evaluation/retriever.py         <- Dense vector indexing & cosine similarity retrieval
│   ├── benchmark/latency.py            <- High-precision timer (warmups, p50, p95, p99)
│   ├── benchmark/throughput.py         <- Multi-batch document throughput profiler
│   ├── benchmark/memory.py             <- Resident memory (RAM RSS & GPU VRAM) tracker
│   └── utils/
│       ├── logger.py                   <- Standardized logger with timestamps
│       └── visualization.py            <- Publication-quality Matplotlib chart generator
├── scripts/
│   ├── run_benchmark.py                <- End-to-end benchmark CLI runner
│   └── generate_visualizations.py      <- Standalone chart generator
├── notebooks/
│   └── 01_exploratory_benchmark.ipynb  <- Interactive exploration notebook
├── tests/
│   ├── test_environment.py            <- 4 passed
│   ├── test_dataset.py                <- 4 passed
│   ├── test_models.py                 <- 2 passed
│   ├── test_metrics.py                <- 7 passed
│   └── test_retrieval.py              <- 2 passed
├── results/
│   ├── raw/                            <- Benchmark JSON dumps
│   ├── tables/                         <- Markdown & CSV summary tables
│   └── figures/                        <- Pareto frontiers & comparison plots
└── docs/
    ├── research_report.md              <- Formal research report
    └── interview_prep.md               <- Technical interview Q&A guide
```

---

## 2. Automated Test Verification

All unit tests were executed with `pytest` inside the project virtual environment:

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Embedding Model Benchmark Lab
collected 19 items

tests/test_dataset.py::test_loader_parses_corpus_correctly PASSED        [  5%]
tests/test_dataset.py::test_loader_parses_queries_correctly PASSED       [ 10%]
tests/test_dataset.py::test_loader_parses_qrels_correctly PASSED         [ 15%]
tests/test_dataset.py::test_validation_reports_accurate_stats PASSED     [ 21%]
tests/test_environment.py::test_python_version PASSED                    [ 26%]
tests/test_environment.py::test_critical_imports PASSED                  [ 31%]
tests/test_environment.py::test_config_parsing PASSED                    [ 36%]
tests/test_environment.py::test_directory_structure PASSED               [ 42%]
tests/test_metrics.py::test_recall_at_k_exact_values PASSED              [ 47%]
tests/test_metrics.py::test_recall_empty_or_zero PASSED                  [ 52%]
tests/test_metrics.py::test_precision_at_k_exact_values PASSED           [ 57%]
tests/test_metrics.py::test_mrr_at_k_first_hit PASSED                    [ 63%]
tests/test_metrics.py::test_ndcg_at_k_hand_calculated PASSED             [ 68%]
tests/test_metrics.py::test_ndcg_perfect_ranking_equals_one PASSED       [ 73%]
tests/test_metrics.py::test_ndcg_zero_relevant_items PASSED              [ 78%]
tests/test_models.py::test_model_wrapper_encoding_and_normalization PASSED [ 84%]
tests/test_models.py::test_model_wrapper_prefix_handling PASSED          [ 89%]
tests/test_retrieval.py::test_retriever_ranking_exact_similarity PASSED  [ 94%]
tests/test_retrieval.py::test_retriever_evaluation_macro_averages PASSED [100%]

============================= 19 passed in 19.62s =============================
```

---

## 3. Real Empirical Benchmark Results (All 3 Models Complete)

The automated benchmark runner evaluated all 3 open-source models on the BEIR NFCorpus dataset (3,633 documents, 323 queries) on a 16-Core CPU:

| Model Architecture | Params | Dim | Memory | Latency $p50$ | Latency $p95$ | Peak Throughput | nDCG@10 | MRR@10 | Recall@10 | Precision@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all-MiniLM-L6-v2`** | 22.7M | 384 | 86.7 MB | **11.82 ms** | **13.29 ms** | **37.97 docs/s** | 0.3166 | 0.5046 | 0.1550 | 0.2433 |
| **`bge-small-en-v1.5`** | 33.4M | 384 | 127.3 MB | 21.19 ms | 24.82 ms | 19.01 docs/s | **0.3453** | **0.5330** | 0.1628 | **0.2563** |
| **`all-mpnet-base-v2`** | 109.5M | 768 | 417.7 MB | 50.84 ms | 59.96 ms | 5.32 docs/s | 0.3353 | 0.5240 | **0.1648** | 0.2533 |

---

## 4. Key Scientific & Engineering Insights

1. **`bge-small-en-v1.5` Outperforms the Larger `all-mpnet-base-v2`**:
   - Despite having only 33.4M parameters ($3.3\times$ smaller) and 384 dimensions (half the size), BGE-small achieves **0.3453 nDCG@10** versus MPNet's **0.3353 nDCG@10**.
   - BGE achieves this while delivering **$2.4\times$ faster single-query latency** (21.2 ms vs 50.8 ms) and **$3.6\times$ higher document encoding throughput** (19.0 docs/s vs 5.3 docs/s).
   - This empirically confirms that modern retrieval-oriented pre-training (RetroMAE and instruction-tuned contrastive loss) yields higher retrieval density than raw parameter scale.

2. **Vector Indexing Footprint (50% Cost Reduction)**:
   - 384-dimensional models require **1.54 GB** per million documents, compared to **3.07 GB** for 768-dimensional models.
   - For an enterprise RAG system with 10M documents, choosing BGE-small saves **15.36 GB of RAM** in the vector database while improving search quality.

3. **CPU Batching Dynamics**:
   - All models reached peak throughput at **batch size 8** on CPU. As batch size increased to 64, working memory exceeded on-chip CPU caches, incurring DDR memory bus latency and slightly reducing throughput.

---

## 5. Artifacts & Deliverables Generated

- **Raw Data**: [results/raw/benchmark_results_20261016_223500.json](file:///d:/Embedding%20Model%20Benchmark%20Lab/results/raw/benchmark_results_20261016_223500.json)
- **Summary Tables**:
  - [results/tables/benchmark_summary_latest.md](file:///d:/Embedding%20Model%20Benchmark%20Lab/results/tables/benchmark_summary_latest.md)
  - [results/tables/benchmark_summary_latest.csv](file:///d:/Embedding%20Model%20Benchmark%20Lab/results/tables/benchmark_summary_latest.csv)
- **Generated Visualizations**:
  - [results/figures/pareto_frontier.png](file:///d:/Embedding%20Model%20Benchmark%20Lab/results/figures/pareto_frontier.png)
  - [results/figures/throughput_scaling.png](file:///d:/Embedding%20Model%20Benchmark%20Lab/results/figures/throughput_scaling.png)
  - [results/figures/retrieval_comparison.png](file:///d:/Embedding%20Model%20Benchmark%20Lab/results/figures/retrieval_comparison.png)
- **Research Paper**: [docs/research_report.md](file:///d:/Embedding%20Model%20Benchmark%20Lab/docs/research_report.md)
- **Interview Prep Guide**: [docs/interview_prep.md](file:///d:/Embedding%20Model%20Benchmark%20Lab/docs/interview_prep.md)
- **Interactive Notebook**: [notebooks/01_exploratory_benchmark.ipynb](file:///d:/Embedding%20Model%20Benchmark%20Lab/notebooks/01_exploratory_benchmark.ipynb)
