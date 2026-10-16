"""
Dense vector retrieval and ranking evaluation engine.
Performs inner product (cosine similarity) ranking and calculates macro-averaged IR metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.evaluation.metrics import ndcg_at_k, mrr_at_k, precision_at_k, recall_at_k
from src.utils.logger import setup_logger

logger = setup_logger("dense_retriever")


@dataclass
class RetrievalEvaluationReport:
    """Consolidated Information Retrieval evaluation results across all queries."""

    model_name: str
    num_queries_evaluated: int
    num_corpus_documents: int
    metrics: Dict[str, float]  # e.g., "Recall@1", "nDCG@10", "MRR@10", etc.


class DenseRetriever:
    """Indexes document embeddings, retrieves top-K candidates, and scores retrieval quality."""

    def __init__(self, doc_ids: List[str], doc_embeddings: np.ndarray):
        """Initializes retriever with document IDs and normalized embeddings.

        Args:
            doc_ids: Ordered list of document IDs corresponding to embedding rows.
            doc_embeddings: Normalized 2D float array (num_docs, dimension).
        """
        if len(doc_ids) != doc_embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(doc_ids)} doc IDs vs {doc_embeddings.shape[0]} embeddings"
            )

        self.doc_ids = doc_ids
        self.doc_embeddings = doc_embeddings.astype(np.float32)

    def retrieve_top_k(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
    ) -> List[List[str]]:
        """Retrieves top-K document IDs for each query using cosine similarity (dot product).

        Args:
            query_embeddings: Normalized 2D float array (num_queries, dimension).
            k: Number of candidates to retrieve.

        Returns:
            List of length num_queries, each containing top-K doc_ids ordered by score descending.
        """
        # Similarity matrix: (num_queries, num_docs)
        # Assumes embeddings are already L2-normalized
        sim_matrix = np.dot(query_embeddings, self.doc_embeddings.T)

        all_top_k_ids: List[List[str]] = []
        for i in range(sim_matrix.shape[0]):
            scores = sim_matrix[i]
            # Use argpartition for fast top-K selection, then sort top-K slice
            if k < len(scores):
                cand_indices = np.argpartition(-scores, k)[:k]
                sorted_cand_indices = cand_indices[np.argsort(-scores[cand_indices])]
            else:
                sorted_cand_indices = np.argsort(-scores)

            top_ids = [self.doc_ids[idx] for idx in sorted_cand_indices]
            all_top_k_ids.append(top_ids)

        return all_top_k_ids

    def evaluate(
        self,
        query_ids: List[str],
        query_embeddings: np.ndarray,
        qrels: Dict[str, Dict[str, int]],
        top_k_values: Optional[List[int]] = None,
        model_name: str = "model",
    ) -> RetrievalEvaluationReport:
        """Evaluates retrieval quality against gold qrels over all queries.

        Args:
            query_ids: Ordered list of query IDs.
            query_embeddings: Normalized 2D float array (num_queries, dimension).
            qrels: Gold relevance mapping {query_id -> {doc_id -> relevance_score}}.
            top_k_values: List of cut-off thresholds (default: [1, 3, 5, 10]).
            model_name: Name of model for reporting.

        Returns:
            RetrievalEvaluationReport with macro-averaged scores.
        """
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

        max_k = max(top_k_values)
        retrieved_results = self.retrieve_top_k(query_embeddings, k=max_k)

        # Accumulators for macro-averaging
        metric_sums: Dict[str, float] = {}
        for k in top_k_values:
            metric_sums[f"Recall@{k}"] = 0.0
            metric_sums[f"Precision@{k}"] = 0.0
            metric_sums[f"MRR@{k}"] = 0.0
            metric_sums[f"nDCG@{k}"] = 0.0

        valid_query_count = 0

        for q_idx, q_id in enumerate(query_ids):
            if q_id not in qrels or not qrels[q_id]:
                continue  # Skip queries without relevance judgments

            gold_relevance = qrels[q_id]
            # Set of doc_ids with relevance > 0 (binary relevant)
            relevant_docs = {did for did, score in gold_relevance.items() if score > 0}
            top_retrieved = retrieved_results[q_idx]

            valid_query_count += 1

            for k in top_k_values:
                metric_sums[f"Recall@{k}"] += recall_at_k(top_retrieved, relevant_docs, k)
                metric_sums[f"Precision@{k}"] += precision_at_k(top_retrieved, relevant_docs, k)
                metric_sums[f"MRR@{k}"] += mrr_at_k(top_retrieved, relevant_docs, k)
                metric_sums[f"nDCG@{k}"] += ndcg_at_k(top_retrieved, gold_relevance, k)

        if valid_query_count == 0:
            raise ValueError("No valid queries found with non-empty gold qrels")

        # Compute macro-averages (mean across all queries)
        avg_metrics = {
            metric_name: round(total / valid_query_count, 4)
            for metric_name, total in metric_sums.items()
        }

        report = RetrievalEvaluationReport(
            model_name=model_name,
            num_queries_evaluated=valid_query_count,
            num_corpus_documents=len(self.doc_ids),
            metrics=avg_metrics,
        )

        # Format display string for available metrics safely
        primary_k = 10 if 10 in top_k_values else top_k_values[-1]
        logger.info(
            "Evaluation for '%s' (%d queries): nDCG@%d=%.4f | MRR@%d=%.4f | Recall@%d=%.4f | Precision@%d=%.4f",
            model_name,
            valid_query_count,
            primary_k,
            avg_metrics.get(f"nDCG@{primary_k}", 0.0),
            primary_k,
            avg_metrics.get(f"MRR@{primary_k}", 0.0),
            primary_k,
            avg_metrics.get(f"Recall@{primary_k}", 0.0),
            primary_k,
            avg_metrics.get(f"Precision@{primary_k}", 0.0),
        )
        return report
