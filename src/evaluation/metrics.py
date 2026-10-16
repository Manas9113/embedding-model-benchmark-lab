"""
Information Retrieval (IR) evaluation metrics for semantic search benchmarks.
Implements mathematically rigorous Recall@K, Precision@K, MRR@K, and nDCG@K.
"""

from typing import Dict, List, Set, Union
import numpy as np


def recall_at_k(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]], k: int) -> float:
    """Computes Recall@K: fraction of relevant documents retrieved in the top K.

    Formula:
        Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (top 1 at index 0).
        relevant_ids: Set or list of document IDs known to be relevant.
        k: Cut-off rank.

    Returns:
        Recall score in [0.0, 1.0]. Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in target_set)
    return hits / float(len(target_set))


def precision_at_k(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]], k: int) -> float:
    """Computes Precision@K: fraction of top K retrieved documents that are relevant.

    Formula:
        Precision@K = |Retrieved_K ∩ Relevant| / K

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set or list of document IDs known to be relevant.
        k: Cut-off rank.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in target_set)
    return hits / float(k)


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Union[Set[str], List[str]], k: int) -> float:
    """Computes Mean Reciprocal Rank (MRR@K) for a single query.

    Formula:
        RR@K = 1 / rank_first_relevant (if rank <= K else 0.0)

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: Set or list of relevant document IDs.
        k: Cut-off rank.

    Returns:
        Reciprocal rank score in [0.0, 1.0].
    """
    if not relevant_ids or k <= 0:
        return 0.0

    target_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in target_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, Union[int, float]], k: int) -> float:
    """Computes Discounted Cumulative Gain (DCG@K) with logarithmic rank discount.

    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevance_scores: Mapping from doc_id to graded relevance score.
        k: Cut-off rank.

    Returns:
        DCG score >= 0.0.
    """
    if k <= 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = float(relevance_scores.get(doc_id, 0.0))
        if rel > 0.0:
            # (2^rel - 1) / log2(rank + 1)
            gain = (2.0 ** rel) - 1.0
            discount = np.log2(rank + 1.0)
            dcg += gain / discount

    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, Union[int, float]], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K).

    Formula:
        nDCG@K = DCG@K / IDCG@K
        where IDCG@K is the Ideal DCG obtained by sorting all relevance scores descending.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevance_scores: Mapping from doc_id to graded relevance score.
        k: Cut-off rank.

    Returns:
        nDCG score in [0.0, 1.0]. Returns 0.0 if no relevant items exist.
    """
    if not relevance_scores or k <= 0:
        return 0.0

    # Calculate actual DCG
    actual_dcg = dcg_at_k(retrieved_ids, relevance_scores, k)
    if actual_dcg == 0.0:
        return 0.0

    # Calculate Ideal DCG (IDCG) by sorting gold relevance scores descending
    ideal_scores = sorted(
        [float(score) for score in relevance_scores.values() if float(score) > 0.0],
        reverse=True,
    )[:k]

    idcg = 0.0
    for rank, rel in enumerate(ideal_scores, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = np.log2(rank + 1.0)
        idcg += gain / discount

    if idcg == 0.0:
        return 0.0

    return actual_dcg / idcg
