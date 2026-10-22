"""Retrieval quality metrics: nDCG@k, MRR, Recall@k, Precision@k.

Uses pytrec_eval when available; falls back to pure-Python implementations
that are transparent, testable, and interview-explainable.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# pytrec_eval path (preferred – matches the TREC/BEIR evaluation standard)
# ---------------------------------------------------------------------------

def _try_pytrec_eval(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, float] | None:
    try:
        import pytrec_eval  # type: ignore[import]
    except ImportError:
        return None

    measures = (
        {f"ndcg_cut_{k}" for k in k_values}
        | {f"recall_{k}" for k in k_values}
        | {"recip_rank"}
    )
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(results)

    n = len(per_query)
    if n == 0:
        zeros: dict[str, float] = {}
        for k in k_values:
            zeros[f"ndcg@{k}"] = 0.0
            zeros[f"recall@{k}"] = 0.0
        zeros["mrr"] = 0.0
        return zeros

    agg: dict[str, float] = {}
    for k in k_values:
        agg[f"ndcg@{k}"] = sum(v[f"ndcg_cut_{k}"] for v in per_query.values()) / n
        agg[f"recall@{k}"] = sum(v[f"recall_{k}"] for v in per_query.values()) / n
    agg["mrr"] = sum(v["recip_rank"] for v in per_query.values()) / n
    return agg


# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------

def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ideal_dcg(relevances: list[int], k: int) -> float:
    return _dcg(sorted(relevances, reverse=True), k)


def _ndcg_at_k(ranked_doc_ids: list[str], relevant: dict[str, int], k: int) -> float:
    rels = [relevant.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    idcg = _ideal_dcg(list(relevant.values()), k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


def _mrr(ranked_doc_ids: list[str], relevant: dict[str, int]) -> float:
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / i
    return 0.0


def _recall_at_k(ranked_doc_ids: list[str], relevant: dict[str, int], k: int) -> float:
    n_relevant = sum(1 for v in relevant.values() if v > 0)
    if n_relevant == 0:
        return 0.0
    retrieved_relevant = sum(
        1 for doc_id in ranked_doc_ids[:k] if relevant.get(doc_id, 0) > 0
    )
    return retrieved_relevant / n_relevant


def _precision_at_k(ranked_doc_ids: list[str], relevant: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_relevant = sum(
        1 for doc_id in ranked_doc_ids[:k] if relevant.get(doc_id, 0) > 0
    )
    return retrieved_relevant / k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Return averaged retrieval metrics across all queries.

    Args:
        qrels:    {query_id: {doc_id: relevance_grade}}
        results:  {query_id: {doc_id: score}}
        k_values: rank cutoffs; defaults to [1, 3, 5, 10]
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    fast = _try_pytrec_eval(qrels, results, k_values)
    if fast is not None:
        return fast

    metrics_sum: dict[str, float] = {f"ndcg@{k}": 0.0 for k in k_values}
    metrics_sum |= {f"recall@{k}": 0.0 for k in k_values}
    metrics_sum |= {f"precision@{k}": 0.0 for k in k_values}
    metrics_sum["mrr"] = 0.0

    n = 0
    for q_id, relevant in qrels.items():
        if q_id not in results:
            continue
        ranked = sorted(results[q_id].items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        for k in k_values:
            metrics_sum[f"ndcg@{k}"] += _ndcg_at_k(ranked_ids, relevant, k)
            metrics_sum[f"recall@{k}"] += _recall_at_k(ranked_ids, relevant, k)
            metrics_sum[f"precision@{k}"] += _precision_at_k(ranked_ids, relevant, k)
        metrics_sum["mrr"] += _mrr(ranked_ids, relevant)
        n += 1

    if n == 0:
        return metrics_sum
    return {k: v / n for k, v in metrics_sum.items()}


def per_query_ndcg(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, float]:
    """nDCG@k per evaluated query (linear gain, log2 discount, as in trec_eval).

    A query that retrieved nothing scores 0.0; queries without qrels are skipped.
    """
    out: dict[str, float] = {}
    for q_id, retrieved in results.items():
        relevant = qrels.get(q_id)
        if relevant is None:
            continue
        ranked = sorted(retrieved.items(), key=lambda x: x[1], reverse=True)
        out[q_id] = _ndcg_at_k([doc_id for doc_id, _ in ranked], relevant, k)
    return out
