"""Reciprocal Rank Fusion — merges N ranked lists into one without ML.

Score(doc) = Σ  1 / (k + rank_i)   for each list i where doc appears.

A doc that ranks highly in multiple lists accumulates more score than one
that ranks #1 in only a single list. k=60 dampens extreme rank-1 dominance.
"""

from __future__ import annotations

from collections import defaultdict

from retrieval.base import RetrievedDoc


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedDoc]],
    k: int = 60,
    top_n: int = 10,
) -> list[RetrievedDoc]:
    """Merge ranked result lists using RRF.

    Args:
        result_lists: Each inner list is one strategy's ranked output.
                      Docs are expected to have 1-based ``rank`` already set
                      (``_make_result`` guarantees this); falls back to
                      positional rank if ``doc.rank == 0``.
        k:            Smoothing constant; 60 is the TREC standard default.
        top_n:        How many docs to return after merging.

    Returns:
        Merged list sorted by RRF score descending, length ≤ top_n.
        ``doc.score`` is the RRF score; original retrieval scores are lost.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_store: dict[str, RetrievedDoc] = {}  # first-seen doc per id

    for result_list in result_lists:
        for pos, doc in enumerate(result_list, start=1):
            rank = doc.rank if doc.rank > 0 else pos
            scores[doc.doc_id] += 1.0 / (k + rank)
            if doc.doc_id not in doc_store:
                doc_store[doc.doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    merged = []
    for rank_pos, (doc_id, rrf_score) in enumerate(ranked, start=1):
        src = doc_store[doc_id]
        merged.append(
            RetrievedDoc(
                doc_id=doc_id,
                score=rrf_score,
                text=src.text,
                rank=rank_pos,
                metadata=src.metadata,
            )
        )
    return merged
