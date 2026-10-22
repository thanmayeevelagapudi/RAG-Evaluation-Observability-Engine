"""Phase 2 unit tests — RRF, hybrid, HyDE, multi-query, parent-child, rerankers.

All tests use mocks; no Qdrant, Elasticsearch, Cohere, or Anthropic needed.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval.base import QueryType, RetrievedDoc, RetrievalResult
from retrieval.fusion.reciprocal_rank_fusion import reciprocal_rank_fusion
from retrieval.reranker.bge_reranker import BGEReranker
from retrieval.reranker.cohere_reranker import CohereReranker
from retrieval.strategies.dense import DenseRetriever
from retrieval.strategies.hybrid import HybridRetriever
from retrieval.strategies.hyde import HyDERetriever
from retrieval.strategies.multi_query import MultiQueryRetriever
from retrieval.strategies.parent_child import ParentChildRetriever, _word_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docs(*ids_and_scores: tuple[str, float]) -> list[RetrievedDoc]:
    return [
        RetrievedDoc(doc_id=did, score=score, text=f"text for {did}", rank=i + 1)
        for i, (did, score) in enumerate(ids_and_scores)
    ]


def _result(docs: list[RetrievedDoc], strategy: str = "stub") -> RetrievalResult:
    return RetrievalResult(
        query="q", docs=docs, strategy=strategy,
        query_type=QueryType.KEYWORD, latency_ms=5.0,
    )


def _mock_dense(docs: list[RetrievedDoc] | None = None):
    dense = AsyncMock()
    dense.retrieve = AsyncMock(return_value=_result(docs or _docs(("d1", 0.9), ("d2", 0.8))))
    dense.warm_up = AsyncMock()
    dense.upsert_documents = AsyncMock()
    dense.embed_text = AsyncMock(return_value=[0.1] * 384)
    dense.retrieve_by_embedding = AsyncMock(return_value=_docs(("d1", 0.9), ("d2", 0.7)))
    return dense


def _mock_bm25(docs: list[RetrievedDoc] | None = None):
    bm25 = AsyncMock()
    bm25.retrieve = AsyncMock(return_value=_result(docs or _docs(("d2", 0.85), ("d3", 0.7))))
    bm25.warm_up = AsyncMock()
    bm25.upsert_documents = AsyncMock()
    return bm25


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_single_list_passthrough(self):
        docs = _docs(("d1", 0.9), ("d2", 0.8))
        result = reciprocal_rank_fusion([docs], top_n=2)
        assert [d.doc_id for d in result] == ["d1", "d2"]

    def test_agreement_amplification(self):
        """A doc ranked high in both lists should outscore one ranked #1 in only one."""
        list1 = _docs(("d_top", 1.0), ("d_both", 0.9), ("d_only1", 0.5))
        list2 = _docs(("d_both", 0.95), ("d_top", 0.3))
        result = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)
        ids = [d.doc_id for d in result]
        # d_both ranks in both lists; d_top only gets high contribution from list1
        assert "d_both" in ids[:2]

    def test_doc_in_only_one_list(self):
        list1 = _docs(("d1", 1.0))
        list2 = _docs(("d2", 1.0))
        result = reciprocal_rank_fusion([list1, list2], top_n=2)
        assert {d.doc_id for d in result} == {"d1", "d2"}

    def test_scores_are_rrf_not_original(self):
        docs = _docs(("d1", 999.0))  # absurdly high original score
        result = reciprocal_rank_fusion([docs], k=60, top_n=1)
        # RRF score for rank-1 with k=60 is 1/61 ≈ 0.0164
        assert result[0].score == pytest.approx(1 / 61, rel=1e-3)

    def test_top_n_truncation(self):
        docs = _docs(*[(f"d{i}", 1.0 / (i + 1)) for i in range(10)])
        result = reciprocal_rank_fusion([docs], top_n=3)
        assert len(result) == 3

    def test_empty_lists_ignored(self):
        docs = _docs(("d1", 0.9))
        result = reciprocal_rank_fusion([[], docs, []], top_n=5)
        assert len(result) == 1
        assert result[0].doc_id == "d1"

    def test_ranks_are_one_based(self):
        docs = _docs(("d1", 0.9), ("d2", 0.8), ("d3", 0.7))
        result = reciprocal_rank_fusion([docs], top_n=3)
        assert [d.rank for d in result] == [1, 2, 3]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_calls_both_retrievers(self):
        dense = _mock_dense()
        bm25 = _mock_bm25()
        retriever = HybridRetriever(dense=dense, bm25=bm25)
        result = await retriever.retrieve("q", top_k=2)
        dense.retrieve.assert_called_once()
        bm25.retrieve.assert_called_once()
        assert isinstance(result, RetrievalResult)
        assert result.strategy == "hybrid"

    @pytest.mark.asyncio
    async def test_metadata_contains_latencies(self):
        retriever = HybridRetriever(dense=_mock_dense(), bm25=_mock_bm25())
        result = await retriever.retrieve("q")
        assert "dense_latency_ms" in result.metadata
        assert "bm25_latency_ms" in result.metadata

    @pytest.mark.asyncio
    async def test_warm_up_delegates_to_both(self):
        dense = _mock_dense()
        bm25 = _mock_bm25()
        await HybridRetriever(dense=dense, bm25=bm25).warm_up()
        dense.warm_up.assert_called_once()
        bm25.warm_up.assert_called_once()

    @pytest.mark.asyncio
    async def test_result_deduplicates_via_rrf(self):
        # Both retrievers return "d2" — it should appear once, ranked high
        shared_doc = RetrievedDoc(doc_id="d2", score=0.9, text="shared", rank=1)
        dense = _mock_dense(_docs(("d1", 1.0), ("d2", 0.9)))
        bm25 = _mock_bm25(_docs(("d2", 0.85), ("d3", 0.7)))
        result = await HybridRetriever(dense=dense, bm25=bm25).retrieve("q", top_k=5)
        doc_ids = [d.doc_id for d in result.docs]
        assert doc_ids.count("d2") == 1


# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------

class TestHyDERetriever:
    def _mock_anthropic(self, text: str = "Hypothesis text."):
        client = AsyncMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        client.messages.create = AsyncMock(return_value=msg)
        return client

    @pytest.mark.asyncio
    async def test_hypothesis_stored_in_metadata(self):
        dense = _mock_dense()
        retriever = HyDERetriever(dense=dense, anthropic_client=self._mock_anthropic("Some answer."))
        result = await retriever.retrieve("What causes inflation?", top_k=2)
        assert result.metadata.get("hypothesis") == "Some answer."

    @pytest.mark.asyncio
    async def test_uses_embed_text_not_retrieve(self):
        dense = _mock_dense()
        retriever = HyDERetriever(dense=dense, anthropic_client=self._mock_anthropic())
        await retriever.retrieve("q", top_k=2)
        dense.embed_text.assert_called_once()
        dense.retrieve_by_embedding.assert_called_once()
        dense.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_on_anthropic_failure(self):
        dense = _mock_dense()
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("rate limit"))
        retriever = HyDERetriever(dense=dense, anthropic_client=client)
        result = await retriever.retrieve("q", top_k=2)
        # Falls back to standard dense retrieve
        assert result.metadata.get("fallback") is True
        dense.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# MultiQueryRetriever
# ---------------------------------------------------------------------------

class TestMultiQueryRetriever:
    def _mock_anthropic(self, variants: list[str]):
        client = AsyncMock()
        import json
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(variants))]
        client.messages.create = AsyncMock(return_value=msg)
        return client

    @pytest.mark.asyncio
    async def test_runs_all_variants(self):
        dense = _mock_dense()
        variants = ["variant 1", "variant 2", "variant 3"]
        client = self._mock_anthropic(variants)
        retriever = MultiQueryRetriever(dense=dense, n_variants=3, anthropic_client=client)
        result = await retriever.retrieve("original q", top_k=5)
        # 1 original + 3 variants = 4 retrieve calls
        assert dense.retrieve.call_count == 4

    @pytest.mark.asyncio
    async def test_fallback_on_json_parse_error(self):
        dense = _mock_dense()
        client = AsyncMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="not json at all")]
        client.messages.create = AsyncMock(return_value=msg)
        retriever = MultiQueryRetriever(dense=dense, anthropic_client=client)
        result = await retriever.retrieve("q", top_k=5)
        # Falls back to original query only — 1 retrieve call
        assert dense.retrieve.call_count == 1

    @pytest.mark.asyncio
    async def test_metadata_contains_variants(self):
        dense = _mock_dense()
        client = self._mock_anthropic(["v1", "v2"])
        retriever = MultiQueryRetriever(dense=dense, n_variants=2, anthropic_client=client)
        result = await retriever.retrieve("original", top_k=5)
        assert "variants" in result.metadata
        assert "original" in result.metadata["variants"]


# ---------------------------------------------------------------------------
# ParentChildRetriever
# ---------------------------------------------------------------------------

class TestWordChunks:
    def test_single_chunk_under_limit(self):
        text = "hello world"
        chunks = _word_chunks(text, 10)
        assert chunks == ["hello world"]

    def test_splits_at_word_boundary(self):
        words = ["w"] * 10
        text = " ".join(words)
        chunks = _word_chunks(text, 4)
        assert len(chunks) == 3  # 4 + 4 + 2

    def test_empty_text(self):
        assert _word_chunks("", 5) == [""]


class TestParentChildRetriever:
    def _mock_dense_with_children(self):
        dense = AsyncMock()
        child_docs = [
            RetrievedDoc(
                doc_id="doc1_p0_c0",
                score=0.9,
                text="child text",
                rank=1,
                metadata={"parent_id": "doc1", "parent_text": "full parent text"},
            ),
            RetrievedDoc(
                doc_id="doc1_p0_c1",  # same parent as above
                score=0.7,
                text="another child",
                rank=2,
                metadata={"parent_id": "doc1", "parent_text": "full parent text"},
            ),
            RetrievedDoc(
                doc_id="doc2_p0_c0",
                score=0.5,
                text="doc2 child",
                rank=3,
                metadata={"parent_id": "doc2", "parent_text": "doc2 parent text"},
            ),
        ]
        dense.retrieve = AsyncMock(return_value=_result(child_docs))
        dense.warm_up = AsyncMock()
        dense.upsert_documents = AsyncMock()
        return dense

    @pytest.mark.asyncio
    async def test_deduplicates_children_to_parents(self):
        dense = self._mock_dense_with_children()
        retriever = ParentChildRetriever(dense=dense)
        result = await retriever.retrieve("q", top_k=5)
        # Two children of doc1 should collapse to one parent
        parent_ids = [d.doc_id for d in result.docs]
        assert parent_ids.count("doc1") == 1
        assert "doc2" in parent_ids

    @pytest.mark.asyncio
    async def test_best_child_score_used_for_parent(self):
        dense = self._mock_dense_with_children()
        retriever = ParentChildRetriever(dense=dense)
        result = await retriever.retrieve("q", top_k=5)
        doc1 = next(d for d in result.docs if d.doc_id == "doc1")
        assert doc1.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_returns_parent_text(self):
        dense = self._mock_dense_with_children()
        retriever = ParentChildRetriever(dense=dense)
        result = await retriever.retrieve("q", top_k=5)
        doc1 = next(d for d in result.docs if d.doc_id == "doc1")
        assert doc1.text == "full parent text"

    @pytest.mark.asyncio
    async def test_upsert_creates_child_chunks(self):
        dense = AsyncMock()
        dense.warm_up = AsyncMock()
        dense.upsert_documents = AsyncMock()
        retriever = ParentChildRetriever(dense=dense)
        docs = [{"id": "p1", "text": " ".join(["word"] * 300)}]
        await retriever.upsert_documents(docs)
        dense.upsert_documents.assert_called_once()
        indexed = dense.upsert_documents.call_args[0][0]
        # 300 words → multiple 128-word children
        assert len(indexed) > 1
        # Every child must have parent_id and parent_text
        for child in indexed:
            assert child["parent_id"] == "p1"
            assert "parent_text" in child


# ---------------------------------------------------------------------------
# BGEReranker
# ---------------------------------------------------------------------------

class TestBGEReranker:
    def _make_docs(self) -> list[RetrievedDoc]:
        return [
            RetrievedDoc(doc_id="d1", score=0.9, text="highly relevant", rank=1),
            RetrievedDoc(doc_id="d2", score=0.8, text="somewhat relevant", rank=2),
            RetrievedDoc(doc_id="d3", score=0.7, text="not relevant", rank=3),
        ]

    @pytest.mark.asyncio
    async def test_reorders_by_cross_encoder_score(self):
        reranker = BGEReranker()
        # Inject a mock model that reverses the order
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.1, 0.5, 0.9])  # d3 is now best
        reranker._model = mock_model
        docs = self._make_docs()
        result = await reranker.rerank("q", docs, top_n=3)
        assert result[0].doc_id == "d3"
        assert result[1].doc_id == "d2"
        assert result[2].doc_id == "d1"

    @pytest.mark.asyncio
    async def test_respects_top_n(self):
        reranker = BGEReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.9, 0.8, 0.7])
        reranker._model = mock_model
        result = await reranker.rerank("q", self._make_docs(), top_n=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        reranker = BGEReranker()
        result = await reranker.rerank("q", [], top_n=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_score_in_metadata(self):
        reranker = BGEReranker()
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.42, 0.21])
        reranker._model = mock_model
        docs = self._make_docs()[:2]
        result = await reranker.rerank("q", docs, top_n=2)
        assert "rerank_score" in result[0].metadata


# ---------------------------------------------------------------------------
# CohereReranker
# ---------------------------------------------------------------------------

class TestCohereReranker:
    def _mock_cohere_response(self, order: list[int], scores: list[float]):
        results = []
        for idx, score in zip(order, scores):
            r = MagicMock()
            r.index = idx
            r.relevance_score = score
            results.append(r)
        response = MagicMock()
        response.results = results
        return response

    @pytest.mark.asyncio
    async def test_maps_response_to_docs(self):
        reranker = CohereReranker(api_key="test")
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(
            return_value=self._mock_cohere_response([1, 0], [0.95, 0.80])
        )
        reranker._client = mock_client
        docs = [
            RetrievedDoc(doc_id="d0", score=0.9, text="doc 0", rank=1),
            RetrievedDoc(doc_id="d1", score=0.8, text="doc 1", rank=2),
        ]
        result = await reranker.rerank("q", docs, top_n=2)
        # Response index 1 → d1 should be rank 1
        assert result[0].doc_id == "d1"
        assert result[0].score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self):
        reranker = CohereReranker(api_key="test")
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(side_effect=RuntimeError("API down"))
        reranker._client = mock_client
        docs = [RetrievedDoc(doc_id="d0", score=0.9, text="t", rank=1)]
        result = await reranker.rerank("q", docs, top_n=1)
        # Falls back to original order
        assert result[0].doc_id == "d0"


# ---------------------------------------------------------------------------
# DenseRetriever — doc_id regression tests (critical benchmark correctness)
# ---------------------------------------------------------------------------

class TestDenseRetrieverDocId:
    """Verify that retrieve() returns the original string doc_id, not the numeric hash.

    Bug: hit.id is the stable_qdrant_id integer (e.g. 8234…), but qrels use
    the original string ID (e.g. "MED-10"). Without this fix nDCG is 0 for
    any corpus with non-integer document IDs.
    """

    def _make_qdrant_hit(self, numeric_id: int, doc_id: str, text: str, score: float):
        hit = MagicMock()
        hit.id = numeric_id
        hit.score = score
        hit.payload = {"doc_id": doc_id, "text": text}
        return hit

    @pytest.mark.asyncio
    async def test_retrieve_uses_payload_doc_id(self):
        retriever = DenseRetriever()
        mock_client = AsyncMock()
        hit = self._make_qdrant_hit(823456789, "MED-10", "medical text", 0.9)
        mock_client.search = AsyncMock(return_value=[hit])
        retriever._client = mock_client
        retriever._model = MagicMock()
        retriever._model.encode = MagicMock(return_value=__import__("numpy").zeros(384))

        with patch.object(retriever, "_encode", AsyncMock(return_value=[0.0] * 384)):
            result = await retriever.retrieve("query", top_k=1)

        assert result.docs[0].doc_id == "MED-10", (
            f"Expected 'MED-10', got '{result.docs[0].doc_id}' — "
            "retrieve() must use payload['doc_id'], not str(hit.id)"
        )

    @pytest.mark.asyncio
    async def test_retrieve_by_embedding_uses_payload_doc_id(self):
        retriever = DenseRetriever()
        mock_client = AsyncMock()
        hit = self._make_qdrant_hit(99887766, "MED-42", "another text", 0.85)
        mock_client.search = AsyncMock(return_value=[hit])
        retriever._client = mock_client

        result = await retriever.retrieve_by_embedding([0.0] * 384, top_k=1)
        assert result[0].doc_id == "MED-42"

    @pytest.mark.asyncio
    async def test_retrieve_falls_back_to_str_id_when_no_payload(self):
        retriever = DenseRetriever()
        mock_client = AsyncMock()
        hit = MagicMock()
        hit.id = 12345
        hit.score = 0.7
        hit.payload = None
        mock_client.search = AsyncMock(return_value=[hit])
        retriever._client = mock_client

        with patch.object(retriever, "_encode", AsyncMock(return_value=[0.0] * 384)):
            result = await retriever.retrieve("q", top_k=1)

        assert result.docs[0].doc_id == "12345"

    def test_warm_up_is_idempotent(self):
        """warm_up called twice must not reload the model (checks the guard condition)."""
        retriever = DenseRetriever()
        sentinel = MagicMock()
        retriever._model = sentinel
        retriever._client = MagicMock()

        import asyncio
        asyncio.run(retriever.warm_up())  # second call with model already set

        assert retriever._model is sentinel, "warm_up() overwrote the existing model"
