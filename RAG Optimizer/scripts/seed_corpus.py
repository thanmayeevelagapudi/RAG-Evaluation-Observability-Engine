"""Download NFCorpus from BEIR and seed both Qdrant and Elasticsearch."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from retrieval.strategies.bm25 import BM25Retriever
from retrieval.strategies.dense import DenseRetriever
from retrieval.strategies.parent_child import ParentChildRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET = "nfcorpus"
DATA_DIR = Path("data")
QRELS_OUT = DATA_DIR / "qrels.json"
BATCH_SIZE = 256


async def seed() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s …", DATASET)
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
        f"/{DATASET}.zip"
    )
    data_path = util.download_and_unzip(url, str(DATA_DIR))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    log.info(
        "Corpus: %d docs | Queries: %d | Qrels: %d",
        len(corpus),
        len(queries),
        len(qrels),
    )

    QRELS_OUT.write_text(json.dumps(qrels, indent=2))
    log.info("Saved qrels → %s", QRELS_OUT)

    docs = [
        {"id": doc_id, "text": doc["text"], "title": doc.get("title", "")}
        for doc_id, doc in corpus.items()
    ]

    dense = DenseRetriever()
    bm25 = BM25Retriever()

    log.info("Warming up retrievers …")
    await asyncio.gather(dense.warm_up(), bm25.warm_up())

    log.info(
        "Indexing %d documents into Qdrant (batches of %d) …", len(docs), BATCH_SIZE
    )
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        await dense.upsert_documents(batch)
        if i % (BATCH_SIZE * 4) == 0:
            log.info("  dense: %d / %d", min(i + BATCH_SIZE, len(docs)), len(docs))

    log.info("Indexing %d documents into Elasticsearch …", len(docs))
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        await bm25.upsert_documents(batch)
        if i % (BATCH_SIZE * 4) == 0:
            log.info("  bm25: %d / %d", min(i + BATCH_SIZE, len(docs)), len(docs))

    # parent_child searches a separate child-chunk collection that only this step populates.
    parent_child = ParentChildRetriever(dense=DenseRetriever(collection="rag_dense_children"))
    await parent_child.warm_up()
    log.info("Indexing %d documents as parent/child chunks into Qdrant …", len(docs))
    for i in range(0, len(docs), BATCH_SIZE):
        await parent_child.upsert_documents(docs[i : i + BATCH_SIZE])
    await parent_child.close()

    await bm25.close()
    log.info("Seeding complete.")


if __name__ == "__main__":
    asyncio.run(seed())
