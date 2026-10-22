"""Shared fixtures for retrieval tests."""

from __future__ import annotations

SAMPLE_DOCS = [
    {"id": "d1", "text": "The quick brown fox jumps over the lazy dog."},
    {"id": "d2", "text": "Machine learning models require large amounts of training data."},
    {"id": "d3", "text": "Information retrieval systems rank documents by relevance."},
    {"id": "d4", "text": "Neural networks are the foundation of modern AI."},
    {"id": "d5", "text": "BM25 is a bag-of-words retrieval function used in search engines."},
]

SAMPLE_QRELS = {
    "q1": {"d3": 2, "d5": 1},
    "q2": {"d2": 2, "d4": 1},
}

SAMPLE_RESULTS = {
    "q1": {"d3": 0.95, "d5": 0.80, "d1": 0.60, "d2": 0.40, "d4": 0.20},
    "q2": {"d2": 0.90, "d4": 0.85, "d1": 0.50, "d5": 0.30, "d3": 0.10},
}
