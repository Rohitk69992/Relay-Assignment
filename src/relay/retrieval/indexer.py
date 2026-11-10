"""Lexical BM25 indexer for document chunks with scientific notation normalization."""

import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.relay.schemas import DocumentChunk


def normalize_scientific_notation(text: str) -> str:
    """Normalizes scientific notation and mathematical symbols across queries and passages."""
    t = text
    # Normalize square root variants
    t = t.replace("√", " sqrt ")
    t = re.sub(r'\bsquare\s+root\b', 'sqrt', t, flags=re.IGNORECASE)
    # Normalize subscripted dimensions to include both concatenated and separated forms (e.g. d_k -> dk d_k)
    t = re.sub(r'\bd_([a-zA-Z0-9]+)\b', r'd\1 d_\1', t, flags=re.IGNORECASE)
    return t


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens with mathematical normalization."""
    normalized = normalize_scientific_notation(text)
    return re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', normalized.lower())


class DocumentIndex:
    """Manages an in-memory BM25 index over ingested document chunks."""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
